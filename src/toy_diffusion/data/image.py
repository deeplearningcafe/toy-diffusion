import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from PIL import Image
from torchvision.transforms import InterpolationMode, v2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


class TieredBatchSampler(Sampler):
    def __init__(self, tiers, batch_size, drop_last=False, generator=None):
        self.tiers = tiers
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        batches = []
        for tier_max_len, indices in self.tiers.items():
            indices_copy = list(indices)

            if self.generator is not None:
                rand_idx = torch.randperm(
                    len(indices_copy), generator=self.generator
                ).tolist()
                indices_copy = [indices_copy[i] for i in rand_idx]
            else:
                random.shuffle(indices_copy)

            for i in range(0, len(indices_copy), self.batch_size):
                batch = indices_copy[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.generator is not None:
            rand_idx = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[i] for i in rand_idx]
        else:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        length = 0
        for indices in self.tiers.values():
            if self.drop_last:
                length += len(indices) // self.batch_size
            else:
                length += (len(indices) + self.batch_size - 1) // self.batch_size
        return length


class ImageDataset(Dataset):
    """
    Pytorch Dataset for loading images
    """

    def __init__(
        self,
        root_dir: str | Path,
        dtype=torch.float32,
        num_workers: int = 4,
        resize_dim: int = None,
        load_into_ram: bool = True,
        conditional: bool = False,
        is_latents: bool = False,
        vae_scale: float = 1.0,
        vae_shift: float = 0.0,
        compute_normalization: bool = False,
        exclude_tags: list = [],
        is_finetune: bool = False,
        finetune_orig_ratio: float = 0.05,
        shuffle_tags: bool = False,
        cfg_dropout_prob: float = 0.0,
        tag_dropout_prob: float = 0.0,
        use_short_prompts: bool = False,
        tiers_len: list = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.num_workers = num_workers
        self.load_into_ram = load_into_ram
        self.conditional = conditional
        self.is_latents = is_latents
        self.vae_scale = vae_scale
        self.vae_shift = vae_shift
        self.compute_normalization = compute_normalization
        self.exclude_tags = exclude_tags
        self.is_finetune = is_finetune
        self.finetune_orig_ratio = finetune_orig_ratio
        self.shuffle_tags = shuffle_tags
        self.cfg_dropout_prob = cfg_dropout_prob
        self.tag_dropout_prob = tag_dropout_prob
        self.use_short_prompts = use_short_prompts
        self.tiers_len = tiers_len

        print(
            f"Using shuffling {self.shuffle_tags}, cfg prob: {self.cfg_dropout_prob} and tag prob: {self.tag_dropout_prob}"
        )

        # computing normalization from dataset
        if self.compute_normalization and self.is_latents:
            self.vae_scale = 1.0
            self.vae_shift = 0.0

        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"H5 root directory not found: {self.root_dir}")

        if not self.is_latents:
            # they support cuda, so we could transform to tensor and operate on cuda
            # also transfers using uint8 are cheaper
            transforms = [
                v2.PILToTensor(),
                v2.ToDtype(dtype, scale=True),
            ]
            if resize_dim is not None:
                transforms.append(
                    v2.Resize(
                        size=(resize_dim, resize_dim),
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=True,
                    )
                )
            transforms.append(v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            self.transform = v2.Compose(transforms)
        else:
            self.transform = None

        self.img_paths = self._scan_directory_images()

        # Filter paths based on tags before loading to RAM
        self.img_paths = self._filter_paths(self.img_paths)

        # Check if text files exist
        if self.conditional:
            has_txt = any(
                p.with_name(f"{p.stem}_short.txt").exists()
                or p.with_suffix(".txt").exists()
                for p in self.img_paths
            )
            if not has_txt and len(self.img_paths) > 0:
                raise FileNotFoundError(
                    "Conditional is True but no .txt or _short.txt files were found."
                )

        if self.load_into_ram:
            self.tensors_list, self.img_paths = self._load_to_ram()

        if self.conditional:
            self._build_vocab()

        if self.compute_normalization and self.is_latents:
            self._compute_latents_factors()

    def _scan_directory_images(self):
        return [
            p for p in self.root_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        ]

    def _get_prompt(self, p: Path) -> str:
        """Helper to read text prompt for a given file path."""
        short_txt = p.with_name(f"{p.stem}_short.txt")
        standard_txt = p.with_suffix(".txt")
        txt_path = (
            short_txt
            if (self.use_short_prompts and short_txt.exists())
            else standard_txt
        )
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def _load_latent_tensor(self, p: Path):
        """Helper to load unscaled latent tensor from .npz file."""
        npz_path = p.with_suffix(".npz")
        if npz_path.exists():
            with np.load(npz_path) as data:
                key = "latent" if "latent" in data else data.files[0]
                return torch.from_numpy(data[key]).float()
        return None

    def _filter_paths(self, all_paths):
        """
        Filters the dataset paths based on the exclude_tags list.
        Uses set operations for optimal O(1) tag lookup performance.
        """
        # empty list returns True
        if not self.exclude_tags:
            return all_paths

        exclude_set = set([t.strip().lower() for t in self.exclude_tags])
        excluded_paths = []
        normal_paths = []

        for p in all_paths:
            prompt = self._get_prompt(p).lower()
            if prompt:
                tags = set([t.strip() for t in prompt.split(",") if t.strip()])
                if not exclude_set.isdisjoint(tags):
                    excluded_paths.append(p)
                else:
                    normal_paths.append(p)
            else:
                normal_paths.append(p)

        if not self.is_finetune:
            print(
                f"Pretrain mode: Kept {len(normal_paths)} images, "
                f"Excluded {len(excluded_paths)} images."
            )
            return normal_paths
        else:
            if len(excluded_paths) == 0:
                print("Warning: No excluded paths found for finetuning!")
                return normal_paths

            # ratio = normal / (normal + excluded)
            # normal = excluded * ratio / (1 - ratio)
            ratio = self.finetune_orig_ratio
            num_normal = int(len(excluded_paths) * (ratio / (1.0 - ratio)))
            num_normal = min(num_normal, len(normal_paths))

            sampled_normal = random.sample(normal_paths, num_normal)
            final_paths = excluded_paths + sampled_normal
            random.shuffle(final_paths)

            print(
                f"Finetune mode: Using {len(excluded_paths)} excluded images "
                f"and {len(sampled_normal)} normal images."
            )
            return final_paths

    def _build_vocab(self):
        """
        Builds a simple vocabulary dictionary from the loaded prompts.
        """
        print("Building vocabulary from prompts...")
        self.vocab = {"<pad>": 0, "<unk>": 1}
        # ignore, max is always biggest tier
        self.max_seq_len = 0
        prompt_lengths = []

        if self.load_into_ram:
            prompts = [item[1] for item in self.tensors_list]
        else:
            prompts = [self._get_prompt(p) for p in self.img_paths]

        for prompt in prompts:
            tags = [t.strip() for t in prompt.split(",") if t.strip()]
            prompt_lengths.append(len(tags))
            self.max_seq_len = max(self.max_seq_len, len(tags))
            for tag in tags:
                if tag not in self.vocab:
                    self.vocab[tag] = len(self.vocab)

        print(
            f"Vocabulary size: {len(self.vocab)}, Max sequence length: {self.max_seq_len}"
        )

        lengths_arr = np.array(prompt_lengths)
        mean_len = np.mean(lengths_arr)
        median_len = np.median(lengths_arr)
        quantiles = np.quantile(lengths_arr, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

        print(f"Prompt Length Stats - Mean: {mean_len:.2f}, Median: {median_len:.2f}")
        print(f"Quantiles (25%, 50%, 75%, 90%, 95%, 99%): {quantiles}")

        if self.tiers_len:
            tier_1_boundary = self.tiers_len[0]
            tier_2_boundary = self.tiers_len[1]
        else:
            tier_1_boundary = int(median_len)
            tier_2_boundary = int(quantiles[-1])
            self.tiers_len = [tier_1_boundary, tier_2_boundary]

        # minimum sequence length for stability
        self.max_seq_len = max(16, self.tiers_len[-1])

        self.tiers = {tier_1_boundary: [], tier_2_boundary: []}
        for idx, length in enumerate(prompt_lengths):
            if length <= tier_1_boundary:
                self.tiers[tier_1_boundary].append(idx)
            else:
                self.tiers[tier_2_boundary].append(idx)

        print(f"Tier <= {tier_1_boundary}: {len(self.tiers[tier_1_boundary])} samples")
        print(f"Tier <= {tier_2_boundary}: {len(self.tiers[tier_2_boundary])} samples")

    def load_entry(self, p: Path):
        """
        Loads an image file (or latent) and applies the transforms

        Args:
            p: Path to the image file

        Returns:
            Tuple containing the data (and prompt if conditional) and path
        """
        prompt = self._get_prompt(p) if self.conditional else ""

        if self.is_latents:
            latent = self._load_latent_tensor(p)
            if latent is not None:
                latent = self.vae_scale * (latent - self.vae_shift)
                if self.conditional:
                    return (latent, prompt), p
                return latent, p
            else:
                return None, p

        _img = Image.open(p)
        img = None
        if _img.mode == "RGB":
            img = _img
        elif _img.mode == "RGBA":
            baimg = Image.new("RGB", _img.size, (255, 255, 255))
            baimg.paste(_img, (0, 0), _img)
            img = baimg
        else:
            img = _img.convert("RGB")

        if img is not None:
            img_tensor = self.transform(img)
            if self.conditional:
                return (img_tensor, prompt), p
            return img_tensor, p
        return None, p

    def _load_to_ram(self):
        """
        Loads using multiple workers the images with the transformation applied to ram
        """

        tensors_list = []
        paths_list = []

        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.load_entry, p) for p in self.img_paths]

            for future in tqdm(
                futures,
                desc="Processing images",
                total=len(self.img_paths),
                leave=False,
                ascii=True,
            ):
                data, path = future.result()
                if data is not None:
                    tensors_list.append(data)
                    paths_list.append(path)
                else:
                    print(f"Skipped: Error processing image {path.name}: ")

        if len(tensors_list) > 0:
            sample = tensors_list[0][0] if self.conditional else tensors_list[0]
            print(f"Loaded {len(tensors_list)} items.")
            print(f"Sample Stats - Shape: {sample.shape}")

            if not self.is_latents:
                print(f"Min: {sample.min():.3f}, Max: {sample.max():.3f}")
                print(f"Mean: {sample.mean():.3f}, Std: {sample.std():.3f}")

                if sample.min() > -0.05:
                    print(
                        "WARNING: Data min is > -0.05. "
                        "Normalization to [-1, 1] likely FAILED."
                    )

                if sample.min() < -1.1 or sample.max() > 1.1:
                    print("WARNING: Data outside expected [-1, 1] range.")

                if abs(sample.mean()) > 0.3:
                    print(
                        f"Info: Data mean is {sample.mean():.3f}. "
                        "Normal for bright (anime/white bg) or dark datasets."
                    )

        return tensors_list, paths_list

    def _compute_latents_factors(self, chunk_size: int = 5000):
        """
        Empirically calculates mean and std of latents using batched
        vectorized PyTorch operations and multi-threaded loading.
        """
        print("Calculating empirical statistics for latents...")
        total_items = (
            len(self.tensors_list) if self.load_into_ram else len(self.img_paths)
        )
        if total_items == 0:
            return

        num_samples = min(80000, total_items)

        count = 0
        mean = 0.0
        m2 = 0.0

        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)

            if self.load_into_ram:
                chunk_latents = [
                    self.tensors_list[j][0]
                    if self.conditional
                    else self.tensors_list[j]
                    for j in range(i, end_idx)
                ]
            else:
                chunk_paths = self.img_paths[i:end_idx]
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    results = list(executor.map(self._load_latent_tensor, chunk_paths))
                chunk_latents = [lat for lat in results if lat is not None]

            if not chunk_latents:
                continue

            chunk_tensor = torch.stack(chunk_latents).to(torch.float64)
            n_b = chunk_tensor.numel()
            mean_b = chunk_tensor.mean().item()
            m2_b = ((chunk_tensor - mean_b) ** 2).sum().item()

            if count == 0:
                count = n_b
                mean = mean_b
                m2 = m2_b
            else:
                # Chan's parallel combination update
                delta = mean_b - mean
                count_next = count + n_b
                mean = mean + delta * (n_b / count_next)
                m2 = m2 + m2_b + (delta**2) * (count * n_b / count_next)
                count = count_next

        if count > 1:
            empirical_mean = mean
            empirical_std = (m2 / (count - 1)) ** 0.5
        else:
            empirical_mean = 0.0
            empirical_std = 1.0

        self.vae_shift = empirical_mean
        self.vae_scale = 1.0 / empirical_std if empirical_std > 0 else 1.0

        print(f"Calculated Empirical Shift (Mean): {self.vae_shift:.4f}")
        print(f"Calculated Empirical Scale (1/Std): {self.vae_scale:.4f}")

        if self.load_into_ram:
            print("Applying empirical normalization to loaded latents...")
            for idx in range(len(self.tensors_list)):
                if self.conditional:
                    latent, prompt = self.tensors_list[idx]
                    norm_lat = self.vae_scale * (latent - self.vae_shift)
                    self.tensors_list[idx] = (norm_lat, prompt)
                else:
                    latent = self.tensors_list[idx]
                    norm_lat = self.vae_scale * (latent - self.vae_shift)
                    self.tensors_list[idx] = norm_lat

    def _create_attention_mask(self, prompt):
        if self.cfg_dropout_prob > 0.0 and random.random() < self.cfg_dropout_prob:
            tags = []
        else:
            tags = [t.strip() for t in prompt.split(",") if t.strip()]

            if len(tags) > 5:
                first_tags = tags[:5]
                middle_tags = tags[5:]

                if self.tag_dropout_prob > 0.0:
                    middle_tags = [
                        t
                        for t in middle_tags
                        if random.random() >= self.tag_dropout_prob
                    ]

                if self.shuffle_tags:
                    random.shuffle(middle_tags)

                tags = first_tags + middle_tags

        unk_id = self.vocab.get("<unk>", 1)
        ids = [self.vocab.get(tag, unk_id) for tag in tags]
        ids = ids[: self.max_seq_len]

        pad_id = self.vocab.get("<pad>", 0)
        padded_ids = ids + [pad_id] * (self.max_seq_len - len(ids))

        tokens_tensor = torch.tensor(padded_ids, dtype=torch.long)

        # Attention Mask
        not_pad_mask = tokens_tensor != pad_id
        shifted_mask = torch.roll(not_pad_mask, shifts=1, dims=0)
        shifted_mask[0] = True
        attention_mask = not_pad_mask | shifted_mask
        return tokens_tensor, attention_mask

    def __len__(self):
        if self.load_into_ram:
            return len(self.tensors_list)
        else:
            return len(self.img_paths)

    def __getitem__(self, idx):
        if self.load_into_ram:
            # TODO: cache instead of prompts?
            if self.conditional:
                data, prompt = self.tensors_list[idx]
                tokens_tensor, attention_mask = self._create_attention_mask(prompt)

                return data, tokens_tensor, attention_mask

            return self.tensors_list[idx]
        else:
            path = self.img_paths[idx]
            entry = self.load_entry(path)
            data = entry[0]
            if self.conditional:
                prompt = entry[1]
                tokens_tensor, attention_mask = self._create_attention_mask(prompt)

                return data, tokens_tensor, attention_mask

            return data
