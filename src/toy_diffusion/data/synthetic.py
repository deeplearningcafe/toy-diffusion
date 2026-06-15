import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image, ImageDraw, ImageFont


class SyntheticDataset(Dataset):
    def __init__(
        self,
        name="spiral",
        n_samples=50000,
        projection_dim=0,
        image_path: str = None,
        font_path: str = None,
    ):
        """
        Args:
            name: Dataset name (spiral, pinwheel, kanji_love, etc.)
            n_samples: Number of points to sample
            projection_dim: If > 0, projects 2D data into high-dim space
            image_path: Path for image-based datasets (nerv/asuka)
            font_path: Path to a .ttf/.otf file supporting Japanese characters
        """
        self.data = self._generate_data(name, n_samples, image_path, font_path)
        self.P = None

        # projection matrix for High-Dim embedding
        if projection_dim > 0:
            rand_mat = np.random.rand(projection_dim, 2)
            self.P, _ = np.linalg.qr(rand_mat)  # Orthogonal projection
            self.data = self.data @ self.P.T

    def _generate_data(self, name, n_samples, image_path, font_path):
        if name == "spiral":
            theta = np.sqrt(np.random.rand(n_samples)) * 4 * np.pi  # 2 turns
            r_a = 2 * theta + np.pi
            data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
            x = data_a + np.random.randn(n_samples, 2) * 0.2

            x = (x - x.mean(0)) / x.std(0)
            return x.astype(np.float32)

        elif name == "gmm":
            # 8 Gaussians in a circle
            centers = []
            for i in range(8):
                angle = 2 * np.pi * i / 8
                centers.append([np.cos(angle) * 3, np.sin(angle) * 3])
            centers = np.array(centers)

            indices = np.random.choice(8, n_samples)
            x = centers[indices] + np.random.randn(n_samples, 2) * 0.3

            x = (x - x.mean(0)) / x.std(0)
            return x.astype(np.float32)

        elif name == "gmm_finetune":
            # We keep only 2 gaussians (indices 0 and 4 - opposite sides)
            centers = []
            indices_to_keep = [0, 4]
            for i in indices_to_keep:
                angle = 2 * np.pi * i / 8
                centers.append([np.cos(angle) * 3, np.sin(angle) * 3])
            centers = np.array(centers)

            indices = np.random.choice(len(centers), n_samples)

            noise = np.random.randn(n_samples, 2)

            scale = np.array([1.0, 0.5])
            noise = noise * scale

            theta = np.radians(20)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            noise = noise @ R.T

            x = centers[indices] + noise

            # Use max std to preserve aspect ratio.
            x = x - x.mean(0)
            scale_factor = x.std(0).max()
            x = x / scale_factor

            return x.astype(np.float32)

        elif name == "gmm_imbalanced":
            # Finetuning dataset: Mode collapse test
            # 8 Gaussians in a circle, but highly imbalanced to test if
            centers = []
            for i in range(8):
                angle = 2 * np.pi * i / 8
                centers.append([np.cos(angle) * 3, np.sin(angle) * 3])
            centers = np.array(centers)

            # 1 blob has 65% probability, others 5%
            p = [0.65] + [0.35 / 7] * 7
            indices = np.random.choice(8, n_samples, p=p)
            x = centers[indices] + np.random.randn(n_samples, 2) * 0.3

            x = (x - x.mean(0)) / x.std(0)
            return x.astype(np.float32)

        elif name == "gmm_extreme_imbalanced":
            # 8 Gaussians in a circle.
            centers = []
            for i in range(8):
                angle = 2 * np.pi * i / 8
                centers.append([np.cos(angle) * 3, np.sin(angle) * 3])
            centers = np.array(centers)

            # 1% of data per blob
            p = [0.93] + [(0.07 / 7)] * 7
            indices = np.random.choice(8, n_samples, p=p)
            x = centers[indices] + np.random.randn(n_samples, 2) * 0.3

            x = (x - x.mean(0)) / x.std(0)
            return x.astype(np.float32)

        elif name == "gmm_long_tail":
            # 10 Gaussians in a line, exponentially decaying probabilities.
            num_modes = 10
            centers = []
            for i in range(num_modes):
                # Spread them evenly along the x-axis
                centers.append([i * 2.0 - (num_modes - 1), 0.0])
            centers = np.array(centers)

            # Exponential decay: p_i = 1 / 2^(i+1)
            p = np.array([1.0 / (2 ** (i + 1)) for i in range(num_modes)])
            p = p / p.sum()  # Normalize to exactly 1.0

            indices = np.random.choice(num_modes, n_samples, p=p)
            x = centers[indices] + np.random.randn(n_samples, 2) * 0.2

            x = (x - x.mean(0)) / x.std(0)
            return x.astype(np.float32)

        elif name == "pinwheel":
            # 5-arm Pinwheel distribution.
            radial_std = 0.3
            tangential_std = 0.1
            num_classes = 5
            num_per_class = n_samples // num_classes
            rate = 0.25
            rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

            features = np.random.randn(num_classes * num_per_class, 2) * np.array(
                [radial_std, tangential_std]
            )
            features[:, 0] += 1
            labels = np.repeat(np.arange(num_classes), num_per_class)

            angles = rads[labels] + rate * np.exp(features[:, 0])
            rotations = np.stack(
                [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
            )
            rotations = np.reshape(rotations.T, (-1, 2, 2))

            x = np.einsum("ti,tij->tj", features, rotations)

            perm = np.random.permutation(len(x))
            x = x[perm]

            x = (x - x.mean(0)) / x.std(0)
            return x.astype(np.float32)

        elif name == "pinwheel_finetune":
            # Finetuning task: Single arm of the pinwheel, slightly shifted/rotated.
            radial_std = 0.3
            tangential_std = 0.1

            num_classes = 1
            rate = 0.25

            features = np.random.randn(n_samples, 2) * np.array(
                [radial_std, tangential_std]
            )
            features[:, 0] += 1

            base_angle = 0.0 + np.radians(20)

            angles = base_angle + rate * np.exp(features[:, 0])
            rotations = np.stack(
                [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
            )
            rotations = np.reshape(rotations.T, (-1, 2, 2))

            x = np.einsum("ti,tij->tj", features, rotations)

            # Use max std to preserve the elongated shape
            x = x - x.mean(0)
            scale_factor = x.std(0).max()
            x = x / scale_factor

            return x.astype(np.float32)

        elif name == "kanji":
            return self._generate_kanji_data("あ", n_samples, font_path, size=512)

        elif name == "kanji_finetune":
            return self._generate_kanji_data("お", n_samples, font_path, size=512)

        elif name in ["nerv", "asuka"]:
            return self._generate_dataset_image(
                image_path=image_path, num_samples=n_samples
            )
        else:
            raise ValueError("Unknown dataset")

    def _generate_kanji_data(self, character, n_samples, font_path, size=512):
        """
        Renders a Kanji character and samples points from it.
        """
        if font_path is None:
            raise ValueError("font_path must be provided for Kanji datasets.")

        # 1. Create Canvas
        # Mode 'L' (8-bit pixels, black and white)
        image = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(image)

        # 2. Load Font
        try:
            font = ImageFont.truetype(font_path, size=int(size * 0.8))
        except OSError:
            raise OSError(f"Could not load font at {font_path}. Please check path.")

        # getbbox returns (left, top, right, bottom)
        bbox = draw.textbbox((0, 0), character, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x_pos = (size - text_w) // 2
        y_pos = (size - text_h) // 2 - (bbox[1] // 2)  # Adjust for baseline

        draw.text((x_pos, y_pos), character, font=font, fill=255)

        img_np = np.array(image)
        # Get coordinates where pixel value > threshold
        y_idxs, x_idxs = np.where(img_np > 128)

        if len(x_idxs) == 0:
            raise ValueError(
                f"No pixels found for character {character}. Check font support."
            )

        indices = np.random.choice(len(x_idxs), n_samples, replace=True)

        x_sampled = x_idxs[indices].astype(np.float32)
        y_sampled = y_idxs[indices].astype(np.float32)

        y_sampled = size - y_sampled

        # This turns the discrete grid of pixels into a continuous distribution
        x_sampled += np.random.uniform(-0.5, 0.5, size=n_samples)
        y_sampled += np.random.uniform(-0.5, 0.5, size=n_samples)

        x_sampled -= x_sampled.mean()
        y_sampled -= y_sampled.mean()

        scale = max(x_sampled.std(), y_sampled.std())
        x_sampled /= scale
        y_sampled /= scale

        return np.stack([x_sampled, y_sampled], axis=1)

    def _generate_dataset_image(self, image_path, num_samples=50000):
        """
        Processes the reference image to create a 2D point cloud dataset
        of Asuka with Mean 0 and Std 1.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not find image at {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Canny edge detection to get the outlines
        edges = cv2.Canny(gray, 100, 200)

        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        combined = cv2.addWeighted(edges, 0.7, thresh, 0.3, 0)

        y_coords, x_coords = np.where(combined > 0)

        y_coords = -y_coords

        indices = np.arange(len(x_coords))
        sampled_indices = np.random.choice(indices, size=num_samples, replace=True)

        x = x_coords[sampled_indices].astype(np.float32)
        y = y_coords[sampled_indices].astype(np.float32)

        # "smooth" the pixelated grid
        x += np.random.normal(0, 0.5, size=num_samples)
        y += np.random.normal(0, 0.5, size=num_samples)

        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)

        dataset = np.stack([x, y], axis=1)

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).to(torch.float32)
