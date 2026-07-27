import abc
import random
import torch
from transformers import AutoTokenizer


class BaseTokenizer(abc.ABC):
    """
    Abstract base class defining the interchangeable tokenizer interface.
    """

    @abc.abstractmethod
    def get_length(self, prompt: str) -> int:
        """
        Calculate length of a single prompt for stats and tiers.
        """
        pass

    @abc.abstractmethod
    def encode(
        self,
        prompt: str,
        max_len: int,
        cfg_dropout_prob: float = 0.0,
        tag_dropout_prob: float = 0.0,
        shuffle_tags: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a prompt string into a token ID tensor and an attention mask.
        """
        pass

    @property
    @abc.abstractmethod
    def vocab(self) -> dict:
        """
        Return the vocabulary mapping.
        """
        pass


class CommaSeparatedTokenizer(BaseTokenizer):
    """
    Comma-separated tag vocabulary tokenizer.
    """

    def __init__(self, vocab: dict = None):
        self._vocab = vocab if vocab is not None else {"<pad>": 0, "<unk>": 1}
        self.pad_id = self._vocab.get("<pad>", 0)
        self.unk_id = self._vocab.get("<unk>", 1)

    def build_vocab(self, prompts: list[str]) -> None:
        """
        Build custom vocabulary dynamically from raw prompts.
        """
        self._vocab = {"<pad>": 0, "<unk>": 1}
        for prompt in prompts:
            tags = [t.strip() for t in prompt.split(",") if t.strip()]
            for tag in tags:
                if tag not in self._vocab:
                    self._vocab[tag] = len(self._vocab)
        self.pad_id = self._vocab.get("<pad>", 0)
        self.unk_id = self._vocab.get("<unk>", 1)

    @property
    def vocab(self) -> dict:
        return self._vocab

    def get_length(self, prompt: str) -> int:
        tags = [t.strip() for t in prompt.split(",") if t.strip()]
        return len(tags)

    def encode(
        self,
        prompt: str,
        max_len: int,
        cfg_dropout_prob: float = 0.0,
        tag_dropout_prob: float = 0.0,
        shuffle_tags: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cfg_dropout_prob > 0.0 and random.random() < cfg_dropout_prob:
            tags = []
        else:
            tags = [t.strip() for t in prompt.split(",") if t.strip()]
            if len(tags) > 5:
                first_tags = tags[:5]
                middle_tags = tags[5:]

                if tag_dropout_prob > 0.0:
                    middle_tags = [
                        t for t in middle_tags if random.random() >= tag_dropout_prob
                    ]

                if shuffle_tags:
                    random.shuffle(middle_tags)

                tags = first_tags + middle_tags

        ids = [self._vocab.get(tag, self.unk_id) for tag in tags]
        ids = ids[:max_len]
        padded_ids = ids + [self.pad_id] * (max_len - len(ids))

        tokens_tensor = torch.tensor(padded_ids, dtype=torch.long)

        # Attention Mask
        not_pad_mask = tokens_tensor != self.pad_id
        shifted_mask = torch.roll(not_pad_mask, shifts=1, dims=0)
        shifted_mask[0] = True
        attention_mask = not_pad_mask | shifted_mask

        return tokens_tensor, attention_mask


class HFLLMTokenizer(BaseTokenizer):
    """
    HuggingFace pretrained LLM tokenizer adapter.
    """

    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
        self.pad_id = self.tokenizer.pad_token_id

    @property
    def vocab(self) -> dict:
        # there is no need to save it as defined by llm
        return None

    def get_length(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=True))

    def encode(
        self,
        prompt: str,
        max_len: int,
        cfg_dropout_prob: float = 0.0,
        tag_dropout_prob: float = 0.0,
        shuffle_tags: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cfg_dropout_prob > 0.0 and random.random() < cfg_dropout_prob:
            processed_prompt = ""
        else:
            tags = [t.strip() for t in prompt.split(",") if t.strip()]
            if len(tags) > 5:
                first_tags = tags[:5]
                middle_tags = tags[5:]

                if tag_dropout_prob > 0.0:
                    middle_tags = [
                        t for t in middle_tags if random.random() >= tag_dropout_prob
                    ]

                if shuffle_tags:
                    random.shuffle(middle_tags)

                tags = first_tags + middle_tags
            processed_prompt = ", ".join(tags)

        encoded = self.tokenizer(
            processed_prompt,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        tokens_tensor = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0).bool()

        return tokens_tensor, attention_mask
