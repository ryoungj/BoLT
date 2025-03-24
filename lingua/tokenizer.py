# Copyright (c) Meta Platforms, Inc. and affiliates.

import abc
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import os
from enum import Enum

from sentencepiece import SentencePieceProcessor
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers import processors

logger = logging.getLogger(__name__)


class SpecialTokens(str, Enum):
    START_OF_LATENT = "<|START_OF_LATENT|>"
    END_OF_LATENT = "<|END_OF_LATENT|>"
    PRIOR_PREFIX = "<|PRIOR_PREFIX|>"
    POSTERIOR_PREFIX = "<|POSTERIOR_PREFIX|>"


@dataclass
class TokenizerArgs:
    name: str = "bytes"
    path: Optional[str] = None


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, tokens, add_bos, add_eos):
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass


class MockTokenizer(Tokenizer):
    n_words: int = 256

    def encode(self, tokens, add_bos, add_eos):
        return tokens


class ByteTokenizer(Tokenizer):
    def __init__(self):
        self.bos_id = 256
        self.eos_id = 257
        self.n_words = 258

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + list(s.encode()) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int]):
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode("utf-8", errors="backslashreplace")

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is None:
            tokens = self.encode(text)

        decoded_chars, offsets = [], []
        byte_pos = 0
        for token in tokens:
            if token < 256:
                char = bytes([token]).decode("utf-8", errors="ignore")
                if char:
                    decoded_chars.append(char)
                    offsets.append(byte_pos)
                byte_pos += len(char.encode("utf-8"))

        return decoded_chars, offsets


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert type(s) is str
        tokens = (
            [self.bos_id] * add_bos + self.sp_model.encode(s) + [self.eos_id] * add_eos
        )
        return tokens

    def decode(self, tokens: List[int]):
        return self.sp_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        pieces = self.sp_model.encode_as_immutable_proto(text).pieces
        substrs = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
        return substrs, offsets


DEFAULT_TIKTOKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
DEFAULT_TIKTOKEN_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 0,
    "<|end_of_text|>": 1,
    "<|fim_prefix|>": 2,
    "<|fim_middle|>": 3,
    "<|fim_end_fill|>": 253,
    "<|fim_pad|>": 254,
    "<|fim_suffix|>": 255,
}
TIKTOKEN_MAX_ENCODE_CHARS = 400_000


class TikTokenTokenizer(Tokenizer):

    def __init__(self, model_path: str) -> None:
        mergeable_ranks = load_tiktoken_bpe(model_path)
        all_special_tokens_with_ids = copy(DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
        missing_ids = set(range(256)) - set(all_special_tokens_with_ids.values())
        for id in missing_ids:
            all_special_tokens_with_ids[f"<|reserved_special_token_{id}|>"] = id
        for name in all_special_tokens_with_ids:
            all_special_tokens_with_ids[name] += len(mergeable_ranks)

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=DEFAULT_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=all_special_tokens_with_ids,
        )

        self.bos_id: int = self.tkt_model.encode_single_token("<|begin_of_text|>")
        self.eos_id: int = self.tkt_model.encode_single_token("<|end_of_text|>")

        self.n_words: int = self.tkt_model.n_vocab

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert isinstance(s, str)

        subs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(s[i : i + TIKTOKEN_MAX_ENCODE_CHARS])
        return (
            [self.bos_id] * add_bos
            + sum(self.tkt_model.encode_ordinary_batch(subs), start=[])
            + [self.eos_id] * add_eos
        )

    def decode(self, tokens: List[int]):
        return self.tkt_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is not None:
            token_bytes = self.tkt_model.decode_tokens_bytes(tokens)
        else:
            token_bytes = self.tkt_model.decode_tokens_bytes(
                self.tkt_model.encode(text, allowed_special="all")
            )

        text_len, offsets = 0, []
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        substrs = [text[s:e] for s, e in zip(offsets, offsets[1:] + [None])]
        return substrs, offsets


def force_support(tokenizer: PreTrainedTokenizerFast) -> None:
    """
    Hack to incorporate:

    https://github.com/huggingface/transformers/pull/31316

    Fix the bug of setting add_bos_token or add_eos_token has no effect for PreTrainedTokenizerFast (e.g., Llama-3) tokenizers
    """

    text = "a"
    tokens_default: list[int] = tokenizer(text)["input_ids"]

    # We need to initialize these correctly, not None. The reason is that if we update
    # set add_eos/bos_token later, and then reset it back to None, we'll always have
    # False-y values instead of the original behavior.
    tokenizer._add_eos_token = tokens_default[-1] == getattr(
        tokenizer, "eos_token_id", None
    )
    tokenizer._add_bos_token = tokens_default[0] == getattr(
        tokenizer, "bos_token_id", None
    )

    class _PreTrainedTokenizerFastPatched(type(tokenizer)):
        @property
        def add_eos_token(self):
            return self._add_eos_token

        @property
        def add_bos_token(self):
            return self._add_bos_token

        @add_eos_token.setter
        def add_eos_token(self, value: bool):
            self._add_eos_token = value
            self.update_post_processor()

        @add_bos_token.setter
        def add_bos_token(self, value: bool):
            self._add_bos_token = value
            self.update_post_processor()

        def update_post_processor(self):
            """
            Overwrites the underlying post processor with the current `bos_token` and
            `eos_token`.
            """
            if not isinstance(
                self._tokenizer.post_processor, processors.TemplateProcessing
            ) and not isinstance(self._tokenizer.post_processor, processors.Sequence):
                return

            bos = self.bos_token
            bos_token_id = self.bos_token_id
            if bos is None and self.add_bos_token:
                raise ValueError("add_bos_token = True but bos_token = None")

            eos = self.eos_token
            eos_token_id = self.eos_token_id
            if eos is None and self.add_eos_token:
                raise ValueError("add_eos_token = True but eos_token = None")

            single = (
                f"{(bos + ':0 ') if self.add_bos_token else ''}"
                "$A:0"
                f"{(' ' + eos + ':0') if self.add_eos_token else ''}"
            )
            pair = (
                f"{single}{(' ' + bos + ':1') if self.add_bos_token else ''} "
                "$B:1"
                f"{(' ' + eos + ':1') if self.add_eos_token else ''}"
            )

            special_tokens = []
            if self.add_bos_token:
                special_tokens.append((bos, bos_token_id))
            if self.add_eos_token:
                special_tokens.append((eos, eos_token_id))
            self._tokenizer.post_processor = processors.TemplateProcessing(
                single=single, pair=pair, special_tokens=special_tokens
            )

    # https://stackoverflow.com/questions/31590152/monkey-patching-a-property
    tokenizer.__class__ = _PreTrainedTokenizerFastPatched


def hf_add_special_tokens(tokenizer, special_tokens):
    # Get all values from the enum members
    special_tokens_list = [token.value for token in special_tokens]
    special_tokens_dict = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=True)
    return tokenizer


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        if "llama" in model_path.lower():
            tokenizer_kwargs = {"use_fast": False, "legacy": False, "add_prefix_space": False}
        else:
            tokenizer_kwargs = {}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)

        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            force_support(self.tokenizer)
        
        # BOS / EOS token IDs
        self.n_words: int = len(self.tokenizer)
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id
        self.pad_id: int = self.tokenizer.pad_token_id
        self.special_tokens = self.tokenizer.special_tokens_map
        
        logger.info(f"Loaded HuggingFace tokenizer from {model_path}")
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - Special tokens: {self.special_tokens}"
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert isinstance(s, str)

        self.tokenizer.add_bos_token = add_bos
        self.tokenizer.add_eos_token = add_eos

        tokens = self.tokenizer.encode(s)
        return tokens

    def decode(self, tokens: List[int]):
        return self.tokenizer.decode(tokens, spaces_between_special_tokens=False)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_bos=False, add_eos=False)
        offsets = [offset[0] for offset in encoding.offset_mapping]
        substrs = [text[start:end] for start, end in encoding.offset_mapping]
        return substrs, offsets
    
    def add_special_tokens(self, special_tokens: Enum):
        hf_add_special_tokens(self.tokenizer, special_tokens)
        
        # update attributes
        self.n_words = len(self.tokenizer)
        self.special_tokens = self.tokenizer.special_tokens_map
        return self


def build_tokenizer(name: str, path: Optional[str] = None) -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    elif name == "mock":
        return MockTokenizer()
    elif name == "sp":
        return SentencePieceTokenizer(path)
    elif name == "tiktoken":
        return TikTokenTokenizer(path)
    elif name == "hf":
        return HuggingFaceTokenizer(path)
    else:
        raise NotImplementedError(f"{name} tokenizer type is not implemented")
