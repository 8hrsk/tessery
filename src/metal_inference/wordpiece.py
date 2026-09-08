"""Bounded BERT normalization and greedy WordPiece from tokenizer data."""

import unicodedata as ud
from typing import Any

import numpy as np
import regex
from numpy.typing import NDArray

from .errors import InvalidInputError, UnsupportedProfileError


def _chinese(char: str) -> bool:
    code = ord(char)
    return any(
        start <= code <= end
        for start, end in (
            (0x4E00, 0x9FFF),
            (0x3400, 0x4DBF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B920, 0x2CEAF),
            (0xF900, 0xFAFF),
            (0x2F800, 0x2FA1F),
        )
    )


class WordPieceTokenizer:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            model, normalizer = config["model"], config["normalizer"]
            if (
                model["type"] != "WordPiece"
                or normalizer["type"] != "BertNormalizer"
                or config["pre_tokenizer"] != {"type": "BertPreTokenizer"}
                or model["continuing_subword_prefix"] != "##"
                or model["unk_token"] != "[UNK]"
                or type(model["max_input_chars_per_word"]) is not int
                or not 1 <= model["max_input_chars_per_word"] <= 1000
            ):
                raise UnsupportedProfileError()
            self.vocab: dict[str, int] = model["vocab"]
            if (
                not isinstance(self.vocab, dict)
                or not 5 <= len(self.vocab) <= 200000
                or any(
                    not isinstance(k, str) or not k or type(v) is not int
                    for k, v in self.vocab.items()
                )
                or set(self.vocab.values()) != set(range(len(self.vocab)))
            ):
                raise UnsupportedProfileError()
            self.vocab_size = len(self.vocab)
            self.word_limit: int = model["max_input_chars_per_word"]
            self.clean: bool = normalizer["clean_text"]
            self.chinese: bool = normalizer["handle_chinese_chars"]
            self.lowercase: bool = normalizer["lowercase"]
            strip = normalizer["strip_accents"]
            if any(
                type(flag) is not bool for flag in (self.clean, self.chinese, self.lowercase)
            ) or (strip is not None and type(strip) is not bool):
                raise UnsupportedProfileError()
            self.strip_accents: bool = self.lowercase if strip is None else strip
            self.added = {row["content"]: row["id"] for row in config["added_tokens"]}
            if set(self.added) != {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"} or any(
                row["id"] != self.vocab[row["content"]]
                or any(
                    row[key] is not False
                    for key in ("single_word", "lstrip", "rstrip", "normalized")
                )
                or row["special"] is not True
                for row in config["added_tokens"]
            ):
                raise UnsupportedProfileError()
            self.pad_id, self.unk_id, self.cls_id, self.sep_id = (
                self.added[k] for k in ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
            )
            post = config["post_processor"]
            if (
                post["type"] != "TemplateProcessing"
                or post["single"]
                != [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                ]
                or any(
                    post["special_tokens"][k]["ids"] != [self.added[k]] for k in ("[CLS]", "[SEP]")
                )
            ):
                raise UnsupportedProfileError()
            self._special = regex.compile("(" + "|".join(regex.escape(k) for k in self.added) + ")")
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            raise UnsupportedProfileError() from None

    def _normalize(self, text: str) -> str:
        parts = []
        for char in text:
            if self.clean:
                if char in ("\x00", "\ufffd") or (
                    ud.category(char) in {"Cc", "Cf", "Cs", "Co"} and char not in "\t\n\r"
                ):
                    continue
                if char.isspace():
                    char = " "
            parts.append(" " + char + " " if self.chinese and _chinese(char) else char)
        text = "".join(parts)
        if self.strip_accents:
            text = "".join(c for c in ud.normalize("NFD", text) if ud.category(c) != "Mn")
        if self.lowercase:
            text = "".join(c.lower() for c in text)
        return text

    def _pieces(self, word: str) -> list[int]:
        if len(word) > self.word_limit:
            return [self.unk_id]
        result = []
        start = 0
        while start < len(word):
            end = len(word)
            while end > start:
                piece = ("##" if start else "") + word[start:end]
                if piece in self.vocab:
                    break
                end -= 1
            if end == start:
                return [self.unk_id]
            result.append(self.vocab[piece])
            start = end
        return result

    def encode(self, text: str, *, max_length: int = 512) -> list[int]:
        if (
            not isinstance(text, str)
            or not text.strip()
            or type(max_length) is not int
            or not 2 <= max_length <= 512
        ):
            raise InvalidInputError()
        try:
            if len(text.encode("utf-8")) > 1024 * 1024:
                raise InvalidInputError()
        except UnicodeError:
            raise InvalidInputError() from None
        output: list[int] = []
        for chunk in self._special.split(text):
            if len(output) >= max_length - 2:
                break
            if chunk in self.added:
                output.append(self.added[chunk])
                continue
            word: list[str] = []
            for char in self._normalize(chunk) + " ":
                code = ord(char)
                punctuation = (
                    ud.category(char).startswith("P")
                    or 33 <= code <= 47
                    or 58 <= code <= 64
                    or 91 <= code <= 96
                    or 123 <= code <= 126
                )
                if char.isspace() or punctuation:
                    output.extend(self._pieces("".join(word)))
                    word.clear()
                    if punctuation:
                        output.extend(self._pieces(char))
                    if len(output) >= max_length - 2:
                        break
                else:
                    word.append(char)
        return [self.cls_id] + output[: max_length - 2] + [self.sep_id]

    def batch(
        self, texts: list[str], *, max_length: int
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
        encoded = [self.encode(text, max_length=max_length) for text in texts]
        lengths = np.array([len(row) for row in encoded], np.uint32)
        ids = np.full((len(texts), int(lengths.max())), self.pad_id, np.uint32)
        for i, row in enumerate(encoded):
            ids[i, : len(row)] = row
        return ids, lengths
