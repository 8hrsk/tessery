"""Local Qwen byte-level BPE, independently implemented from tokenizer.json data.

Only the registered NFC/Split/ByteLevel/BPE/template profile is supported. There
is no Hugging Face runtime, remote-code hook, network path or persistent cache.
"""

import heapq
import unicodedata
from typing import Any

import numpy as np
import regex
from numpy.typing import NDArray

from .cancellation import CancelCheck, checkpoint
from .errors import InvalidInputError, UnsupportedProfileError

QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def validate_qwen_profile(config: dict[str, Any]) -> None:
    """Reject tokenizer behavior the adapter does not implement, before constructing it."""
    byte_level = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }
    try:
        if config["pre_tokenizer"] != {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": QWEN_SPLIT},
                    "behavior": "Isolated",
                    "invert": False,
                },
                byte_level,
            ],
        }:
            raise UnsupportedProfileError()
        model = config["model"]
        for name, expected in {
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
        }.items():
            if model.get(name, expected) != expected:
                raise UnsupportedProfileError()
        post = config["post_processor"]
        if (
            post["type"] != "Sequence"
            or len(post["processors"]) != 2
            or post["processors"][0] != byte_level
        ):
            raise UnsupportedProfileError()
        template = post["processors"][1]
        if template["type"] != "TemplateProcessing" or template["single"] != [
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}},
        ]:
            raise UnsupportedProfileError()
        suffix = template["special_tokens"]["<|endoftext|>"]["ids"]
        added = {row["content"]: row["id"] for row in config["added_tokens"]}
        if suffix != [added["<|endoftext|>"]]:
            raise UnsupportedProfileError()
    except (KeyError, TypeError, ValueError, IndexError):
        raise UnsupportedProfileError() from None


def byte_alphabet() -> tuple[str, ...]:
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    extra = 256
    result = []
    for byte in range(256):
        if byte in visible:
            result.append(chr(byte))
        else:
            result.append(chr(extra))
            extra += 1
    return tuple(result)


class QwenTokenizer:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            if config["normalizer"] != {"type": "NFC"} or config["model"]["type"] != "BPE":
                raise UnsupportedProfileError()
            self.vocab: dict[str, int] = config["model"]["vocab"]
            if (
                not isinstance(self.vocab, dict)
                or not 256 <= len(self.vocab) <= 200000
                or any(
                    not isinstance(k, str) or not k or type(v) is not int or not 0 <= v < 200000
                    for k, v in self.vocab.items()
                )
            ):
                raise UnsupportedProfileError()
            self.ranks = {tuple(pair): i for i, pair in enumerate(config["model"]["merges"])}
            self._split = regex.compile(
                config["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
            )
            self.added = {row["content"]: row["id"] for row in config["added_tokens"]}
            if not self.added or any(
                not isinstance(k, str) or not k or type(v) is not int or not 0 <= v < 200000
                for k, v in self.added.items()
            ):
                raise UnsupportedProfileError()
            # Added tokens are matched before normalization. All flags in this
            # registered pack are false; they are not approximated for other packs.
            if any(
                row[key]
                for row in config["added_tokens"]
                for key in ("single_word", "lstrip", "rstrip", "normalized")
            ):
                raise UnsupportedProfileError()
            self._special = regex.compile(
                "("
                + "|".join(regex.escape(x) for x in sorted(self.added, key=len, reverse=True))
                + ")"
            )
            self.suffix_id: int = config["post_processor"]["processors"][1]["special_tokens"][
                "<|endoftext|>"
            ]["ids"][0]
            self.pad_id = self.suffix_id
            self.vocab_size = max((*self.vocab.values(), *self.added.values())) + 1
            if set(self.vocab.values()) | set(self.added.values()) != set(range(self.vocab_size)):
                raise UnsupportedProfileError()
            self._alphabet = byte_alphabet()
        except (KeyError, TypeError, ValueError, IndexError):
            raise UnsupportedProfileError() from None

    def _bpe(self, piece: str, canceled: CancelCheck = None) -> list[int]:
        checkpoint(canceled)
        raw = piece.encode("utf-8")
        if len(raw) > 65536:
            raise InvalidInputError()
        symbols = [self._alphabet[b] for b in raw]
        if not symbols:
            return []
        size = len(symbols)
        previous = list(range(-1, size - 1))
        following = list(range(1, size + 1))
        version = [0] * size
        alive = [True] * size
        pending: list[tuple[int, int, int, int, int]] = []

        def enqueue(left: int) -> None:
            right = following[left]
            if right < size:
                rank = self.ranks.get((symbols[left], symbols[right]))
                if rank is not None:
                    heapq.heappush(pending, (rank, left, right, version[left], version[right]))

        for left in range(size - 1):
            if left % 256 == 0:
                checkpoint(canceled)
            enqueue(left)
        operations = 0
        while pending:
            if operations % 256 == 0:
                checkpoint(canceled)
            operations += 1
            _, left, right, lv, rv = heapq.heappop(pending)
            if (
                not alive[left]
                or not alive[right]
                or following[left] != right
                or version[left] != lv
                or version[right] != rv
            ):
                continue
            symbols[left] += symbols[right]
            alive[right] = False
            version[left] += 1
            following[left] = following[right]
            if following[right] < size:
                previous[following[right]] = left
            if previous[left] >= 0:
                enqueue(previous[left])
            enqueue(left)
        return [self.vocab[symbols[i]] for i in range(size) if alive[i]]

    def encode(
        self, text: str, *, max_length: int = 512, canceled: CancelCheck = None
    ) -> list[int]:
        checkpoint(canceled)
        if not isinstance(text, str) or not text.strip() or not 1 <= max_length <= 512:
            raise InvalidInputError()
        output: list[int] = []
        try:
            # Preserve the special-token template even at the truncation boundary.
            for chunk in self._special.split(text):
                checkpoint(canceled)
                if chunk in self.added:
                    output.append(self.added[chunk])
                else:
                    normalized = unicodedata.normalize("NFC", chunk)
                    for match in self._split.finditer(normalized):
                        output.extend(self._bpe(match.group(), canceled))
                        if len(output) >= max_length - 1:
                            break
                if len(output) >= max_length - 1:
                    break
        except (UnicodeError, KeyError):
            raise InvalidInputError() from None
        return output[: max_length - 1] + [self.suffix_id]

    def batch(
        self, texts: list[str], *, max_length: int, canceled: CancelCheck = None
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
        encoded = [self.encode(text, max_length=max_length, canceled=canceled) for text in texts]
        lengths = np.array([len(ids) for ids in encoded], dtype=np.uint32)
        ids = np.full((len(encoded), int(lengths.max())), self.pad_id, dtype=np.uint32)
        for row, tokens in enumerate(encoded):
            ids[row, : len(tokens)] = tokens
        return ids, lengths
