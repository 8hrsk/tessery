import copy

import pytest

from metal_inference.errors import InvalidInputError, UnsupportedProfileError
from metal_inference.wordpiece import WordPieceTokenizer


def tiny_config():
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {
        word: i
        for i, word in enumerate(
            specials + ["hello", "world", "##s", "cafe", ",", "中", "文", "a", "##b"]
        )
    }
    return {
        "model": {
            "type": "WordPiece",
            "vocab": vocab,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
        },
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": None,
            "lowercase": True,
        },
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "added_tokens": [
            {
                "content": word,
                "id": vocab[word],
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for word in specials
        ],
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
            "special_tokens": {word: {"ids": [vocab[word]]} for word in ("[CLS]", "[SEP]")},
        },
    }


def test_normalization_wordpiece_special_tokens_and_boundaries():
    t = WordPieceTokenizer(tiny_config())
    assert t.encode("HELLO, worlds Café") == [2, 5, 9, 6, 7, 8, 3]
    assert t.encode("中文") == [2, 10, 11, 3]
    assert t.encode("hello[MASK]worlds") == [2, 5, 4, 6, 7, 3]
    assert t.encode("hello", max_length=2) == [2, 3]
    assert t.encode("hello worlds", max_length=3) == [2, 5, 3]
    assert t.encode("a" * 101) == [2, 1, 3]
    assert t.encode("a\x00b") == [2, 12, 13, 3]
    assert t.encode("\u200b") == [2, 3]
    ids, lengths = t.batch(["hello", "worlds"], max_length=512)
    assert ids.tolist() == [[2, 5, 3, 0], [2, 6, 7, 3]]
    assert lengths.tolist() == [3, 4]
    assert t._normalize("ΣΟΣ İ \ue000 \u0378") == "σοσ i  \u0378"


def test_normalizer_flags():
    config = tiny_config()
    config["normalizer"].update(
        clean_text=False, handle_chinese_chars=False, strip_accents=False, lowercase=False
    )
    t = WordPieceTokenizer(config)
    assert t._normalize("Café\x00中文") == "Café\x00中文"


@pytest.mark.parametrize(
    "text,limit",
    [
        ("", 512),
        (" ", 512),
        ("\ud800", 512),
        ("x" * 1048577, 512),
        ("a", 1),
        ("a", True),
        ("a", 513),
    ],
)
def test_input_rejection(text, limit):
    with pytest.raises(InvalidInputError):
        WordPieceTokenizer(tiny_config()).encode(text, max_length=limit)


@pytest.mark.parametrize(
    "path,value",
    [
        (("model", "type"), "BPE"),
        (("model", "continuing_subword_prefix"), "@@"),
        (("model", "max_input_chars_per_word"), True),
        (("model", "max_input_chars_per_word"), 1001),
        (("model", "vocab"), {"[UNK]": 1}),
        (("normalizer", "lowercase"), 1),
        (("normalizer", "strip_accents"), "true"),
        (("added_tokens", 0, "lstrip"), True),
        (("added_tokens", 0, "id"), 9),
        (("post_processor", "single"), []),
        (("post_processor", "special_tokens", "[CLS]", "ids"), [9]),
    ],
)
def test_unsupported_profiles(path, value):
    config = copy.deepcopy(tiny_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(UnsupportedProfileError):
        WordPieceTokenizer(config)


def test_invalid_config():
    with pytest.raises(UnsupportedProfileError):
        WordPieceTokenizer({})


@pytest.mark.parametrize("operation", ["normalize", "pieces", "batch"])
def test_cancel_inside_wordpiece_and_recover(operation):
    from metal_inference.errors import CanceledError

    tokenizer = WordPieceTokenizer(tiny_config())
    checks = 0

    def canceled():
        nonlocal checks
        checks += 1
        return checks == 3

    with pytest.raises(CanceledError):
        if operation == "normalize":
            tokenizer._normalize("Café " * 10000, canceled)
        elif operation == "pieces":
            tokenizer._pieces("a" + "b" * 90, canceled)
        else:
            tokenizer.batch(["hello " * 10000], max_length=512, canceled=canceled)
    assert checks == 3
    assert tokenizer.encode("hello") == [2, 5, 3]
