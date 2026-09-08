import pytest

from metal_inference.errors import InvalidInputError, UnsupportedProfileError
from metal_inference.tokenizer import QwenTokenizer, byte_alphabet


def tiny_config():
    vocab = {v: i for i, v in enumerate(byte_alphabet())}
    vocab.update({"ab": 256, "aba": 257})
    return {
        "normalizer": {"type": "NFC"},
        "model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"], ["ab", "a"]]},
        "pre_tokenizer": {"pretokenizers": [{"pattern": {"Regex": r".+"}}]},
        "added_tokens": [
            {
                "content": "<eot>",
                "id": 258,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            }
        ],
        "post_processor": {
            "processors": [{}, {"special_tokens": {"<|endoftext|>": {"ids": [258]}}}]
        },
    }


def test_heap_bpe_and_added_token():
    tokenizer = QwenTokenizer(tiny_config())
    assert tokenizer.encode("aba") == [257, 258]
    assert tokenizer.encode("ababa") == [256, 257, 258]
    assert tokenizer.encode("a<eot>aba") == [97, 258, 257, 258]
    assert tokenizer.encode("aba", max_length=1) == [258]
    assert tokenizer.encode("abab", max_length=2) == [256, 258]
    assert tokenizer.encode("e\u0301") == tokenizer.encode("é")
    ids, lengths = tokenizer.batch(["aba", "xaba"], max_length=512)
    assert lengths.tolist() == [2, 3]
    assert ids.tolist() == [[257, 258, 258], [120, 257, 258]]
    assert len(set(byte_alphabet())) == 256
    assert tokenizer._bpe("") == []


@pytest.mark.parametrize("text", ["", " ", "\ud800", "x" * 65537])
def test_bpe_errors(text):
    with pytest.raises(InvalidInputError):
        QwenTokenizer(tiny_config()).encode(text)


def test_unknown_tokenizer_profile():
    with pytest.raises(UnsupportedProfileError):
        QwenTokenizer({})
    config = tiny_config()
    config["normalizer"] = None
    with pytest.raises(UnsupportedProfileError):
        QwenTokenizer(config)
    config = tiny_config()
    config["added_tokens"][0]["lstrip"] = True
    with pytest.raises(UnsupportedProfileError):
        QwenTokenizer(config)
