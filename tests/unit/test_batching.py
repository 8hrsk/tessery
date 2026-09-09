import numpy as np
import pytest

from metal_inference.batching import execution_batches, length_batches


def test_buckets_bound_padding_preserve_identity_and_equal_length_order():
    lengths = np.array([512] + [2] * 31, np.uint32)
    groups = list(length_batches(lengths, 4096))
    assert [(rows.tolist(), width) for rows, width in groups] == [
        (list(range(1, 32)), 2),
        ([0], 512),
    ]
    assert sum(len(rows) * width for rows, width in groups) == 574
    for length_set in ([3, 8, 2, 8, 2], [512] * 32, [1, 2, 3, 4, 8, 9, 64, 128, 512]):
        values = np.array(length_set, np.uint32)
        groups = list(length_batches(values, 4096))
        assert sorted(np.concatenate([r for r, _ in groups]).tolist()) == list(range(len(values)))
        for rows, width in groups:
            assert width == max(values[rows])
            assert width <= 2 * min(values[rows])
            assert width * len(rows) <= 4096


def test_empty_and_uniform_batches():
    assert list(length_batches(np.array([], np.uint32), 4096)) == []
    groups = list(length_batches(np.ones(5, np.uint32), 2))
    assert [r.tolist() for r, _ in groups] == [[0, 1], [2, 3], [4]]


@pytest.mark.parametrize("architecture", ["qwen3_uint4", "bert_f32"])
def test_execution_padding_keeps_membership_and_all_resource_bounds(architecture):
    for maximum in (7, 31, 65, 127, 511, 512):
        for count in (1, 3, 8, 9, 32):
            for longest in range(1, maximum + 1):
                lengths = np.full(count, longest, np.uint32)
                if count > 1:
                    lengths[0] = max(1, longest // 2)
                before = list(length_batches(lengths, 4096))
                after = list(execution_batches(lengths, 4096, maximum, architecture))
                assert len(before) == len(after)
                for (old_rows, old_width), (rows, width) in zip(before, after, strict=True):
                    np.testing.assert_array_equal(rows, old_rows)
                    assert old_width <= width <= maximum
                    assert width <= 2 * int(lengths[rows].min())
                    assert width * len(rows) <= 4096
    assert list(execution_batches(np.array([], np.uint32), 4096, 512, architecture)) == []


@pytest.mark.parametrize(
    "length, count, architecture, expected",
    [
        (4, 1, "qwen3_uint4", 4),
        (7, 1, "qwen3_uint4", 8),
        (33, 8, "qwen3_uint4", 33),
        (33, 8, "bert_f32", 33),
        (63, 8, "qwen3_uint4", 64),
        (129, 1, "qwen3_uint4", 129),
        (129, 1, "bert_f32", 136),
        (511, 8, "qwen3_uint4", 512),
        (455, 9, "bert_f32", 455),
        (7, 1, "future_backend", 7),
    ],
)
def test_execution_padding_selection(length, count, architecture, expected):
    groups = list(execution_batches(np.full(count, length, np.uint32), 4096, 512, architecture))
    assert len(groups) == 1
    assert groups[0][1] == expected
