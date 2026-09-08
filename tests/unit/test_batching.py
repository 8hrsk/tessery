import numpy as np

from metal_inference.batching import length_batches


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
