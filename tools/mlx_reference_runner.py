"""Development-only compilation boundary; no tokenizer or NumPy work is traced."""

import threading
import time

import numpy as np


class ReferenceRunner:
    def __init__(self, mx, *, compiled=False):
        self.mx = mx
        self.compiled = compiled
        self.functions = {}
        self.events = []
        self.traces = 0

    def run(self, graph, ids, lengths, mask, dimensions):
        if not self.compiled:
            result = graph(ids, lengths, mask, dimensions)
            self.mx.eval(result)
            return np.array(result)
        dynamic_mask = mask is not None and not isinstance(mask, str)
        key = (
            threading.get_ident(),
            tuple(ids.shape),
            str(ids.dtype),
            tuple(lengths.shape),
            str(lengths.dtype),
            (tuple(mask.shape), str(mask.dtype)) if dynamic_mask else mask,
            dimensions,
        )
        cold = key not in self.functions
        if cold:
            # One function per static shape/mask kind. Array values never enter the key.
            if dynamic_mask:

                def traced(token_ids, token_lengths, attention_mask):
                    self.traces += 1
                    return graph(token_ids, token_lengths, attention_mask, dimensions)
            else:

                def traced(token_ids, token_lengths):
                    self.traces += 1
                    return graph(token_ids, token_lengths, mask, dimensions)

            self.functions[key] = self.mx.compile(traced)
        before = self.traces
        started = time.perf_counter()
        fn = self.functions[key]
        result = fn(ids, lengths, mask) if dynamic_mask else fn(ids, lengths)
        self.mx.eval(result)
        output = np.array(result)
        elapsed = time.perf_counter() - started
        if cold:
            self.events.append(
                {
                    "ids_shape": list(ids.shape),
                    "thread_id": threading.get_ident(),
                    "dimensions": dimensions,
                    "mask_kind": "dynamic_additive" if dynamic_mask else mask,
                    "first_call_seconds": elapsed,
                    "traces": self.traces - before,
                }
            )
        elif self.traces != before:
            raise AssertionError("MLX retraced an already-warm shape")
        return output

    def diagnostics(self):
        return {
            "mode": "compiled" if self.compiled else "uncompiled",
            "specializations": len(self.functions),
            "python_traces": self.traces,
            "first_calls": list(self.events),
            "cold_note": (
                "First evaluation includes tracing, compilation and execution; "
                "not pure compile time"
            ),
        }

    def clear(self):
        self.functions.clear()
