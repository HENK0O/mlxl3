"""FP16 reserve screen with old snapshots alive; no mutable runtime cache.

Measures functional slice updates plus unchanged per-row attention against one
concat. It deliberately retains old buffers, as real speculative snapshots do.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
from pathlib import Path

from benchmark_lmhead_mb4 import conditions
from benchmark_verify_attention import compare_outputs


def check_cache(reference, candidate):
    import numpy as np

    a, b = np.asarray(reference), np.asarray(candidate)
    if (
        a.ndim != 4
        or a.shape != b.shape
        or a.shape[:2] != (1, 4)
        or a.shape[-1] != 256
        or a.size == 0
    ):
        raise ValueError("invalid KV shape")
    if (
        a.dtype != np.float16
        or b.dtype != a.dtype
        or not (np.isfinite(a).all() and np.isfinite(b).all())
    ):
        raise ValueError("invalid FP16 KV values")
    if a.tobytes() != b.tobytes():
        raise ValueError("KV snapshot bytes differ")


def validate(report, iterations):
    cells = report["cells"]
    expected = set(itertools.product([8192, 16384, 32768], [2, 3, 4]))
    if len(cells) != 9 or {(cell.get("past"), cell.get("m")) for cell in cells} != expected:
        raise ValueError("incomplete reserve matrix")
    for cell in cells:
        if (
            cell.get("bit_exact") is not True
            or cell.get("retained_prefixes") != cell["m"]
            or cell.get("old_snapshot_unchanged") is not True
            or cell.get("fork_rollback_exact") is not True
        ):
            raise ValueError("incomplete reserve parity")
        arms = cell.get("samples_ms", [])
        if len(arms) != 2 or any(len(arm) != iterations for arm in arms):
            raise ValueError("incomplete reserve samples")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for arm in arms
            for value in arm
        ):
            raise ValueError("invalid reserve timing")


def screen(report, iterations, persist):
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(20261007)
    for past, rows in itertools.product([8192, 16384, 32768], [2, 3, 4]):

        def fixture(shape):
            return mx.array(rng.normal(0, 0.3, shape).astype(np.float16))

        q = fixture((1, 24, rows, 256))
        base = [fixture((1, 4, past, 256)) for _ in range(2)]
        new = [fixture((1, 4, rows, 256)) for _ in range(2)]
        # Live original + reserved snapshots disallow unsafe buffer donation.
        storage = [
            mx.concatenate([value, mx.zeros((1, 4, 1024, 256), dtype=mx.float16)], axis=2)
            for value in base
        ]
        mx.eval(q, *base, *new, *storage)
        original = [np.asarray(value).copy() for value in storage]
        index = mx.array([past], dtype=mx.int32)

        def compute(
            reserved, q=q, storage=storage, index=index, base=base, new=new, past=past, rows=rows
        ):
            values = [
                mx.slice_update(value, update, index, axes=[2])
                if reserved
                else mx.concatenate([value, update], axis=2)
                for value, update in zip(storage if reserved else base, new)
            ]
            output = mx.concatenate(
                [
                    mx.fast.scaled_dot_product_attention(
                        q[:, :, row : row + 1, :],
                        values[0][:, :, : past + row + 1, :],
                        values[1][:, :, : past + row + 1, :],
                        scale=1 / 16,
                    )
                    for row in range(rows)
                ],
                axis=2,
            )
            return output, *values

        stock = compute(False)
        candidate = compute(True)
        mx.eval(*stock, *candidate)
        if compare_outputs(stock[0], candidate[0])["bit_exact"] is not True:
            raise ValueError("reserve changed attention output")
        for expected, actual in zip(stock[1:], candidate[1:]):
            check_cache(expected, actual[:, :, : past + rows, :])
            # A rollback changes only valid length. Forking the retained
            # snapshot must preserve the later snapshot's complete contents.
            full = np.asarray(actual).copy()
            for retained in range(1, rows + 1):
                check_cache(
                    np.asarray(expected)[:, :, : past + retained, :],
                    actual[:, :, : past + retained, :],
                )
                probe = fixture((1, 4, 1, 256))
                fork = mx.slice_update(
                    actual, probe, mx.array([past + retained], dtype=mx.int32), axes=[2]
                )
                mx.eval(fork)
                oracle = np.concatenate(
                    [np.asarray(expected)[:, :, : past + retained, :], np.asarray(probe)], axis=2
                )
                check_cache(oracle, fork[:, :, : past + retained + 1, :])
                check_cache(full, actual)
        for old, live in zip(original, storage):
            check_cache(old, live)
        del stock, candidate, full, fork, oracle, expected, actual
        mx.clear_cache()
        before = {
            "active": mx.get_active_memory(),
            "cache": mx.get_cache_memory(),
            "peak": mx.get_peak_memory(),
        }
        samples = [[], []]
        for repeat in range(iterations + 3):
            for arm in [0, 1] if repeat % 2 == 0 else [1, 0]:
                mx.synchronize()
                start = time.perf_counter_ns()
                outputs = compute(arm == 1)
                mx.eval(*outputs)
                mx.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if repeat >= 3:
                    samples[arm].append(elapsed)
                del outputs
        final_stock, final_candidate = compute(False), compute(True)
        mx.eval(*final_stock, *final_candidate)
        if compare_outputs(final_stock[0], final_candidate[0])["bit_exact"] is not True:
            raise ValueError("reserve changed final attention output")
        for expected, actual in zip(final_stock[1:], final_candidate[1:]):
            check_cache(expected, actual[:, :, : past + rows, :])
        for old, live in zip(original, storage):
            check_cache(old, live)
        cell = {
            "past": past,
            "m": rows,
            "capacity": past + 1024,
            "bit_exact": True,
            "retained_prefixes": rows,
            "old_snapshot_unchanged": True,
            "fork_rollback_exact": True,
            "samples_ms": samples,
            "medians_ms": [statistics.median(arm) for arm in samples],
            "paired_gain_pct": statistics.median([(a / b - 1) * 100 for a, b in zip(*samples)]),
            "memory_before": before,
            "memory_after": {
                "active": mx.get_active_memory(),
                "cache": mx.get_cache_memory(),
                "peak": mx.get_peak_memory(),
            },
        }
        report["cells"].append(cell)
        persist()
        print(f"B past={past} M={rows} exact", flush=True)
        del (
            q,
            base,
            new,
            storage,
            original,
            probe,
            final_stock,
            final_candidate,
            expected,
            actual,
            live,
            compute,
        )
        mx.clear_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=12)
    args = parser.parse_args(argv)
    if args.iterations < 3:
        parser.error("at least three pairs required")
    with args.output.open("x") as stream:
        report = {
            "status": "running",
            "parity": False,
            "cells": [],
            "conditions_before": conditions(),
            "boundary": "one synthetic FP16 attention/cache component, old snapshots live; not native cache or model throughput",
        }

        def persist():
            stream.seek(0)
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.truncate()
            stream.flush()

        persist()
        try:
            screen(report, args.iterations, persist)
            validate(report, args.iterations)
            report.update(status="complete", parity=True)
        except BaseException as error:
            report.update(status="failed", parity=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["conditions_after"] = conditions()
            persist()


if __name__ == "__main__":
    main()
