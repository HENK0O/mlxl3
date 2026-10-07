"""True four-row lm_head screen; uses the stock shader and original weights.

Synchronized projection timings and a feedback proxy are not model throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from benchmark_smallm import Checkpoint, assert_exact, layout, paired, qmv


def head_shape(checkpoint):
    """Reject other geometries before constructing a kernel.

    raises: ValueError
    """
    matches = [shape for shape in checkpoint.inventory() if shape["prefixes"] == ["lm_head"]]
    if len(matches) != 1:
        raise ValueError("expected exactly one separate lm_head")
    shape = matches[0]
    if (shape["input"], shape["widths"], shape["k"], shape["cb"]) != (5120, [248320], 3, 2):
        raise ValueError("MB4 screen only supports the measured Qwen27B head")
    return shape


def conditions():
    result = {}
    for key, cmd in {
        "power": ["pmset", "-g", "batt"],
        "swap": ["sysctl", "vm.swapusage"],
        "therm": ["pmset", "-g", "therm"],
    }.items():
        try:
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
            result[key] = (
                run.stdout
                if run.returncode == 0
                else f"unavailable (exit {run.returncode}): {run.stderr}"
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            result[key] = f"unavailable: {type(error).__name__}: {error}"
    return result


def validate_result(report: dict, iterations: int, validation_only=False):
    """A partial run, fake parity or invalid timing must never be complete.

    raises: ValueError
    """
    if any(
        report.get(key) is not True
        for key in ("parity", "fp32_partials_bit_exact", "fp16_forward_and_chain_bit_exact")
    ):
        raise ValueError("incomplete MB4 parity")
    if report.get("validated_projection_steps") != 16:
        raise ValueError("incomplete MB4 dependent projection validation")
    if validation_only:
        return
    for key, steps in [("isolated", 1), ("dependent", 8)]:
        timing = report.get(key, {})
        samples, medians = timing.get("samples_ms", []), timing.get("medians_ms", [])
        if len(samples) != 2 or len(medians) != 2 or timing.get("steps") != steps:
            raise ValueError("incomplete MB4 timing")
        if any(len(arm) != iterations for arm in samples):
            raise ValueError("incomplete MB4 samples")
        values = medians + [value for arm in samples for value in arm]
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in values
        ):
            raise ValueError("invalid MB4 timing")
        gain = timing.get("paired_gain_pct")
        if type(gain) not in (int, float) or not math.isfinite(gain):
            raise ValueError("invalid MB4 gain")


def check_projection(value):
    import numpy as np

    array = np.asarray(value)
    if array.shape != (4, 248320) or array.dtype != np.float16 or not np.isfinite(array).all():
        raise ValueError("invalid dependent lm_head projection")


def screen(model: Path, iterations: int, report: dict, validation_only=False):
    import mlx.core as mx
    import numpy as np

    checkpoint = Checkpoint(model)
    shape = head_shape(checkpoint)
    report.update(
        shape=shape,
        mlx_version=mx.__version__ if hasattr(mx, "__version__") else "see command provenance",
    )
    weights = mx.array(checkpoint.load("lm_head.trellis").view(np.uint32))
    suh_np = checkpoint.load("lm_head.suh").astype(np.float16)
    svh_np = checkpoint.load("lm_head.svh").astype(np.float16)
    if (
        suh_np.shape != (5120,)
        or svh_np.shape != (248320,)
        or not (np.isfinite(suh_np).all() and np.isfinite(svh_np).all())
    ):
        raise ValueError("invalid lm_head transforms")
    report["tensor_sha256"] = {
        name: hashlib.sha256(checkpoint.load(name).tobytes()).hexdigest()
        for name in ["lm_head.trellis", "lm_head.suh", "lm_head.svh"]
    }
    suh, svh = mx.array(suh_np), mx.array(svh_np)
    x = mx.array(np.random.default_rng(20261007).normal(0, 0.2, (4, 5120)).astype(np.float16))
    kernels = [qmv(mx, shape, 4), qmv(mx, shape, 4, mb=4, nt=1)]
    oracle = qmv(mx, shape, 1, mb=1, nt=1)

    def transform(value):
        return mx.hadamard_transform(
            (value * suh).reshape(value.shape[0], 40, 128), scale=1 / 128**0.5
        ).reshape(value.shape[0], 5120)

    xhat = transform(x)
    partials = [kernel(xhat, weights, None) for kernel in kernels]
    reference = mx.concatenate([oracle(xhat[i : i + 1], weights, None) for i in range(4)], axis=0)
    mx.eval(*partials, reference)
    for partial in partials:
        assert_exact(partial, reference)
    report["fp32_partials_bit_exact"] = True

    def forward(value, kernel):
        y = kernel(transform(value), weights, None).sum(axis=1).astype(mx.float16)
        return (
            mx.hadamard_transform(y.reshape(4, 1940, 128), scale=1 / 128**0.5).reshape(4, 248320)
            * svh
        )

    assert_exact(forward(x, kernels[0]), forward(x, kernels[1]))

    def chain(steps, kernel, validate=False):
        value = x
        for _ in range(steps):
            y = forward(value, kernel)
            if validate:
                check_projection(y)
            value = (mx.tanh(y[:, :5120]) * 0.2).astype(mx.float16)
        return value

    assert_exact(chain(8, kernels[0], validate=True), chain(8, kernels[1], validate=True))
    report["validated_projection_steps"] = 16
    report["fp16_forward_and_chain_bit_exact"] = True
    report["parity"] = True
    report["baseline"] = layout(4, False, 248320, 5120, 3)
    report["candidate"] = {"mb": 4, "nt": 1, "groups": 1, "sg": 8, "splits": 1}
    if validation_only:
        report["boundary"] = (
            "correctness only; all sixteen complete dependent projections checked, no timing"
        )
        return
    report["isolated"] = paired(
        mx, [lambda _, kernel=kernel: kernel(xhat, weights, None) for kernel in kernels], iterations
    )
    report["dependent"] = paired(
        mx, [lambda steps, kernel=kernel: chain(steps, kernel) for kernel in kernels], iterations, 8
    )
    report["boundary"] = (
        "host+MLX+GPU synchronization; eight-head feedback proxy, not target verification or tokens/s"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations < 3:
        parser.error("at least three paired iterations required")
    # Reserve the path before running; never replace previous evidence.
    with args.output.open("x") as stream:
        report = {"status": "running", "parity": False, "conditions_before": conditions()}
        try:
            screen(args.model, args.iterations, report, args.validation_only)
            validate_result(report, args.iterations, args.validation_only)
            report["status"] = "complete"
        except BaseException as error:
            report.update(status="failed", parity=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            report["conditions_after"] = conditions()
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
