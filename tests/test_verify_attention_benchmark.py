"""Numerical/reporting gate errors; no MLX or GPU needed."""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import benchmark_verify_attention as bench


def test_signed_zero_difference_is_not_bit_exact():
    a = np.zeros((1, 24, 2, 256), dtype=np.float16)
    b = a.copy()
    b[0, 0, 0, 0] = -0.0
    result = bench.compare_outputs(a, b)
    assert (
        result["bit_exact"] is False
        and result["different_values"] == 1
        and result["max_abs_diff"] == 0
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_output_rejected(bad):
    a = np.zeros((1, 24, 2, 256), dtype=np.float16)
    b = a.copy()
    b[0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        bench.compare_outputs(a, b)


def test_empty_or_different_dtype_rejected():
    for a, b in [
        (np.zeros((1, 24, 0, 256), np.float16), np.zeros((1, 24, 0, 256), np.float16)),
        (np.zeros((1, 24, 2, 256), np.float16), np.zeros((1, 24, 2, 256), np.float32)),
    ]:
        with pytest.raises(ValueError):
            bench.compare_outputs(a, b)


def matrix(nax=False):
    cells = [
        {
            "kind": "tail_stock",
            "past": past,
            "m": rows,
            "seed": seed,
            "bit_exact": True,
            "different_values": 0,
            "max_abs_diff": 0.0,
        }
        for past, rows, seed in itertools.product(
            [0, 1, 23, 257, 4096, 16384], range(1, 9), [0, 37]
        )
    ]
    if nax:
        cells += [
            {
                "kind": "nax",
                "past": past,
                "m": rows,
                "ks": ks,
                "blocks": blocks,
                "q_staging": qtg,
                "status": "unsupported",
                "bit_exact": False,
            }
            for past, rows, ks, blocks, qtg in itertools.product(
                [8192, 16384, 32768], [2, 3, 4], [1, 2, 4, 8], [32, 64, 128], [0, 1]
            )
        ]
    return {"cells": cells}


def test_incomplete_matrix_cannot_validate():
    with pytest.raises(ValueError):
        bench.finish({"cells": []}, False)
    report = matrix(True)
    bench.finish(report, True)
    assert report["parity"] is False


def test_duplicate_cells_cannot_claim_complete_matrix():
    report = matrix()
    report["cells"][-1] = report["cells"][0].copy()
    with pytest.raises(ValueError, match="matrix"):
        bench.finish(report, False)


@pytest.mark.parametrize(
    "key,bad", [("bit_exact", 1), ("max_abs_diff", float("nan")), ("different_values", True)]
)
def test_invalid_cell_metrics_cannot_validate(key, bad):
    report = matrix()
    report["cells"][0][key] = bad
    with pytest.raises(ValueError):
        bench.finish(report, False)


@pytest.mark.parametrize("failure", [ValueError("divergence"), KeyboardInterrupt()])
def test_partial_evidence_preserved_on_failure(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(bench, "conditions", dict)

    def run(report, *args):
        report["cells"].append({"kind": "tail_stock", "bit_exact": False})
        raise failure

    monkeypatch.setattr(bench, "run", run)
    output = tmp_path / "failure.json"
    with pytest.raises(type(failure)):
        bench.main(["--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False and len(report["cells"]) == 1
