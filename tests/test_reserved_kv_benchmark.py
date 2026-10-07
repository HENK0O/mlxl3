"""Cache oracle and partial evidence errors without GPU loading."""

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import benchmark_reserved_kv as bench


def test_entire_snapshot_bytes_and_original_dtype_required():
    good = np.zeros((1, 4, 12, 256), np.float16)
    for bad in [good.astype(np.float32), good[:, :, :0, :], good.copy()]:
        if bad.shape == good.shape and bad.dtype == good.dtype:
            bad[0, 0, -1, -1] = -0.0
        with pytest.raises(ValueError):
            bench.check_cache(good, bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_snapshot_is_rejected(value):
    bad = np.zeros((1, 4, 12, 256), np.float16)
    bad[0, 0, -1, -1] = value
    with pytest.raises(ValueError):
        bench.check_cache(bad, bad)


def matrix():
    return {
        "cells": [
            {
                "past": past,
                "m": rows,
                "bit_exact": True,
                "retained_prefixes": rows,
                "old_snapshot_unchanged": True,
                "fork_rollback_exact": True,
                "samples_ms": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            }
            for past, rows in itertools.product([8192, 16384, 32768], [2, 3, 4])
        ]
    }


def test_partial_or_duplicate_matrix_cannot_complete():
    good = matrix()
    bench.validate(good, 3)
    duplicate = copy.deepcopy(good)
    duplicate["cells"][-1] = duplicate["cells"][0].copy()
    for bad in [{"cells": []}, duplicate]:
        with pytest.raises(ValueError):
            bench.validate(bad, 3)


@pytest.mark.parametrize("value", [True, 0.0, float("nan"), float("inf")])
def test_invalid_timing_cannot_complete(value):
    bad = matrix()
    bad["cells"][0]["samples_ms"][1][2] = value
    with pytest.raises(ValueError):
        bench.validate(bad, 3)


@pytest.mark.parametrize("error", [ValueError("snapshot overwritten"), KeyboardInterrupt()])
def test_partial_failure_is_saved(tmp_path, monkeypatch, error):
    monkeypatch.setattr(bench, "conditions", dict)

    def screen(report, *args):
        report["cells"].append({"past": 8192, "m": 2})
        raise error

    monkeypatch.setattr(bench, "screen", screen)
    output = tmp_path / "failure.json"
    with pytest.raises(type(error)):
        bench.main(["--output", str(output), "--iterations", "3"])
    saved = json.loads(output.read_text())
    assert saved["status"] == "failed" and saved["parity"] is False and len(saved["cells"]) == 1


def test_previous_evidence_is_preserved(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("previous error")
    with pytest.raises(FileExistsError):
        bench.main(["--output", str(output)])
    assert output.read_text() == "previous error"
