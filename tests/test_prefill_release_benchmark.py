"""Real driver control flow with isolated simulated transport, no GPU."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import benchmark_prefill_release as bench


def response(kind, cached):
    return {
        "type": "complete",
        "token_hash": kind,
        "cache_context": kind,
        "stats": {
            "generated_tokens": 16,
            "decode_tps": 8.0,
            "decode_seconds": 2.0,
            "mtp_accepted_tokens": 5,
            "mtp_proposed_tokens": 7,
            "mtp_blocks": 4,
            "prompt_tokens": 16710 if kind == "long" else 1110,
            "cached_prompt_tokens": 1024 if cached else 0,
            "memory": {
                "mlx_active_bytes": 10,
                "mlx_cache_bytes": 0,
                "mlx_peak_bytes": 12,
                "process_footprint_bytes": 14,
            },
        },
    }


@pytest.mark.parametrize(
    "key,bad",
    [
        ("cached_prompt_tokens", 0),
        ("cached_prompt_tokens", True),
        ("prompt_tokens", True),
        ("generated_tokens", 0),
    ],
)
def test_missing_cache_or_invalid_counts_rejected(key, bad):
    event = response("short", True)
    event["stats"][key] = bad
    with pytest.raises((ValueError, AssertionError)):
        bench.check_response(event, "text", "short", True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, -1])
def test_invalid_memory_cannot_validate(bad):
    event = response("long", False)
    event["stats"]["memory"]["mlx_cache_bytes"] = bad
    with pytest.raises(ValueError, match="memory"):
        bench.check_response(event, "text", "long", False)


def transport(monkeypatch, fail=None):
    owned = []

    class Process:
        def __init__(self, command, **kwargs):
            self.queue = [{"type": "ready", "runtime_version": "1.4.2"}]
            self.process = SimpleNamespace(wait=lambda timeout: 0)
            self.requests = []
            self.closed = False
            self.env = kwargs["env"]
            owned.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def send(self, request, deadline):
            if request["type"] == "shutdown":
                return
            self.requests.append(request)
            kind = request["request_id"].split("-")[0]
            cached = request["request_id"].endswith("-1")
            event = response(kind, cached)
            event["request_id"] = request["request_id"]
            if fail:
                fail(event, self, owned)
            self.queue.extend(
                [
                    {
                        "type": "context_usage",
                        "request_id": request["request_id"],
                        "used_tokens": event["stats"]["prompt_tokens"],
                    },
                    {"type": "delta", "request_id": request["request_id"], "text": kind},
                    event,
                ]
            )

        def receive(self, deadline):
            return self.queue.pop(0)

    monkeypatch.setattr(bench, "JsonProcess", Process)
    monkeypatch.setattr(bench, "conditions", dict)
    return owned


def arguments(tmp_path):
    engine = tmp_path / "engine"
    engine.write_bytes(b"fixture")
    output = tmp_path / "report.json"
    return [
        "--engine",
        str(engine),
        "--model",
        str(tmp_path),
        "--head",
        str(tmp_path),
        "--output",
        str(output),
    ], output


@pytest.mark.parametrize("repeat,arms", [(False, 3), (True, 2)])
def test_whole_driver_preserves_two_oracles_and_closes_all_children(
    tmp_path, monkeypatch, repeat, arms
):
    owned = transport(monkeypatch)
    argv, output = arguments(tmp_path)
    bench.main(argv + (["--memory-repeat"] if repeat else []))
    report = json.loads(output.read_text())
    assert report["status"] == "complete" and report["parity"] is True
    assert len(owned) == arms and all(p.closed and len(p.requests) == 3 for p in owned)
    assert all(p.env["MLXL3_MTP_LOOKUP"] == "0" for p in owned)
    assert all(len(row["measurements"]) == 3 for row in report["runs"])
    assert all(
        [m["kind"] for m in row["measurements"]] == ["long", "short", "short"]
        for row in report["runs"]
    )


@pytest.mark.parametrize("failure", ["cache", "ids", "context", "cancel", "interrupt", "silent"])
def test_partial_or_divergent_driver_cannot_claim_parity(tmp_path, monkeypatch, failure):
    def corrupt(event, process, owned):
        if failure == "cache" and event["stats"]["cached_prompt_tokens"]:
            event["stats"]["cached_prompt_tokens"] = 0
        if failure == "ids" and len(owned) == 2:
            event["token_hash"] = "different"
        if failure == "context":
            event["stats"]["prompt_tokens"] = 1
        if failure == "cancel":
            event["type"] = "cancelled"
        if failure == "interrupt":
            raise KeyboardInterrupt()
        if failure == "silent":
            raise TimeoutError("silent child reached its deadline")

    owned = transport(monkeypatch, corrupt)
    argv, output = arguments(tmp_path)
    with pytest.raises((ValueError, RuntimeError, KeyboardInterrupt, TimeoutError)):
        bench.main(argv)
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert owned and all(p.closed for p in owned)
    if failure == "cache":
        assert report["runs"][0]["measurements"][-1]["event"]["stats"]["cached_prompt_tokens"] == 0


def test_never_overwrites_previous_evidence(tmp_path):
    argv, output = arguments(tmp_path)
    output.write_text("previous failed evidence")
    with pytest.raises(FileExistsError):
        bench.main(argv)
    assert output.read_text() == "previous failed evidence"


def test_wrong_runtime_is_saved_and_rejected_before_generation(tmp_path, monkeypatch):
    owned = transport(monkeypatch)
    monkeypatch.setattr(
        bench, "until", lambda *args: ({"type": "ready", "runtime_version": "1.4.1"}, "")
    )
    argv, output = arguments(tmp_path)
    with pytest.raises(ValueError, match="version"):
        bench.main(argv)
    assert owned[0].closed and not owned[0].requests
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert report["runs"][0]["ready"]["runtime_version"] == "1.4.1"
