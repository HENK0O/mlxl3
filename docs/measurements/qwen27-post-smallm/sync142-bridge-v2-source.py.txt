"""Real bridge parity and allocator observation; never sustained throughput.

The dense 16k snapshot exceeds the existing 256 MiB prompt-cache budget.
Use a separate short prompt to exercise reuse without changing that budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

from benchmark_lmhead_mb4 import conditions
from benchmark_smallm_bridge import JsonProcess, fingerprint, until

PASSAGE = (
    "Une comparaison fiable garde les mêmes données et vérifie les résultats. "
    "Les mesures longues doivent conserver les erreurs et contrôler les conditions.\n"
)
QUESTION = (
    "\nExplique longuement en français comment analyser ce document et répondre avec des exemples."
)


def check_response(event, text, kind, cached):
    signature = fingerprint(event, text, 16)
    stats = event["stats"]
    prompt, reused = stats["prompt_tokens"], stats["cached_prompt_tokens"]
    lower, upper = (16384, 32768) if kind == "long" else (256, 2049)
    if (
        type(prompt) is not int
        or not lower <= prompt < upper
        or type(reused) is not int
        or not 0 <= reused <= prompt
        or (reused > 0) != cached
    ):
        raise ValueError("unexpected prompt length or cache reuse")
    memory = stats["memory"]
    for key in (
        "mlx_active_bytes",
        "mlx_cache_bytes",
        "mlx_peak_bytes",
        "process_footprint_bytes",
    ):
        value = memory[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid memory observation")
    if memory["mlx_active_bytes"] == 0 or memory["process_footprint_bytes"] == 0:
        raise ValueError("empty memory observation")
    return signature


def run(args, report, persist):
    prompts = {"long": PASSAGE * 640 + QUESTION, "short": PASSAGE * 40 + QUESTION}
    report.update(
        engine_sha256=hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        model=str(args.model),
        head=str(args.head),
        prompt_sha256={
            key: hashlib.sha256(value.encode()).hexdigest() for key, value in prompts.items()
        },
        inherited_options={
            name: os.environ.get(name)
            for name in (
                "MLXL3_QWEN_PIPELINE",
                "MLX_MAX_MB_PER_BUFFER",
                "MTPLX_MLX_COMMAND_BUFFER_MB",
            )
        },
    )
    persist()
    references = {}
    arms = (
        [(False, True), (False, False)]
        if args.memory_repeat
        else [(False, False), (False, True), (True, True)]
    )
    for enabled, release in arms:
        row = {
            "enabled": enabled,
            "release_scratch": release,
            "status": "running",
            "conditions_before": conditions(),
            "measurements": [],
        }
        report["runs"].append(row)
        persist()
        environment = os.environ | {
            "MLXL3_EXPERIMENTAL_BATCHED_VERIFY": "1" if enabled else "0",
            "MLXL3_EXPERIMENTAL_PREFILL_CACHE_RELEASE": "1" if release else "0",
            "MLXL3_MTP_LOOKUP": "0",
        }
        stderr_path = args.output.with_name(
            f"{args.output.stem}-{int(enabled)}-{int(release)}.stderr"
        )
        with tempfile.TemporaryDirectory(prefix="post-smallm-bridge-") as directory:
            command = [
                str(args.engine),
                "--registry",
                str(Path(directory) / "models.json"),
                "bridge",
                str(args.model),
                "--context-length",
                "32768",
            ]
            row.update(
                command=command,
                options={
                    name: environment[name]
                    for name in (
                        "MLXL3_EXPERIMENTAL_BATCHED_VERIFY",
                        "MLXL3_EXPERIMENTAL_PREFILL_CACHE_RELEASE",
                        "MLXL3_MTP_LOOKUP",
                    )
                },
            )
            persist()
            with (
                stderr_path.open("xb") as stderr,
                JsonProcess(command, env=environment, stderr=stderr) as process,
            ):
                row["ready"], _ = until(process, "ready", time.monotonic() + 120)
                persist()
                if row["ready"].get("runtime_version") != "1.4.2":
                    raise ValueError("requires the measured engine version 1.4.2")
                for kind, cached in [("long", False), ("short", False), ("short", True)]:
                    request = f"{kind}-{int(cached)}"
                    measurement = {"kind": kind, "cached_request": cached, "status": "running"}
                    row["measurements"].append(measurement)
                    persist()
                    deadline = time.monotonic() + (600 if kind == "long" else 180)
                    process.send(
                        {
                            "type": "generate",
                            "request_id": request,
                            "conversation_id": f"post-smallm-{kind}",
                            "messages": [{"role": "user", "content": prompts[kind]}],
                            "max_tokens": 16,
                            "temperature": 0,
                            "top_k": 1,
                            "repetition_penalty": 1,
                            "mtp": True,
                            "mtp_depth": 2,
                            "mtp_head_path": str(args.head),
                            "reuse_prompt_cache": True,
                        },
                        deadline,
                    )
                    text = ""
                    while True:
                        event = process.receive(deadline)
                        if event.get("type") in ("error", "cancelled"):
                            measurement["terminal_error"] = event
                            persist()
                            raise RuntimeError(event)
                        if event.get("request_id") != request:
                            continue
                        if event.get("type") == "context_usage":
                            measurement["context_usage"] = event
                            persist()
                            used = event["used_tokens"]
                            lower, upper = (16384, 32768) if kind == "long" else (256, 2049)
                            if type(used) is not int or not lower <= used < upper:
                                raise ValueError("prompt outside measured context range")
                        if event.get("type") == "delta":
                            text += event.get("text", "")
                            if len(text) > 4 * 1024 * 1024:
                                raise ValueError("unbounded response")
                        if event.get("type") == "complete":
                            break
                    measurement.update(event=event, text=text)
                    persist()
                    signature = check_response(event, text, kind, cached)
                    context = measurement.get("context_usage", {})
                    if context.get("used_tokens") != event["stats"]["prompt_tokens"]:
                        raise ValueError("missing or inconsistent real context usage")
                    if kind in references and signature != references[kind]:
                        raise ValueError("bridge IDs/text/history/MTP counters differ")
                    references.setdefault(kind, signature)
                    measurement.update(
                        status="complete",
                        parity=True,
                        fingerprint=signature,
                        conditions_after=conditions(),
                    )
                    persist()
                process.send({"type": "shutdown"}, time.monotonic() + 5)
                if process.process.wait(timeout=10) != 0:
                    raise RuntimeError("bridge shutdown failed")
        row.update(status="complete", parity=True)
        persist()
        print(f"bridge 1.4.2 A1={enabled} H={release}: long/short/cache exact", flush=True)
    if len(report["runs"]) != len(arms) or any(
        len(row["measurements"]) != 3 for row in report["runs"]
    ):
        raise ValueError("incomplete bridge matrix")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "model", "head", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--memory-repeat", action="store_true")
    args = parser.parse_args(argv)
    with args.output.open("x") as stream:
        report = {
            "status": "running",
            "parity": False,
            "runs": [],
            "boundary": "16-token requests; cold long-context allocator observation plus short cold/cached parity, not sustained throughput",
        }

        def persist():
            stream.seek(0)
            json.dump(report, stream, indent=2)
            stream.write("\n")
            stream.truncate()
            stream.flush()

        persist()
        try:
            run(args, report, persist)
            report.update(status="complete", parity=True)
        except BaseException as error:
            report.update(status="failed", parity=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            persist()


if __name__ == "__main__":
    main()
