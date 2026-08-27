"""Shared step-runner for the one-command pipeline scripts.

Runs named steps in order, printing a plan first. A step can auto-skip
(e.g. its output already exists), the user can skip steps (--skip) or resume
from a step (--from-step), and a failure stops the run with the exact command
that failed plus the flag to resume.
"""

import json
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass


class PipelineError(RuntimeError):
    """A step (or pre-flight check) failed; message is user-facing."""


@dataclass
class Step:
    name: str
    desc: str
    run: Callable[[], None]
    auto_skip: Callable[[], str | None] | None = None  # returns skip reason or None


def run_cmd(cmd: list) -> None:
    """Run one subprocess, streaming its output; raise PipelineError on failure."""
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    try:
        result = subprocess.run([str(c) for c in cmd], check=False)
    except FileNotFoundError as exc:
        raise PipelineError(f"executable not found: {cmd[0]}") from exc
    if result.returncode != 0:
        raise PipelineError(f"exit code {result.returncode} from: {printable}")


def require_exe(name: str, install_hint: str) -> str:
    path = shutil.which(name)
    if not path:
        raise PipelineError(f"'{name}' not found on PATH. Install it: {install_hint}")
    return path


def ollama_model_names(url: str = "http://localhost:11434") -> list:
    """Names of models available on the local Ollama server."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=10) as resp:
            data = json.load(resp)
    except OSError as exc:
        raise PipelineError(
            f"Ollama server not reachable at {url} — start the Ollama app "
            f"or run 'ollama serve' ({exc})"
        ) from exc
    return [m.get("name", "") for m in data.get("models", [])]


def run_pipeline(steps: list, skip: list, start_from: str | None, dry_run: bool) -> int:
    names = [s.name for s in steps]
    for unknown in [*([start_from] if start_from else []), *skip]:
        if unknown not in names:
            print(f"Unknown step '{unknown}'. Steps: {', '.join(names)}", file=sys.stderr)
            return 2

    print("Plan:")
    started = start_from is None
    plan_started = started
    for s in steps:
        if s.name == start_from:
            plan_started = True
        state = "run" if plan_started and s.name not in skip else "skip"
        print(f"  [{state:4}] {s.name:8} {s.desc}")
    if dry_run:
        print("\nDry run — nothing executed.")
        return 0

    for s in steps:
        if s.name == start_from:
            started = True
        if not started or s.name in skip:
            print(f"\n-- {s.name}: skipped (by request)")
            continue
        reason = s.auto_skip() if s.auto_skip else None
        if reason:
            print(f"\n-- {s.name}: skipped ({reason})")
            continue
        print(f"\n== {s.name}: {s.desc}")
        try:
            s.run()
        except PipelineError as exc:
            print(f"\nStep '{s.name}' FAILED: {exc}", file=sys.stderr)
            print(f"Fix the issue, then resume with: --from-step {s.name}", file=sys.stderr)
            return 1
    print("\nAll steps complete.")
    return 0
