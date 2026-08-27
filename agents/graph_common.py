"""Planner-agent toolbox: hardware probe, fit planning, and tool checks.

Used by the LangGraph pipelines (pipeline_agents.py) and the chatbot (chat.py).
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from _pipeline import (  # noqa: E402  # pylint: disable=wrong-import-position
    PipelineError,
    ollama_model_names,
    run_cmd,
)


def probe_hardware() -> dict:
    """GPU name/VRAM (nvidia-smi), RAM (psutil if present), free disk."""
    hw = {"gpu": None, "vram_gb": 0.0, "ram_gb": None, "disk_free_gb": 0.0}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False)
        if out.returncode == 0 and out.stdout.strip():
            name, mem = out.stdout.strip().splitlines()[0].rsplit(",", 1)
            hw["gpu"] = name.strip()
            hw["vram_gb"] = round(float(mem.strip()) / 1024, 1)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    if importlib.util.find_spec("psutil"):
        import psutil  # pylint: disable=import-outside-toplevel
        hw["ram_gb"] = round(psutil.virtual_memory().total / 1e9)
    hw["disk_free_gb"] = round(shutil.disk_usage(".").free / 1e9)
    return hw


def plan_fit(pipeline: str, hw: dict) -> tuple[bool, list, dict]:
    """Decide whether the pipeline fits this machine.

    Returns (ok, plan_lines, tweaks). ok=False means do not run; plan_lines
    always explain the decision; tweaks holds adjustments (e.g. MinerU backend).
    """
    vram = hw["vram_gb"]
    lines = [f"GPU: {hw['gpu'] or 'none detected'} ({vram} GB VRAM), "
             f"RAM: {hw['ram_gb'] or '?'} GB, free disk: {hw['disk_free_gb']} GB"]
    tweaks = {}

    if hw["disk_free_gb"] < 40:
        lines.append("WARNING: under 40 GB free disk — model downloads alone need 15-40 GB.")

    if pipeline == "embedding":
        ok = True
        if vram >= 8:
            lines.append("Plan: MinerU vlm-transformers backend, qwen3-vl:8b captions, "
                         "qwen2.5:7b questions, BGE training on GPU.")
        elif vram >= 4:
            tweaks["backend"] = "pipeline"
            lines.append(f"Only {vram} GB VRAM: switching MinerU to the lighter "
                         "'pipeline' backend; captioning/question models will be slow "
                         "(they spill to system RAM).")
        else:
            tweaks["backend"] = "pipeline"
            lines.append("No usable GPU: extraction will run on CPU (slow); captioning and "
                         "question generation via Ollama will be very slow. Consider a "
                         "smaller caption model or skipping captions (--skip caption).")
    else:  # llm
        if vram >= 8:
            ok = True
            lines.append("Plan: QLoRA fine-tune of a 7B model (measured 7.3 GB peak) → "
                         "GGUF → Ollama.")
        elif vram >= 6:
            ok = True
            lines.append(f"Only {vram} GB VRAM: 7B QLoRA is borderline — close every other "
                         "GPU app; if it still fails, lower batch_size to 1 and "
                         "max_length to 512 in the config.")
        else:
            ok = False
            lines.append(f"{vram} GB VRAM cannot QLoRA-train a 7B model. Options: "
                         "(a) a small base like TinyLlama-1.1B, (b) Soup layer-streaming "
                         "(8B proven on a 4 GB card, ~1.4x slower), (c) a cloud GPU.")
    return ok, lines, tweaks


def check_tools(pipeline: str, ollama_url: str = "http://localhost:11434") -> list:
    """Missing tools/models as (name, how_to_fix, auto_install_cmd_or_None)."""
    missing = []
    if pipeline == "embedding" and not shutil.which("mineru"):
        missing.append(("mineru", 'pip install "mineru[core]"',
                        [sys.executable, "-m", "pip", "install", "mineru[core]"]))
    if not shutil.which("soup"):
        missing.append(("soup", 'pip install -e "Soup[train]" (from the Soup checkout)', None))
    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        missing.append(("ollama", "winget install Ollama.Ollama", None))
    elif pipeline == "embedding":
        try:
            have = ollama_model_names(ollama_url)
        except PipelineError:
            missing.append(("ollama server", "start the Ollama app or run 'ollama serve'", None))
        else:
            for model in ("qwen3-vl:8b", "qwen2.5:7b"):
                if not any(m.startswith(model) for m in have):
                    missing.append((f"ollama model {model}", f"ollama pull {model}",
                                    [ollama_exe, "pull", model]))
    return missing


def auto_install(missing: list, log) -> list:
    """Try the auto-installable fixes; return what is still missing."""
    still = []
    for name, how, cmd in missing:
        if cmd is None:
            still.append((name, how, None))
            continue
        log(f"installing {name} ...")
        try:
            run_cmd(cmd)
        except PipelineError as exc:
            log(f"auto-install of {name} failed ({exc}) — do it manually: {how}")
            still.append((name, how, None))
    return still


def recommend_models(pipeline: str, hw: dict) -> list:
    """(model_id, why, fits_this_machine) — first fitting entry is the recommendation."""
    vram = hw["vram_gb"]
    if pipeline == "llm":
        return [
            ("Qwen/Qwen2.5-7B-Instruct", "best all-round 7B chat model", vram >= 8),
            ("mistralai/Mistral-7B-Instruct-v0.3", "classic 7B alternative", vram >= 8),
            ("Qwen/Qwen2.5-3B-Instruct", "smaller and faster, still capable", vram >= 5),
            ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tiny — runs almost anywhere", vram >= 3),
        ]
    return [
        ("BAAI/bge-base-en-v1.5", "English docs, fastest to train and serve", True),
        ("BAAI/bge-large-en-v1.5", "a few recall points better, ~3x the size", vram >= 4),
        ("BAAI/bge-m3", "multilingual docs and long chunks (8k tokens)", vram >= 6),
    ]


def troubleshoot_gpu() -> dict:
    """Diagnose common GPU problems.

    Returns {"auto": [(description, command_list)], "manual": [advice]} —
    'auto' fixes this program can run itself (with the user's approval),
    'manual' ones need the user (usually admin rights: drivers, closing apps).
    """
    auto, manual = [], []
    if shutil.which("nvidia-smi") is None:
        manual.append("No NVIDIA driver found (nvidia-smi missing). Install the driver "
                      "from https://www.nvidia.com/drivers (needs admin), then reopen "
                      "the terminal.")
        return {"auto": auto, "manual": manual}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=False)
        used, total = (float(x) for x in out.stdout.strip().splitlines()[0].split(","))
        if total and used / total > 0.5:
            manual.append(f"{used:.0f} of {total:.0f} MB VRAM is already in use by other "
                          "programs — close games, video calls, and other AI tools "
                          "before training.")
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    if importlib.util.find_spec("torch"):
        import torch  # pylint: disable=import-outside-toplevel
        if not torch.cuda.is_available():
            auto.append((
                "PyTorch cannot see the GPU (CPU-only build installed) — reinstall the "
                "CUDA build",
                [sys.executable, "-m", "pip", "install", "--force-reinstall", "torch",
                 "--index-url", "https://download.pytorch.org/whl/cu126"]))
    return {"auto": auto, "manual": manual}
