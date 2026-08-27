"""One command for the whole LLM pipeline (data → QLoRA train → deploy to Ollama).

Serialized steps: check → validate → train → verify → eval → export → ollama.
Finished stages auto-skip (e.g. the GGUF already exists), so rerunning after an
interruption is safe.

Usage (from the repo root, Soup venv active):
    python scripts/run_llm_pipeline.py --data data/sft_train.jsonl --name my-model
    python scripts/run_llm_pipeline.py --from-step export --name my-model   # resume
    python scripts/run_llm_pipeline.py --dry-run
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline import (  # noqa: E402  # pylint: disable=wrong-import-position
    PipelineError,
    Step,
    ollama_model_names,
    require_exe,
    run_cmd,
    run_pipeline,
)

SCRIPTS = Path(__file__).resolve().parent


def parse_config(config_path: Path) -> dict:
    """Pull base / data.train / output out of the Soup YAML without a YAML dependency."""
    found = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"^base:\s*(\S+)", line):
            found["base"] = m.group(1).strip("'\"")
        elif m := re.match(r"^\s+train:\s*(\S+)", line):
            found["train"] = m.group(1).strip("'\"")
        elif m := re.match(r"^output:\s*(\S+)", line):
            found["output"] = m.group(1).strip("'\"")
    return found


def _local_like(data: str) -> bool:
    """A path-looking data value (vs a bare Hugging Face dataset id like 'org/name')."""
    return (Path(data).suffix in {".jsonl", ".json", ".csv", ".parquet", ".txt"}
            or data.startswith((".", "/", "~")) or "\\" in data or Path(data).exists())


def find_ollama() -> str:
    path = shutil.which("ollama")
    if path:
        return path
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if local.is_file():
        return str(local)
    raise PipelineError("'ollama' not found — install it: winget install Ollama.Ollama")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/soup_qwen7b.yaml")
    ap.add_argument("--data", help="Training JSONL (default: the config's data.train)")
    ap.add_argument("--name", default="my-model", help="Name for the model in Ollama")
    ap.add_argument("--quant", default="q4_K_M", help="Ollama import quantization")
    ap.add_argument("--resume", action="store_true", help="Pass --resume to soup train")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--skip", default="", help="Comma-separated step names to skip")
    ap.add_argument("--from-step", dest="from_step", help="Resume from this step")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, run nothing")
    args = ap.parse_args()
    return run_pipeline(build_steps(args), skip=[s for s in args.skip.split(",") if s],
                        start_from=args.from_step, dry_run=args.dry_run)


def build_steps(args) -> list:
    """The pipeline steps; shared by this CLI and the agent graph."""

    config = Path(args.config)
    cfg = parse_config(config) if config.is_file() else {}
    base = cfg.get("base", "")
    data = args.data or cfg.get("train", "")
    output = Path(cfg.get("output", "./output_qwen7b"))
    gguf = output.parent / (output.name + ".f16.gguf")

    def check() -> None:
        require_exe("soup", 'pip install -e "Soup[train]"')
        if not config.is_file():
            raise PipelineError(f"config not found: {config} (run from the repo root)")
        if not base:
            raise PipelineError(f"could not read 'base:' from {config}")
        if not data:
            raise PipelineError(f"no training data: pass --data or set data.train in {config}")
        if _local_like(data) and not Path(data).is_file():
            raise PipelineError(f"training data not found: {data}")
        if args.data and args.data != cfg.get("train"):
            print(f"  note: --data is '{args.data}' but {config} trains on "
                  f"'{cfg.get('train')}' — update the config's data.train to match")

    def validate() -> None:
        run_cmd(["soup", "data", "validate", data])
        run_cmd(["soup", "data", "doctor", data, "--model", base])

    def validate_skip():
        if not _local_like(data) and not Path(data).is_file():
            return f"'{data}' is a Hugging Face dataset id, not a local file"
        return None

    def train() -> None:
        cmd = ["soup", "train", "--config", config, "--yes"]
        if args.resume:
            cmd.append("--resume")
        run_cmd(cmd)

    def verify() -> None:
        run_cmd([sys.executable, SCRIPTS / "verify_adapter.py",
                 "--base", base, "--adapter", output])

    def evaluate() -> None:
        try:
            run_cmd(["soup", "eval", "auto", "--config", config])
        except PipelineError as exc:
            print(f"  soup eval failed ({exc}) — optional step, continuing")

    def export() -> None:
        run_cmd(["soup", "export", "--model", output, "--format", "gguf", "--quant", "f16"])
        if not gguf.is_file():
            raise PipelineError(f"export finished but {gguf} was not created")

    def export_skip():
        return f"{gguf} already exists (delete it to re-export)" if gguf.is_file() else None

    def ollama() -> None:
        exe = find_ollama()
        ollama_model_names(args.ollama_url)  # raises if the server is down
        modelfile = output / "Modelfile.generated"
        modelfile.parent.mkdir(parents=True, exist_ok=True)
        modelfile.write_text(f"FROM {gguf.resolve()}\n", encoding="utf-8")
        run_cmd([exe, "create", args.name, "-q", args.quant, "-f", modelfile])
        print(f"\nDone — chat with it:  ollama run {args.name}")

    steps = [
        Step("check", "verify tools, config, and training data", check),
        Step("validate", "soup data validate + doctor", validate, validate_skip),
        Step("train", f"QLoRA fine-tune {base or '<base>'}", train),
        Step("verify", "generation sanity check (4-bit base + adapter)", verify),
        Step("eval", "soup eval auto (optional)", evaluate),
        Step("export", "merge adapter and convert to f16 GGUF", export, export_skip),
        Step("ollama", f"import into Ollama as '{args.name}' ({args.quant})", ollama),
    ]
    return steps



if __name__ == "__main__":
    sys.exit(main())
