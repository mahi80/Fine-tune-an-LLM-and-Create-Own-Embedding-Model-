"""Run a fine-tuning pipeline through the LangGraph agents from the terminal.

Usage (from the repo root, Soup venv active):
    python agents/run_agents.py embedding --pdfs pdfs/
    python agents/run_agents.py embedding --sft --auto-install
    python agents/run_agents.py llm --data data/sft_train.jsonl --name my-model
    python agents/run_agents.py llm --auto            # unattended: no checkpoints

The planner agent runs first (hardware + tools + fit). Human-in-the-loop
checkpoints ask you to approve key outputs; --auto approves everything.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_agents import (  # noqa: E402  # pylint: disable=wrong-import-position
    build_graph,
    default_args,
)


def console_gate(step_name: str, question: str, preview: str) -> bool:
    print(f"\n----- CHECKPOINT after '{step_name}' -----")
    if preview:
        print(preview)
    answer = input(f"{question} [Y/n] ").strip().lower()
    return answer in ("", "y", "yes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pipeline", choices=["embedding", "llm"])
    ap.add_argument("--pdfs", help="(embedding) folder with the source PDFs")
    ap.add_argument("--sft", action="store_true", help="(embedding) also emit LLM data")
    ap.add_argument("--data", help="(llm) training JSONL")
    ap.add_argument("--name", help="(llm) Ollama model name")
    ap.add_argument("--auto", action="store_true",
                    help="No human checkpoints — approve everything")
    ap.add_argument("--auto-install", action="store_true",
                    help="Let the planner pip-install / ollama-pull missing pieces")
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k in ("pdfs", "sft", "data", "name") and v}
    ns = default_args(args.pipeline, **overrides)
    gate = None if args.auto else console_gate
    graph = build_graph(args.pipeline, ns, gate=gate, install=args.auto_install)
    final = graph.invoke({"pipeline": args.pipeline})
    return 1 if final.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
