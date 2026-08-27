"""One command for the whole embedding pipeline (PDF → tuned embedder → eval).

Serialized steps: check → extract → caption → pairs → train → eval.
Finished stages auto-skip (e.g. extraction output already present; captioning
resumes where it stopped), so rerunning after an interruption is safe.

Usage (from the repo root, Soup venv active, Ollama running):
    python scripts/run_embedding_pipeline.py --pdfs pdfs/
    python scripts/run_embedding_pipeline.py --pdfs pdfs/ --sft        # also emit LLM data
    python scripts/run_embedding_pipeline.py --from-step pairs         # resume
    python scripts/run_embedding_pipeline.py --dry-run                 # show the plan
"""

import argparse
import importlib.util
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdfs", default="pdfs", help="Folder with the source PDFs")
    ap.add_argument("--extracted", default="extracted", help="MinerU output folder")
    ap.add_argument("--data", default="data", help="Training-data output folder")
    ap.add_argument("--config", default="configs/soup_embedding.yaml")
    ap.add_argument("--backend", default="vlm-transformers",
                    help="MinerU backend (use 'pipeline' on small GPUs / CPU)")
    ap.add_argument("--caption-model", default="qwen3-vl:8b")
    ap.add_argument("--pairs-model", default="qwen2.5:7b")
    ap.add_argument("--questions-per-chunk", type=int, default=2)
    ap.add_argument("--sft", action="store_true",
                    help="Also write alpaca SFT data for the LLM pipeline")
    ap.add_argument("--tuned-output", default="./output_embedding")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--skip", default="", help="Comma-separated step names to skip")
    ap.add_argument("--from-step", dest="from_step", help="Resume from this step")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, run nothing")
    args = ap.parse_args()
    return run_pipeline(build_steps(args), skip=[s for s in args.skip.split(",") if s],
                        start_from=args.from_step, dry_run=args.dry_run)


def build_steps(args) -> list:
    """The pipeline steps; shared by this CLI and the agent graph."""

    pdfs, extracted, data = Path(args.pdfs), Path(args.extracted), Path(args.data)

    def check() -> None:
        require_exe("mineru", 'pip install "mineru[core]"')
        require_exe("soup", 'pip install -e "Soup[train]"')
        if not Path(args.config).is_file():
            raise PipelineError(f"config not found: {args.config} (run from the repo root)")
        models = ollama_model_names(args.ollama_url)
        for need in (args.caption_model, args.pairs_model):
            if not any(m.startswith(need) for m in models):
                raise PipelineError(f"Ollama model '{need}' missing — run: ollama pull {need}")
        if not list(extracted.rglob("*.md")) and not list(pdfs.glob("*.pdf")):
            raise PipelineError(f"no PDFs in {pdfs}/ and no extracted Markdown in {extracted}/")
        if str(data) != "data":
            print(f"  note: --data is '{data}' — make sure {args.config} points its "
                  f"data.train there")

    def extract() -> None:
        run_cmd(["mineru", "-p", pdfs, "-o", extracted, "-b", args.backend])

    def extract_skip():
        if list(extracted.rglob("*.md")):
            return f"{extracted}/ already contains extracted Markdown (delete it to redo)"
        return None

    def caption() -> None:
        raw = [m for m in sorted(extracted.rglob("*.md"))
               if not m.name.endswith(".captioned.md")]
        if not raw:
            raise PipelineError(f"no .md files under {extracted}/")
        for md in raw:
            cap = md.with_suffix(".captioned.md")
            if cap.exists():  # resume: re-run over the captioned file, skips finished images
                run_cmd([sys.executable, SCRIPTS / "caption_images.py", cap, "--output", cap,
                         "--model", args.caption_model, "--url", args.ollama_url])
            else:
                run_cmd([sys.executable, SCRIPTS / "caption_images.py", md,
                         "--model", args.caption_model, "--url", args.ollama_url])

    def pairs() -> None:
        cmd = [sys.executable, SCRIPTS / "build_pairs.py", extracted, "--output-dir", data,
               "--model", args.pairs_model, "--url", args.ollama_url,
               "--questions-per-chunk", args.questions_per_chunk]
        if args.sft:
            cmd.append("--sft")
        run_cmd(cmd)

    def train() -> None:
        run_cmd(["soup", "train", "--config", args.config, "--yes"])

    def evaluate() -> None:
        run_cmd([sys.executable, SCRIPTS / "eval_embedder.py", data / "embedding_val.jsonl",
                 "--tuned", args.tuned_output])

    def eval_skip():
        if importlib.util.find_spec("sentence_transformers") is None:
            return "sentence-transformers not installed (pip install sentence-transformers)"
        if not (data / "embedding_val.jsonl").is_file():
            return f"no {data}/embedding_val.jsonl to evaluate against"
        return None

    steps = [
        Step("check", "verify tools, Ollama models, and inputs", check),
        Step("extract", f"MinerU: {pdfs}/ -> {extracted}/", extract, extract_skip),
        Step("caption", "describe screenshots into the Markdown", caption),
        Step("pairs", "generate anchor/positive/negative pairs", pairs),
        Step("train", "fine-tune the embedding model with Soup", train),
        Step("eval", "compare tuned vs base recall", evaluate, eval_skip),
    ]
    return steps



if __name__ == "__main__":
    sys.exit(main())
