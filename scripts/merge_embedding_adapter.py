"""Merge a Soup-trained embedding LoRA adapter into a standalone model.

Soup's `task: embedding` saves a LoRA adapter over a frozen base — fine for
local evaluation (transformers loads it when peft is installed), but production
servers (TEI) and plain sentence-transformers deployments want ONE merged
model. Note: `soup export`'s own merge path only handles causal LMs, so
BERT-family embedding adapters must be merged here.

Usage:
    python merge_embedding_adapter.py ./output_embedding --output ./output_embedding_merged
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adapter", help="Soup output dir containing adapter_config.json")
    ap.add_argument("--output", default="./output_embedding_merged")
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"],
                    help="Must match the training config's embedding_pooling")
    ap.add_argument("--max-seq", type=int, default=512)
    args = ap.parse_args()

    adapter_dir = Path(args.adapter)
    cfg_file = adapter_dir / "adapter_config.json"
    if not cfg_file.is_file():
        print(f"{cfg_file} not found — is this a Soup embedding output dir?",
              file=sys.stderr)
        return 1
    base = json.loads(cfg_file.read_text(encoding="utf-8")).get("base_model_name_or_path")
    if not base:
        print("adapter_config.json has no base_model_name_or_path", file=sys.stderr)
        return 1
    print(f"merging adapter {adapter_dir} into base {base} ...")

    from peft import PeftModel  # pylint: disable=C0415,E0401
    from transformers import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel,import-error
        AutoModel,
        AutoTokenizer,
    )

    model = AutoModel.from_pretrained(base)
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()
    out = Path(args.output)
    merged.save_pretrained(out)
    try:
        AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out)
    except (OSError, ValueError):
        AutoTokenizer.from_pretrained(base).save_pretrained(out)

    # wrap as a full sentence-transformers model when ST is available
    try:
        from sentence_transformers import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel,import-error
            SentenceTransformer,
            models,
        )
    except ImportError:
        print(f"merged HF model -> {out}  (install sentence-transformers and rerun "
              "to add the ST wrapper for TEI)")
        return 0
    transformer = models.Transformer(str(out), max_seq_length=args.max_seq)
    pooling = models.Pooling(getattr(transformer, "get_embedding_dimension",
                                     transformer.get_word_embedding_dimension)(),
                             pooling_mode=args.pooling)
    SentenceTransformer(modules=[transformer, pooling]).save(str(out))
    print(f"merged + sentence-transformers-ready model -> {out}")
    print("Serve it:  docker run -p 8080:80 -v "
          f"{out.resolve()}:/model ghcr.io/huggingface/text-embeddings-inference:"
          "cpu-latest --model-id /model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
