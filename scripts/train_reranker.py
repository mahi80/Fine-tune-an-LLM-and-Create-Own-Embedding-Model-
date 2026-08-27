"""Cross-encoder reranker training (the Cohere-rerank counterpart).

A reranker reads the query and a candidate TOGETHER (cross-attention), so it
judges relevance far better than embedding distance — at the cost of scoring
only the top-k retrieved candidates. Trains on the hard-negative pairs from
mine_hard_negatives.py: (anchor, positive) -> 1.0, (anchor, negative) -> 0.0.

Usage:
    python train_reranker.py data/embedding_train.hard.jsonl --output ./output_reranker
    python train_reranker.py data/embedding_train.hard.jsonl \
        --base cross-encoder/ms-marco-MiniLM-L6-v2 --epochs 2
"""

import argparse
import json
import sys
from pathlib import Path


def load_rows(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("anchor") and row.get("positive") and row.get("negative"):
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pairs", help="JSONL with anchor/positive/negative (mined)")
    ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L6-v2",
                    help="Starting cross-encoder (any BERT-family id works too)")
    ap.add_argument("--output", default="./output_reranker")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--eval-split", type=float, default=0.1,
                    help="Held-out fraction for the pairwise accuracy report")
    args = ap.parse_args()

    from sentence_transformers import InputExample  # pylint: disable=C0415,E0401
    from sentence_transformers.cross_encoder import CrossEncoder  # pylint: disable=C0415,E0401
    from torch.utils.data import DataLoader  # pylint: disable=C0415,E0401

    rows = load_rows(Path(args.pairs))
    if not rows:
        print(f"no anchor/positive/negative rows in {args.pairs} — "
              "run mine_hard_negatives.py first", file=sys.stderr)
        return 1
    n_eval = max(1, int(len(rows) * args.eval_split)) if len(rows) > 4 else 0
    train_rows, eval_rows = rows[n_eval:], rows[:n_eval]

    examples = []
    for row in train_rows:
        examples.append(InputExample(texts=[row["anchor"], row["positive"]], label=1.0))
        examples.append(InputExample(texts=[row["anchor"], row["negative"]], label=0.0))
    print(f"{len(train_rows)} triples -> {len(examples)} labeled pairs "
          f"({n_eval} held out)")

    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_seq)
    model.fit(train_dataloader=DataLoader(examples, shuffle=True,
                                          batch_size=args.batch_size),
              epochs=args.epochs,
              optimizer_params={"lr": args.lr},
              show_progress_bar=True)
    model.save(args.output)
    print(f"saved reranker -> {args.output}")

    if eval_rows:
        pos_scores = model.predict([(r["anchor"], r["positive"]) for r in eval_rows])
        neg_scores = model.predict([(r["anchor"], r["negative"]) for r in eval_rows])
        wins = sum(p > n for p, n in zip(pos_scores, neg_scores, strict=False))
        print(f"pairwise accuracy on held-out triples: {wins}/{len(eval_rows)} "
              f"({wins / len(eval_rows):.1%}) — positive should outscore its hard negative")
    print("Next: python scripts/eval_embedder.py data/embedding_val.jsonl "
          f"--tuned <embedder> --reranker {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
