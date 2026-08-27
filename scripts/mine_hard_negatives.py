"""Hard-negative mining (Cohere-recipe stage 2 fuel).

Uses the current embedder to find, for every anchor, the highest-ranked chunk
that is NOT its positive — the "almost right" documents that teach the model
fine distinctions. A cosine-similarity guard against the positive filters
near-duplicates (false negatives). Rewrites the pairs JSONL with a `negative`
field, ready for a second training round and for the reranker.

Usage:
    python mine_hard_negatives.py data/embedding_train.jsonl \
        --model ./output_embedding_st --output data/embedding_train.hard.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pairs", help="anchor/positive JSONL to enrich")
    ap.add_argument("--model", required=True, help="Embedder dir (ST format) or HF id")
    ap.add_argument("--output", help="Default: <pairs>.hard.jsonl")
    ap.add_argument("--corpus", help="Optional chunk JSONL dir to widen the candidate pool")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--false-negative-sim", type=float, default=0.95,
                    help="Candidates this similar to the positive are assumed true and skipped")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    from sentence_transformers import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel,import-error
        SentenceTransformer,
        util,
    )

    rows = []
    with Path(args.pairs).open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("anchor") and row.get("positive"):
                rows.append(row)
    if not rows:
        print(f"no pairs in {args.pairs}", file=sys.stderr)
        return 1

    pool = sorted({r["positive"] for r in rows})
    if args.corpus:
        for f in sorted(Path(args.corpus).rglob("*.jsonl")):
            if f.name.startswith("route_"):
                continue
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        text = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    if len(text) >= 100:
                        pool.append(text)
        pool = sorted(set(pool))
    index = {text: i for i, text in enumerate(pool)}
    print(f"{len(rows)} anchors against a pool of {len(pool)} chunks")

    model = SentenceTransformer(args.model)
    doc_emb = model.encode(pool, batch_size=args.batch_size, convert_to_tensor=True,
                           normalize_embeddings=True, show_progress_bar=True)
    query_emb = model.encode([r["anchor"] for r in rows], batch_size=args.batch_size,
                             convert_to_tensor=True, normalize_embeddings=True,
                             show_progress_bar=True)
    hits = util.semantic_search(query_emb, doc_emb, top_k=args.top_k + 1)

    mined = skipped = 0
    for row, ranked in zip(rows, hits, strict=False):
        gold = index[row["positive"]]
        gold_emb = doc_emb[gold]
        negative = None
        for hit in ranked:
            cid = hit["corpus_id"]
            if cid == gold:
                continue
            if float(util.cos_sim(doc_emb[cid], gold_emb)) >= args.false_negative_sim:
                continue  # near-duplicate of the positive -> likely a true answer
            negative = pool[cid]
            break
        if negative:
            row["negative"] = negative
            mined += 1
        else:
            skipped += 1

    # negatives must be all-or-nothing for training -> drop rows we couldn't mine
    kept = [r for r in rows if r.get("negative")]
    out = Path(args.output) if args.output else Path(args.pairs).with_suffix(".hard.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"mined hard negatives for {mined} rows ({skipped} skipped) -> {out}")
    return 0 if mined else 1


if __name__ == "__main__":
    sys.exit(main())
