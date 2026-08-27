"""Compare a fine-tuned embedder against its base model on held-out pairs.

Uses the val split from build_pairs.py: each anchor (question) should retrieve
its own positive (chunk) from the corpus of all positives. Reports Recall@k
and MRR for both models so you can see whether fine-tuning actually helped.

Usage:
    pip install sentence-transformers
    python eval_embedder.py data/embedding_val.jsonl --tuned ./output_embedding

Exit code 0 if the tuned model wins on Recall@5, 1 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path


def load_pairs(path: Path) -> tuple[list[str], list[str]]:
    anchors, positives = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            anchors.append(row["anchor"])
            positives.append(row["positive"])
    return anchors, positives


def evaluate(model_name: str, anchors: list[str], corpus: list[str],
             gold: list[int], ks: tuple[int, ...]) -> dict:
    # heavy optional dep, loaded lazily (install the .[eval] extra)
    from sentence_transformers import (  # pylint: disable=import-outside-toplevel,import-error
        SentenceTransformer,
        util,
    )
    model = SentenceTransformer(model_name)
    q = model.encode(anchors, convert_to_tensor=True, show_progress_bar=False,
                     normalize_embeddings=True)
    d = model.encode(corpus, convert_to_tensor=True, show_progress_bar=False,
                     normalize_embeddings=True)
    hits = util.semantic_search(q, d, top_k=max(ks))
    recall = {k: 0 for k in ks}
    mrr = 0.0
    for i, ranked in enumerate(hits):
        ids = [h["corpus_id"] for h in ranked]
        for k in ks:
            if gold[i] in ids[:k]:
                recall[k] += 1
        if gold[i] in ids:
            mrr += 1.0 / (ids.index(gold[i]) + 1)
    n = len(anchors)
    return {**{f"recall@{k}": recall[k] / n for k in ks}, "mrr": mrr / n}


def evaluate_reranked(model_name: str, reranker_dir: str,  # pylint: disable=R0913,R0917
                      anchors: list, corpus: list,
                      gold: list, ks: tuple, retrieve_k: int = 20) -> dict:
    """Retrieve with the embedder, rescore the top candidates with a cross-encoder."""
    # heavy optional deps, loaded lazily (install the .[eval] extra)
    from sentence_transformers import (  # pylint: disable=import-outside-toplevel,import-error
        SentenceTransformer,
        util,
    )
    from sentence_transformers.cross_encoder import (  # pylint: disable=import-outside-toplevel,import-error
        CrossEncoder,
    )
    model = SentenceTransformer(model_name)
    reranker = CrossEncoder(reranker_dir)
    q = model.encode(anchors, convert_to_tensor=True, show_progress_bar=False,
                     normalize_embeddings=True)
    d = model.encode(corpus, convert_to_tensor=True, show_progress_bar=False,
                     normalize_embeddings=True)
    hits = util.semantic_search(q, d, top_k=max(retrieve_k, *ks))
    recall = {k: 0 for k in ks}
    mrr = 0.0
    for i, ranked in enumerate(hits):
        ids = [h["corpus_id"] for h in ranked]
        scores = reranker.predict([(anchors[i], corpus[cid]) for cid in ids])
        ids = [cid for _, cid in sorted(zip(scores, ids, strict=False), key=lambda x: -x[0])]
        for k in ks:
            if gold[i] in ids[:k]:
                recall[k] += 1
        if gold[i] in ids:
            mrr += 1.0 / (ids.index(gold[i]) + 1)
    n = len(anchors)
    return {**{f"recall@{k}": recall[k] / n for k in ks}, "mrr": mrr / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("val_pairs", help="embedding_val.jsonl from build_pairs.py")
    ap.add_argument("--base", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--tuned", default="./output_embedding")
    ap.add_argument("--reranker", help="Cross-encoder dir: adds a tuned+rerank row")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    args = ap.parse_args()

    anchors, positives = load_pairs(Path(args.val_pairs))
    if not anchors:
        print("No pairs in file.", file=sys.stderr)
        return 1
    # dedup corpus (several anchors can share a positive chunk)
    corpus = sorted(set(positives))
    index = {c: i for i, c in enumerate(corpus)}
    gold = [index[p] for p in positives]
    ks = tuple(sorted(args.k))
    print(f"{len(anchors)} queries against {len(corpus)} unique chunks\n")

    results = {}
    for label, name in (("base", args.base), ("tuned", args.tuned)):
        print(f"evaluating {label}: {name}")
        results[label] = evaluate(name, anchors, corpus, gold, ks)
    if args.reranker:
        print(f"evaluating tuned+rerank: {args.reranker}")
        results["tuned+rr"] = evaluate_reranked(args.tuned, args.reranker,
                                                anchors, corpus, gold, ks)

    header = ["model"] + [f"recall@{k}" for k in ks] + ["mrr"]
    print("\n" + "  ".join(f"{h:>10}" for h in header))
    for label, res in results.items():
        row = [label] + [f"{res[f'recall@{k}']:.3f}" for k in ks] \
              + [f"{res['mrr']:.3f}"]
        print("  ".join(f"{v:>10}" for v in row))

    k_ref = 5 if 5 in ks else ks[0]
    won = results["tuned"][f"recall@{k_ref}"] >= results["base"][f"recall@{k_ref}"]
    print(f"\ntuned model {'matches or beats' if won else 'LOSES to'} base on recall@{k_ref}")
    return 0 if won else 1


if __name__ == "__main__":
    sys.exit(main())
