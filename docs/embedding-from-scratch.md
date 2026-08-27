# Build a Cohere-style embedding model + reranker

You asked: *can we build an embedding model like Cohere's, since theirs (and their
rerankers) are the best?* Honest answer: their **general-domain** quality comes
from a data + compute moat — billions of curated pairs, huge-batch distributed
training — not from secret techniques. The *recipe* is public, it scales down,
and on **your own domain** (e.g. 200 GB of PLM documentation) a well-executed
scaled-down version can genuinely beat general-purpose APIs, because domain
specificity is exactly where generic models are weakest.

| | Cohere-class (general) | This repo on your 200 GB corpus | True from-scratch toy |
|---|---|---|---|
| Pairs | ~billions, curated, multilingual | millions (structure-mined) + 100k+ (LLM-generated) | thousands |
| Compute | GPU fleets, weeks | laptop + one cloud GPU for days | laptop, hours |
| Result | best everywhere | **best on your domain** | educational only |

## The recipe (what Cohere-style training actually is)

1. **Strong pretrained backbone** — nobody random-inits production embedders.
2. **Stage 1: large-batch contrastive pretraining** on huge weakly-supervised pairs.
   The batch *is* the negative pool: every other example in the batch is a negative,
   so effective batch size directly sets quality.
3. **Stage 2: hard-negative fine-tune** — mine "almost right" candidates with the
   stage-1 model and train against them.
4. **Reranker**: a cross-encoder trained on the same mined data — reads query and
   candidate together, catches what embedding distance can't.
5. **Evaluate every stage** — retriever recall, then rerank lift.

Soup can't run this (its embedding task is LoRA-only, ignores hard negatives, and
outputs an adapter); the scripts below train **all weights** via sentence-transformers,
with GradCache making 1024+ effective batches fit in 12 GB VRAM.

## Step 0 — extraction at corpus scale (routed, not one-engine)

At millions of pages, forcing every page through one OCR engine destroys either
throughput or fidelity. Route each document to the cheapest engine that does it
justice — engine ranking for technical documentation: **#1 MinerU** (best first
PoC), **#2 PaddleOCR PP-StructureV3** (the production ingestion engine),
**#3 Surya OCR** (lighter fallback), **#4 olmOCR/VLM** (recovery only):

```
Fast Path        digital PDF                  → native extraction (no OCR)
OCR Path         scanned document             → PaddleOCR PP-StructureV3
Complex Path     tables + equations + diagrams → MinerU / Surya
Recovery Path    very difficult pages         → olmOCR / VLM
```

Bad structural extraction cannot be repaired later by BM25 + RRF + reranking —
this routing decision is one of the highest-leverage choices in the whole stack.

[`scripts/bulk_extract.py`](../scripts/bulk_extract.py) implements the router: it
classifies every PDF (text layer? figure-dense?), runs the Fast Path itself in
parallel (resumable), writes `route_ocr.txt` / `route_complex.txt` /
`route_recovery.txt` manifests for the heavier engines, and can drive MinerU over
the complex+OCR queues directly:

```bash
pip install pypdf
python scripts/bulk_extract.py D:/plm_corpus --output corpus_chunks/ --workers 8
python scripts/bulk_extract.py D:/plm_corpus --output corpus_chunks/ --run-mineru   # quality tier
```

Throughput: the Fast Path handles hundreds of GB in days on a multicore CPU; the
MinerU/Paddle tiers work through their (much smaller) queues on the GPU.

## Step 1 — millions of pairs without an LLM

[`scripts/build_weak_pairs.py`](../scripts/build_weak_pairs.py) mines training
pairs from document *structure* (the E5/GTE trick): title ↔ first chunk,
section heading ↔ section text, adjacent passages. CPU-speed, scales to the
whole corpus, consumes both the router's JSONL and MinerU Markdown:

```bash
python scripts/build_weak_pairs.py corpus_chunks/ --output-dir data_weak/
```

(Your LLM-generated pairs from `build_pairs.py` stay the premium stage-2 data.)

## Step 2 — large-batch contrastive training

[`scripts/train_embedder.py`](../scripts/train_embedder.py) — full-parameter
InfoNCE with GradCache:

```bash
pip install sentence-transformers
python scripts/train_embedder.py --pairs data_weak/embedding_train.jsonl \
    --base BAAI/bge-base-en-v1.5 --batch-size 512 --cache-chunk 16 --output ./embedder_s1
```

- `--batch-size` is the quality lever (it's the in-batch negative pool);
  `--cache-chunk` is the VRAM lever — raise batch, lower chunk, until it fits.
- `--base bert-base-uncased` for the never-was-an-embedder path;
  `--random-init --layers 4 --hidden 256` plus `--mlm-epochs 2 --corpus corpus_chunks/`
  for the true from-scratch educational tier (expect it to lose — see the table).
- Laptop: subset runs and the full recipe rehearsal. Corpus scale: one 24–48 GB
  cloud GPU (A10G/L4/L40S) for hours–days — see [deployment.md](deployment.md).

## Step 3 — hard negatives, then round 2

```bash
python scripts/mine_hard_negatives.py data/embedding_train.jsonl \
    --model ./embedder_s1 --corpus corpus_chunks/
python scripts/train_embedder.py --pairs data/embedding_train.hard.jsonl \
    --base ./embedder_s1 --batch-size 128 --epochs 2 --output ./embedder_final
```

The miner takes each anchor's top-ranked non-gold chunks (with a similarity guard
against false negatives) — the "almost right" answers that teach fine distinctions.

## Step 4 — the reranker

```bash
python scripts/train_reranker.py data/embedding_train.hard.jsonl --output ./output_reranker
```

Starts from `cross-encoder/ms-marco-MiniLM-L6-v2` by default (or any BERT id).
Serving: TEI serves rerankers too (same container family as the embedder).

## Step 5 — prove the whole stack

```bash
python scripts/eval_embedder.py data/embedding_val.jsonl \
    --tuned ./embedder_final --reranker ./output_reranker
```

```
     model    recall@1    recall@5     mrr
      base       0.41        0.70     0.53
     tuned       0.55        0.86     0.67
  tuned+rr       0.63        0.90     0.74   ← the reranker's lift
```

Each stage must beat the previous one on the held-out split — if it doesn't,
fix the data before adding compute.

## Where the Soup pipeline fits

The [fine-tuning guide](finetune-embedding.md) (Soup + LoRA) remains the fastest
path to a good domain embedder. Its adapter output needs
[`scripts/merge_embedding_adapter.py`](../scripts/merge_embedding_adapter.py)
before TEI/production serving. This guide is the next rung: full-parameter,
staged, reranked — the Cohere shape.
