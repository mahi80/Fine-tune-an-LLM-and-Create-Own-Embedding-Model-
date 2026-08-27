# Fine-tune an embedding model (PDF → RAG)

Turn a folder of PDFs (e.g. enterprise software manuals or product documentation) into a domain-tuned sentence-embedding model. Best used for retrieval (RAG): the tuned embedder finds the right manual chunks, and an LLM answers over them. Complete the [environment setup](../README.md#setup-windows-11-nvidia-gpu) first.

The stack (all local, fits 12 GB VRAM): **MinerU** for extraction → **Qwen3-VL** for screenshot captions → chunk → contrastive pairs → **BGE** fine-tune with Soup.

## 1. Extract the PDFs with MinerU

[MinerU](https://github.com/opendatalab/MinerU) beats naive text extraction (pypdf) on everything that matters for technical docs: it outputs Markdown with **math as LaTeX**, **tables as HTML**, correct reading order, and it **extracts embedded images/screenshots to files** with their references linked in the Markdown. It OCRs scanned pages automatically. Native Windows, needs Python 3.10–3.12 (3.13 breaks), ~8 GB VRAM for the high-accuracy VLM backend:

```bash
pip install "mineru[core]"
mineru -p manual.pdf -o extracted/ -b vlm-transformers
```

(First run downloads the models. `-b pipeline` is the lighter classic backend — runs on ~4 GB VRAM or CPU.)

Fallbacks: `soup data ingest manual.pdf` (pypdf, text-only — fine for simple prose PDFs); [Docling](https://github.com/docling-project/docling) (IBM, MIT-licensed) if strict licensing matters — slightly weaker math fidelity.

## 2. Caption screenshots with a local VLM

MinerU extracts UI screenshots as image files but doesn't describe them. Run a second pass with Qwen3-VL via Ollama (Apache-2.0, trained for GUI/screen understanding — needs Ollama ≥ 0.12.7):

```bash
ollama pull qwen3-vl:8b
```

For each image MinerU extracted, ask it to *"describe this screenshot from software documentation: which application, which dialog/menu, which options are visible"*, and splice the description into the Markdown at the image reference. This makes screenshots searchable text. Run it as a separate pass so MinerU and the VLM don't contend for VRAM.

Math needs no extra step — MinerU already emits LaTeX, and embedding/LLM models understand LaTeX natively. Tip: for formula-dense chunks, also add a one-line plain-language gloss next to the equation — users query in words, not LaTeX.

## 3. Build contrastive pairs

The embedding trainer wants anchor/positive JSONL (optionally with a negative):

```json
{"anchor": "How do I create a revision rule?", "positive": "<the doc chunk that answers it>"}
{"anchor": "query", "positive": "relevant chunk", "negative": "unrelated chunk"}
```

Two ways to produce them from the extracted chunks:

- **Synthetic questions via a local LLM** (recommended): for each chunk, have a local Ollama model generate 1–3 questions that the chunk answers → each `(question, chunk)` becomes an `(anchor, positive)` pair. Soup's `soup data forge --docs <dir> --task sft --judge-provider ollama --judge-model <model>` automates Q&A synthesis (always pass `--judge-provider` — the offline default generates placeholder junk).
- **Structure-based (no LLM)**: use each section heading as the anchor and the section body as the positive. Weaker but free.

Add negatives by sampling chunks from a *different* document (e.g. a chunk from one product's manual as the negative for another product's anchor).

## 4. Configure and train

```bash
soup init --template embedding -o soup_embedding.yaml   # or use configs/soup_embedding.yaml
soup train --config soup_embedding.yaml --yes
```

Key config (see [configs/soup_embedding.yaml](../configs/soup_embedding.yaml)): base `BAAI/bge-base-en-v1.5` (~110M params — trains fast at full precision on any GPU we'd use here), `task: embedding`, `format: embedding`, `embedding_loss: contrastive` (InfoNCE; also `triplet` with `embedding_margin`, or `cosine`), `embedding_pooling: mean`.

## 5. Use the tuned embedder

The output is a Hugging Face model directory — no GGUF/Ollama step; load it in your RAG stack:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("./output_embedding")
vectors = model.encode(["How do I create a new design revision?"])
```

Point your vector store (Chroma, Qdrant, pgvector, …) at it and index the same chunks you extracted in step 1.

**If screenshot-centric queries still retrieve poorly** even with captions: consider late-interaction *visual* retrieval (ColPali-style — e.g. [ColModernVBERT](https://github.com/illuin-tech/modernvbert), 250M params, MIT), which embeds page images directly and skips extraction loss entirely — at the cost of replacing the text-vector design (needs a multi-vector-capable store like Qdrant ≥ 1.10). And if the answering LLM sits behind a reranker, prefer a local one (a BGE reranker; Soup can fine-tune `task: reranker`) over cloud rerank APIs when the docs are confidential.
