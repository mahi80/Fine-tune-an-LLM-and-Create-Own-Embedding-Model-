# Getting started (no experience needed)

This walkthrough assumes you have never trained a model before. You only need to copy-paste commands into **PowerShell** (press the Windows key, type `powershell`, press Enter).

You can build two things with this repo:

- **Path A — your own chat model**: an AI you can talk to, taught with your question/answer examples. ([full guide](finetune-llm.md))
- **Path B — smart document search**: turn a folder of PDF manuals into a search engine that understands meaning, not just keywords. ([full guide](finetune-embedding.md))

They combine naturally: B finds the right page, A writes the answer. Do them in either order.

## 1. What computer do you need?

| | Minimum | Comfortable |
|---|---|---|
| **Graphics card (GPU)** | NVIDIA with **8 GB VRAM** (e.g. RTX 3060 Ti, 4060) | NVIDIA with **12 GB+ VRAM** (e.g. RTX 3060 12GB, 4070, 4080 laptop) |
| **RAM** | 16 GB | 32 GB |
| **Free disk** | 60 GB | 150 GB+ |
| **OS** | Windows 10/11 (this repo's instructions), Linux also works | |
| **Internet** | Needed once, to download models (15–40 GB total) | |

To check your GPU: press Ctrl+Shift+Esc → Performance tab → GPU. It must say NVIDIA and show "Dedicated GPU memory" of 8 GB or more.

- **8 GB VRAM**: everything works, with the smaller settings noted inline below.
- **4–6 GB VRAM**: Path B works (use MinerU's `-b pipeline` mode); for Path A use a small model (TinyLlama) — or Soup's layer-streaming feature, which has trained 8B models on a 4 GB card (~1.4× slower).
- **No NVIDIA GPU**: extraction (step B1) can run on CPU; training realistically cannot. Consider a cloud GPU rental for the training steps.

## 2. One-time setup (~30 min, mostly downloads)

Copy-paste each block into PowerShell, one at a time. If a step prints an error, stop and fix it before continuing.

**2.1 Install Python 3.12 and Ollama:**

```bash
winget install Python.Python.3.12 Ollama.Ollama
```

Close PowerShell and open a new one (so the installs are picked up).

**2.2 Make a project folder with an isolated Python environment:**

```bash
cd $HOME; mkdir ai-lab; cd ai-lab; py -3.12 -m venv .venv; .venv\Scripts\activate
```

Your prompt now starts with `(.venv)`. **Every later command assumes this** — if you open a new PowerShell, run `cd $HOME\ai-lab; .venv\Scripts\activate` first.

**2.3 Install the training tools** (~3 GB download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

```bash
git clone https://github.com/MakazhanAlpamys/Soup.git; pip install -e "Soup[train]"
```

```bash
git clone https://github.com/mahi80/LLMFinetuning.git
```

**2.4 Check everything:**

```bash
soup doctor
```

Every row must say OK and your GPU must be listed. If not, the output tells you exactly what to fix.

## 3. Path A — your own chat model

Rough total time on a 12 GB card: **2–4 hours**, most of it unattended.

| Step | Command | Time (12 GB GPU) |
|---|---|---|
| A1. Smoke test — proves the setup works with a tiny model | `soup quickstart --dry-run` then `soup train --config quickstart_soup.yaml --yes` | ~5 min (incl. 2 GB download) |
| A2. Get training data — 1,000+ question/answer examples in a `train.jsonl` file (format below), or generate them from PDFs via Path B's `--sft` flag | — | varies |
| A3. Train | `soup train --config LLMFinetuning/configs/soup_qwen7b.yaml --yes` (first edit `data.train` in that file to point at your data, set `epochs: 3`) | 15 GB download once, then **~1–2 h per 1,000 examples × 3 epochs** |
| A4. Test it | `python LLMFinetuning/scripts/verify_adapter.py` and `soup chat --model ./output_qwen7b` | minutes |
| A5. Make it a local app | `soup export --model output_qwen7b --format gguf --quant f16` then `ollama create my-model -q q4_K_M -f LLMFinetuning/Modelfile` then `ollama run my-model` | ~20–30 min |

Data format for A2 — one line per example in a text file saved as `train.jsonl`:

```json
{"instruction": "How do I reset a user password?", "input": "", "output": "Open the admin panel, select the user, click Reset Password..."}
```

**8 GB card?** In `configs/soup_qwen7b.yaml` keep `batch_size: 2` and `max_length: 1024` (as shipped). Peak measured use is 7.3 GB — close other GPU-using apps (games, browser video).

## 4. Path B — smart search over your PDF manuals

Rough total time for **one 300-page manual with ~150 screenshots**: **2–4 hours**, almost all unattended.

Good to know: several AI models help along the way (reading PDFs, describing screenshots, writing practice questions), but they are all frozen tools — **the only model that actually gets trained is the small search model (`BAAI/bge-base-en-v1.5`) in step B4**, and the trained copy lands in `./output_embedding`. (Likewise in Path A, the model being trained is Qwen2.5-7B-Instruct.)

**B0. One-time extras:**

```bash
pip install "mineru[core]"; ollama pull qwen3-vl:8b; ollama pull qwen2.5:7b
```

(~15 GB of downloads: MinerU's extraction models + two local AI models.)

**B1. Extract the PDFs** — put your PDFs in a folder called `pdfs`, then:

```bash
mineru -p pdfs/ -o extracted/ -b vlm-transformers
```

- Speed: very roughly **500–1,500 pages/hour** on a 12 GB card (first run is slower — model downloads).
- **How many PDFs in one go? There is no hard limit** — it processes them one after another; the limits are time and disk. Practical advice: run **one PDF first** end-to-end to validate the whole pipeline, then batch the rest (50 manuals ≈ an overnight run). Extracted output on disk ≈ the size of the PDFs themselves.
- On 8 GB (or no) GPU: use `-b pipeline` instead — lighter, still good.

**B2. Describe the screenshots** (so pictures become searchable text) — once per extracted manual:

```bash
python LLMFinetuning/scripts/caption_images.py extracted/<manual-name>/auto/<manual-name>.md
```

Speed: **~5–15 s per image** → 150 screenshots ≈ 15–40 min. Safe to interrupt — rerunning skips what's done.

**B3. Generate training pairs** from everything extracted:

```bash
python LLMFinetuning/scripts/build_pairs.py extracted/ --output-dir data/
```

Speed: **~5–15 s per chunk of text** → a 300-page manual (≈ 400 chunks) ≈ 1–2 h. Add `--sft` to also produce Path A training data in the same run (roughly doubles the time). Aim for 1,000+ pairs before training.

**B4. Train the search model** (this one is light — works even on small GPUs):

```bash
soup train --config LLMFinetuning/configs/soup_embedding.yaml --yes
```

Speed: **~10–30 min** for a few thousand pairs.

**B5. Prove it worked** — compares your tuned model against the original on held-back questions:

```bash
pip install sentence-transformers; python LLMFinetuning/scripts/eval_embedder.py data/embedding_val.jsonl --tuned ./output_embedding
```

You want the `tuned` row's numbers higher than `base`. If not, generate more pairs in B3 (`--questions-per-chunk 3`).

## 5. Cheat-sheet: what runs where, how long

| Stage | GPU memory it needs | Time (12 GB card) |
|---|---|---|
| MinerU extraction (`vlm-transformers`) | ~8 GB | ~500–1,500 pages/h |
| MinerU extraction (`pipeline`) | ~4 GB or CPU | similar or faster on GPU |
| Screenshot captioning (qwen3-vl:8b) | ~6 GB | 5–15 s per image |
| Pair/Q&A generation (qwen2.5:7b) | ~5 GB | 5–15 s per chunk |
| Embedding training (BGE-base) | ~2–4 GB | 10–30 min |
| LLM QLoRA training (Qwen2.5-7B) | ~7.3 GB measured | ~1–2 h per 1k examples × 3 epochs |
| GGUF export + Ollama import | CPU/RAM mostly | ~20–30 min |

Run stages **one at a time** — two GPU stages at once will fight over memory. All time figures are rough; the first run of anything is slower because models download.

## 6. When something goes wrong

- **"CUDA out of memory"** → close other GPU apps; in the config lower `batch_size` (e.g. to 1) or `max_length`.
- **`ollama`/`soup`/`mineru` "not recognized"** → open a new PowerShell and re-activate: `cd $HOME\ai-lab; .venv\Scripts\activate`.
- **Training crashes mid-way** → `soup train --config ... --resume`. Captioning (B2) resumes by just rerunning it.
- **A step needs confirmation and hangs in a script** → add `--yes` to `soup train` commands.
- **Model gives poor answers after training** → almost always a data problem: more examples, better examples.
