# LLMFinetuning

Local fine-tuning pipelines running entirely on a consumer laptop GPU (RTX 4080 Laptop, 12 GB VRAM, Windows 11), built with [Soup](https://github.com/MakazhanAlpamys/Soup) (`soup-cli`, Apache-2.0), a CLI-first fine-tuning tool. The files in this repo (configs, scripts, docs) are MIT-licensed.

Two guides, sharing one environment setup:

| Guide | What you get |
|---|---|
| **[Fine-tune an LLM](docs/finetune-llm.md)** | A chat model tuned on your instruction data: Qwen2.5-7B-Instruct → QLoRA SFT → GGUF → Ollama |
| **[Fine-tune an embedding model](docs/finetune-embedding.md)** | A domain-tuned retriever for RAG from a folder of PDFs: MinerU extraction → VLM screenshot captions → contrastive pairs → BGE |

They're complementary: the embedding model retrieves the right doc chunks, the LLM answers over them.

## ⚠️ Security note

This project originally started from `github.com/jochi2018/Soup`, which turned out to be a **malicious clone** of the real Soup repository. That fork adds a `src/3.0.zip` containing a Windows malware dropper (`Application.cmd` → `util.exe` + obfuscated Lua payload disguised as `cert.txt`) and replaces the README with a fake "Download Soup" download badge, while deleting the CI workflows to keep the tampering quiet. The Python source itself was unmodified from upstream.

**Use the real upstream instead: <https://github.com/MakazhanAlpamys/Soup>** — and in general, never run a "release zip" from a repo whose project is installed via `pip`.

## Setup (Windows 11, NVIDIA GPU)

Soup requires Python >=3.10,<3.13.

```bash
winget install Python.Python.3.12
py -3.12 -m venv .venv
.venv\Scripts\activate

# CUDA torch first (plain pip torch can be CPU-only on Windows)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Soup from the genuine upstream
git clone https://github.com/MakazhanAlpamys/Soup.git
pip install -e "Soup[train]"

soup doctor   # everything must be green
```

Versions that worked: `soup-cli 0.73.3`, `torch 2.13.0+cu126`, `transformers 5.16.1`, `trl 0.29.1`, `peft 0.20.0`, `bitsandbytes 0.50.2`.

Then pick your guide: **[LLM fine-tuning](docs/finetune-llm.md)** or **[embedding fine-tuning](docs/finetune-embedding.md)**.

## Windows gotchas hit along the way

- **`batch_size: auto` hard-OOMs**: Soup's auto batch probe can crash with `cudaErrorMemoryAllocation` on Windows/WDDM instead of backing off. Use a fixed `batch_size`.
- **Relative paths in configs** resolve against the shell's cwd — use absolute paths in `data.train` and `output`.
- **`soup train` prompts for confirmation** — pass `--yes` in scripts/CI.
- **`soup infer` loads fp16** (no 4-bit option), so a 7B won't fit in 12 GB for inference — use `scripts/verify_adapter.py` or the exported GGUF instead.
- **transformers 5.x**: `apply_chat_template(..., return_tensors="pt")` returns a dict — pass `return_dict=True` and take `["input_ids"]`.

## Repo contents

| Path | What |
|---|---|
| `docs/finetune-llm.md` | LLM fine-tuning guide (QLoRA SFT → GGUF → Ollama) |
| `docs/finetune-embedding.md` | Embedding fine-tuning guide (PDF → RAG) |
| `configs/soup_qwen7b.yaml` | Qwen2.5-7B QLoRA SFT config |
| `configs/quickstart_soup.yaml` | TinyLlama smoke-test config |
| `configs/soup_embedding.yaml` | Embedding-model fine-tune config |
| `scripts/verify_adapter.py` | 4-bit + LoRA adapter generation sanity check |
| `data/test_prompts.jsonl` | Prompts for inference sanity checks |
| `Modelfile` | Ollama import file for the exported GGUF |

Model weights, adapters, and GGUF files are git-ignored (multi-GB).

## License

MIT — see [LICENSE](LICENSE). Soup itself is Apache-2.0, © its authors.
