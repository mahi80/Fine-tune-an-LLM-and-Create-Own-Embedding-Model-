# LLMFinetuning

Local LLM fine-tuning pipeline: **Qwen2.5-7B-Instruct → QLoRA SFT → GGUF → Ollama**, running entirely on a consumer laptop GPU (RTX 4080 Laptop, 12 GB VRAM, Windows 11).

Built with [Soup](https://github.com/MakazhanAlpamys/Soup) (`soup-cli`, Apache-2.0), a CLI-first fine-tuning tool. The files in this repo (configs, scripts, docs) are MIT-licensed.

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

## Pipeline

### 1. Smoke test (TinyLlama, ~1 min)

```bash
soup quickstart --dry-run          # writes quickstart_data.jsonl + quickstart_soup.yaml
soup train --config configs/quickstart_soup.yaml --yes
```

### 2. QLoRA fine-tune (Qwen2.5-7B-Instruct)

```bash
soup profile --config configs/soup_qwen7b.yaml   # pre-flight: ~7.2 GB VRAM estimate
soup train  --config configs/soup_qwen7b.yaml --yes
```

Measured: **7.3 GB peak VRAM** on the 12 GB card, LoRA adapter (r=16, alpha=32, 4-bit NF4 base) saved to `output_qwen7b/`.

The demo config trains on Soup's 20-example quickstart data — swap `data.train` for your own dataset (alpaca `{"instruction", "input", "output"}` or chatml `{"messages": [...]}` JSONL) and raise `epochs` / `max_length` for a real run.

### 3. Verify the adapter

```bash
python scripts/verify_adapter.py   # loads base 4-bit + adapter, generates test replies (~6 GB VRAM)
```

### 4. Export to GGUF + Ollama

```bash
# f16 GGUF needs no llama.cpp build (Soup auto-clones the convert script)
soup export --model output_qwen7b --format gguf --quant f16

# Ollama quantizes on import — no C++ toolchain needed
ollama create soup-qwen7b -q q4_K_M -f Modelfile
ollama run soup-qwen7b
```

## Windows gotchas hit along the way

- **`batch_size: auto` hard-OOMs**: Soup's auto batch probe can crash with `cudaErrorMemoryAllocation` on Windows/WDDM instead of backing off. Use a fixed `batch_size`.
- **Relative paths in configs** resolve against the shell's cwd — use absolute paths in `data.train` and `output`.
- **`soup train` prompts for confirmation** — pass `--yes` in scripts/CI.
- **`soup infer` loads fp16** (no 4-bit option), so a 7B won't fit in 12 GB for inference — use `scripts/verify_adapter.py` or the exported GGUF instead.
- **transformers 5.x**: `apply_chat_template(..., return_tensors="pt")` returns a dict — pass `return_dict=True` and take `["input_ids"]`.

## Repo contents

| Path | What |
|---|---|
| `configs/soup_qwen7b.yaml` | Qwen2.5-7B QLoRA SFT config (the main one) |
| `configs/quickstart_soup.yaml` | TinyLlama smoke-test config |
| `scripts/verify_adapter.py` | 4-bit + LoRA adapter generation sanity check |
| `data/test_prompts.jsonl` | Prompts for inference sanity checks |
| `Modelfile` | Ollama import file for the exported GGUF |

Model weights, adapters, and GGUF files are git-ignored (multi-GB).

## License

MIT — see [LICENSE](LICENSE). Soup itself is Apache-2.0, © its authors.
