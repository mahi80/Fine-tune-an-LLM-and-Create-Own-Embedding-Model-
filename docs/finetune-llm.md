# Fine-tune an LLM (QLoRA SFT → GGUF → Ollama)

Fine-tune a chat LLM on your own instruction data and deploy it locally. Reference run: **Qwen2.5-7B-Instruct** on an RTX 4080 Laptop (12 GB VRAM), Windows 11. Complete the [environment setup](../README.md#setup-windows-11-nvidia-gpu) first.

## 1. Smoke test (TinyLlama, ~1 min)

Verifies the whole stack — download → train → save — before committing to a big model:

```bash
soup quickstart --dry-run          # writes quickstart_data.jsonl + quickstart_soup.yaml
soup train --config configs/quickstart_soup.yaml --yes
```

## 2. Prepare your data

Supported JSONL formats (auto-detected via `format: auto`):

```json
{"instruction": "What does the export command do?", "input": "", "output": "It converts a trained model to..."}
```

or chat format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Validate before training:

```bash
soup data validate ./data/train.jsonl
soup data doctor ./data/train.jsonl --model Qwen/Qwen2.5-7B-Instruct
```

## 3. QLoRA fine-tune

```bash
soup profile --config configs/soup_qwen7b.yaml   # pre-flight: ~7.2 GB VRAM estimate
soup train  --config configs/soup_qwen7b.yaml --yes
```

Measured: **7.3 GB peak VRAM** on the 12 GB card, LoRA adapter (r=16, alpha=32, 4-bit NF4 base) saved to `output_qwen7b/`.

The config ([configs/soup_qwen7b.yaml](../configs/soup_qwen7b.yaml)) ships pointed at Soup's 20-example demo data — swap `data.train` for your dataset and raise `epochs` to 3 and `max_length` to 2048 for a real run. Sizing guide for other cards: 8 GB → ~7B QLoRA, 12 GB → 7–8B comfortable, 16 GB → ~14B.

## 4. Verify the adapter

```bash
python scripts/verify_adapter.py   # loads base 4-bit + adapter, generates test replies (~6 GB VRAM)
```

(Soup's own `soup infer` loads fp16 with no 4-bit option, so a 7B doesn't fit in 12 GB that way — hence this script.)

## 5. Export to GGUF + deploy to Ollama

```bash
# f16 GGUF needs no llama.cpp build (Soup auto-clones the convert script)
soup export --model output_qwen7b --format gguf --quant f16

# Ollama quantizes on import — no C++ toolchain needed
ollama create soup-qwen7b -q q4_K_M -f Modelfile
ollama run soup-qwen7b
```

The f16 GGUF (~14 GB for a 7B) can be deleted after `ollama create` — Ollama stores its own quantized copy (~4.7 GB).
