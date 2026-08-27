"""Sanity check: load Qwen2.5-7B in 4-bit + trained LoRA adapter, generate a reply."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = r"C:\PROJECTS_MAHI\SOUP\output_qwen7b"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="cuda:0")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

for q in ["What is the capital of France?", "Write a haiku about soup."]:
    msgs = [{"role": "user", "content": q}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=80, do_sample=False)
    print("Q:", q)
    print("A:", tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip())
    print("-" * 60)
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
