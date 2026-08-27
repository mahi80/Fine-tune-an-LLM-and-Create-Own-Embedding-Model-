"""Sanity check: load a base LLM in 4-bit + a trained LoRA adapter, generate replies.

Usage:
    python verify_adapter.py                       # defaults below
    python verify_adapter.py --base Qwen/Qwen2.5-7B-Instruct --adapter ./output_qwen7b
"""

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=r"C:\PROJECTS_MAHI\SOUP\output_qwen7b")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("prompts", nargs="*",
                    default=["What is the capital of France?", "Write a haiku about soup."])
    args = ap.parse_args()

    # heavy deps come from the Soup training venv, loaded lazily
    import torch  # pylint: disable=import-outside-toplevel,import-error
    from peft import PeftModel  # pylint: disable=import-outside-toplevel,import-error
    from transformers import (  # pylint: disable=import-outside-toplevel,import-error
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    for q in args.prompts:
        msgs = [{"role": "user", "content": q}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to("cuda:0")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        print("Q:", q)
        print("A:", tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip())
        print("-" * 60)
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
