"""Full-parameter contrastive embedder training (the Cohere-style core).

Unlike the Soup pipeline (LoRA adapter over a frozen base), this trains ALL
weights with InfoNCE + in-batch negatives, and uses GradCache
(CachedMultipleNegativesRankingLoss) so the effective batch — which directly
sets embedding quality — can reach 1024+ on a 12 GB GPU. Output is a complete
sentence-transformers model directory, ready for ST, TEI, and vector stores.

Backbone options (--base):
  - BAAI/bge-base-en-v1.5 (default)  continue-training the best open embedder
  - bert-base-uncased / ModernBERT   a plain encoder that was never an embedder
  - a local dir                      e.g. the --random-init or MLM output below

Extra stages for the true from-scratch path:
  --random-init --layers 4 --hidden 256   materialize a small random encoder
  --mlm-epochs 2 --corpus corpus_chunks/  MLM warm-up on your extracted corpus

Usage:
    pip install sentence-transformers
    python train_embedder.py --pairs data/embedding_train.jsonl --batch-size 256
    python train_embedder.py --pairs data_weak/embedding_train.jsonl \
        --base bert-base-uncased --epochs 1 --batch-size 512
"""

import argparse
import json
import sys
from pathlib import Path


def load_pairs(paths: list) -> tuple:
    """-> (anchors, positives, negatives_or_None). Negatives all-or-nothing."""
    rows = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("anchor") and row.get("positive"):
                    rows.append(row)
    if not rows:
        raise SystemExit(f"no usable pairs in {paths}")
    negatives = [r.get("negative") for r in rows]
    use_neg = all(isinstance(n, str) and n for n in negatives)
    return ([r["anchor"] for r in rows], [r["positive"] for r in rows],
            negatives if use_neg else None)


def make_random_encoder(out_dir: Path, layers: int, hidden: int, tokenizer_name: str) -> str:
    from transformers import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel,import-error
        AutoTokenizer,
        BertConfig,
        BertForMaskedLM,
    )

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    config = BertConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=tok.vocab_size, hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=max(2, hidden // 32), intermediate_size=hidden * 4,
        max_position_embeddings=512)
    BertForMaskedLM(config).save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"random-initialized encoder ({layers} layers, hidden {hidden}) -> {out_dir}")
    return str(out_dir)


def collect_corpus(corpus_dir: Path) -> list:
    """Paragraph texts from bulk_extract JSONL and Markdown under corpus_dir."""
    texts = []
    for f in sorted(corpus_dir.rglob("*")):
        if f.suffix == ".jsonl" and not f.name.startswith("route_"):
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        text = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    if len(text) >= 80:
                        texts.append(text)
        elif f.suffix in (".md", ".txt"):
            for para in f.read_text(encoding="utf-8").split("\n\n"):
                if len(para.strip()) >= 80:
                    texts.append(para.strip())
    return texts


def mlm_pretrain(base: str, corpus_dir: Path, out_dir: Path, epochs: int,  # pylint: disable=R0913,R0917
                 max_seq: int, batch: int) -> str:
    import torch  # noqa: PLC0415  # pylint: disable=import-outside-toplevel,import-error
    from datasets import Dataset  # pylint: disable=C0415,E0401
    from transformers import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel,import-error
        AutoModelForMaskedLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    texts = collect_corpus(corpus_dir)
    if not texts:
        raise SystemExit(f"no corpus text under {corpus_dir}")
    print(f"MLM warm-up: {len(texts)} paragraphs, {epochs} epoch(s)")
    tok = AutoTokenizer.from_pretrained(base)

    def tokenize(batch_rows):
        return tok(batch_rows["text"], truncation=True, max_length=max_seq)

    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True,
                                                remove_columns=["text"])
    model = AutoModelForMaskedLM.from_pretrained(base)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "_ckpt"), num_train_epochs=epochs,
            per_device_train_batch_size=batch, report_to=[], save_strategy="no",
            logging_steps=100, fp16=torch.cuda.is_available()),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=True))
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"MLM-pretrained encoder -> {out_dir}")
    return str(out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", nargs="+", default=["data/embedding_train.jsonl"],
                    help="Anchor/positive[/negative] JSONL file(s)")
    ap.add_argument("--base", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--output", default="./output_embedding_st")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="EFFECTIVE batch = the in-batch negative pool; bigger is better")
    ap.add_argument("--cache-chunk", type=int, default=16,
                    help="GradCache mini-batch (VRAM knob); 0 = plain MNRL loss")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--random-init", action="store_true",
                    help="Start from random weights (educational tier)")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--tokenizer", default="bert-base-uncased",
                    help="Tokenizer reused for --random-init")
    ap.add_argument("--mlm-epochs", type=int, default=0,
                    help="MLM warm-up epochs before contrastive training")
    ap.add_argument("--corpus", help="Corpus dir for the MLM stage (bulk_extract output)")
    ap.add_argument("--workdir", default=".encoder_work")
    args = ap.parse_args()

    import torch  # noqa: PLC0415  # pylint: disable=import-outside-toplevel,import-error
    from datasets import Dataset  # pylint: disable=C0415,E0401
    from sentence_transformers import (  # pylint: disable=C0415,E0401,E0611
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
        models,
    )

    workdir = Path(args.workdir)
    base = args.base
    if args.random_init:
        base = make_random_encoder(workdir / "random_encoder", args.layers,
                                   args.hidden, args.tokenizer)
    if args.mlm_epochs > 0:
        if not args.corpus:
            raise SystemExit("--mlm-epochs needs --corpus <dir>")
        base = mlm_pretrain(base, Path(args.corpus), workdir / "mlm_encoder",
                            args.mlm_epochs, args.max_seq, batch=8)

    anchors, positives, negatives = load_pairs(args.pairs)
    print(f"{len(anchors)} pairs" + (" with hard negatives" if negatives else ""))
    columns = {"anchor": anchors, "positive": positives}
    if negatives:
        columns["negative"] = negatives
    ds = Dataset.from_dict(columns)

    transformer = models.Transformer(base, max_seq_length=args.max_seq)
    pooling = models.Pooling(getattr(transformer, "get_embedding_dimension",
                                     transformer.get_word_embedding_dimension)(),
                             pooling_mode="mean")
    model = SentenceTransformer(modules=[transformer, pooling])

    if args.cache_chunk > 0:
        loss = losses.CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=args.cache_chunk)
    else:
        loss = losses.MultipleNegativesRankingLoss(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(workdir / "_ckpt"), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, learning_rate=args.lr,
            report_to=[], save_strategy="no", logging_steps=50,
            fp16=torch.cuda.is_available()),
        train_dataset=ds,
        loss=loss)
    trainer.train()
    model.save(args.output)
    print(f"\nSaved sentence-transformers model -> {args.output}")
    print("Next: python scripts/mine_hard_negatives.py --model", args.output,
          "then a second training round, then scripts/train_reranker.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
