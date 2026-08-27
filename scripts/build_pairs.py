"""Build training data from extracted Markdown: embedding pairs and optional SFT rows.

Chunks Markdown files by headings, then uses a local Ollama text model to
generate questions each chunk answers. Emits:

  embedding_train.jsonl / embedding_val.jsonl
      {"anchor": <question>, "positive": <chunk>, "negative": <chunk from another doc>}
      -> train with Soup: task: embedding, format: embedding

  sft_train.jsonl (only with --sft)
      {"instruction": <question>, "input": "", "output": <model-written answer>}
      -> train with Soup: task: sft, format: alpaca

Usage:
    python build_pairs.py extracted/ --output-dir data/
    python build_pairs.py extracted/ --output-dir data/ --sft --model qwen2.5:7b

Requires: a running Ollama server with a text model pulled
(`ollama pull qwen2.5:7b`). Stdlib only — no pip installs.
"""

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

CAPTION_ONLY_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")

Q_PROMPT = (
    "You generate search queries for a documentation retrieval system.\n"
    "Given the documentation chunk below, write {n} distinct questions a user "
    "might type that THIS chunk answers. Vary phrasing and specificity. "
    'Reply with JSON only: {{"questions": ["...", "..."]}}\n\n'
    "CHUNK:\n{chunk}"
)
A_PROMPT = (
    "Answer the question using ONLY the documentation chunk below. "
    "Be direct and complete; do not mention the chunk itself.\n\n"
    "QUESTION: {question}\n\nCHUNK:\n{chunk}"
)


def ollama_chat(url: str, model: str, prompt: str, timeout: int, json_mode: bool = False) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.7},
    }
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"]


def chunk_markdown(text: str, max_chars: int, min_chars: int) -> list[str]:
    """Split on headings; merge small sections; split oversized ones on paragraphs."""
    sections, current = [], []
    for line in text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    # drop image-ref-only lines (captions were already spliced in as text)
    cleaned = []
    for sec in sections:
        kept = [ln for ln in sec.splitlines() if not CAPTION_ONLY_RE.match(ln.strip())]
        sec = "\n".join(kept).strip()
        if sec:
            cleaned.append(sec)

    chunks: list[str] = []
    for sec in cleaned:
        if len(sec) <= max_chars:
            if chunks and len(chunks[-1]) < min_chars and len(chunks[-1]) + len(sec) <= max_chars:
                chunks[-1] = chunks[-1] + "\n\n" + sec  # merge tiny section into previous
            else:
                chunks.append(sec)
            continue
        paras, buf = sec.split("\n\n"), ""
        for p in paras:
            if buf and len(buf) + len(p) + 2 > max_chars:
                chunks.append(buf.strip())
                buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf.strip():
            chunks.append(buf.strip())
    return [c for c in chunks if len(c) >= min_chars]


def parse_questions(raw: str, want: int) -> list[str]:
    try:
        qs = json.loads(raw).get("questions", [])
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in qs if isinstance(q, str) and q.strip()][:want]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="Directory of .md files (searched recursively), or one file")
    ap.add_argument("--output-dir", default="data", help="Where the JSONL files go")
    ap.add_argument("--model", default="qwen2.5:7b", help="Ollama text model")
    ap.add_argument("--url", default="http://localhost:11434", help="Ollama server URL")
    ap.add_argument("--questions-per-chunk", type=int, default=2)
    ap.add_argument("--max-chunk-chars", type=int, default=3000)
    ap.add_argument("--min-chunk-chars", type=int, default=200)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--sft", action="store_true",
                    help="Also write alpaca SFT rows (model answers each question)")
    ap.add_argument("--timeout", type=int, default=300, help="Seconds per model call")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.input)
    files = sorted(root.rglob("*.md")) if root.is_dir() else [root]
    # prefer captioned versions when both exist
    captioned = {f for f in files if f.name.endswith(".captioned.md")}
    files = [f for f in files
             if f in captioned or f.with_suffix("").with_suffix(".captioned.md") not in captioned]
    if not files:
        print(f"No .md files under {root}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    chunks: list[tuple[str, str]] = []  # (source_file, chunk_text)
    for f in files:
        for c in chunk_markdown(f.read_text(encoding="utf-8"),
                                args.max_chunk_chars, args.min_chunk_chars):
            chunks.append((str(f), c))
    print(f"{len(files)} files -> {len(chunks)} chunks")
    if not chunks:
        return 1

    pairs, sft_rows, failed = [], [], 0
    for idx, (src, chunk) in enumerate(chunks):
        try:
            raw = ollama_chat(args.url, args.model,
                              Q_PROMPT.format(n=args.questions_per_chunk, chunk=chunk),
                              args.timeout, json_mode=True)
            questions = parse_questions(raw, args.questions_per_chunk)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            print(f"  [failed] chunk {idx}: {exc}", file=sys.stderr)
            failed += 1
            continue
        others = ([c for s, c in chunks if s != src and c != chunk]
                  or [c for _, c in chunks if c != chunk])
        for q in questions:
            row = {"anchor": q, "positive": chunk}
            if others:
                row["negative"] = rng.choice(others)
            pairs.append(row)
            if args.sft:
                try:
                    answer = ollama_chat(args.url, args.model,
                                         A_PROMPT.format(question=q, chunk=chunk),
                                         args.timeout).strip()
                    sft_rows.append({"instruction": q, "input": "", "output": answer})
                except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
                    pass
        if (idx + 1) % 10 == 0 or idx + 1 == len(chunks):
            print(f"  {idx + 1}/{len(chunks)} chunks, {len(pairs)} pairs so far")

    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_split)) if len(pairs) > 1 else 0
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list[dict]) -> None:
        p = out / name
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):5d} rows -> {p}")

    dump("embedding_train.jsonl", pairs[n_val:])
    if n_val:
        dump("embedding_val.jsonl", pairs[:n_val])
    if args.sft:
        dump("sft_train.jsonl", sft_rows)
    if failed:
        print(f"{failed} chunks failed question generation", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
