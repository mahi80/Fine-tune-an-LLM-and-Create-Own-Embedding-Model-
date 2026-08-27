"""Structure-mined training pairs at corpus scale (no LLM — the E5/GTE trick).

Reads chunk JSONL from bulk_extract.py (fast path) AND MinerU Markdown (complex/
OCR path output) and derives (anchor, positive) pairs
from document structure alone, so it scales to millions of chunks at CPU speed:

  - title    : document title        <-> its first chunk
  - section  : section heading       <-> a chunk from that section
  - adjacent : a chunk               <-> the next chunk of the same document

These weak pairs power the large-batch stage-1 contrastive run (in-batch
negatives supply the contrast — no explicit negatives needed here). The
LLM-generated pairs from build_pairs.py remain the higher-quality stage-2 data.

Usage:
    python build_weak_pairs.py corpus_chunks/ --output-dir data_weak/
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_CHUNK = 3000
MIN_CHUNK = 200


def rows_from_markdown(path: Path) -> list:
    """MinerU .md -> the same row shape bulk_extract emits (heading = section)."""
    import build_pairs  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    rows = []
    for chunk in build_pairs.chunk_markdown(path.read_text(encoding="utf-8"),
                                            MAX_CHUNK, MIN_CHUNK):
        first = chunk.splitlines()[0]
        section = first.lstrip("#").strip() if first.startswith("#") else ""
        rows.append({"title": path.stem.replace(".captioned", ""),
                     "section": section, "text": chunk})
    return rows


def pairs_from_doc(rows: list, use_adjacent: bool) -> list:
    """Derive structure pairs from one document's chunk rows (in file order)."""
    out = []
    if not rows:
        return out
    title = (rows[0].get("title") or "").strip()
    if title and len(title) > 3:
        out.append({"anchor": title, "positive": rows[0]["text"], "kind": "title"})
    seen_sections = set()
    for row in rows:
        section = (row.get("section") or "").strip()
        if section and len(section) > 3 and section not in seen_sections:
            seen_sections.add(section)
            out.append({"anchor": section, "positive": row["text"], "kind": "section"})
    if use_adjacent:
        for a, b in zip(rows, rows[1:], strict=False):
            out.append({"anchor": a["text"], "positive": b["text"], "kind": "adjacent"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="Folder of chunk JSONL files (bulk_extract.py output)")
    ap.add_argument("--output-dir", default="data_weak")
    ap.add_argument("--no-adjacent", action="store_true",
                    help="Skip adjacent-passage pairs (biggest but noisiest source)")
    ap.add_argument("--max-pairs", type=int, default=0, help="Cap total pairs (0 = all)")
    ap.add_argument("--val-split", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.input)
    files = sorted(root.rglob("*.jsonl"))
    mds = sorted(root.rglob("*.md"))
    captioned = {m for m in mds if m.name.endswith(".captioned.md")}
    files += [m for m in mds
              if m in captioned or m.with_suffix(".captioned.md") not in captioned]
    files = [f for f in files if not f.name.startswith("route_")]
    if not files:
        print(f"No .jsonl or .md under {args.input}", file=sys.stderr)
        return 1

    seen, pairs, counts = set(), [], {"title": 0, "section": 0, "adjacent": 0}
    for f in files:
        if f.suffix == ".md":
            rows = rows_from_markdown(f)
        else:
            rows = []
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        for pair in pairs_from_doc(rows, use_adjacent=not args.no_adjacent):
            key = hashlib.sha1(
                (pair["anchor"][:200] + "\x00" + pair["positive"][:200]).encode()
            ).digest()
            if key in seen or pair["anchor"] == pair["positive"]:
                continue
            seen.add(key)
            counts[pair.pop("kind")] += 1
            pairs.append(pair)
            if args.max_pairs and len(pairs) >= args.max_pairs:
                break
        if args.max_pairs and len(pairs) >= args.max_pairs:
            break

    if not pairs:
        print("No pairs derived — are the chunk files empty?", file=sys.stderr)
        return 1
    random.Random(args.seed).shuffle(pairs)
    n_val = (max(1, int(len(pairs) * args.val_split))
             if args.val_split > 0 and len(pairs) > 1 else 0)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("embedding_train.jsonl", pairs[n_val:]),
                       ("embedding_val.jsonl", pairs[:n_val])):
        with (out / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):7d} rows -> {out / name}")
    print(f"pair sources: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
