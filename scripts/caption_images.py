"""Caption images in a MinerU-extracted Markdown file with a local Ollama VLM.

For every image reference in the Markdown, sends the image to a vision model
(default qwen3-vl:8b) and inserts the returned description right after the
reference as a blockquote. Idempotent: already-captioned images are skipped,
so the script can be re-run after interruptions.

Usage:
    python caption_images.py extracted/manual/auto/manual.md
    python caption_images.py manual.md --model qwen3-vl:8b --output manual.captioned.md

Requires: a running Ollama server (`ollama serve`) with the model pulled
(`ollama pull qwen3-vl:8b`). Stdlib only — no pip installs.
"""

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

IMAGE_RE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)\s*$")
CAPTION_MARK = "> **Screenshot:**"

PROMPT = (
    "Describe this image from software documentation. If it is a UI screenshot, "
    "name the application area, the dialog/menu/tab shown, and the visible options, "
    "fields, and buttons. If it is a diagram or table, summarize what it shows. "
    "If it contains a formula, transcribe it as LaTeX. Reply with 2-4 sentences, "
    "no preamble."
)


def ollama_caption(url: str, model: str, image_path: Path, timeout: int) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": PROMPT,
            "images": [base64.b64encode(image_path.read_bytes()).decode()],
        }],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return " ".join(data["message"]["content"].split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("markdown", help="Markdown file produced by MinerU")
    ap.add_argument("--model", default="qwen3-vl:8b", help="Ollama vision model")
    ap.add_argument("--url", default="http://localhost:11434", help="Ollama server URL")
    ap.add_argument("--output", help="Output path (default: <name>.captioned.md)")
    ap.add_argument("--timeout", type=int, default=300, help="Seconds per image")
    args = ap.parse_args()

    md_path = Path(args.markdown)
    out_path = Path(args.output) if args.output else md_path.with_suffix(".captioned.md")
    lines = md_path.read_text(encoding="utf-8").splitlines()

    out_lines, captioned, skipped, failed = [], 0, 0, 0
    for i, line in enumerate(lines):
        out_lines.append(line)
        m = IMAGE_RE.match(line.strip())
        if not m:
            continue
        # already captioned (from a previous run of this script)?
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        nxt2 = lines[i + 2] if i + 2 < len(lines) else ""
        if nxt.startswith(CAPTION_MARK) or nxt2.startswith(CAPTION_MARK):
            skipped += 1
            continue
        img = (md_path.parent / m.group(1)).resolve()
        if not img.is_file():
            print(f"  [missing] {m.group(1)}", file=sys.stderr)
            failed += 1
            continue
        try:
            caption = ollama_caption(args.url, args.model, img, args.timeout)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            print(f"  [failed]  {img.name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        out_lines.append("")
        out_lines.append(f"{CAPTION_MARK} {caption}")
        captioned += 1
        print(f"  [ok] {img.name}: {caption[:80]}...")

    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\n{captioned} captioned, {skipped} already done, {failed} failed -> {out_path}")
    return 1 if (failed and not captioned) else 0


if __name__ == "__main__":
    sys.exit(main())
