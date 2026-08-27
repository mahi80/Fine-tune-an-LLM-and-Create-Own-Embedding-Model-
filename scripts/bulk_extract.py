"""Corpus-scale routed PDF ingestion (tier-1 of the Cohere-style recipe).

Every document is classified and routed to the cheapest engine that can do it
justice — at millions-of-pages scale, forcing one OCR engine on everything
destroys either throughput (all-MinerU) or fidelity (all-native):

    Fast Path      digital PDF                 -> native extraction here (pypdf, no OCR)
    OCR Path       scanned document            -> PaddleOCR PP-StructureV3 queue
    Complex Path   tables/equations/diagrams   -> MinerU (or Surya) queue
    Recovery Path  pages that defeat the above -> olmOCR / VLM queue

This script runs the Fast Path itself (parallel, resumable) and writes queue
manifests (route_ocr.txt, route_complex.txt, route_recovery.txt) plus the exact
commands to run the heavier engines on their queues. MinerU output feeds
build_weak_pairs.py directly (it accepts both this JSONL and MinerU Markdown).

Fast-path rows:  {"doc": ..., "title": ..., "section": ..., "text": ...}

Usage:
    pip install pypdf
    python bulk_extract.py D:/plm_corpus --output corpus_chunks/ --workers 8
    python bulk_extract.py D:/plm_corpus --output corpus_chunks/ --run-mineru
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

MAX_CHUNK = 3000
MIN_CHUNK = 200
SAMPLE_PAGES = 8           # pages inspected to classify a document
MIN_CHARS_PER_PAGE = 200   # below this on sampled pages => scanned => OCR path
COMPLEX_IMG_PER_PAGE = 2.5  # dense figures/diagrams => complex path


def chunk_text(paragraphs: list, max_chars: int, min_chars: int) -> list:
    chunks, buf = [], ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if buf and len(buf) + len(p) + 2 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= min_chars]


def _outline_page_map(reader) -> dict:
    """page index -> nearest outline heading, from the PDF bookmarks."""
    toc = {}

    def walk(items):
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            try:
                toc[reader.get_destination_page_number(item)] = str(item.title).strip()
            except (AttributeError, KeyError, TypeError, ValueError):
                continue

    try:
        walk(reader.outline)
    except (AttributeError, TypeError, ValueError):
        pass
    return toc


def classify(reader) -> str:
    """'native' | 'ocr' | 'complex' from a sample of pages."""
    n = min(len(reader.pages), SAMPLE_PAGES)
    chars = images = 0
    for i in range(n):
        page = reader.pages[i]
        try:
            chars += len(page.extract_text() or "")
        except (KeyError, TypeError, ValueError):
            pass
        try:
            images += len(page.images)
        except (KeyError, TypeError, ValueError, OSError):
            pass
    if n and chars / n < MIN_CHARS_PER_PAGE:
        return "ocr"                      # no real text layer -> scanned
    if n and images / n > COMPLEX_IMG_PER_PAGE:
        return "complex"                  # figure/diagram-dense -> MinerU tier
    return "native"


def extract_one(job: tuple) -> tuple:
    """(pdf, out, root) -> (pdf, route, n_chunks, error)."""
    pdf_path, out_path, root = job
    from pypdf import PdfReader  # pylint: disable=C0415,E0401

    try:
        reader = PdfReader(pdf_path)
        route = classify(reader)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return (pdf_path, "recovery", 0, str(exc))  # unreadable -> recovery queue
    if route != "native":
        return (pdf_path, route, 0, None)

    rows, toc, section = [], _outline_page_map(reader), ""
    title = ((reader.metadata or {}).get("/Title") or "").strip() or Path(pdf_path).stem
    for page_no, page in enumerate(reader.pages):
        section = toc.get(page_no, section)
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            return (pdf_path, "recovery", 0, f"page {page_no}: {exc}")
        for chunk in chunk_text(text.split("\n\n"), MAX_CHUNK, MIN_CHUNK):
            rows.append({"doc": str(Path(pdf_path).relative_to(root)),
                         "title": title, "section": section, "text": chunk})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return (pdf_path, "native", len(rows), None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="Root folder to scan recursively for PDFs")
    ap.add_argument("--output", default="corpus_chunks", help="Output folder (mirrors input tree)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="Stop after N PDFs (0 = all)")
    ap.add_argument("--run-mineru", action="store_true",
                    help="After routing, run MinerU on the complex+ocr queues (slow, high quality)")
    ap.add_argument("--mineru-backend", default="vlm-transformers")
    args = ap.parse_args()

    try:
        import pypdf  # noqa: F401,PLC0415  # pylint: disable=import-outside-toplevel,import-error,unused-import
    except ImportError:
        print("pypdf missing — pip install pypdf", file=sys.stderr)
        return 1

    root, out_root = Path(args.input), Path(args.output)
    jobs = []
    for pdf in sorted(root.rglob("*.pdf")):
        out = out_root / pdf.relative_to(root).with_suffix(".jsonl")
        if out.exists():
            continue
        jobs.append((str(pdf), out, str(root)))
        if args.limit and len(jobs) >= args.limit:
            break
    if not jobs:
        print("Nothing to do (no PDFs found, or all already extracted).")
        return 0
    print(f"{len(jobs)} PDFs to route with {args.workers} workers ...")

    start, counts, chunks = time.time(), {"native": 0, "ocr": 0, "complex": 0, "recovery": 0}, 0
    queues: dict = {"ocr": [], "complex": [], "recovery": []}
    with Pool(args.workers) as pool:
        for done, (pdf, route, n, err) in enumerate(pool.imap_unordered(extract_one, jobs), 1):
            counts[route] += 1
            chunks += n
            if route != "native":
                queues[route].append(pdf)
                if err:
                    print(f"  [{route}] {pdf}: {err}", file=sys.stderr)
            if done % 25 == 0 or done == len(jobs):
                rate = done / max(time.time() - start, 1) * 3600
                print(f"  {done}/{len(jobs)} routed, {chunks} fast-path chunks, "
                      f"~{rate:.0f} PDFs/hour")

    out_root.mkdir(parents=True, exist_ok=True)
    for route, files in queues.items():
        manifest = out_root / f"route_{route}.txt"
        manifest.write_text("\n".join(files) + ("\n" if files else ""), encoding="utf-8")
    print(f"\nRouting: {counts['native']} native (extracted), {counts['ocr']} scanned -> OCR, "
          f"{counts['complex']} complex -> MinerU/Surya, {counts['recovery']} -> recovery")
    if queues["ocr"]:
        print(f"OCR queue      ({out_root}/route_ocr.txt): run PaddleOCR PP-StructureV3, "
              f"or fold into MinerU with --run-mineru")
    if queues["complex"]:
        print(f"Complex queue  ({out_root}/route_complex.txt): "
              f"mineru -p <pdf> -o {out_root}/mineru/ -b {args.mineru_backend}")
    if queues["recovery"]:
        print(f"Recovery queue ({out_root}/route_recovery.txt): olmOCR / VLM territory — "
              f"inspect these by hand first")

    if args.run_mineru and (queues["complex"] or queues["ocr"]):
        from _pipeline import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
            PipelineError,
            run_cmd,
        )
        for pdf in queues["complex"] + queues["ocr"]:
            try:
                run_cmd(["mineru", "-p", pdf, "-o", out_root / "mineru",
                         "-b", args.mineru_backend])
            except PipelineError as exc:
                print(f"  [mineru failed -> recovery] {pdf}: {exc}", file=sys.stderr)
                with (out_root / "route_recovery.txt").open("a", encoding="utf-8") as fh:
                    fh.write(pdf + "\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
