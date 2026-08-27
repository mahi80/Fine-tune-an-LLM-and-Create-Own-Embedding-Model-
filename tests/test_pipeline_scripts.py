"""Unit tests for the pipeline scripts (network calls are mocked)."""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_pairs  # noqa: E402  # pylint: disable=wrong-import-position
import caption_images  # noqa: E402  # pylint: disable=wrong-import-position
import eval_embedder  # noqa: E402  # pylint: disable=wrong-import-position

# ---------------------------------------------------------------- build_pairs


def test_chunk_markdown_splits_on_headings():
    text = "# One\n" + "alpha " * 50 + "\n## Two\n" + "beta " * 50
    chunks = build_pairs.chunk_markdown(text, max_chars=3000, min_chars=100)
    assert len(chunks) == 2
    assert chunks[0].startswith("# One")
    assert chunks[1].startswith("## Two")


def test_chunk_markdown_merges_small_sections():
    text = "# Tiny\nshort\n# Big\n" + "word " * 60
    chunks = build_pairs.chunk_markdown(text, max_chars=3000, min_chars=100)
    # tiny section merges into the next append cycle rather than standing alone
    assert all(len(c) >= 100 for c in chunks)


def test_chunk_markdown_splits_oversized_sections():
    paras = "\n\n".join("paragraph " * 30 for _ in range(10))
    chunks = build_pairs.chunk_markdown("# Big\n" + paras, max_chars=800, min_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_markdown_drops_image_only_lines():
    text = "# Sec\n![](images/a.jpg)\n" + "content " * 40
    chunks = build_pairs.chunk_markdown(text, max_chars=3000, min_chars=50)
    assert "images/a.jpg" not in chunks[0]
    assert "content" in chunks[0]


def test_parse_questions_valid_json():
    raw = json.dumps({"questions": ["How?", "  Why?  ", 42, ""]})
    assert build_pairs.parse_questions(raw, 5) == ["How?", "Why?"]


def test_parse_questions_caps_at_want():
    raw = json.dumps({"questions": ["a", "b", "c"]})
    assert build_pairs.parse_questions(raw, 2) == ["a", "b"]


def test_parse_questions_bad_json():
    assert build_pairs.parse_questions("not json at all", 3) == []


def test_build_pairs_end_to_end(tmp_path, monkeypatch):
    doc = tmp_path / "manual.md"
    doc.write_text("# A\n" + "alpha " * 80 + "\n# B\n" + "beta " * 80, encoding="utf-8")

    def fake_chat(_url, _model, prompt, _timeout, json_mode=False):  # pylint: disable=unused-argument
        if json_mode:
            return json.dumps({"questions": ["q1?", "q2?"]})
        return "an answer"

    monkeypatch.setattr(build_pairs, "ollama_chat", fake_chat)
    out = tmp_path / "data"
    monkeypatch.setattr(
        sys, "argv",
        ["build_pairs.py", str(tmp_path), "--output-dir", str(out), "--sft",
         "--min-chunk-chars", "100"],
    )
    assert build_pairs.main() == 0

    train = [json.loads(x) for x in (out / "embedding_train.jsonl").read_text().splitlines()]
    val = [json.loads(x) for x in (out / "embedding_val.jsonl").read_text().splitlines()]
    assert len(train) + len(val) == 4  # 2 chunks x 2 questions
    row = train[0]
    assert set(row) == {"anchor", "positive", "negative"}
    assert row["negative"] != row["positive"]

    sft = [json.loads(x) for x in (out / "sft_train.jsonl").read_text().splitlines()]
    assert sft and set(sft[0]) == {"instruction", "input", "output"}
    assert sft[0]["output"] == "an answer"


def test_build_pairs_prefers_captioned(tmp_path, monkeypatch):
    (tmp_path / "m.md").write_text("# Raw\n" + "raw " * 100, encoding="utf-8")
    (tmp_path / "m.captioned.md").write_text("# Cap\n" + "cap " * 100, encoding="utf-8")
    seen = []

    def fake_chat(_url, _model, prompt, _timeout, json_mode=False):  # pylint: disable=unused-argument
        seen.append(prompt)
        return json.dumps({"questions": ["q?"]})

    monkeypatch.setattr(build_pairs, "ollama_chat", fake_chat)
    monkeypatch.setattr(sys, "argv", ["build_pairs.py", str(tmp_path),
                                      "--output-dir", str(tmp_path / "d")])
    assert build_pairs.main() == 0
    assert any("cap" in p for p in seen)
    assert not any("raw raw" in p for p in seen)

# ------------------------------------------------------------- caption_images


def test_caption_insertion_and_idempotence(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "shot.jpg").write_bytes(b"\xff\xd8fake")
    md = tmp_path / "doc.md"
    md.write_text("# T\n\n![](images/shot.jpg)\n\ntext\n", encoding="utf-8")

    calls = []

    def fake_caption(_url, _model, image_path, _timeout):  # pylint: disable=unused-argument
        calls.append(image_path.name)
        return "A settings dialog with an OK button."

    monkeypatch.setattr(caption_images, "ollama_caption", fake_caption)
    out = tmp_path / "doc.captioned.md"

    monkeypatch.setattr(sys, "argv", ["caption_images.py", str(md)])
    assert caption_images.main() == 0
    text = out.read_text(encoding="utf-8")
    assert "> **Screenshot:** A settings dialog" in text
    assert text.index("![](images/shot.jpg)") < text.index("> **Screenshot:**")
    assert calls == ["shot.jpg"]

    # second run over the captioned file: nothing re-captioned
    monkeypatch.setattr(sys, "argv", ["caption_images.py", str(out), "--output", str(out)])
    assert caption_images.main() == 0
    assert calls == ["shot.jpg"]
    assert out.read_text(encoding="utf-8").count("> **Screenshot:**") == 1


def test_caption_missing_image_fails_gracefully(tmp_path, monkeypatch):
    md = tmp_path / "doc.md"
    md.write_text("![](images/nope.jpg)\n", encoding="utf-8")
    monkeypatch.setattr(caption_images, "ollama_caption",
                        lambda *a, **k: pytest.fail("should not be called"))
    monkeypatch.setattr(sys, "argv", ["caption_images.py", str(md)])
    assert caption_images.main() == 1  # nothing captioned, one failure

# -------------------------------------------------------------- eval_embedder


def test_eval_load_pairs(tmp_path):
    p = tmp_path / "val.jsonl"
    rows = [{"anchor": f"q{i}", "positive": f"chunk{i}", "negative": "x"} for i in range(3)]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    anchors, positives = eval_embedder.load_pairs(p)
    assert anchors == ["q0", "q1", "q2"]
    assert positives == ["chunk0", "chunk1", "chunk2"]


def test_eval_metrics_with_stub_model():
    pytest.importorskip("sentence_transformers")
    anchors = ["a", "b"]
    corpus = ["a doc", "b doc"]
    res = eval_embedder.evaluate("sentence-transformers/all-MiniLM-L6-v2",
                                 anchors, corpus, gold=[0, 1], ks=(1,))
    assert 0.0 <= res["recall@1"] <= 1.0
    assert 0.0 <= res["mrr"] <= 1.0
