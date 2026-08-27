"""Tests for the Cohere-style stack (tiny CPU models, synthetic data)."""
# pylint: disable=redefined-outer-name,import-outside-toplevel

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

st = pytest.importorskip("sentence_transformers")

import build_weak_pairs  # noqa: E402  # pylint: disable=wrong-import-position
import bulk_extract  # noqa: E402  # pylint: disable=wrong-import-position
import eval_embedder  # noqa: E402  # pylint: disable=wrong-import-position
import merge_embedding_adapter  # noqa: E402  # pylint: disable=wrong-import-position
import mine_hard_negatives  # noqa: E402  # pylint: disable=wrong-import-position
import train_embedder  # noqa: E402  # pylint: disable=wrong-import-position

WORDS = ["revision", "assembly", "harness", "workflow", "schematic",
         "release", "variant", "baseline", "routing", "solver"]


def _pairs(n=8):
    return [{"anchor": f"how to configure {WORDS[i % 10]} number {i}",
             "positive": f"To configure {WORDS[i % 10]} number {i}, open the "
                         f"settings dialog and adjust the {WORDS[(i + 3) % 10]} value. "
                         + "Details follow. " * 10}
            for i in range(n)]


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


@pytest.fixture(scope="module")
def tiny_embedder(tmp_path_factory):
    """A tiny random-init embedder trained for 1 epoch on synthetic pairs."""
    root = tmp_path_factory.mktemp("emb")
    _write_jsonl(root / "pairs.jsonl", _pairs())
    out = root / "model"
    argv = ["train_embedder.py", "--pairs", str(root / "pairs.jsonl"),
            "--output", str(out), "--random-init", "--layers", "2", "--hidden", "32",
            "--batch-size", "4", "--cache-chunk", "2", "--epochs", "1",
            "--max-seq", "64", "--workdir", str(root / "work")]
    old = sys.argv
    sys.argv = argv
    try:
        assert train_embedder.main() == 0
    finally:
        sys.argv = old
    return str(out)


def test_trained_embedder_loads_and_encodes(tiny_embedder):
    model = st.SentenceTransformer(tiny_embedder)
    vectors = model.encode(["how do I branch a design?"])
    assert vectors.shape == (1, 32)


def test_mlm_stage_runs(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_jsonl(corpus / "doc.jsonl",
                 [{"text": "The routing solver computes wire paths. " * 6}
                  for _ in range(6)])
    base = train_embedder.make_random_encoder(tmp_path / "rand", 2, 32,
                                              "bert-base-uncased")
    out = train_embedder.mlm_pretrain(base, corpus, tmp_path / "mlm", epochs=1,
                                      max_seq=64, batch=2)
    from transformers import AutoModel
    assert AutoModel.from_pretrained(out).config.hidden_size == 32


def test_mine_hard_negatives(tmp_path, tiny_embedder, monkeypatch):
    pairs = tmp_path / "pairs.jsonl"
    _write_jsonl(pairs, _pairs(8))
    out = tmp_path / "hard.jsonl"
    monkeypatch.setattr(sys, "argv",
                        ["mine_hard_negatives.py", str(pairs), "--model", tiny_embedder,
                         "--output", str(out), "--false-negative-sim", "0.999"])
    assert mine_hard_negatives.main() == 0
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:
        assert row["negative"] and row["negative"] != row["positive"]


def test_eval_with_reranker_path(tmp_path, tiny_embedder, monkeypatch):
    val = tmp_path / "val.jsonl"
    _write_jsonl(val, _pairs(4))
    scores = iter([[1.0] * 20] * 10)
    monkeypatch.setattr(
        "sentence_transformers.cross_encoder.CrossEncoder",
        lambda *a, **k: type("FakeCE", (), {"predict": lambda self, pairs:
                                            [1.0] * len(pairs)})(),
    )
    res = eval_embedder.evaluate_reranked(tiny_embedder, "unused",
                                          [r["anchor"] for r in _pairs(4)],
                                          [r["positive"] for r in _pairs(4)],
                                          gold=[0, 1, 2, 3], ks=(1, 4))
    del scores
    assert 0.0 <= res["recall@4"] <= 1.0


def test_merge_soup_adapter(tmp_path, monkeypatch):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

    base_dir = tmp_path / "base"
    cfg = BertConfig(vocab_size=30522, hidden_size=32, num_hidden_layers=2,  # pylint: disable=unexpected-keyword-arg
                     num_attention_heads=2, intermediate_size=64)
    BertModel(cfg).save_pretrained(base_dir)
    AutoTokenizer.from_pretrained("bert-base-uncased").save_pretrained(base_dir)

    peft_model = get_peft_model(
        AutoModel.from_pretrained(base_dir),
        LoraConfig(r=4, lora_alpha=8, task_type=TaskType.FEATURE_EXTRACTION,
                   target_modules=["query", "value"]))
    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)
    AutoTokenizer.from_pretrained(base_dir).save_pretrained(adapter_dir)

    out = tmp_path / "merged"
    monkeypatch.setattr(sys, "argv",
                        ["merge_embedding_adapter.py", str(adapter_dir),
                         "--output", str(out), "--max-seq", "64"])
    assert merge_embedding_adapter.main() == 0
    assert not (out / "adapter_config.json").exists()
    model = st.SentenceTransformer(str(out))
    assert model.encode(["hello"]).shape == (1, 32)


def test_bulk_extract_chunker():
    paragraphs = ["alpha " * 60, "beta " * 60, "tiny", "gamma " * 60]
    chunks = bulk_extract.chunk_text(paragraphs, max_chars=400, min_chars=100)
    assert chunks
    assert all(len(c) >= 100 for c in chunks)
    assert all(len(c) <= 400 + 2 for c in chunks)


def test_weak_pairs_from_chunks_and_markdown(tmp_path, monkeypatch):
    docs = tmp_path / "chunks"
    docs.mkdir()
    _write_jsonl(docs / "a.jsonl",
                 [{"title": "Harness Design Guide", "section": "Routing wires",
                   "text": "Routing content. " * 20},
                  {"title": "Harness Design Guide", "section": "Splice points",
                   "text": "Splice content. " * 20}])
    (docs / "b.md").write_text(
        "# Simulation Manual\n" + "intro " * 50 +
        "\n## Solver settings\n" + "solver text " * 40, encoding="utf-8")
    out = tmp_path / "weak"
    monkeypatch.setattr(sys, "argv",
                        ["build_weak_pairs.py", str(docs), "--output-dir", str(out),
                         "--val-split", "0.0"])
    assert build_weak_pairs.main() == 0
    rows = [json.loads(x) for x in
            (out / "embedding_train.jsonl").read_text(encoding="utf-8").splitlines()]
    anchors = {r["anchor"] for r in rows}
    assert "Harness Design Guide" in anchors            # title pair (jsonl)
    assert "Routing wires" in anchors                   # section pair (jsonl)
    assert any("Solver settings" in a for a in anchors)  # section pair (markdown)
    assert all(set(r) == {"anchor", "positive"} for r in rows)


def test_bulk_extract_routes_unreadable_to_recovery(tmp_path, monkeypatch):
    (tmp_path / "bad.pdf").write_bytes(b"not a real pdf")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv",
                        ["bulk_extract.py", str(tmp_path), "--output", str(out),
                         "--workers", "1"])
    assert bulk_extract.main() == 0
    recovery = (out / "route_recovery.txt").read_text(encoding="utf-8")
    assert "bad.pdf" in recovery
