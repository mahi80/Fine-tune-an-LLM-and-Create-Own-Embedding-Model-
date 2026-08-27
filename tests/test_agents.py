"""Tests for the LangGraph agent layer (hardware + subprocesses mocked)."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "agents"))

pytest.importorskip("langgraph")

import graph_common  # noqa: E402  # pylint: disable=wrong-import-position
import pipeline_agents  # noqa: E402  # pylint: disable=wrong-import-position
import run_embedding_pipeline  # noqa: E402  # pylint: disable=wrong-import-position
import run_llm_pipeline  # noqa: E402  # pylint: disable=wrong-import-position


def _quiet(_line):
    pass


def _fake_embedding_repo(tmp_path):
    ex = tmp_path / "extracted" / "m" / "auto"
    ex.mkdir(parents=True)
    (ex / "m.md").write_text("# t\n" + "words " * 100, encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "soup_embedding.yaml").write_text("base: b\n", encoding="utf-8")


def _mock_env(monkeypatch, vram_gb=12.0):
    monkeypatch.setattr(graph_common, "probe_hardware",
                        lambda: {"gpu": "FakeGPU", "vram_gb": vram_gb,
                                 "ram_gb": 32, "disk_free_gb": 500})
    monkeypatch.setattr(graph_common, "check_tools", lambda *a, **k: [])


def test_planner_aborts_when_llm_cannot_fit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _mock_env(monkeypatch, vram_gb=4.0)
    ran = []
    monkeypatch.setattr(run_llm_pipeline, "run_cmd", ran.append)
    ns = pipeline_agents.default_args("llm")
    graph = pipeline_agents.build_graph("llm", ns, gate=None, log=_quiet)
    final = graph.invoke({"pipeline": "llm"})
    assert final["aborted"] is True
    assert "cannot" in final["error"]
    assert not ran  # no worker agent ever executed


def test_planner_aborts_on_missing_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _mock_env(monkeypatch)
    monkeypatch.setattr(graph_common, "check_tools",
                        lambda *a, **k: [("mineru", "pip install mineru", None)])
    _fake_embedding_repo(tmp_path)
    ns = pipeline_agents.default_args("embedding")
    graph = pipeline_agents.build_graph("embedding", ns, gate=None, log=_quiet)
    final = graph.invoke({"pipeline": "embedding"})
    assert final["aborted"] is True
    assert "mineru" in final["error"]


def test_embedding_agents_run_sequentially(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_embedding_repo(tmp_path)
    _mock_env(monkeypatch)
    ran = []
    monkeypatch.setattr(run_embedding_pipeline, "run_cmd",
                        lambda cmd: ran.append(" ".join(str(c) for c in cmd)))
    monkeypatch.setattr(run_embedding_pipeline, "require_exe", lambda n, h: n)
    monkeypatch.setattr(run_embedding_pipeline, "ollama_model_names",
                        lambda url: ["qwen3-vl:8b", "qwen2.5:7b"])
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    approvals = []

    def gate(step, _question, _preview):
        approvals.append(step)
        return True

    ns = pipeline_agents.default_args("embedding")
    graph = pipeline_agents.build_graph("embedding", ns, gate=gate, log=_quiet)
    final = graph.invoke({"pipeline": "embedding"})
    assert not final.get("error")
    joined = "\n".join(ran)
    assert joined.index("caption_images") < joined.index("build_pairs") < joined.index("train")
    assert "extract (skipped)" in final["done"]      # already-extracted layout
    # skipped stages don't re-ask for approval; eval auto-skipped too -> only pairs gates
    assert approvals == ["pairs"]


def test_gate_rejection_stops_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_embedding_repo(tmp_path)
    _mock_env(monkeypatch)
    ran = []
    monkeypatch.setattr(run_embedding_pipeline, "run_cmd",
                        lambda cmd: ran.append(" ".join(str(c) for c in cmd)))
    monkeypatch.setattr(run_embedding_pipeline, "require_exe", lambda n, h: n)
    monkeypatch.setattr(run_embedding_pipeline, "ollama_model_names",
                        lambda url: ["qwen3-vl:8b", "qwen2.5:7b"])

    ns = pipeline_agents.default_args("embedding")
    graph = pipeline_agents.build_graph(
        "embedding", ns, gate=lambda s, q, p: False, log=_quiet)  # human says no
    final = graph.invoke({"pipeline": "embedding"})
    assert "checkpoint" in final["error"]
    assert "soup train" not in "\n".join(ran)        # nothing after the rejected stage


def test_plan_fit_tiers():
    ok, _, tweaks = graph_common.plan_fit(
        "embedding", {"gpu": "x", "vram_gb": 6.0, "ram_gb": 16, "disk_free_gb": 100})
    assert ok and tweaks["backend"] == "pipeline"
    ok, lines, _ = graph_common.plan_fit(
        "llm", {"gpu": None, "vram_gb": 0.0, "ram_gb": 16, "disk_free_gb": 100})
    assert not ok
    assert any("TinyLlama" in ln or "layer-streaming" in ln for ln in lines)
