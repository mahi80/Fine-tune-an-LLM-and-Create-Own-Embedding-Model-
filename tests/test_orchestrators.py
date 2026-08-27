"""Unit tests for the one-command pipeline orchestrators (subprocesses mocked)."""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _pipeline  # noqa: E402  # pylint: disable=wrong-import-position
import run_embedding_pipeline  # noqa: E402  # pylint: disable=wrong-import-position
import run_llm_pipeline  # noqa: E402  # pylint: disable=wrong-import-position

# ------------------------------------------------------------------ _pipeline


def _steps(log, fail_at=None):
    def make(name):
        def runner():
            if name == fail_at:
                raise _pipeline.PipelineError("boom")
            log.append(name)
        return runner
    return [_pipeline.Step(n, n, make(n)) for n in ("a", "b", "c")]


def test_runner_runs_in_order():
    log = []
    assert _pipeline.run_pipeline(_steps(log), skip=[], start_from=None, dry_run=False) == 0
    assert log == ["a", "b", "c"]


def test_runner_skip_and_from_step():
    log = []
    assert _pipeline.run_pipeline(_steps(log), skip=["c"], start_from="b", dry_run=False) == 0
    assert log == ["b"]


def test_runner_failure_stops_and_reports():
    log = []
    assert _pipeline.run_pipeline(_steps(log, fail_at="b"), skip=[], start_from=None,
                                  dry_run=False) == 1
    assert log == ["a"]  # c never ran


def test_runner_auto_skip():
    log = []
    steps = _steps(log)
    steps[1].auto_skip = lambda: "already done"
    assert _pipeline.run_pipeline(steps, skip=[], start_from=None, dry_run=False) == 0
    assert log == ["a", "c"]


def test_runner_dry_run_executes_nothing():
    log = []
    assert _pipeline.run_pipeline(_steps(log), skip=[], start_from=None, dry_run=True) == 0
    assert not log


def test_runner_rejects_unknown_step():
    assert _pipeline.run_pipeline(_steps([]), skip=["nope"], start_from=None, dry_run=False) == 2
    assert _pipeline.run_pipeline(_steps([]), skip=[], start_from="nope", dry_run=False) == 2

# ------------------------------------------------------ embedding orchestrator


def test_embedding_dry_run_runs_nothing(tmp_path, monkeypatch):
    (tmp_path / "pdfs").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_embedding_pipeline, "run_cmd",
                        lambda cmd: (_ for _ in ()).throw(AssertionError("ran a command")))
    monkeypatch.setattr(sys, "argv", ["run_embedding_pipeline.py", "--dry-run"])
    assert run_embedding_pipeline.main() == 0


def test_embedding_step_order_with_mocks(tmp_path, monkeypatch):
    # a fake already-extracted layout: extract should auto-skip
    ex = tmp_path / "extracted" / "manual" / "auto"
    ex.mkdir(parents=True)
    (ex / "manual.md").write_text("# hi\n" + "text " * 100, encoding="utf-8")
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    (cfgdir / "soup_embedding.yaml").write_text("base: BAAI/bge-base-en-v1.5\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    ran = []
    monkeypatch.setattr(run_embedding_pipeline, "run_cmd",
                        lambda cmd: ran.append(" ".join(str(c) for c in cmd)))
    monkeypatch.setattr(run_embedding_pipeline, "require_exe", lambda n, h: n)
    monkeypatch.setattr(run_embedding_pipeline, "ollama_model_names",
                        lambda url: ["qwen3-vl:8b", "qwen2.5:7b"])
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)  # eval auto-skips
    monkeypatch.setattr(sys, "argv", ["run_embedding_pipeline.py"])

    assert run_embedding_pipeline.main() == 0
    joined = "\n".join(ran)
    assert "mineru" not in joined                      # extract auto-skipped
    assert "caption_images.py" in joined
    assert "build_pairs.py" in joined
    assert "soup train" in joined
    assert "eval_embedder.py" not in joined            # no sentence-transformers
    # serialized order: caption before pairs before train
    assert joined.index("caption_images") < joined.index("build_pairs") < joined.index("train")

# ------------------------------------------------------------ llm orchestrator


def test_llm_parse_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "base: Qwen/Qwen2.5-7B-Instruct\ntask: sft\n\ndata:\n  train: ./data/train.jsonl\n"
        "output: ./out_dir\n", encoding="utf-8")
    parsed = run_llm_pipeline.parse_config(cfg)
    assert parsed == {"base": "Qwen/Qwen2.5-7B-Instruct",
                      "train": "./data/train.jsonl", "output": "./out_dir"}


def test_llm_step_order_with_mocks(tmp_path, monkeypatch):
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    (cfgdir / "soup_qwen7b.yaml").write_text(
        "base: Qwen/Qwen2.5-7B-Instruct\ndata:\n  train: ./train.jsonl\noutput: ./out\n",
        encoding="utf-8")
    (tmp_path / "train.jsonl").write_text('{"instruction":"q","input":"","output":"a"}\n',
                                          encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    ran = []

    def fake_run(cmd):
        line = " ".join(str(c) for c in cmd)
        ran.append(line)
        if "export" in line:  # simulate soup export creating the GGUF
            Path("out.f16.gguf").write_bytes(b"gguf")

    monkeypatch.setattr(run_llm_pipeline, "run_cmd", fake_run)
    monkeypatch.setattr(run_llm_pipeline, "require_exe", lambda n, h: n)
    monkeypatch.setattr(run_llm_pipeline, "find_ollama", lambda: "ollama")
    monkeypatch.setattr(run_llm_pipeline, "ollama_model_names", lambda url: [])
    monkeypatch.setattr(sys, "argv", ["run_llm_pipeline.py", "--name", "test-model"])

    assert run_llm_pipeline.main() == 0
    joined = "\n".join(ran)
    for expected in ("soup data validate", "soup data doctor", "soup train",
                     "verify_adapter.py", "soup eval auto", "soup export",
                     "ollama create test-model"):
        assert expected in joined
    assert joined.index("soup train") < joined.index("soup export")
    assert Path("out/Modelfile.generated").read_text(encoding="utf-8").startswith("FROM ")


def test_llm_missing_data_fails_check(tmp_path, monkeypatch):
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    (cfgdir / "soup_qwen7b.yaml").write_text(
        "base: Qwen/Qwen2.5-7B-Instruct\ndata:\n  train: ./missing.jsonl\noutput: ./out\n",
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_llm_pipeline, "require_exe", lambda n, h: n)
    monkeypatch.setattr(run_llm_pipeline, "run_cmd",
                        lambda cmd: (_ for _ in ()).throw(AssertionError("ran a command")))
    monkeypatch.setattr(sys, "argv", ["run_llm_pipeline.py"])
    assert run_llm_pipeline.main() == 1  # check step fails: data file absent
