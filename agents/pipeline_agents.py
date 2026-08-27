"""LangGraph agent graphs for both fine-tuning pipelines.

Sequential agents managed by a LangGraph orchestrator:

    planner -> step agents (one per pipeline stage, in order) -> report

- The **planner agent** probes the hardware, checks/installs tools, and decides
  whether the chosen pipeline fits this machine. If it cannot fit, it stops and
  the report tells the user their options.
- Each **step agent** wraps one stage from scripts/run_*_pipeline.py (the exact
  same Step objects, so agent runs and manual runs behave identically).
- **Human-in-the-loop**: after checkpoint stages the graph pauses and asks the
  provided `gate` callback to approve the output before continuing.

Errors at any node route straight to the report node — later agents never run.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "agents"))

import graph_common  # noqa: E402  # pylint: disable=wrong-import-position
import run_embedding_pipeline  # noqa: E402  # pylint: disable=wrong-import-position
import run_llm_pipeline  # noqa: E402  # pylint: disable=wrong-import-position
from _pipeline import PipelineError  # noqa: E402  # pylint: disable=wrong-import-position
from langgraph.graph import END, StateGraph  # noqa: E402  # pylint: disable=wrong-import-position

# after these stages, pause and ask the human to check the output
CHECKPOINTS = {
    "embedding": {
        "extract": "Open the extracted Markdown — does the text look complete and readable?",
        "pairs": "Here is a sample of the generated training pairs — do they look sensible?",
        "eval": "Did the tuned model beat the base in the table above?",
    },
    "llm": {
        "verify": "Read the generated answers above — do they look coherent?",
    },
}


class AgentState(TypedDict, total=False):
    pipeline: str
    hw: dict
    plan: list
    aborted: bool
    error: str | None
    done: list
    report: str


def _preview(pipeline: str, step_name: str, args) -> str:
    """A short peek at a step's output, shown at the human checkpoint."""
    try:
        if pipeline == "embedding" and step_name == "extract":
            mds = sorted(Path(args.extracted).rglob("*.md"))
            if mds:
                text = mds[0].read_text(encoding="utf-8")[:400]
                return f"--- {mds[0]} (first 400 chars) ---\n{text}"
        if pipeline == "embedding" and step_name == "pairs":
            f = Path(args.data) / "embedding_train.jsonl"
            if f.is_file():
                rows = f.read_text(encoding="utf-8").splitlines()[:2]
                sample = [{k: str(v)[:120] for k, v in json.loads(r).items()} for r in rows]
                return "--- sample pairs ---\n" + "\n".join(json.dumps(s) for s in sample)
    except (OSError, json.JSONDecodeError):
        pass
    return ""


def build_graph(pipeline: str, args, gate=None, log=print, install: bool = False):
    """Compile the sequential agent graph for 'embedding' or 'llm'."""
    build = (run_embedding_pipeline if pipeline == "embedding" else run_llm_pipeline)
    steps = build.build_steps(args)
    checkpoints = CHECKPOINTS[pipeline]

    def planner(state: AgentState) -> AgentState:  # pylint: disable=unused-argument
        log("[planner] probing hardware and tools ...")
        hw = graph_common.probe_hardware()
        ok, plan_lines, tweaks = graph_common.plan_fit(pipeline, hw)
        for line in plan_lines:
            log(f"[planner] {line}")
        if tweaks.get("backend") and hasattr(args, "backend"):
            args.backend = tweaks["backend"]
            log(f"[planner] adjusted MinerU backend -> {args.backend}")
        missing = graph_common.check_tools(pipeline, getattr(args, "ollama_url",
                                                             "http://localhost:11434"))
        if missing and install:
            missing = graph_common.auto_install(missing, log)
        if missing:
            fixes = "; ".join(f"{n}: {how}" for n, how, _ in missing)
            return {"hw": hw, "plan": plan_lines, "aborted": True,
                    "error": f"missing tools — {fixes}"}
        if not ok:
            return {"hw": hw, "plan": plan_lines, "aborted": True,
                    "error": "this machine cannot run the plan (see planner advice above)"}
        return {"hw": hw, "plan": plan_lines, "aborted": False, "done": []}

    def make_step_node(step):
        def node(state: AgentState) -> AgentState:
            reason = step.auto_skip() if step.auto_skip else None
            if reason:
                log(f"[{step.name}] skipped ({reason})")
                return {"done": [*state.get("done", []), f"{step.name} (skipped)"]}
            log(f"[{step.name}] {step.desc}")
            try:
                step.run()
            except PipelineError as exc:
                return {"error": f"step '{step.name}' failed: {exc}"}
            if step.name in checkpoints and gate is not None:
                question = checkpoints[step.name]
                if not gate(step.name, question, _preview(pipeline, step.name, args)):
                    return {"error": f"stopped by user at the '{step.name}' checkpoint"}
                log(f"[{step.name}] approved by user")
            return {"done": [*state.get("done", []), step.name]}
        return node

    def report(state: AgentState) -> AgentState:
        if state.get("error"):
            text = (f"Pipeline stopped: {state['error']}\n"
                    f"Completed stages: {', '.join(state.get('done', [])) or 'none'}")
        else:
            text = ("All agents finished. Stages: "
                    + ", ".join(state.get("done", [])))
        log(f"[report] {text}")
        return {"report": text}

    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("report", report)
    for step in steps:
        g.add_node(step.name, make_step_node(step))

    g.set_entry_point("planner")
    g.add_conditional_edges(
        "planner", lambda s: "report" if s.get("aborted") else steps[0].name)
    for current, nxt in zip(steps, steps[1:], strict=False):
        g.add_conditional_edges(
            current.name,
            lambda s, target=nxt.name: "report" if s.get("error") else target)
    g.add_edge(steps[-1].name, "report")
    g.add_edge("report", END)
    return g.compile()


def default_args(pipeline: str, **overrides) -> argparse.Namespace:
    """The same defaults the CLI orchestrators use, as a Namespace."""
    if pipeline == "embedding":
        ns = argparse.Namespace(
            pdfs="pdfs", extracted="extracted", data="data",
            config="configs/soup_embedding.yaml", backend="vlm-transformers",
            caption_model="qwen3-vl:8b", pairs_model="qwen2.5:7b",
            questions_per_chunk=2, sft=False, tuned_output="./output_embedding",
            ollama_url="http://localhost:11434", skip="", from_step=None, dry_run=False)
    else:
        ns = argparse.Namespace(
            config="configs/soup_qwen7b.yaml", data=None, name="my-model",
            quant="q4_K_M", resume=False, ollama_url="http://localhost:11434",
            skip="", from_step=None, dry_run=False)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns
