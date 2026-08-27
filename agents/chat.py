"""Chatbot that guides you through fine-tuning and runs the agents in the background.

Start it (from the repo root, Soup venv active):
    python agents/chat.py

It asks what you want to build, tells you where to put your PDFs, checks your
hardware with the planner agent, then launches the LangGraph pipeline in a
background thread. You only hear from it again at human-in-the-loop
checkpoints (approve the extracted text, the training pairs, the eval result)
— no other manual steps.
"""

import queue
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graph_common  # noqa: E402  # pylint: disable=wrong-import-position
from pipeline_agents import (  # noqa: E402  # pylint: disable=wrong-import-position
    build_graph,
    default_args,
)

BOT = "🤖"


def say(text: str) -> None:
    print(f"{BOT} {text}")


def ask(text: str) -> str:
    return input(f"{BOT} {text}\n> ").strip()


def choose_pipeline() -> str:
    say("Hi! I can build two things for you, fully locally:")
    say("  1. smart search over your PDF manuals (embedding model + RAG)")
    say("  2. your own chat model (LLM fine-tune)")
    while True:
        choice = ask("Which one? [1/2, Enter = 1]")
        if choice in ("", "1"):
            return "embedding"
        if choice == "2":
            return "llm"
        say("Please answer 1 or 2.")


def collect_pdfs() -> str:
    pdf_dir = Path("pdfs")
    pdf_dir.mkdir(exist_ok=True)
    while True:
        count = len(list(pdf_dir.glob("*.pdf")))
        if count:
            say(f"Found {count} PDF(s) in {pdf_dir.resolve()} — good.")
            return str(pdf_dir)
        say(f"Please copy your PDF files into this folder:\n    {pdf_dir.resolve()}")
        answer = ask("Type 'done' when the PDFs are there (or 'quit' to stop).")
        if answer.lower() in ("quit", "q", "exit"):
            raise SystemExit(0)


def collect_llm_data() -> str:
    while True:
        path = ask("Where is your training JSONL? [Enter = data/sft_train.jsonl]") \
               or "data/sft_train.jsonl"
        if Path(path).is_file():
            return path
        say(f"I can't find {path}. Tip: the embedding pipeline with --sft creates "
            "data/sft_train.jsonl from your PDFs. Type another path, or 'quit'.")
        if path.lower() in ("quit", "q", "exit"):
            raise SystemExit(0)


def preflight(pipeline: str) -> bool:
    say("Let me check your machine ...")
    hw = graph_common.probe_hardware()
    ok, lines, _ = graph_common.plan_fit(pipeline, hw)
    for line in lines:
        say(line)
    missing = graph_common.check_tools(pipeline)
    if missing:
        say("Some tools are missing. I can install the automatic ones when we start "
            "(the rest need one command from you):")
        for name, how, cmd in missing:
            say(f"  - {name}: {how}" + ("  (I can do this one)" if cmd else ""))
    if not ok:
        say("I have to stop here — see the options above, then run me again.")
        return False
    return True


def run_in_background(pipeline: str, ns) -> int:
    """Launch the agent graph in a worker thread; relay logs and checkpoints."""
    events: queue.Queue = queue.Queue()

    def log(line: str) -> None:
        events.put(("log", str(line)))

    def gate(step_name: str, question: str, preview: str) -> bool:
        holder: dict = {}
        done = threading.Event()
        events.put(("ask", step_name, question, preview, holder, done))
        done.wait()
        return holder.get("ok", False)

    result: dict = {}

    def work() -> None:
        graph = build_graph(pipeline, ns, gate=gate, log=log, install=True)
        result.update(graph.invoke({"pipeline": pipeline}))
        events.put(("end",))

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    say("The agents are running in the background. I'll only interrupt you at "
        "checkpoints. (Ctrl+C aborts.)")

    while True:
        event = events.get()
        if event[0] == "log":
            print(f"   {event[1]}")
        elif event[0] == "ask":
            _, step_name, question, preview, holder, done = event
            print(f"\n{BOT} CHECKPOINT after '{step_name}':")
            if preview:
                print(preview)
            answer = ask(f"{question} [Y/n]").lower()
            holder["ok"] = answer in ("", "y", "yes")
            done.set()
        elif event[0] == "end":
            break
    thread.join()

    if result.get("error"):
        say(f"We stopped early: {result['error']}")
        say("Fix that and run me again — finished stages are skipped automatically.")
        return 1
    say("All done! " + result.get("report", ""))
    if pipeline == "embedding":
        say("Your tuned search model is in ./output_embedding — load it with "
            "sentence-transformers in your RAG stack.")
    else:
        say(f"Chat with your model:  ollama run {ns.name}")
    return 0


def choose_model(pipeline: str, hw: dict) -> str:
    """Advise which base model fits this machine and let the user pick."""
    options = graph_common.recommend_models(pipeline, hw)
    fitting = [o for o in options if o[2]]
    say("Based on your hardware, here are the base models I can fine-tune "
        "(* = fits your machine):")
    for i, (model, why, fits) in enumerate(options, 1):
        say(f"  {i}. {'*' if fits else ' '} {model} — {why}")
    default = fitting[0][0] if fitting else options[-1][0]
    say(f"My recommendation for you: {default}")
    while True:
        answer = ask("Pick a number, paste a Hugging Face model id, or press "
                     "Enter for the recommendation.")
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            model, _, fits = options[int(answer) - 1]
            if not fits:
                say("Heads up: that one likely does NOT fit your GPU — I'll try anyway "
                    "if you insist, but the recommended one is safer.")
                if ask("Use it anyway? [y/N]").lower() not in ("y", "yes"):
                    continue
            return model
        if "/" in answer:
            return answer
        say("Please give a number from the list or a full model id like 'org/name'.")


def write_session_config(pipeline: str, base: str, data_path: str | None) -> str:
    """Copy the template config with the chosen base model (and data path) baked in."""
    template = Path("configs") / ("soup_embedding.yaml" if pipeline == "embedding"
                                  else "soup_qwen7b.yaml")
    out = Path("configs") / f"chat_{pipeline}.yaml"
    text = template.read_text(encoding="utf-8")
    text = re.sub(r"^base:.*$", f"base: {base}", text, count=1, flags=re.M)
    if data_path:
        text = re.sub(r"^(\s+)train:.*$", rf"\g<1>train: {data_path}", text,
                      count=1, flags=re.M)
    out.write_text(text, encoding="utf-8")
    say(f"I wrote your run configuration to {out} (base model: {base}).")
    return str(out)


def explain_steps(pipeline: str, ns) -> None:
    """The exhaustive, layman-friendly walkthrough of what is about to happen."""
    say("Here is exactly what I will do, in order. You can walk away — I only stop "
        "at the checkpoints marked [CHECKPOINT], where I show you the result and "
        "ask if it looks good.")
    if pipeline == "embedding":
        for line in (
            "1. CHECK — I verify the tools, the Ollama models, and your folders. "
            "Anything missing that I can install myself, I install.",
            f"2. EXTRACT — MinerU reads every PDF in {ns.pdfs}/ and turns it into "
            "text files (with math and tables preserved). Roughly 500-1,500 "
            "pages/hour; the first run also downloads models (~15 GB).",
            "   [CHECKPOINT] I show you a sample of the extracted text.",
            "3. CAPTION — a local vision model looks at every screenshot in your "
            "PDFs and writes a description into the text (5-15 s per image).",
            "4. PAIRS — a local language model writes practice questions for every "
            "section of your documents (5-15 s per section). This becomes the "
            "training data.",
            "   [CHECKPOINT] I show you a couple of the generated pairs.",
            "5. TRAIN — the embedding model learns your domain (10-30 min).",
            "6. EVAL — I measure the tuned model against the original and show you "
            "the score table.",
            "   [CHECKPOINT] you confirm the tuned model actually won.",
            "After that, your search model is in ./output_embedding — nothing else "
            "to do.",
        ):
            say(line)
    else:
        for line in (
            "1. CHECK — I verify the tools, your config, and your training data.",
            "2. VALIDATE — Soup checks every training example for format problems.",
            "3. TRAIN — the QLoRA fine-tune runs (roughly 1-2 hours per 1,000 "
            "examples; the first run downloads the base model, ~15 GB).",
            "4. VERIFY — I load the result and generate a few test answers.",
            "   [CHECKPOINT] you read them and confirm they are coherent.",
            "5. EVAL — Soup runs its automatic evaluation (optional; failures here "
            "do not stop us).",
            "6. EXPORT — the model is merged and converted (~20-30 min).",
            f"7. DEPLOY — it lands in Ollama as '{ns.name}'; afterwards you chat "
            f"with it via:  ollama run {ns.name}",
        ):
            say(line)


def gpu_selfcare() -> None:
    """Fix GPU problems automatically where safe; hand admin steps to the user."""
    issues = graph_common.troubleshoot_gpu()
    for advice in issues["manual"]:
        say(f"NEEDS YOU (admin): {advice}")
    for desc, cmd in issues["auto"]:
        say(f"I found a problem I can fix myself: {desc}.")
        if ask("Fix it now? [Y/n]").lower() in ("", "y", "yes"):
            try:
                graph_common.run_cmd(cmd)
                say("Fixed.")
            except graph_common.PipelineError as exc:
                say(f"That fix failed ({exc}) — run it manually: "
                    + " ".join(str(c) for c in cmd))



def main() -> int:
    pipeline = choose_pipeline()
    if pipeline == "embedding":
        pdfs = collect_pdfs()
        want_sft = ask("Also produce chat-model training data from these PDFs? "
                       "[y/N]").lower() in ("y", "yes")
        ns = default_args("embedding", pdfs=pdfs, sft=want_sft)
    else:
        data = collect_llm_data()
        name = ask("What should the finished model be called in Ollama? "
                   "[Enter = my-model]") or "my-model"
        ns = default_args("llm", data=data, name=name)

    if not preflight(pipeline):
        return 1
    gpu_selfcare()

    hw = graph_common.probe_hardware()
    base = choose_model(pipeline, hw)
    ns.config = write_session_config(
        pipeline, base, ns.data if pipeline == "llm" else None)

    explain_steps(pipeline, ns)
    if ask("Ready to start? [Y/n]").lower() not in ("", "y", "yes"):
        say("Okay — run me again whenever you're ready.")
        return 0
    return run_in_background(pipeline, ns)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print()
        say("Stopped. Progress is kept — finished stages are skipped next time.")
        sys.exit(130)
