"""Label contract clauses with a LoRA-tuned local model, and compare it with the base model and Claude."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import predict
from .labels import TITLES

DEMO_APP = Path(__file__).with_name("demo_app.py")


def demo() -> None:
    """Start the local web demo on http://localhost:8501 and open the browser."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit('The demo needs Streamlit and MLX: pip install -e ".[demo]"')
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(DEMO_APP), "--server.address", "localhost",
                    "--server.port", "8501", "--browser.gatherUsageStats", "false",
                    "--client.toolbarMode", "minimal",  # hides Streamlit's "Deploy" button
                    "--server.fileWatcherType", "none"])  # no code reloading needed, and no "install Watchdog" hint


def write_mlx_data(out: Path = Path(".mlx-data")) -> None:
    """data/{train,valid}.jsonl -> the chat format mlx_lm.lora reads."""
    out.mkdir(exist_ok=True)
    for split in ("train", "valid"):
        rows = [json.loads(line) for line in open(f"data/{split}.jsonl", encoding="utf-8")]
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(predict.to_chat(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{split}: {len(rows)} examples -> {out / split}.jsonl")


def train(extra: list[str]) -> int:
    write_mlx_data()
    return subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--config", "lora.yaml", *extra]).returncode


def show(name: str, p: predict.Prediction) -> None:
    risks = "invalid answer: " + p.raw if p.risks is None else (
        ", ".join(f"{r} ({TITLES[r]['en']})" for r in p.risks) or "fair (no risk)")
    cost = f", ${p.cost:.4f}" if p.cost else ""
    print(f"{name:>6}: {risks}\n        {p.prompt_tokens} prompt + {p.output_tokens} output tokens, "
          f"{p.seconds:.2f}s{cost}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="lora-clauses", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="label one clause with the base model, the LoRA model and optionally Claude")
    p.add_argument("clause", help="the clause text, or - to read it from stdin")
    p.add_argument("--adapter", default=str(predict.ADAPTER), help="LoRA adapter folder (default: adapters)")
    p.add_argument("--claude", action="store_true", help="also ask Claude (needs ANTHROPIC_API_KEY)")
    sub.add_parser("train", help="fine-tune with mlx_lm.lora using lora.yaml; extra flags go to mlx_lm",
                   add_help=False)
    sub.add_parser("demo", help="start the local web demo on localhost:8501")
    args, extra = ap.parse_known_args()

    if args.cmd == "demo":
        demo()
    elif args.cmd == "train":
        raise SystemExit(train(extra))
    else:
        if extra:
            ap.error(f"unrecognized arguments: {' '.join(extra)}")
        clause = sys.stdin.read() if args.clause == "-" else args.clause
        show("base", predict.local(clause))
        if Path(args.adapter, "adapters.safetensors").exists():
            show("LoRA", predict.local(clause, args.adapter))
        else:
            print(f"  LoRA: no adapter in {args.adapter}/ yet, run `lora-clauses train` first")
        if args.claude:
            show("Claude", predict.claude(clause))
