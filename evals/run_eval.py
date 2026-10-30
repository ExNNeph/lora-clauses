"""Benchmark base vs LoRA vs Claude. Usage: python evals/run_eval.py [base] [lora] [claude]   (default: all three)

Two test sets, neither used in training:
- real:  47 clauses from the six hand-labeled contracts of contract-red-flags (34 risks, 15 fair clauses)
- synth: Claude-written clauses from separate requests with other contract types than the training data

Per system and set: exact match (the clause's risk set is exactly right), micro precision/recall/F1 over
(clause, risk) pairs, fair clauses flagged, invalid answers (not JSON with known risk ids; counted as []),
mean latency per clause, output tokens per second, and cost per 1,000 clauses. Local models cost $0 in API
fees. The Claude row needs ANTHROPIC_API_KEY. Results go to evals/results.json (merged with earlier rows,
so the local rows and the Claude row can be run separately) and per-clause answers to evals/predictions/.
"""
import json
import sys
from pathlib import Path

from lora_clauses import predict

HERE = Path(__file__).parent
SETS = ("real", "synth")


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in open(HERE / f"test_{name}.jsonl", encoding="utf-8")]


def score(rows: list[dict], preds: list[predict.Prediction]) -> dict:
    tp = fp = fn = exact = fair = fair_flagged = 0
    for r, p in zip(rows, preds):
        gold, got = set(r["risks"]), set(p.risks or [])
        tp, fp, fn = tp + len(gold & got), fp + len(got - gold), fn + len(gold - got)
        exact += gold == got
        fair += not gold
        fair_flagged += not gold and bool(got)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    n = len(rows)
    return {"n": n, "exact": round(exact / n, 3), "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 3),
            "fair_flagged": f"{fair_flagged}/{fair}", "invalid": sum(p.risks is None for p in preds),
            "latency_s": round(sum(p.seconds for p in preds) / n, 3),
            "tokens_per_s": round(sum(p.tokens_per_second for p in preds) / n, 1),
            "prompt_tokens": round(sum(p.prompt_tokens for p in preds) / n),
            "cost_per_1k": round(1000 * sum(p.cost for p in preds) / n, 2)}


def runner(system: str):
    if system == "base":
        return lambda c: predict.local(c)
    if system == "lora":
        return lambda c: predict.local(c, predict.ADAPTER)
    import anthropic

    client = anthropic.Anthropic()
    return lambda c: predict.claude(c, client)


def main() -> None:
    systems = sys.argv[1:] or ["base", "lora", "claude"]
    path = HERE / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    (HERE / "predictions").mkdir(exist_ok=True)
    for system in systems:
        run = runner(system)
        if system != "claude":
            run("warm-up")  # loads the model, so loading time is not counted as latency
        for name in SETS:
            rows = load(name)
            preds = []
            for i, r in enumerate(rows):
                preds.append(run(r["clause"]))
                print(f"\r{system} {name} {i + 1}/{len(rows)}", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
            (HERE / "predictions" / f"{system}-{name}.jsonl").write_text("".join(
                json.dumps({"id": r["id"], "gold": r["risks"], **p.model_dump()}, ensure_ascii=False) + "\n"
                for r, p in zip(rows, preds)), encoding="utf-8")
            results.setdefault(system, {})[name] = score(rows, preds)
            print(system, name, results[system][name])
        path.write_text(json.dumps(results, indent=1) + "\n")
    print_table(results)


def print_table(results: dict) -> None:
    label = {"base": f"base {predict.BASE_MODEL.split('/')[1]}", "lora": "base + LoRA adapter",
             "claude": f"Claude {predict.CLAUDE}"}
    print("\n| system | set | exact match | precision | recall | F1 | fair flagged | invalid | latency | "
          "tokens/s | prompt tokens | $ per 1k clauses |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for system in ("base", "lora", "claude"):
        for name, s in results.get(system, {}).items():
            print(f"| {label[system]} | {name} ({s['n']}) | {s['exact']:.0%} | {s['precision']:.0%} | "
                  f"{s['recall']:.0%} | {s['f1']:.0%} | {s['fair_flagged']} | {s['invalid']} | "
                  f"{s['latency_s']:.2f} s | {s['tokens_per_s']:.0f} | {s['prompt_tokens']} | "
                  f"${s['cost_per_1k']:.2f} |")


if __name__ == "__main__":
    main()
