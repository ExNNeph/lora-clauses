"""Local web demo, started with `lora-clauses demo`: base model vs LoRA model (and optionally Claude) side by side."""
import json
import os
from pathlib import Path

import streamlit as st

from lora_clauses import predict
from lora_clauses.labels import TITLES

ROOT = Path(__file__).parents[2]
EXAMPLES = [json.loads(line) for line in open(ROOT / "evals/test_real.jsonl", encoding="utf-8")]
RESULTS = ROOT / "evals/results.json"
HAS_ADAPTER = (predict.ADAPTER / "adapters.safetensors").exists()

st.set_page_config(page_title="LoRA clause classifier", page_icon="🧩", layout="wide")
st.title("🧩 A 1.7B model fine-tuned with LoRA on a MacBook")
st.caption(f"`{predict.BASE_MODEL}` runs locally with MLX and labels contract clauses with 18 freelancer risks "
           "(unlimited liability, pay-when-paid, auto-renewal, …). The **base model** gets the whole taxonomy in its "
           "prompt. The **LoRA model** was fine-tuned on 750 synthetic clauses and gets only a one-line instruction. "
           "The examples below come from six real labeled contracts that were never used in training.")

with st.sidebar:
    api_key = st.text_input("Anthropic API key (optional)", type="password",
                            help="Adds a Claude column. Used only for your requests in this local session, never stored.")
    use_claude = bool(api_key or os.getenv("ANTHROPIC_API_KEY"))
    if use_claude:
        st.success(f"🔑 Key set: Claude ({predict.CLAUDE}) answers too, at about half a cent per clause.")
    else:
        st.info("No key: base vs LoRA only, everything on this machine. Paste a key and press Enter to add Claude.")
    if not HAS_ADAPTER:
        st.warning("No adapter in `adapters/` yet. Train it with `lora-clauses train` (about 20 minutes).")

labels = [f"{e['id']} · " + (", ".join(e["risks"]) or "fair") for e in EXAMPLES]
pick = st.selectbox("Example clause from a real contract (label after the dot)", range(len(EXAMPLES)),
                    format_func=lambda i: labels[i], index=10)
clause = st.text_area("Clause (edit it or paste your own, English or German)", EXAMPLES[pick]["clause"], height=120)
gold = EXAMPLES[pick]["risks"] if clause == EXAMPLES[pick]["clause"] else None


def show(title: str, p: predict.Prediction) -> None:
    st.markdown(f"#### {title}")
    if p.risks is None:
        st.error("Invalid answer (not JSON with known risk ids)")
    elif not p.risks:
        st.success("Fair: no risk found")
    for r in p.risks or []:
        hit = "" if gold is None else (" ✅" if r in gold else " ❌ not in the label")
        st.warning(f"**{TITLES[r]['en']}** · `{r}`{hit}")
    for r in sorted(set(gold or []) - set(p.risks or [])):
        st.caption(f"missed: {TITLES[r]['en']} (`{r}`)")
    cost = f" · ${p.cost:.4f}" if p.cost else " · $0 (local)"
    st.caption(f"{p.seconds:.2f} s · {p.prompt_tokens} prompt + {p.output_tokens} output tokens{cost}")
    with st.expander("raw answer"):
        st.code(p.raw or "(empty)", language="json")


if st.button("Label the clause", type="primary") and clause.strip():
    cols = st.columns(3 if use_claude else 2)
    with cols[0], st.spinner("base model …"):
        show("Base model · full taxonomy in prompt", predict.local(clause))
    with cols[1]:
        if HAS_ADAPTER:
            with st.spinner("LoRA model …"):
                show("LoRA model · one-line prompt", predict.local(clause, predict.ADAPTER))
        else:
            st.subheader("Base + LoRA adapter")
            st.info("Train the adapter first: `lora-clauses train`.")
    if use_claude:
        with cols[2], st.spinner("Claude …"):
            import anthropic

            show(f"Claude ({predict.CLAUDE})", predict.claude(clause, anthropic.Anthropic(api_key=api_key or None)))

if RESULTS.exists():
    st.divider()
    st.subheader("Benchmark (evals/run_eval.py)")
    results = json.loads(RESULTS.read_text())
    name = {"base": "base", "lora": "base + LoRA", "claude": f"Claude {predict.CLAUDE}"}
    st.dataframe([{"system": name[s], "test set": f"{t} ({r['n']})", "exact match": f"{r['exact']:.0%}",
                   "F1": f"{r['f1']:.0%}", "fair flagged": r["fair_flagged"], "invalid": r["invalid"],
                   "latency": f"{r['latency_s']:.2f} s", "prompt tokens": r["prompt_tokens"],
                   "$ per 1k": f"${r['cost_per_1k']:.2f}"}
                  for s in name if s in results for t, r in results[s].items()], hide_index=True)
