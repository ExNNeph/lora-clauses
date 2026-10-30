"""Generate the synthetic clauses with Claude. Usage: python data/generate.py   (needs ANTHROPIC_API_KEY, about $4)

Every request asks for a batch of clauses with a fixed label, so the label is known by construction:
- risky clauses for each of the 18 risks, and a few with two risks at once,
- fair "near misses" for each risk (the same topic, but harmless: capped liability, 30-day payment, …),
- neutral clauses (scope, deliverables, definitions).
Train/valid and the synthetic test set come from separate requests with different contract types, so the
test clauses are not paraphrases of training clauses. The real test set (evals/test_real.jsonl, from the six
labeled contracts in contract-red-flags) is never shown to Claude here.

Writes data/train.jsonl, data/valid.jsonl and evals/test_synth.jsonl.
"""
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from lora_clauses.labels import RISKS

MODEL = "claude-opus-5-5"
PRICE = {"input_tokens": 4.00, "output_tokens": 20.00}  # $ per million tokens
ROOT = Path(__file__).parent.parent
CONTEXTS = {
    "train": "freelance web development, graphic design, copywriting, influencer marketing, agency subcontracting, "
             "video production, UX research",
    "test": "photography, translation, management consulting, event staffing, music production, software "
            "licensing, illustration for publishers",
}
PAIRS = [("ip_full_transfer", "rights_before_payment"), ("unlimited_usage", "moral_rights_waiver"),
         ("exclusivity", "broad_noncompete"), ("auto_renewal", "unilateral_changes")]
N = {"train": {"risk": 14, "fair": 6, "pair": 5, "neutral": 30},
     "test": {"risk": 4, "fair": 2, "pair": 2, "neutral": 8}}
LANG = {"en": "English", "de": "German"}
DESC = {r["id"]: r["en"]["description"] for r in RISKS}

SYSTEM = """You write realistic clauses for contracts between a freelancer (or creator, or small agency) and a \
client, used to train and test a clause classifier. Each clause looks like one clause of a real contract: a \
numbered heading ("4. Payment", "§ 6 Vergütung") followed by one to three sentences. Vary the wording, length, \
numbering style, amounts, durations and the contract type a lot; do not reuse phrases between clauses. Write \
plain legal language, no explanations, no placeholders like [Name]."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["clauses"],
          "properties": {"clauses": {"type": "array", "items": {"type": "string"}}}}


def jobs(split: str) -> list[dict]:
    n, out = N[split], []
    for lang in LANG:
        for rid, desc in DESC.items():
            out.append({"lang": lang, "risks": [rid], "n": n["risk"],
                        "ask": f"Each clause must clearly contain this risk for the freelancer: {desc} "
                               "It must not contain any of the other typical freelancer risks."})
            out.append({"lang": lang, "risks": [], "n": n["fair"],
                        "ask": f"Each clause is about the same topic as this risk, but is fair and does NOT contain "
                               f"it: {desc} For example a cap, a short or reasonable period, a mutual right, a "
                               "clear limit or compensation. No other freelancer risk either."})
        for a, b in PAIRS:
            out.append({"lang": lang, "risks": [a, b], "n": n["pair"],
                        "ask": f"Each clause contains both of these risks and no other: (1) {DESC[a]} (2) {DESC[b]}"})
        out.append({"lang": lang, "risks": [], "n": n["neutral"],
                    "ask": "Each clause is a neutral, standard clause with no risk for the freelancer: scope, "
                           "deliverables, definitions, contact persons, notices, severability, reasonable payment "
                           "within 14 or 30 days, a liability cap, mutual termination with notice, and so on."})
    return out


def generate(client: anthropic.Anthropic, split: str, job: dict) -> tuple[list[str], float]:
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if a safety classifier declines, the API retries on a fallback model
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": (
            f"Write {job['n']} different contract clauses in {LANG[job['lang']]}. {job['ask']}\n"
            f"Contract types to spread across: {CONTEXTS[split]}.")}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"declined: {job}")
    text = next(b.text for b in resp.content if b.type == "text")
    usage = resp.usage.model_dump()
    return json.loads(text)["clauses"], sum((usage.get(k) or 0) * p for k, p in PRICE.items()) / 1e6


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    client, cost, rows = anthropic.Anthropic(), 0.0, {}
    real = {norm(json.loads(line)["clause"]) for line in open(ROOT / "evals/test_real.jsonl", encoding="utf-8")}
    for split in ("test", "train"):
        todo = jobs(split)
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda j: generate(client, split, j), todo))
        seen, rows[split] = set(real), []
        for job, (clauses, c) in zip(todo, results):
            cost += c
            for text in clauses:
                if norm(text) not in seen:
                    seen.add(norm(text))
                    rows[split].append({"lang": job["lang"], "clause": text.strip(), "risks": sorted(job["risks"])})
        print(f"{split}: {len(rows[split])} clauses from {len(todo)} requests, ${cost:.2f} so far", file=sys.stderr)
    test_norm = {norm(r["clause"]) for r in rows["test"]}
    train = [r for r in rows["train"] if norm(r["clause"]) not in test_norm]
    random.Random(7).shuffle(train)
    n_valid = len(train) // 10
    for name, split_rows in (("test", rows["test"]), ("valid", train[:n_valid]), ("train", train[n_valid:])):
        for i, r in enumerate(split_rows):
            r["id"] = f"{name}-{i + 1:04d}"
    write(ROOT / "evals/test_synth.jsonl", [{"id": r.pop("id"), **r} for r in rows["test"]])
    write(ROOT / "data/valid.jsonl", [{"id": r.pop("id"), **r} for r in train[:n_valid]])
    write(ROOT / "data/train.jsonl", [{"id": r.pop("id"), **r} for r in train[n_valid:]])
    print(f"train {len(train) - n_valid}, valid {n_valid}, test_synth {len(rows['test'])}; total ${cost:.2f}")


if __name__ == "__main__":
    main()
