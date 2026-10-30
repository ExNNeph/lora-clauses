"""The 18 contract risks (taken from the contract-red-flags repo), the prompts and the answer parser.

The task: one contract clause in, the ids of the risks it contains out, as JSON. A fair clause gets [].
"""
import json
import re
from importlib.resources import files

RISKS = json.loads(files("lora_clauses").joinpath("risks.json").read_text(encoding="utf-8"))
RISK_IDS = [r["id"] for r in RISKS]
TITLES = {r["id"]: {"en": r["en"]["title"], "de": r["de"]["title"]} for r in RISKS}

TASK = ('Which of the known freelancer risks does this contract clause contain? '
        'Answer only with JSON: {"risks": [...]} with the risk ids, or {"risks": []} if the clause is fair.')

# The base model and Claude get the whole taxonomy in the prompt. The LoRA model learned it in training,
# so it gets only the task line (about 60 tokens instead of about 600).
FULL_SYSTEM = (TASK + "\n\nKnown risks (id: what makes a clause risky):\n"
               + "\n".join(f"- {r['id']}: {r['en']['description']}" for r in RISKS)
               + "\n\nA clause that only mentions a topic (a capped liability, confidentiality for a few years, "
                 "payment within 14 days) is fair. A clause can contain more than one risk.")
SHORT_SYSTEM = TASK

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["risks"],
          "properties": {"risks": {"type": "array", "items": {"type": "string", "enum": RISK_IDS}}}}


def messages(clause: str, short: bool) -> list[dict]:
    return [{"role": "system", "content": SHORT_SYSTEM if short else FULL_SYSTEM},
            {"role": "user", "content": clause}]


def answer(risks: list[str]) -> str:
    """The target text the LoRA model is trained to write."""
    return json.dumps({"risks": sorted(risks)})


def parse(text: str) -> list[str] | None:
    """The risk ids in a model answer, or None if it is not valid JSON with known ids only."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        risks = json.loads(m.group(0))["risks"] if m else None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(risks, list) or not all(r in RISK_IDS for r in risks):
        return None
    return sorted(set(risks))
