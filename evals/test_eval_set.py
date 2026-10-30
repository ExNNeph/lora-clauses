"""Checks on the labeled data, so a broken or leaking split fails CI."""
import json
import re
from collections import Counter
from pathlib import Path

from lora_clauses.labels import RISK_IDS
from run_eval import score
from lora_clauses.predict import Prediction

ROOT = Path(__file__).parent.parent
FILES = {"train": ROOT / "data/train.jsonl", "valid": ROOT / "data/valid.jsonl",
         "test_real": ROOT / "evals/test_real.jsonl", "test_synth": ROOT / "evals/test_synth.jsonl"}


def rows(name):
    return [json.loads(line) for line in open(FILES[name], encoding="utf-8")]


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def test_every_row_is_well_formed():
    for name in FILES:
        data = rows(name)
        assert len({r["id"] for r in data}) == len(data), name
        for r in data:
            assert r["lang"] in ("en", "de") and len(r["clause"]) > 20
            assert r["risks"] == sorted(set(r["risks"])) and set(r["risks"]) <= set(RISK_IDS), r["id"]


def test_real_test_set_matches_the_contract_labels():
    real = rows("test_real")
    assert len(real) == 47
    assert sum(len(r["risks"]) for r in real) == 34
    assert sum(not r["risks"] for r in real) == 15


def test_no_clause_is_in_two_splits():
    seen = Counter()
    for name in FILES:
        seen.update({norm(r["clause"]) for r in rows(name)})
    assert [c for c, n in seen.items() if n > 1] == []


def test_training_data_covers_every_risk_in_both_languages():
    have = Counter((r["lang"], risk) for r in rows("train") for risk in r["risks"])
    assert all(have[(lang, risk)] >= 8 for lang in ("en", "de") for risk in RISK_IDS)
    assert sum(not r["risks"] for r in rows("train")) > 100  # fair near misses and neutral clauses


def test_score():
    rows_ = [{"risks": ["a"]}, {"risks": []}, {"risks": ["a", "b"]}]
    p = lambda r: Prediction(risks=r, raw="", prompt_tokens=10, output_tokens=2, seconds=0.5)  # noqa: E731
    s = score(rows_, [p(["a"]), p(["b"]), p(None)])
    assert s["exact"] == round(1 / 3, 3) and s["fair_flagged"] == "1/1" and s["invalid"] == 1
    assert (s["precision"], s["recall"]) == (0.5, 0.333)
