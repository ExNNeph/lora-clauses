from types import SimpleNamespace

from lora_clauses import predict
from lora_clauses.labels import FULL_SYSTEM, RISK_IDS, SHORT_SYSTEM, answer, parse


def test_parse_accepts_plain_and_fenced_json():
    assert parse('{"risks": ["auto_renewal"]}') == ["auto_renewal"]
    assert parse('```json\n{\n  "risks": ["exclusivity", "auto_renewal", "exclusivity"]\n}\n```') == [
        "auto_renewal", "exclusivity"]
    assert parse('{"risks": []}') == []


def test_parse_rejects_unknown_ids_and_broken_json():
    assert parse('{"risks": ["late_fees"]}') is None
    assert parse('{"risks": "auto_renewal"}') is None
    assert parse('{"risks": [') is None
    assert parse("The clause is fair.") is None


def test_taxonomy_and_prompts():
    assert len(RISK_IDS) == 18 and len(set(RISK_IDS)) == 18
    assert all(r in FULL_SYSTEM for r in RISK_IDS)
    assert not any(r in SHORT_SYSTEM for r in RISK_IDS)  # the LoRA model has to know them by heart


def test_training_example_format():
    ex = predict.to_chat({"clause": "§ 4 Zahlung ...", "risks": ["pay_when_paid", "long_payment_terms"]})
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert ex["messages"][0]["content"] == SHORT_SYSTEM
    assert ex["messages"][2]["content"] == answer(["long_payment_terms", "pay_when_paid"])


class FakeClient:
    """Stands in for anthropic.Anthropic: records the request, returns a canned answer."""

    def __init__(self, text, stop_reason="end_turn"):
        self.request, self.text, self.stop_reason = None, text, stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.request = kw
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=self.text)],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=50))


def test_claude_prediction_uses_full_prompt_schema_and_prices():
    fake = FakeClient('{"risks": ["unlimited_usage"]}')
    p = predict.claude("The Brand may use the content forever.", fake)
    assert p.risks == ["unlimited_usage"]
    assert fake.request["model"] == "claude-opus-5-5"
    assert fake.request["system"] == FULL_SYSTEM
    assert fake.request["output_config"]["format"]["schema"]["properties"]["risks"]["items"]["enum"] == RISK_IDS
    assert round(p.cost, 6) == round((1000 * 4 + 50 * 20) / 1e6, 6)


def test_claude_refusal_counts_as_invalid():
    assert predict.claude("x", FakeClient("", stop_reason="refusal")).risks is None
