import pytest
from streamlit.testing.v1 import AppTest

from lora_clauses import cli, predict


def fake_local(clause, adapter=None):
    risks = ["ip_full_transfer", "rights_before_payment"] if adapter else ["ip_full_transfer"]
    return predict.Prediction(risks=risks, raw="{}", prompt_tokens=60 if adapter else 600, output_tokens=12,
                              seconds=0.4)


@pytest.fixture
def app(monkeypatch, tmp_path):
    (tmp_path / "adapters.safetensors").write_bytes(b"")  # the demo only checks that an adapter exists
    monkeypatch.setattr(predict, "ADAPTER", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(predict, "local", fake_local)  # no MLX on CI runners
    return AppTest.from_file(str(cli.DEMO_APP), default_timeout=30)  # the file `lora-clauses demo` starts


def test_compares_base_and_lora_on_a_real_clause(app, monkeypatch):
    at = app.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    text = " ".join(w.value for w in at.warning)
    assert text.count("All rights transferred") == 2  # base and LoRA
    assert "Rights pass before payment" in text or "rights_before_payment" in text
    assert any("missed" in c.value and "rights_before_payment" in c.value for c in at.caption)  # base missed it
