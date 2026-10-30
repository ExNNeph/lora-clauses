"""Three ways to label a clause: the base model, the base model with the LoRA adapter (both local, MLX), and Claude.

Each returns a Prediction with the parsed risks (None if the answer was not valid JSON), the raw text, tokens,
the time it took and the cost.
"""
import time
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from .labels import SCHEMA, answer, messages, parse

BASE_MODEL = "mlx-community/Qwen3-1.7B-4bit"
ADAPTER = Path("adapters")
CLAUDE = "claude-opus-5-5"
PRICE = {"input_tokens": 4.00, "output_tokens": 20.00}  # $ per million tokens, claude-opus-5-5


class Prediction(BaseModel):
    risks: list[str] | None
    raw: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    tokens_per_second: float = 0.0
    cost: float = 0.0


@lru_cache(maxsize=2)
def model(adapter: str | None):
    from mlx_lm import load  # imported here: CI runners are not Apple Silicon and have no MLX

    return load(BASE_MODEL, adapter_path=adapter)


def local(clause: str, adapter: str | Path | None = None) -> Prediction:
    """Greedy decoding. With an adapter the short prompt is used, the one the LoRA model was trained on."""
    from mlx_lm import stream_generate

    m, tok = model(str(adapter) if adapter else None)
    prompt = tok.apply_chat_template(messages(clause, short=adapter is not None), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    start, text, last = time.perf_counter(), "", None
    for last in stream_generate(m, tok, prompt, max_tokens=120):
        text += last.text
    return Prediction(risks=parse(text), raw=text.strip(), prompt_tokens=last.prompt_tokens,
                      output_tokens=last.generation_tokens, seconds=time.perf_counter() - start,
                      tokens_per_second=last.generation_tps)


def claude(clause: str, client=None) -> Prediction:
    import anthropic

    client = client or anthropic.Anthropic()
    start = time.perf_counter()
    resp = client.beta.messages.create(
        model=CLAUDE, max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if a safety classifier declines, the API retries on a fallback model
        system=messages(clause, short=False)[0]["content"],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": clause}],
    )
    seconds = time.perf_counter() - start
    text = next((b.text for b in resp.content if b.type == "text"), "")
    usage = {k: getattr(resp.usage, k) or 0 for k in PRICE}
    return Prediction(risks=None if resp.stop_reason == "refusal" else parse(text), raw=text,
                      prompt_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"], seconds=seconds,
                      tokens_per_second=usage["output_tokens"] / seconds,
                      cost=sum(usage[k] * PRICE[k] for k in PRICE) / 1e6)


def to_chat(row: dict) -> dict:
    """A labeled clause in the chat format mlx_lm.lora trains on."""
    return {"messages": messages(row["clause"], short=True) + [{"role": "assistant", "content": answer(row["risks"])}]}
