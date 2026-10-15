"""Built-in GRPO reward functions"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

DEFAULT_JUDGE_PROMPT = (
    "Rate how well the response answers the prompt on a scale from 0 to 10.\n\n"
    "Prompt:\n{prompt}\n\nResponse:\n{completion}\n\n"
    "Reply with your reasoning, then end with a final line of the form `Score: <number>`."
)

_SCORE_RE = re.compile(r"score\s*[:=]?\s*\**\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _to_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(f"{m['role']}: {m['content']}" for m in value)
    return str(value)


def parse_judge_score(text: str, max_score: float) -> float:
    matches = _SCORE_RE.findall(text)
    if not matches:
        raise ValueError(f"no score found in judge response: {text!r}")
    return min(max(float(matches[-1]) / max_score, 0.0), 1.0)


def llm_judge_reward(
    completions: list,
    prompts: list | None = None,
    *,
    base_url: str,
    model: str,
    prompt_template: str = DEFAULT_JUDGE_PROMPT,
    api_key: str | None = None,
    max_score: float = 10.0,
    max_concurrency: int = 8,
    timeout: float = 60.0,
    **kwargs,
) -> list[float]:
    """Score completions in [0, 1] with an OpenAI-compatible chat endpoint.

    `prompt_template` is formatted with `prompt`, `completion` and any dataset
    columns passed through to rewards (e.g. `{answer}`). Failed requests or
    unparsable responses score 0.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    columns = {k: v for k, v in kwargs.items() if isinstance(v, list)}

    def score(i: int) -> float:
        try:
            content = prompt_template.format(
                prompt=_to_text(prompts[i]) if prompts else "",
                completion=_to_text(completions[i]),
                **{k: v[i] for k, v in columns.items()},
            )
            resp = requests.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.0,
                },
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return parse_judge_score(text, max_score)
        except Exception as exc:
            LOG.warning(f"LLM judge failed, scoring 0: {exc}")
            return 0.0

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        return list(pool.map(score, range(len(completions))))
