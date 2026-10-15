from unittest.mock import MagicMock

import pytest
import requests

from axolotl.core.trainers.grpo import GRPOStrategy, rewards
from axolotl.core.trainers.grpo.rewards import llm_judge_reward, parse_judge_score
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.trl import LLM_JUDGE_REWARD, TRLConfig


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Looks right.\nScore: 8", 0.8),
        ("score = 7.5 then **Score:** 9", 0.9),
        ("Score: 15", 1.0),
        ("Score: -2", 0.0),
    ],
)
def test_parse_judge_score(text, expected):
    assert parse_judge_score(text, max_score=10) == pytest.approx(expected)


def test_parse_judge_score_unparsable():
    with pytest.raises(ValueError):
        parse_judge_score("great answer", max_score=10)


def _judge_reply(text):
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    return resp


def test_llm_judge_reward_formats_prompt(monkeypatch):
    post = MagicMock(return_value=_judge_reply("Score: 6"))
    monkeypatch.setattr(rewards.requests, "post", post)

    scores = llm_judge_reward(
        [[{"role": "assistant", "content": "4"}]],
        [[{"role": "user", "content": "2+2?"}]],
        base_url="http://judge/v1/",
        model="judge",
        prompt_template="Q: {prompt} A: {completion} Gold: {answer}",
        api_key="k",
        answer=["4"],
    )

    assert scores == [pytest.approx(0.6)]
    url, kwargs = post.call_args.args[0], post.call_args.kwargs
    assert url == "http://judge/v1/chat/completions"
    assert kwargs["headers"] == {"Authorization": "Bearer k"}
    assert kwargs["json"]["messages"][0]["content"] == (
        "Q: user: 2+2? A: assistant: 4 Gold: 4"
    )


def test_llm_judge_reward_failures_score_zero(monkeypatch):
    replies = {"a": _judge_reply("no score"), "b": requests.ConnectionError("down")}

    def post(url, json, **kwargs):
        reply = replies[json["messages"][0]["content"]]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(rewards.requests, "post", post)
    scores = llm_judge_reward(
        ["a", "b"],
        base_url="http://judge/v1",
        model="m",
        prompt_template="{completion}",
    )
    assert scores == [0.0, 0.0]


def test_llm_judge_config_required():
    with pytest.raises(ValueError, match="trl.llm_judge"):
        TRLConfig(reward_funcs=[LLM_JUDGE_REWARD])


def test_set_trainer_args_binds_judge_config():
    cfg = DictDefault(
        trl={
            "reward_funcs": [LLM_JUDGE_REWARD],
            "llm_judge": {"base_url": "http://judge/v1", "model": "m", "api_key": None},
        }
    )
    (reward_func,) = GRPOStrategy.set_trainer_args(cfg)[0]
    assert reward_func.func is llm_judge_reward
    assert reward_func.keywords == {"base_url": "http://judge/v1", "model": "m"}
