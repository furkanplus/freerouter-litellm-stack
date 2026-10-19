from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from freerouter.gateway import CHAT_TIMEOUT, PROBE_TAG, TOOL_PROBE_MAX_TOKENS, Gateway
from freerouter.policy import Policy, parse_params
from freerouter.transport import Response

if TYPE_CHECKING:
    from collections.abc import Mapping

    from freerouter.transport import JsonValue


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("org/model-20b-and-70b", 70.0),
        ("MODEL-E4B", 4.0),
        ("org/model-0.3b", 0.3),
        ("org/model-without-size", None),
        ("llama-4-scout-17b-16e-instruct", None),
    ],
)
def test_parameter_scan_preserves_values(model_id: str, expected: float | None) -> None:
    assert parse_params(model_id) == expected


def test_policy_default_mappings_are_independent() -> None:
    first = Policy()
    second = Policy()
    assert first == second
    assert not first.merge
    assert not first.roles
    assert not first.overrides
    assert first.merge is not second.merge
    assert first.roles is not second.roles
    assert first.overrides is not second.overrides


def test_tool_probe_preserves_the_wire_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[JsonValue] = []
    response = Response(status=200, body=b"{}")

    def capture(
        url: str,
        *,
        method: str,
        headers: Mapping[str, str],
        payload: JsonValue,
        timeout: float,
    ) -> Response:
        assert url == "https://gateway.example/v1/chat/completions"
        assert method == "POST"
        assert headers == {"Authorization": "Bearer test-only"}
        assert timeout == CHAT_TIMEOUT
        captured.append(payload)
        assert json.loads(json.dumps(payload)) == payload
        return response

    monkeypatch.setattr("freerouter.gateway.request", capture)
    assert Gateway("https://gateway.example", "test-only").chat_tools("demo") is response
    assert captured == [
        {
            "model": "demo",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Paris right now? Use the get_weather tool.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": TOOL_PROBE_MAX_TOKENS,
            "stream": False,
            "metadata": {"tags": [PROBE_TAG]},
        }
    ]
