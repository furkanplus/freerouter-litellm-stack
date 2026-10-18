"""Local addition: the usability gate, merging, roles, context learning, breaker."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING

from freerouter.discovery import DiscoveryResult, _finish, _model_ids
from freerouter.planning import POOL_ALIAS, canonical_alias, plan_deployments
from freerouter.policy import MIN_CONTEXT, Override, Policy, Role, hard_rejection, parse_params
from freerouter.refresh import CycleContext, Settings, apply_plan, run_cycle
from freerouter.state import (
    HealthStatus,
    ProbeOutcome,
    forget_expired_rejections,
    learn_context,
    load_state,
    record_probe,
    record_tools,
)
from freerouter.traffic import context_signals, learned_ceiling, stated_limit
from tests.conftest import NOW, make_health, make_provider, make_state
from tests.test_refresh import CHATTY, FakeGateway, _wire

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

ENV = {"DEMO_API_KEY": "secret"}
HEALTHY = HealthStatus.HEALTHY


def test_parameter_counts_are_read_from_ids_but_moe_active_tokens_are_not() -> None:
    assert parse_params("llama-3.1-8b-instruct") == 8
    assert parse_params("gpt-oss:20b") == 20
    assert parse_params("ERNIE-4.5-0.3B-PT") == 0.3
    assert parse_params("nemotron-3-super-120b-a12b") == 120
    assert parse_params("nemotron-3-nano-30b-a3b") == 30
    assert parse_params("gemma-4-e4b") == 4
    assert parse_params("deepseek-v4.1-flash") is None
    assert parse_params("qwen-3.8-27b") == 27
    assert parse_params("llama-4-scout-17b-16e-instruct") is None


def test_tiny_and_non_chat_models_are_rejected_by_name() -> None:
    assert hard_rejection("meta/llama-3.2-3b-instruct") is not None
    assert hard_rejection("canopylabs/orpheus-v1-english") is not None
    assert hard_rejection("granite-4.0-h-micro") is not None
    assert hard_rejection("command-r7b-12-2024") is not None
    assert hard_rejection("gpt-oss-20b") is None
    assert hard_rejection("deepseek-v4.1-flash") is None


def test_the_name_filter_ignores_the_vendor_path() -> None:
    # "*micro*" targets granite-…-micro, not the microsoft/ vendor prefix.
    assert hard_rejection("microsoft/mai-ds-r1:free") is None
    assert hard_rejection("microsoft/phi-4-reasoning-plus") is None
    assert hard_rejection("@cf/ibm/granite-4.0-h-micro") is not None


def test_the_cap_keeps_the_biggest_usable_models_not_the_first_alphabetically() -> None:
    # Given: upstream sorted alphabetically and cut, keeping llama-3.2-1b over gpt-oss-120b
    provider = make_provider(max_models=2)
    entries = (
        ("@cf/meta/llama-3.2-1b", None),
        ("@cf/ibm/granite-4.0-h-micro", None),
        ("@cf/openai/gpt-oss-120b", None),
        ("@cf/openai/gpt-oss-20b", None),
        ("@cf/moonshot/kimi-k2.6", 262144),
        ("@cf/small-window-70b", 8192),
    )

    # When
    result = _finish(provider, entries)

    # Then
    assert result.models == ("@cf/moonshot/kimi-k2.6", "@cf/openai/gpt-oss-120b")
    assert "@cf/meta/llama-3.2-1b" in result.dropped_by_gate
    assert "@cf/small-window-70b" in result.dropped_by_gate
    assert result.dropped_by_cap == ("@cf/openai/gpt-oss-20b",)
    assert dict(result.context) == {"@cf/moonshot/kimi-k2.6": 262144}


def test_require_keeps_only_entries_marked_free_by_a_field() -> None:
    provider = make_provider(
        discovery={"mode": "static", "models": [], "require": {"tier": ["turbo"]}}
    )
    payload = {
        "data": [
            {"id": "glm-5.2", "tier": "turbo", "context_window": {"tokens": 976000}},
            {"id": "claude-opus-5", "tier": "pro", "context_window": {"tokens": 1000000}},
        ]
    }
    assert _model_ids(payload, provider.discovery) == (("glm-5.2", 976000),)


def test_a_small_window_deployment_is_rejected_but_the_same_model_elsewhere_is_not() -> None:
    # Given: the same model, a 1M window on one provider and an 8k TPM cap on another
    roomy = make_provider(id="roomy", credential="DEMO_API_KEY")
    capped = make_provider(id="capped", credential="DEMO_API_KEY", limits={"tpm": 8000})
    state = make_state(
        make_health(provider="roomy", model_id="m-70b", status=HEALTHY, context=1_000_000),
        make_health(provider="capped", model_id="m-70b", status=HEALTHY, context=1_000_000),
    )

    # When
    plan = plan_deployments([roomy, capped], state, ENV)

    # Then
    pooled = {(d.provider_id, d.alias) for d in plan if d.alias == POOL_ALIAS}
    assert pooled == {("roomy", POOL_ALIAS)}
    assert ("capped", "fr/capped/m-70b") in {(d.provider_id, d.alias) for d in plan}


def test_traffic_that_proved_a_large_prompt_beats_a_small_declaration() -> None:
    provider = make_provider()
    health = make_health(status=HEALTHY, context=32_000, proven_input=90_000)
    assert Policy().rejection(health, provider, "alpha") is None
    assert Policy().effective_context(health, provider, "alpha") == 90_000


def test_a_small_successful_prompt_never_becomes_the_context() -> None:
    # Regression: a 14-token success on a deployment with no declared window was
    # read as "context 14" and dropped it from the pool.
    provider = make_provider()
    health = make_health(status=HEALTHY, proven_input=14)
    assert Policy().effective_context(health, provider, "alpha") is None
    assert Policy().rejection(health, provider, "alpha") is None


def test_unknown_capability_pools_and_proven_inadequacy_does_not() -> None:
    provider = make_provider()
    assert Policy().rejection(make_health(), provider, "alpha") is None
    assert Policy().rejection(make_health(tools=False), provider, "alpha") == "no tool calling"
    small = make_health(max_input=MIN_CONTEXT - 1)
    assert Policy().rejection(small, provider, "alpha") is not None
    keep = Policy(overrides={"alpha": Override(keep=True)})
    assert keep.rejection(make_health(tools=False), provider, "alpha") is None


def test_merge_map_and_roles_shape_the_aliases() -> None:
    # Given
    provider = make_provider(discovery={"mode": "static", "models": ["gemma-4-31b-it"]})
    policy = Policy(
        merge={"nemotron-3-ultra": "nemotron-3-ultra-550b-a55b"},
        roles={
            "hermes-smart": Role(include=("gemma-4-31b",)),
            "hermes-long": Role(min_context=200_000),
        },
    )
    state = make_state(
        make_health(model_id="google/gemma-4-31b-it", status=HEALTHY, context=262_144),
        make_health(model_id="nemotron-3-ultra", status=HEALTHY),
    )

    # When
    plan = plan_deployments([provider], state, ENV, policy)

    # Then
    aliases = {d.alias for d in plan}
    assert {"gemma-4-31b", "hermes-smart", "hermes-long", "nemotron-3-ultra-550b-a55b"} <= aliases
    assert all("order" not in d.payload()["litellm_params"] for d in plan)
    assert "demo-free" not in aliases
    row = next(d for d in plan if d.alias == "hermes-smart")
    assert row.payload()["model_info"]["max_input_tokens"] == 262_144


def test_tiered_roles_emit_litellm_order_and_skip_unlisted_models() -> None:
    # Given: tier 1 is preferred, tier 2 is the fallback, a third model is unlisted
    provider = make_provider()
    policy = Policy(roles={"hermes-smart": Role(tiers=(("strong",), ("ok",)), include=("other",))})
    state = make_state(
        make_health(model_id="strong", status=HEALTHY),
        make_health(model_id="ok", status=HEALTHY),
        make_health(model_id="other", status=HEALTHY),
    )

    # When
    plan = plan_deployments([provider], state, ENV, policy)

    # Then
    orders = {
        d.model_id: d.payload()["litellm_params"].get("order")
        for d in plan
        if d.alias == "hermes-smart"
    }
    assert orders == {"strong": 1, "ok": 2}


def test_it_and_free_suffixes_merge_with_the_bare_name() -> None:
    assert canonical_alias("google/gemma-4-31b-it") == canonical_alias("gemma-4-31b")
    assert canonical_alias("z-ai/glm-4.7-flash-free") == "glm-4.7-flash"
    assert canonical_alias("qwen/qwen3.8-27b-free") == canonical_alias("qwen/qwen3.8-27b")


def test_a_too_large_prompt_never_counts_as_a_health_failure() -> None:
    health = make_health(status=HEALTHY, consecutive_failures=0)
    updated, event = record_probe(health, ProbeOutcome.TOO_LARGE, "413", NOW)
    assert updated.status is HEALTHY
    assert updated.consecutive_failures == 0
    assert event is None


def test_a_spent_free_allowance_quarantines_at_once_even_when_it_arrives_as_a_429() -> None:
    from freerouter.probe import outcome_for

    spent = "You've used this period's free allowance. Your next rolling 7-day period starts"
    locked = "Daily check-in required to use free models"
    assert outcome_for(429, spent) is ProbeOutcome.EXHAUSTED
    assert outcome_for(402, locked) is ProbeOutcome.EXHAUSTED
    assert outcome_for(429, "rate limit exceeded") is ProbeOutcome.THROTTLED
    health = make_health(status=HEALTHY)
    updated, event = record_probe(health, ProbeOutcome.EXHAUSTED, spent, NOW)
    assert updated.status is HealthStatus.QUARANTINED
    assert event is not None


def test_two_plain_text_answers_reject_tools_and_the_rejection_expires() -> None:
    once = record_tools(make_health(), called=False, now=NOW)
    assert once.tools is None
    twice = record_tools(once, called=False, now=NOW)
    assert twice.tools is False
    later = forget_expired_rejections(make_state(twice), NOW + timedelta(days=8))
    assert next(iter(later.models.values())).tools is None
    assert record_tools(twice, called=True, now=NOW).tools is True


def test_limits_are_read_from_too_large_errors() -> None:
    groq = (
        "Request too large for model `openai/gpt-oss-20b` on tokens per minute (TPM): "
        "Limit 8000, Requested 21544"
    )
    assert stated_limit(groq) == 8000
    assert stated_limit("This model's maximum context length is 32768 tokens.") == 32768
    assert stated_limit("prompt is too long") is None


def test_traffic_learns_both_a_proven_floor_and_a_ceiling() -> None:
    entries = [
        {
            "model_id": "d1",
            "startTime": "2026-08-25T12:01:00",
            "status": "success",
            "prompt_tokens": 70_000,
        },
        {
            "model_id": "d1",
            "startTime": "2026-08-25T12:02:00",
            "status": "failure",
            "metadata": {
                "error_information": {
                    "error_code": "400",
                    "error_message": "maximum context length is 131072 tokens",
                }
            },
        },
    ]
    signal = context_signals(entries, {"d1": "demo::alpha"}, None)["demo::alpha"]
    assert signal.proven == 70_000
    assert learned_ceiling(signal, None) == 131_072
    learned = learn_context(make_health(), proven=70_000, ceiling=131_072, now=NOW)
    assert (learned.proven_input, learned.max_input) == (70_000, 131_072)


def _settings(tmp_path: Path, fraction: float = 0.25) -> Settings:
    return Settings(
        master_key="sk-test",
        providers_dir=tmp_path / "providers",
        state_dir=tmp_path / "state",
        gateway_url="http://gateway",
        policy_dir=tmp_path / "policy",
        max_delete_fraction=fraction,
    )


def test_a_model_that_never_calls_tools_leaves_the_pool_but_stays_probeable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    gateway = FakeGateway(replies={"fr/demo/alpha": CHATTY})
    result = {"demo": DiscoveryResult(provider_id="demo", models=("alpha",))}
    _wire(monkeypatch, gateway, [provider], result)

    # When: two cycles, each answering in plain text
    _ = run_cycle(_settings(tmp_path), ENV)
    _ = run_cycle(_settings(tmp_path), ENV)

    # Then
    assert gateway.aliases() == {"fr/demo/alpha"}
    health = load_state(tmp_path / "state" / "health.json").models["demo::alpha"]
    assert health.drop_reason == "no tool calling"


def test_a_re_id_is_not_a_mass_delete(tmp_path: Path) -> None:
    # Given: 40 live rows whose ids all change (new fingerprint), same routes
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    state = make_state(make_health(status=HEALTHY))
    gateway = FakeGateway()
    for index in range(3):
        gateway.deployments[f"fr-old{index:013d}"] = None  # type: ignore[assignment]
    strict = CycleContext(now=NOW, settings=_settings(tmp_path, 0.25))

    # When
    added, deleted, errors = apply_plan(gateway, [provider], state, ENV, strict)  # type: ignore[arg-type]

    # Then: net change is 0, so the old ids go
    assert (added, deleted, errors) == (3, 3, [])


def test_the_circuit_breaker_refuses_a_mass_delete(tmp_path: Path) -> None:
    # Given: 40 live managed rows and a plan that wants none of them
    gateway = FakeGateway()
    provider = make_provider()
    for index in range(40):
        gateway.deployments[f"fr-{index:016d}"] = None  # type: ignore[assignment]

    strict = CycleContext(now=NOW, settings=_settings(tmp_path, 0.25))
    loose = CycleContext(now=NOW, settings=_settings(tmp_path, 1.0))

    # When
    _, deleted, errors = apply_plan(gateway, [provider], make_state(), ENV, strict)  # type: ignore[arg-type]

    # Then
    assert deleted == 0
    assert len(gateway.deployments) == 40
    assert any("circuit breaker" in error for error in errors)
    _, deleted, _ = apply_plan(gateway, [provider], make_state(), ENV, loose)  # type: ignore[arg-type]
    assert deleted == 40


def test_policy_files_are_loaded(tmp_path: Path) -> None:
    from freerouter.policy import load_policy

    (tmp_path / "aliases.yaml").write_text("merge:\n  a: b\nexclude: [free]\n")
    (tmp_path / "roles.yaml").write_text("roles:\n  hermes-long: {min_context: 200000}\n")
    (tmp_path / "capabilities.yaml").write_text("overrides:\n  x: {context: 9000, keep: false}\n")
    policy = load_policy(tmp_path)
    assert policy.canonical("a") == "b"
    assert "free" in policy.exclude
    assert policy.roles["hermes-long"].min_context == 200_000
    assert policy.overrides["x"] == Override(context=9000, keep=False)
    assert json.dumps(sorted(policy.roles))  # serialisable names
