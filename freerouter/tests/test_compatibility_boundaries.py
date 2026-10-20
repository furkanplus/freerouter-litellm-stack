"""Current-behavior observations, not promises to support malformed legacy input.

Update these observations deliberately when the proposed strict boundary is approved.
All input files, state and gateway replies are synthetic; no provider calls are made.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from freerouter.errors import PolicyError
from freerouter.policy import load_policy, rank_key
from freerouter.refresh import Settings, run_cycle
from freerouter.registry import select_models
from freerouter.state import save_state
from freerouter.transport import JSON_ADAPTER
from tests.conftest import NOW, make_health, make_provider, make_state
from tests.test_refresh import FakeGateway

if TYPE_CHECKING:
    from datetime import datetime

    from freerouter.policy import Policy
    from freerouter.state import HealthState
    from freerouter.transport import JsonValue

FIXTURES = Path(__file__).parent / "fixtures" / "compatibility"


def cases(name: str) -> list[dict[str, JsonValue]]:
    loaded = JSON_ADAPTER.validate_json((FIXTURES / name).read_bytes())
    assert isinstance(loaded, list)
    result: list[dict[str, JsonValue]] = []
    for item in loaded:
        assert isinstance(item, dict)
        result.append(item)
    return result


def snapshot(policy: Policy) -> dict[str, object]:
    return {
        "merge": dict(policy.merge),
        "exclude": sorted(policy.exclude),
        "roles": {
            name: {
                "include": list(role.include),
                "min_context": role.min_context,
                "tiers": [list(tier) for tier in role.tiers],
            }
            for name, role in policy.roles.items()
        },
        "overrides": {
            name: {"context": item.context, "tools": item.tools, "keep": item.keep}
            for name, item in policy.overrides.items()
        },
    }


@pytest.mark.parametrize("case", cases("policy-cases.json"), ids=lambda case: case["id"])
def test_policy_boundary_current_behavior(tmp_path: Path, case: dict[str, JsonValue]) -> None:
    files = case["files"]
    assert isinstance(files, dict)
    for name, source in files.items():
        assert isinstance(source, str)
        _ = (tmp_path / name).write_text(source, encoding="utf-8")
    error = case.get("error")
    if error is not None:
        error_type = OverflowError if error == "OverflowError" else PolicyError
        with pytest.raises(error_type):
            _ = load_policy(tmp_path)
    else:
        assert snapshot(load_policy(tmp_path)) == case["expected"]


def test_shipped_policy_matches_unchanged_baseline() -> None:
    expected = JSON_ADAPTER.validate_json((FIXTURES / "shipped-policy.json").read_bytes())
    assert snapshot(load_policy(Path(__file__).parents[1] / "policy")) == expected


@pytest.mark.parametrize("case", cases("cache-cases.json"), ids=lambda case: case["id"])
def test_fresh_cache_boundary_current_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: dict[str, JsonValue]
) -> None:
    # Isolate the reference read via the public cycle: no routable provider is present.
    # Capture the would-be persisted state before unchecked cached values are written.
    settings = Settings(master_key="test-only", state_dir=tmp_path, policy_dir=tmp_path / "policy")
    state = make_state(make_health(ref_context=70000)).model_copy(update={"reference_fetched": NOW})
    save_state(state, settings.state_path, NOW)
    text = case["text"]
    assert isinstance(text, str)
    _ = settings.reference_path.write_text(text, encoding="utf-8")
    captured: list[HealthState] = []

    def capture(value: HealthState, _path: Path, _now: datetime) -> None:
        captured.append(value)

    def no_fetch(_policy: Policy) -> None:
        pytest.fail("a fresh existing cache must not be fetched in this characterization")

    monkeypatch.setattr("freerouter.refresh.utcnow", lambda: NOW)
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: FakeGateway())
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: ())
    monkeypatch.setattr("freerouter.refresh._fetch_reference", no_fetch)
    monkeypatch.setattr("freerouter.refresh.save_state", capture)
    if case.get("error") is not None:
        with pytest.raises(AttributeError):
            _ = run_cycle(settings, {})
        assert not captured
    else:
        _ = run_cycle(settings, {})
        assert len(captured) == 1
        # Inspect the runtime field; serialization can normalize bool to int.
        actual: object = captured[0].models["demo::alpha"].ref_context
        assert actual == case["expected"]
        assert type(actual) is type(case["expected"])
        assert captured[0].reference_fetched == NOW


@pytest.mark.parametrize("rank", ["default", "text", "length", "policy"])
def test_select_models_accepts_existing_homogeneous_sort_keys(rank: str) -> None:
    # The callable parameter is typed independently in each public invocation below.
    ids = ("model-20b", "model-120b", "model-70b")
    if rank == "default":
        result = select_models(ids, allow=(), deny=(), limit=2)
    elif rank == "text":
        result = select_models(ids, allow=(), deny=(), limit=2, rank=str.lower)
    elif rank == "length":
        result = select_models(ids, allow=(), deny=(), limit=2, rank=len)
    else:
        result = select_models(ids, allow=(), deny=(), limit=2, rank=rank_key)
    expected = {
        "default": (("model-120b", "model-20b"), ("model-70b",)),
        "text": (("model-120b", "model-20b"), ("model-70b",)),
        "policy": (("model-120b", "model-70b"), ("model-20b",)),
    }
    if rank == "length":
        # Equal-length keys inherit set iteration order; no ordering contract exists.
        assert set(result[0]) == {"model-20b", "model-70b"}
        assert result[1] == ("model-120b",)
    else:
        assert result == expected[rank]


@pytest.mark.parametrize("age_hours", [23, 24])
@pytest.mark.parametrize("replacement", [None, {"alpha": 131072}])
def test_corrupt_cache_refresh_follows_age_not_validity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    age_hours: int,
    replacement: dict[str, int] | None,
) -> None:
    settings = Settings(master_key="test-only", state_dir=tmp_path, policy_dir=tmp_path / "policy")
    fetched = NOW - timedelta(hours=age_hours)
    state = make_state(make_health(ref_context=70000)).model_copy(
        update={"reference_fetched": fetched}
    )
    save_state(state, settings.state_path, NOW)
    _ = settings.reference_path.write_text("[]", encoding="utf-8")
    calls: list[Policy] = []
    captured: list[HealthState] = []

    def fetch(policy: Policy) -> dict[str, int] | None:
        calls.append(policy)
        return replacement

    def capture(value: HealthState, _path: Path, _now: datetime) -> None:
        captured.append(value)

    monkeypatch.setattr("freerouter.refresh.utcnow", lambda: NOW)
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: FakeGateway())
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: ())
    monkeypatch.setattr("freerouter.refresh._fetch_reference", fetch)
    monkeypatch.setattr("freerouter.refresh.save_state", capture)
    if age_hours < 24 or replacement is None:
        with pytest.raises(AttributeError):
            _ = run_cycle(settings, {})
        assert not captured
    else:
        _ = run_cycle(settings, {})
        assert captured[0].models["demo::alpha"].ref_context == 131072
        assert captured[0].reference_fetched == NOW
    assert len(calls) == (0 if age_hours < 24 else 1)


def test_string_cached_context_blocks_real_provider_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unlike the shape-isolation cases, include a real routable provider here.
    # Its known ref_context is compared with MIN_CONTEXT during plan creation.
    settings = Settings(master_key="test-only", state_dir=tmp_path, policy_dir=tmp_path / "policy")
    state = make_state(make_health(ref_context=70000)).model_copy(update={"reference_fetched": NOW})
    save_state(state, settings.state_path, NOW)
    before = settings.state_path.read_bytes()
    _ = settings.reference_path.write_text('{"alpha":"131072"}', encoding="utf-8")
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    monkeypatch.setattr("freerouter.refresh.utcnow", lambda: NOW)
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: FakeGateway())
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: (provider,))
    with pytest.raises(TypeError):
        _ = run_cycle(settings, {"DEMO_API_KEY": "test-only"})
    assert settings.state_path.read_bytes() == before
