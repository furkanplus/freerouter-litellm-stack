"""Validated policy and recoverable reference-cache regression coverage.
All input files, state and gateway replies are synthetic; no provider calls are made.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from freerouter.errors import PolicyError
from freerouter.policy import load_policy, rank_key
from freerouter.refresh import Settings, run_cycle
from freerouter.registry import select_models
from freerouter.state import load_state, save_state
from freerouter.transport import JSON_ADAPTER, Response
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
def test_policy_boundary_validation(tmp_path: Path, case: dict[str, JsonValue]) -> None:
    files = case["files"]
    assert isinstance(files, dict)
    for name, source in files.items():
        assert isinstance(source, str)
        _ = (tmp_path / name).write_text(source, encoding="utf-8")
    error = case.get("error")
    if error is not None:
        with pytest.raises(PolicyError):
            _ = load_policy(tmp_path)
    else:
        assert snapshot(load_policy(tmp_path)) == case["expected"]


def test_shipped_policy_matches_unchanged_baseline() -> None:
    expected = JSON_ADAPTER.validate_json((FIXTURES / "shipped-policy.json").read_bytes())
    assert snapshot(load_policy(Path(__file__).parents[1] / "policy")) == expected


@pytest.mark.parametrize("case", cases("cache-cases.json"), ids=lambda case: case["id"])
def test_fresh_cache_boundary_recovery(
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

    fetches: list[Policy] = []

    def fetch(_policy: Policy) -> None:
        fetches.append(_policy)

    monkeypatch.setattr("freerouter.refresh.utcnow", lambda: NOW)
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: FakeGateway())
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: ())
    monkeypatch.setattr("freerouter.refresh._fetch_reference", fetch)
    monkeypatch.setattr("freerouter.refresh.save_state", capture)
    _ = run_cycle(settings, {})
    assert len(captured) == 1
    actual: object = captured[0].models["demo::alpha"].ref_context
    assert actual == case["expected"]
    assert type(actual) is type(case["expected"])
    assert captured[0].reference_fetched == NOW
    assert len(fetches) == (1 if case.get("refresh") else 0)
    assert settings.reference_path.read_text(encoding="utf-8") == text


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
def test_corrupt_cache_refresh_ignores_age(
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
    _ = run_cycle(settings, {})
    assert captured[0].models["demo::alpha"].ref_context == (
        70000 if replacement is None else 131072
    )
    assert captured[0].reference_fetched == (fetched if replacement is None else NOW)
    assert len(calls) == 1
    assert settings.reference_path.read_text(encoding="utf-8") == (
        "[]" if replacement is None else '{\n "alpha": 131072\n}'
    )


def test_string_cached_context_does_not_block_real_provider_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unlike the shape-isolation cases, include a real routable provider here.
    # Its known ref_context is compared with MIN_CONTEXT during plan creation.
    settings = Settings(master_key="test-only", state_dir=tmp_path, policy_dir=tmp_path / "policy")
    state = make_state(make_health(ref_context=70000)).model_copy(update={"reference_fetched": NOW})
    save_state(state, settings.state_path, NOW)
    _ = settings.reference_path.write_text('{"alpha":"131072"}', encoding="utf-8")
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    monkeypatch.setattr("freerouter.refresh.utcnow", lambda: NOW)
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: FakeGateway())
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: (provider,))
    _ = run_cycle(settings, {"DEMO_API_KEY": "test-only"})
    assert load_state(settings.state_path).models["demo::alpha"].ref_context == 70000


@pytest.mark.parametrize(
    ("filename", "source", "field"),
    [
        ("aliases.yaml", "merge: {alpha: 3}", "merge.alpha"),
        ("aliases.yaml", "exclude: [alpha, 1]", "exclude"),
        ("roles.yaml", "roles: {1: {}}", "roles"),
        ("roles.yaml", "roles: {demo: {tiers: [null]}}", "roles.demo.tiers[0]"),
        ("roles.yaml", "roles: {demo: {include: [true]}}", "roles.demo.include"),
        ("roles.yaml", "roles: {demo: {min_context: .inf}}", "roles.demo.min_context"),
        ("capabilities.yaml", "overrides: {demo: {tools: 1}}", "overrides.demo.tools"),
        ("capabilities.yaml", "overrides: {demo: {keep: 'false'}}", "overrides.demo.keep"),
        ("capabilities.yaml", "overrides: {demo: {context: false}}", "overrides.demo.context"),
        ("capabilities.yaml", "overrides: {demo: []}", "overrides.demo"),
    ],
)
def test_policy_errors_have_file_and_field_and_cycle_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, source: str, field: str
) -> None:
    settings = Settings(master_key="test-only", state_dir=tmp_path, policy_dir=tmp_path)
    save_state(make_state(make_health(ref_context=70000)), settings.state_path, NOW)
    before = settings.state_path.read_bytes()
    _ = (tmp_path / filename).write_text(source, encoding="utf-8")
    gateway = FakeGateway()
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: gateway)
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: ())
    with pytest.raises(PolicyError) as caught:
        _ = run_cycle(settings, {})
    assert caught.value.path == str(tmp_path / filename)
    assert field in caught.value.reason
    assert not gateway.deployments
    assert settings.state_path.read_bytes() == before


def test_null_sections_and_metadata_preserve_precedence(tmp_path: Path) -> None:
    _ = (tmp_path / "aliases.yaml").write_text("merge: null\nexclude: null\ndistinct: [[a,b]]\n")
    _ = (tmp_path / "roles.yaml").write_text(
        "roles: {demo: {include: [alpha], tiers: [[beta]], min_context: null, note: anything}}"
    )
    _ = (tmp_path / "capabilities.yaml").write_text(
        "overrides: {demo: {context: null, tools: null, keep: null, why: metadata}}"
    )
    policy = load_policy(tmp_path)
    assert policy.roles_for("alpha", 100000) == ()
    assert policy.roles_for("beta", 100000) == (("demo", 1),)


@pytest.mark.parametrize("age", [23, 24])
@pytest.mark.parametrize("replacement", [None, {}, {"alpha": True}, {"alpha": 131072}])
def test_valid_cache_daily_refresh_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, age: int, replacement: object
) -> None:
    settings = Settings(master_key="test-only", state_dir=tmp_path, policy_dir=tmp_path / "policy")
    fetched = NOW - timedelta(hours=age)
    save_state(
        make_state(make_health(ref_context=70000)).model_copy(
            update={"reference_fetched": fetched}
        ),
        settings.state_path,
        NOW,
    )
    text = '{"alpha": 90000}'
    _ = settings.reference_path.write_text(text)
    calls: list[Policy] = []

    def fetch(policy: Policy) -> object:
        calls.append(policy)
        return replacement

    monkeypatch.setattr("freerouter.refresh.utcnow", lambda: NOW)
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: FakeGateway())
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: ())
    monkeypatch.setattr("freerouter.refresh._fetch_reference", fetch)
    _ = run_cycle(settings, {})
    state = load_state(settings.state_path)
    refreshed = age == 24 and replacement == {"alpha": 131072}
    assert state.models["demo::alpha"].ref_context == (131072 if refreshed else 90000)
    assert state.reference_fetched == (NOW if refreshed else fetched)
    assert len(calls) == (1 if age == 24 else 0)
    if not refreshed:
        assert settings.reference_path.read_text() == text


@pytest.mark.parametrize("replacement", [None, {}, {"alpha": True}, {"alpha": -1}])
def test_invalid_cache_failed_refresh_retries_without_advancing_freshness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: object
) -> None:
    settings = Settings(master_key="test-only", state_dir=tmp_path, policy_dir=tmp_path / "policy")
    save_state(
        make_state(make_health(ref_context=70000)).model_copy(update={"reference_fetched": NOW}),
        settings.state_path,
        NOW,
    )
    text = '{"alpha": 131072, "beta": null}'
    _ = settings.reference_path.write_text(text)
    calls: list[Policy] = []

    def fetch(policy: Policy) -> object:
        calls.append(policy)
        return replacement

    monkeypatch.setattr("freerouter.refresh.utcnow", lambda: NOW)
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: FakeGateway())
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: ())
    monkeypatch.setattr("freerouter.refresh._fetch_reference", fetch)
    for _ in range(2):
        _ = run_cycle(settings, {})
        state = load_state(settings.state_path)
        assert state.models["demo::alpha"].ref_context == 70000
        assert state.reference_fetched == NOW
        assert settings.reference_path.read_text() == text
    assert len(calls) == 2


def test_cache_atomic_replace_failure_retains_file_health_and_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(master_key="test-only", state_dir=tmp_path, policy_dir=tmp_path / "policy")
    save_state(
        make_state(make_health(ref_context=70000)).model_copy(update={"reference_fetched": NOW}),
        settings.state_path,
        NOW,
    )
    _ = settings.reference_path.write_text("[]")
    before_files = set(tmp_path.iterdir())

    def fail_replace(_self: Path, _target: Path) -> None:
        raise PermissionError

    monkeypatch.setattr("freerouter.refresh.utcnow", lambda: NOW)
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: FakeGateway())
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: ())
    monkeypatch.setattr("freerouter.refresh._fetch_reference", lambda _: {"alpha": 131072})
    # Only fail replacement of the reference cache, not health-state persistence.
    original_replace = Path.replace

    def replace(path: Path, target: Path) -> Path:
        if target == settings.reference_path:
            fail_replace(path, target)
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    _ = run_cycle(settings, {})
    assert settings.reference_path.read_text() == "[]"
    assert load_state(settings.state_path).models["demo::alpha"].ref_context == 70000
    assert load_state(settings.state_path).reference_fetched == NOW
    assert set(tmp_path.iterdir()) == before_files


@pytest.mark.parametrize("bad_length", [True, False, -1, 0, "131072", 64000.9, None])
def test_downloaded_reference_filters_invalid_contexts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_length: JsonValue
) -> None:
    # Undo the global offline fetch stub, then stub the transport itself instead.
    monkeypatch.undo()
    settings = Settings(master_key="test-only", state_dir=tmp_path, policy_dir=tmp_path / "policy")
    payload = {
        "data": [
            {"id": "alpha", "context_length": bad_length, "supported_parameters": ["tools"]},
            {"id": "beta", "context_length": 131072, "supported_parameters": ["tools"]},
        ]
    }
    monkeypatch.setattr(
        "freerouter.refresh.request", lambda _: Response(200, json.dumps(payload).encode())
    )
    monkeypatch.setattr("freerouter.refresh.utcnow", lambda: NOW)
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: FakeGateway())
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: ())
    _ = run_cycle(settings, {})
    assert JSON_ADAPTER.validate_json(settings.reference_path.read_bytes()) == {"beta": 131072}
    assert load_state(settings.state_path).reference_fetched == NOW
