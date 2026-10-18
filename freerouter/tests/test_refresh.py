from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from freerouter.discovery import DiscoveryResult
from freerouter.planning import POOL_ALIAS
from freerouter.refresh import Settings, run_cycle
from freerouter.state import EventKind, HealthStatus, load_state
from freerouter.transport import JsonValue, Response
from tests.conftest import make_provider

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pytest

    from freerouter.planning import Deployment
    from freerouter.registry import Provider

TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
OK = Response(
    status=200,
    body=json.dumps(
        {"choices": [{"message": {"content": None, "tool_calls": [TOOL_CALL]}}]}
    ).encode(),
)
CHATTY = Response(
    status=200, body=json.dumps({"choices": [{"message": {"content": "sunny"}}]}).encode()
)
GONE = Response(status=404, body=json.dumps({"error": {"message": "model not found"}}).encode())


class FakeGateway:
    def __init__(self, replies: dict[str, Response] | None = None) -> None:
        self.deployments: dict[str, Deployment] = {}
        self.replies: dict[str, Response] = replies or {}
        self.chats: list[str] = []
        self.logs: tuple[JsonValue, ...] = ()

    def managed_ids(self) -> tuple[str, ...]:
        return tuple(self.deployments)

    def spend_logs(self, _since: datetime | None = None) -> tuple[JsonValue, ...]:
        return self.logs

    def add(self, deployment: Deployment) -> None:
        self.deployments[deployment.deployment_id] = deployment

    def delete(self, identifier: str) -> None:
        del self.deployments[identifier]

    def chat(self, alias: str, _prompt: str, _max_tokens: int) -> Response:
        self.chats.append(alias)
        return self.replies.get(alias, OK)

    def chat_tools(self, alias: str) -> Response:
        return self.chat(alias, "", 0)

    def aliases(self) -> set[str]:
        return {deployment.alias for deployment in self.deployments.values()}


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    gateway: FakeGateway,
    providers: Sequence[Provider],
    results: dict[str, DiscoveryResult],
) -> None:
    monkeypatch.setattr("freerouter.refresh.Gateway", lambda **_: gateway)
    monkeypatch.setattr("freerouter.refresh.load_providers", lambda _: tuple(providers))
    monkeypatch.setattr(
        "freerouter.refresh.discover",
        lambda provider, _env: results[provider.provider_id],
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        master_key="sk-test",
        providers_dir=tmp_path / "providers",
        state_dir=tmp_path / "state",
        gateway_url="http://gateway",
    )


def test_a_new_model_is_probed_before_it_joins_the_shared_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    gateway = FakeGateway()
    _wire(
        monkeypatch,
        gateway,
        [provider],
        {"demo": DiscoveryResult(provider_id="demo", models=("alpha",))},
    )

    # When
    report = run_cycle(_settings(tmp_path), {"DEMO_API_KEY": "k"})

    # Then
    assert gateway.chats == ["fr/demo/alpha"]
    assert gateway.aliases() == {"fr/demo/alpha", POOL_ALIAS, "alpha"}
    assert report.healthy == 1
    assert {event.kind for event in report.events} == {EventKind.ADDED}


def test_a_dead_model_is_pulled_out_of_the_pool_but_stays_addressable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    gateway = FakeGateway({"fr/demo/alpha": GONE})
    _wire(
        monkeypatch,
        gateway,
        [provider],
        {"demo": DiscoveryResult(provider_id="demo", models=("alpha",))},
    )

    # When
    report = run_cycle(_settings(tmp_path), {"DEMO_API_KEY": "k"})

    # Then
    assert gateway.aliases() == {"fr/demo/alpha"}
    assert report.quarantined == 1
    assert EventKind.QUARANTINED in {event.kind for event in report.events}


def test_removing_a_key_takes_that_platform_out_of_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    gateway = FakeGateway()
    settings = _settings(tmp_path)
    _wire(
        monkeypatch,
        gateway,
        [provider],
        {"demo": DiscoveryResult(provider_id="demo", models=("alpha",))},
    )
    _ = run_cycle(settings, {"DEMO_API_KEY": "k"})
    assert gateway.aliases()

    # When the credential goes away
    _wire(
        monkeypatch,
        gateway,
        [provider],
        {"demo": DiscoveryResult(provider_id="demo", skipped="missing DEMO_API_KEY")},
    )
    report = run_cycle(settings, {})

    # Then
    assert gateway.aliases() == set()
    assert report.discovered == 0
    assert EventKind.REMOVED in {event.kind for event in report.events}


def test_a_catalog_outage_never_empties_an_already_working_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    gateway = FakeGateway()
    settings = _settings(tmp_path)
    _wire(
        monkeypatch,
        gateway,
        [provider],
        {"demo": DiscoveryResult(provider_id="demo", models=("alpha",))},
    )
    _ = run_cycle(settings, {"DEMO_API_KEY": "k"})

    # When
    _wire(
        monkeypatch,
        gateway,
        [provider],
        {"demo": DiscoveryResult(provider_id="demo", error="catalog HTTP 503")},
    )
    report = run_cycle(settings, {"DEMO_API_KEY": "k"})

    # Then
    assert POOL_ALIAS in gateway.aliases()
    assert report.healthy == 1
    assert EventKind.PROVIDER_ERROR in {event.kind for event in report.events}


def test_a_repeated_provider_error_is_reported_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    gateway = FakeGateway()
    settings = _settings(tmp_path)
    results = {"demo": DiscoveryResult(provider_id="demo", error="catalog HTTP 503")}
    _wire(monkeypatch, gateway, [provider], results)

    # When
    first = run_cycle(settings, {"DEMO_API_KEY": "k"})
    second = run_cycle(settings, {"DEMO_API_KEY": "k"})

    # Then
    assert EventKind.PROVIDER_ERROR in {event.kind for event in first.events}
    assert second.events == ()


def test_an_expiring_offer_is_announced_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given
    ends_on = (datetime.now(UTC) + timedelta(days=2)).date().isoformat()
    provider = make_provider(
        tier="offer",
        offer={"ends_on": ends_on, "note": "限时不限量"},
        discovery={"mode": "static", "models": ["alpha"]},
    )
    gateway = FakeGateway()
    settings = _settings(tmp_path)
    _wire(
        monkeypatch,
        gateway,
        [provider],
        {"demo": DiscoveryResult(provider_id="demo", models=("alpha",))},
    )

    # When
    first = run_cycle(settings, {"DEMO_API_KEY": "k"})
    second = run_cycle(settings, {"DEMO_API_KEY": "k"})

    # Then
    assert EventKind.OFFER_EXPIRING in {event.kind for event in first.events}
    assert EventKind.OFFER_EXPIRING not in {event.kind for event in second.events}


def test_state_and_changelog_land_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    gateway = FakeGateway()
    settings = _settings(tmp_path)
    _wire(
        monkeypatch,
        gateway,
        [provider],
        {"demo": DiscoveryResult(provider_id="demo", models=("alpha",))},
    )

    # When
    _ = run_cycle(settings, {"DEMO_API_KEY": "k"})

    # Then
    state = load_state(settings.state_path)
    assert state.models["demo::alpha"].status is HealthStatus.HEALTHY
    assert settings.changelog_path.read_text(encoding="utf-8").count("\n") == 1


def test_a_failed_request_in_the_call_log_is_counted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a pooled model, then one failed request logged *after* the next cycle
    # starts (the log is read later in the cycle than its start time).
    provider = make_provider(discovery={"mode": "static", "models": ["alpha"]})
    gateway = FakeGateway()
    settings = _settings(tmp_path)
    _wire(
        monkeypatch,
        gateway,
        [provider],
        {"demo": DiscoveryResult(provider_id="demo", models=("alpha",))},
    )
    _ = run_cycle(settings, {"DEMO_API_KEY": "k"})
    deployment_id = next(iter(gateway.deployments))
    logged_at = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    gateway.logs = (
        {
            "model_id": deployment_id,
            "startTime": logged_at,
            "status": "failure",
            "metadata": {"error_information": {"error_code": "500", "error_message": "boom"}},
        },
    )

    # When
    _ = run_cycle(settings, {"DEMO_API_KEY": "k"})
    _ = run_cycle(settings, {"DEMO_API_KEY": "k"})

    # Then
    health = load_state(settings.state_dir / "health.json").models["demo::alpha"]
    assert health.consecutive_failures == 1
