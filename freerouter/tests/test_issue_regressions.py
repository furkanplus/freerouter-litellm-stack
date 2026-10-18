from __future__ import annotations

import importlib.util
import io
import json
import re
import shutil
import subprocess
from datetime import timedelta
from email.message import Message
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from freerouter.cli import cmd_daemon
from freerouter.errors import PolicyError
from freerouter.planning import plan_deployments
from freerouter.policy import hard_rejection, load_policy
from freerouter.refresh import CycleContext, CycleReport, Settings, observe_traffic
from freerouter.state import HealthStatus, ProbeOutcome
from freerouter.traffic import context_signals, latest_observations
from tests.conftest import NOW, make_health, make_provider, make_state
from tests.test_refresh import FakeGateway

if TYPE_CHECKING:
    from urllib.request import Request

    from freerouter.gateway import Gateway
    from freerouter.policy import Policy
    from freerouter.transport import JsonValue

ROOT = Path(__file__).resolve().parents[2]


def test_root_gitignore_protects_secrets_but_not_templates() -> None:
    secrets = [
        ".env",
        ".env.production",
        "nested/.env.local",
        "private.pem",
        "nested/server.key",
        "id_ed25519",
        "credentials.json",
        "data/postgres/PG_VERSION",
        "logs/gateway.log",
        "freerouter/state/health.json",
    ]
    git = shutil.which("git")
    assert git is not None
    result = subprocess.run(  # noqa: S603 -- fixed git argv and test fixture inputs
        [git, "check-ignore", "--no-index", "--stdin"],
        input="\n".join(secrets) + "\n",
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.splitlines() == secrets
    templates = [".env.example", "freerouter/.env.example", "config.yaml"]
    result = subprocess.run(  # noqa: S603 -- fixed git argv and test fixture inputs
        [git, "check-ignore", "--no-index", "--stdin"],
        input="\n".join(templates) + "\n",
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert not result.stdout


@pytest.mark.parametrize(
    "model", ["microsoft/phi-4", "microsoft/MAI-DS-R1", "microsoft/WizardLM-2-8x22B"]
)
def test_microsoft_namespace_is_not_a_micro_model(model: str) -> None:
    assert hard_rejection(model) is None


@pytest.mark.parametrize("model", ["@cf/ibm/granite-4.0-h-micro", "granite_micro:latest", "micro"])
def test_actual_micro_models_stay_excluded(model: str) -> None:
    assert hard_rejection(model) is not None


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("aliases.yaml", "merge: [unterminated"),
        ("roles.yaml", "roles:\n  long: {min_context: typo}\n"),
        ("capabilities.yaml", "overrides: [not-a-mapping]\n"),
        ("aliases.yaml", "not-a-mapping\n"),
    ],
)
def test_malformed_policy_is_a_recoverable_error(
    tmp_path: Path, filename: str, content: str
) -> None:
    _ = (tmp_path / filename).write_text(content)
    with pytest.raises(PolicyError):
        _ = load_policy(tmp_path)


def test_daemon_keeps_running_and_retries_after_bad_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(master_key="test", policy_dir=tmp_path, interval=300)
    path = tmp_path / "roles.yaml"
    _ = path.write_text("roles: [")
    policies: list[Policy] = []
    attempts: list[str] = []
    reports: list[CycleReport] = []

    def cycle(_settings: Settings) -> CycleReport:
        attempts.append("attempt")
        policies.append(load_policy(_settings.policy_dir))
        return CycleReport()

    def sleep(_seconds: float) -> None:
        assert _seconds == 300
        if len(attempts) == 1:
            assert not policies
            assert not reports
            _ = path.write_text("roles:\n  long: {min_context: 64000}\n")
            return
        assert len(attempts) == 2
        assert policies[0].roles["long"].min_context == 64000
        assert len(reports) == 1
        raise StopIteration

    monkeypatch.setattr("freerouter.cli.Settings.from_env", lambda: settings)
    monkeypatch.setattr(
        "freerouter.cli.Gateway", lambda **_: SimpleNamespace(wait_ready=lambda _: True)
    )
    monkeypatch.setattr("freerouter.cli.run_cycle", cycle)
    monkeypatch.setattr("freerouter.cli._log_report", reports.append)
    monkeypatch.setattr("freerouter.cli.time.sleep", sleep)
    with pytest.raises(StopIteration):
        _ = cmd_daemon([])


def test_traffic_after_snapshot_does_not_count_twice(tmp_path: Path) -> None:
    provider = make_provider()
    health = make_health(status=HealthStatus.HEALTHY)
    state = make_state(health)
    settings = Settings(master_key="test", state_dir=tmp_path, threshold=2)
    gateway = FakeGateway()
    deployment = plan_deployments([provider], state, {"DEMO_API_KEY": "test"})[0]
    at = NOW + timedelta(seconds=1)
    gateway.logs = (
        {
            "model_id": deployment.deployment_id,
            "startTime": at.isoformat(),
            "status": "failure",
            "metadata": {"error_information": {"error_code": "500", "error_message": "boom"}},
        },
    )
    # Logs fetched over HTTP can contain a request newer than cycle start.
    for now in [NOW, NOW + timedelta(minutes=5), NOW + timedelta(minutes=10)]:
        state, _, _ = observe_traffic(
            cast("Gateway", cast("object", gateway)),
            [provider],
            state,
            {"DEMO_API_KEY": "test"},
            CycleContext(now=now, settings=settings),
        )
    assert state.models[health.key].consecutive_failures == 1
    assert state.models[health.key].last_outcome is ProbeOutcome.TRANSIENT
    assert state.models[health.key].status is HealthStatus.HEALTHY
    assert plan_deployments([provider], state, {"DEMO_API_KEY": "test"})


def test_consumed_watermark_prevents_repeated_context_learning() -> None:
    entry: JsonValue = {
        "model_id": "d1",
        "startTime": (NOW + timedelta(seconds=1)).isoformat(),
        "status": "success",
        "prompt_tokens": 90000,
    }
    consumed = NOW + timedelta(seconds=1)
    assert latest_observations([entry], {"d1": "demo::alpha"}, None)
    assert context_signals([entry], {"d1": "demo::alpha"}, None)
    assert not latest_observations([entry], {"d1": "demo::alpha"}, consumed)
    assert not context_signals([entry], {"d1": "demo::alpha"}, consumed)


def _site() -> ModuleType:
    spec = importlib.util.spec_from_file_location("standalone_site", ROOT / "generate_site.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        ("0", True),
        (0, True),
        ("0.000", True),
        (None, False),
        ("NaN", False),
        ("garbage", False),
        (False, False),
        ("0.1", False),
        ("Infinity", False),
    ],
)
def test_standalone_site_requires_explicit_zero_price(price: object, *, expected: bool) -> None:
    assert _site()._is_free({"prompt": price}, "prompt") is expected  # noqa: SLF001


def test_site_generates_real_outputs_without_private_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site = _site()
    marker = "synthetic-private-account-id"
    error = f"https://api.cloudflare.com/client/v4/accounts/{marker}/ai?token=synthetic-token"

    class FixtureHTTPResponse(io.BytesIO):
        def __init__(self, payload: dict[str, object]) -> None:
            super().__init__(json.dumps(payload).encode())
            self.headers: Message = Message()
            self.headers["Content-Type"] = "application/json"

    def open_fixture(request: Request, timeout: int) -> FixtureHTTPResponse:
        assert timeout == 20
        if "api.cloudflare.com" in request.full_url:
            assert marker in request.full_url
            raise RuntimeError(error)
        assert request.full_url == "https://openrouter.ai/api/v1/models"
        return FixtureHTTPResponse(
            {
                "data": [
                    {
                        "id": "demo-70b",
                        "name": "Demo",
                        "context_length": 64000,
                        "pricing": {"prompt": "0", "completion": "0"},
                    },
                    {"id": "paid-70b", "pricing": {"prompt": "0.1", "completion": "0"}},
                    {"id": "unknown-70b"},
                ]
            }
        )

    provider = dict(
        site.PROVIDERS[0],
        key="cloudflare",
        label="Cloudflare",
        env="TEST_SITE_KEY",
        fetch=site.fetch_cloudflare,
    )
    healthy = dict(
        site.PROVIDERS[0],
        key="demo",
        label="Demo",
        env="TEST_SITE_KEY",
        fetch=site.fetch_openrouter,
    )
    monkeypatch.setenv("TEST_SITE_KEY", "fixture-only")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", marker)
    monkeypatch.setattr(site, "PROVIDERS", [provider, healthy])
    monkeypatch.setattr(site._opener, "open", open_fixture)  # noqa: SLF001
    monkeypatch.setattr(site, "OUT_DIR", tmp_path)
    monkeypatch.setattr(site, "fetch_litellm_db", dict)
    site.main()
    payload = json.loads((tmp_path / "models.json").read_text())
    assert payload["providers"]["cloudflare"]["error"] == "Provider temporarily unavailable."
    models = payload["providers"]["demo"]["models"]
    assert [model["id"] for model in models] == ["demo-70b"]
    html = (tmp_path / "index.html").read_text()
    assert "Provider temporarily unavailable." in html
    assert "demo-70b" in html
    outputs = list(tmp_path.rglob("*"))
    assert len([p for p in outputs if p.suffix == ".html"]) > 1
    for path in outputs:
        if path.is_file():
            content = path.read_text()
            assert marker not in content
            assert "synthetic-token" not in content


def test_daemon_retries_unexpected_cycle_errors_but_does_not_swallow_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(master_key="test", interval=300)
    outcomes: list[BaseException | CycleReport] = [
        RuntimeError("unexpected provider error"),
        CycleReport(),
        KeyboardInterrupt(),
    ]
    sleeps: list[float] = []
    reports: list[CycleReport] = []

    def cycle(_settings: Settings) -> CycleReport:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr("freerouter.cli.Settings.from_env", lambda: settings)
    monkeypatch.setattr(
        "freerouter.cli.Gateway", lambda **_: SimpleNamespace(wait_ready=lambda _: True)
    )
    monkeypatch.setattr("freerouter.cli.run_cycle", cycle)
    monkeypatch.setattr("freerouter.cli._log_report", reports.append)
    monkeypatch.setattr("freerouter.cli.time.sleep", sleeps.append)
    with pytest.raises(KeyboardInterrupt):
        _ = cmd_daemon([])
    assert sleeps == [300, 300]
    assert len(reports) == 1
    assert not outcomes


def test_root_env_example_covers_compose_interpolation() -> None:
    variables = set(
        re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", (ROOT / "docker-compose.yml").read_text())
    )
    declared = {
        line.split("=", 1)[0]
        for line in (ROOT / ".env.example").read_text().splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    assert variables <= declared
