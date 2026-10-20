from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Final, Protocol, cast

from pydantic import ValidationError

from freerouter.discovery import discover
from freerouter.errors import GatewayError, MissingMasterKeyError
from freerouter.gateway import Gateway
from freerouter.notify import send
from freerouter.planning import canonical_alias, diff, is_managed, plan_deployments
from freerouter.policy import DEFAULT_POLICY_DIR, Policy, load_policy
from freerouter.probe import probe_one, probe_targets
from freerouter.registry import load_providers
from freerouter.state import (
    Event,
    EventKind,
    HealthState,
    HealthStatus,
    ProbeOutcome,
    append_events,
    forget_expired_rejections,
    learn_context,
    load_state,
    observe,
    record_probe,
    record_tools,
    save_state,
    state_key,
    utcnow,
)
from freerouter.traffic import (
    context_signals,
    latest_observations,
    learned_ceiling,
    newest_entry_time,
)
from freerouter.transport import JSON_ADAPTER, as_items, as_strings, as_text, dig, request

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date, datetime

    from freerouter.discovery import DiscoveryResult
    from freerouter.planning import Deployment
    from freerouter.registry import Provider

LOGGER: Final = logging.getLogger("freerouter")

DEFAULT_INTERVAL: Final = 21600.0
DEFAULT_RECHECK_HOURS: Final = 24.0
DEFAULT_THRESHOLD: Final = 3
DEFAULT_OFFER_WARN_DAYS: Final = 14
DEFAULT_READY_TIMEOUT: Final = 300.0
DEFAULT_WORKERS: Final = 4
MIN_INTERVAL: Final = 300.0
DEFAULT_MAX_DELETE_FRACTION: Final = 0.25
MASS_DELETE_FLOOR: Final = 20
REFERENCE_URL: Final = "https://openrouter.ai/api/v1/models"
REFERENCE_MAX_AGE: Final = timedelta(hours=24)


def _number(env: Mapping[str, str], name: str, fallback: float) -> float:
    """Read a numeric environment variable, falling back when unset or malformed."""
    try:
        return float(env.get(name, "").strip() or fallback)
    except ValueError:
        LOGGER.warning("%s is not a number; using %s", name, fallback)
        return fallback


@dataclass(frozen=True, slots=True)
class Settings:
    master_key: str
    providers_dir: Path = Path("/app/providers")
    state_dir: Path = Path("/app/state")
    gateway_url: str = "http://litellm:4000"
    interval: float = DEFAULT_INTERVAL
    recheck_after: timedelta = timedelta(hours=DEFAULT_RECHECK_HOURS)
    threshold: int = DEFAULT_THRESHOLD
    webhook: str | None = None
    offer_warn_days: int = DEFAULT_OFFER_WARN_DAYS
    ready_timeout: float = DEFAULT_READY_TIMEOUT
    workers: int = DEFAULT_WORKERS
    policy_dir: Path = DEFAULT_POLICY_DIR
    max_delete_fraction: float = DEFAULT_MAX_DELETE_FRACTION

    @property
    def state_path(self) -> Path:
        """Return the path of the persisted health state."""
        return self.state_dir / "health.json"

    @property
    def reference_path(self) -> Path:
        """Return the cached reference catalog (context windows by canonical alias)."""
        return self.state_dir / "reference_catalog.json"

    @property
    def changelog_path(self) -> Path:
        """Return the path of the append-only change log."""
        return self.state_dir / "changelog.jsonl"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from the process environment."""
        source = os.environ if env is None else env
        master_key = source.get("LITELLM_MASTER_KEY", "").strip()
        if not master_key:
            raise MissingMasterKeyError
        return cls(
            master_key=master_key,
            providers_dir=Path(source.get("FREEROUTER_PROVIDERS_DIR", "/app/providers")),
            state_dir=Path(source.get("FREEROUTER_STATE_DIR", "/app/state")),
            gateway_url=source.get("FREEROUTER_GATEWAY_URL", "http://litellm:4000").rstrip("/"),
            interval=max(
                MIN_INTERVAL, _number(source, "FREEROUTER_REFRESH_INTERVAL", DEFAULT_INTERVAL)
            ),
            recheck_after=timedelta(
                hours=_number(source, "FREEROUTER_RECHECK_HOURS", DEFAULT_RECHECK_HOURS)
            ),
            threshold=int(_number(source, "FREEROUTER_FAILURE_THRESHOLD", DEFAULT_THRESHOLD)),
            webhook=source.get("FREEROUTER_NOTIFY_WEBHOOK", "").strip() or None,
            offer_warn_days=int(
                _number(source, "FREEROUTER_OFFER_WARN_DAYS", DEFAULT_OFFER_WARN_DAYS)
            ),
            ready_timeout=_number(source, "FREEROUTER_READY_TIMEOUT", DEFAULT_READY_TIMEOUT),
            workers=max(1, int(_number(source, "FREEROUTER_PROBE_WORKERS", DEFAULT_WORKERS))),
            policy_dir=Path(source.get("FREEROUTER_POLICY_DIR", str(DEFAULT_POLICY_DIR))),
            max_delete_fraction=_number(
                source, "FREEROUTER_MAX_DELETE_FRACTION", DEFAULT_MAX_DELETE_FRACTION
            ),
        )


@dataclass(frozen=True, slots=True)
class CycleContext:
    """Everything one cycle's stages share besides the gateway and the state."""

    now: datetime
    settings: Settings
    verified: frozenset[str] = frozenset()
    policy: Policy = field(default_factory=Policy)


@dataclass(frozen=True, slots=True)
class CycleReport:
    discovered: int = 0
    healthy: int = 0
    quarantined: int = 0
    probed: int = 0
    added: int = 0
    deleted: int = 0
    verified_by_traffic: int = 0
    events: tuple[Event, ...] = ()
    errors: tuple[str, ...] = field(default=())

    def summary(self) -> str:
        """Render a one-line operator summary of the cycle."""
        return (
            f"discovered={self.discovered} healthy={self.healthy} "
            f"quarantined={self.quarantined} traffic-verified={self.verified_by_traffic} "
            f"probed={self.probed} "
            f"added={self.added} deleted={self.deleted} "
            f"events={len(self.events)} errors={len(self.errors)}"
        )


NOTIFIABLE: Final = frozenset(
    {
        EventKind.ADDED,
        EventKind.REMOVED,
        EventKind.QUARANTINED,
        EventKind.REVIVED,
        EventKind.PROVIDER_ERROR,
        EventKind.OFFER_EXPIRING,
        EventKind.OFFER_EXPIRED,
    }
)


def _discover_all(
    providers: Sequence[Provider],
    env: Mapping[str, str],
    workers: int,
) -> tuple[tuple[Provider, DiscoveryResult], ...]:
    """Query every provider's catalog concurrently."""

    def run(provider: Provider) -> tuple[Provider, DiscoveryResult]:
        return provider, discover(provider, env)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return tuple(pool.map(run, providers))


def _ingest(
    state: HealthState,
    pairs: Sequence[tuple[Provider, DiscoveryResult]],
    now: datetime,
) -> tuple[HealthState, list[Event]]:
    """Fold discovery results into the health state, deduplicating provider errors."""
    events: list[Event] = []
    errors = dict(state.provider_errors)
    gate_drops = dict(state.gate_drops)
    for provider, result in pairs:
        name = provider.provider_id
        if result.skipped is not None:
            # A skipped provider is one that is not configured at all — no credential,
            # paused, or retired. Its models must leave the pool. A provider whose
            # catalog merely failed is handled below and keeps everything it had.
            LOGGER.info("skip %s: %s", name, result.skipped)
            state, dropped = observe(state, name, (), now)
            events.extend(dropped)
            continue
        if result.error is not None:
            if errors.get(name) != result.error:
                events.append(
                    Event(kind=EventKind.PROVIDER_ERROR, provider=name, detail=result.error)
                )
            errors[name] = result.error
            LOGGER.warning("%s discovery failed: %s", name, result.error)
            continue
        _ = errors.pop(name, None)
        if result.dropped_by_cap:
            LOGGER.warning(
                "%s: max_models=%d reached, %d models not routed (%s)",
                name,
                provider.max_models,
                len(result.dropped_by_cap),
                ", ".join(result.dropped_by_cap[:5]),
            )
        if result.dropped_by_gate:
            gate_drops[name] = list(result.dropped_by_gate)
        else:
            _ = gate_drops.pop(name, None)
        state, seen = observe(state, name, result.models, now)
        state = _store_context(state, name, dict(result.context))
        events.extend(seen)
    return state.model_copy(update={"provider_errors": errors, "gate_drops": gate_drops}), events


def _store_context(state: HealthState, provider_id: str, context: Mapping[str, int]) -> HealthState:
    """Record the context window each model's catalog entry advertised this cycle."""
    models = dict(state.models)
    for key, health in state.models.items():
        if health.provider == provider_id and health.context != context.get(health.model_id):
            models[key] = health.model_copy(update={"context": context.get(health.model_id)})
    return state.model_copy(update={"models": models})


class PlanGateway(Protocol):
    """Only the gateway operations required to reconcile a deployment plan."""

    def managed_ids(self) -> tuple[str, ...]:
        """List current deployment identifiers."""
        ...

    def add(self, deployment: Deployment) -> None:
        """Register one planned deployment."""
        ...

    def delete(self, identifier: str) -> None:
        """Remove one deployment by identifier."""
        ...


def apply_plan(
    gateway: PlanGateway,
    providers: Sequence[Provider],
    state: HealthState,
    env: Mapping[str, str],
    context: CycleContext | None = None,
) -> tuple[int, int, list[str]]:
    """Reconcile the proxy's live deployments with the plan, without restarting it.

    Circuit breaker: when one pass would *shrink* the managed rows by more than
    ``max_delete_fraction``, the deletes are skipped and reported instead. A
    gateway-side fault (the 2026-09-25 store_model_in_db incident made every probe
    fail) must not be able to empty the pool on its own; adds still go through.
    Net loss, not delete count: a re-id (changed fingerprint) deletes and re-adds the
    same route, which is routine and must not trip it.
    """
    errors: list[str] = []
    try:
        current = gateway.managed_ids()
    except GatewayError as error:
        return 0, 0, [str(error)]

    policy = context.policy if context else None
    fraction = context.settings.max_delete_fraction if context else 1.0
    to_add, to_delete = diff(plan_deployments(providers, state, env, policy), current)
    managed = sum(1 for identifier in current if is_managed(identifier))
    limit = max(MASS_DELETE_FLOOR, int(managed * fraction))
    if len(to_delete) - len(to_add) > limit:
        message = (
            f"circuit breaker: refusing to shrink the pool by {len(to_delete) - len(to_add)} "
            f"of {managed} managed deployments (limit {limit}); set "
            "FREEROUTER_MAX_DELETE_FRACTION=1 for one run if this is intended"
        )
        LOGGER.error(message)
        errors.append(message)
        to_delete = ()
    added = 0
    for deployment in to_add:
        try:
            gateway.add(deployment)
        except GatewayError as error:
            errors.append(str(error))
        else:
            added += 1

    deleted = 0
    for identifier in to_delete:
        try:
            gateway.delete(identifier)
        except GatewayError as error:
            errors.append(str(error))
        else:
            deleted += 1
    return added, deleted, errors


def observe_traffic(
    gateway: Gateway,
    providers: Sequence[Provider],
    state: HealthState,
    env: Mapping[str, str],
    context: CycleContext,
) -> tuple[HealthState, list[Event], frozenset[str]]:
    """Fold real request outcomes into health, so traffic replaces synthetic probes.

    Every model a user actually exercised is already verified for free. Returning
    those keys lets the probe stage skip them and spend its small quota budget on
    the models nobody has tested.
    """
    routes = {
        deployment.deployment_id: state_key(deployment.provider_id, deployment.model_id)
        for deployment in plan_deployments(providers, state, env, context.policy)
    }
    try:
        entries = gateway.spend_logs(state.last_traffic_seen or context.now - timedelta(days=1))
    except GatewayError as error:
        LOGGER.warning("could not read the call log, falling back to probes: %s", error)
        return state, [], frozenset()

    observed = latest_observations(entries, routes, state.last_traffic_seen)
    models = dict(state.models)
    for key, signal in context_signals(entries, routes, state.last_traffic_seen).items():
        health = models.get(key)
        if health is None:
            continue
        ceiling = learned_ceiling(signal, health.max_input or health.context or health.ref_context)
        models[key] = learn_context(health, proven=signal.proven, ceiling=ceiling, now=context.now)
        if ceiling is not None:
            LOGGER.info("learned input ceiling %s for %s", ceiling, key)
    events: list[Event] = []
    for key, observation in observed.items():
        health = models.get(key)
        if health is None:
            continue
        updated, event = record_probe(
            health,
            observation.outcome,
            observation.detail,
            observation.at,
            threshold=context.settings.threshold,
        )
        models[key] = updated
        if event is not None:
            events.append(event)

    if observed:
        LOGGER.info("real traffic verified %d models; no probe needed for them", len(observed))
    newest = newest_entry_time(entries)
    seen = max(filter(None, (newest, state.last_traffic_seen)), default=context.now)
    return (
        state.model_copy(update={"models": models, "last_traffic_seen": seen}),
        events,
        frozenset(observed),
    )


def _probe_provider(
    gateway: Gateway,
    provider: Provider,
    targets: Sequence[str],
) -> tuple[str, tuple[tuple[str, ProbeOutcome, str, bool | None], ...]]:
    """Probe one provider's selected models sequentially to respect its rate limit."""
    outcomes: list[tuple[str, ProbeOutcome, str, bool | None]] = []
    for model_id in targets:
        outcome, detail, called = probe_one(gateway, provider, model_id)
        LOGGER.info(
            "probe %s/%s -> %s%s",
            provider.provider_id,
            model_id,
            outcome.value,
            "" if called is None else f" tools={called}",
        )
        outcomes.append((model_id, outcome, detail, called))
    return provider.provider_id, tuple(outcomes)


def _probe_all(
    gateway: Gateway,
    providers: Sequence[Provider],
    state: HealthState,
    context: CycleContext,
) -> tuple[HealthState, list[Event], int]:
    """Probe every provider's due models and fold the outcomes into the state."""
    plans = [
        (provider, targets)
        for provider in providers
        if (
            targets := probe_targets(
                state,
                provider,
                context.now,
                recheck_after=context.settings.recheck_after,
                verified_by_traffic=context.verified,
            )
        )
    ]
    if not plans:
        return state, [], 0

    def run(
        item: tuple[Provider, Sequence[str]],
    ) -> tuple[str, tuple[tuple[str, ProbeOutcome, str, bool | None], ...]]:
        return _probe_provider(gateway, item[0], item[1])

    with ThreadPoolExecutor(max_workers=context.settings.workers) as pool:
        results = tuple(pool.map(run, plans))

    events: list[Event] = []
    models = dict(state.models)
    probed = 0
    for name, outcomes in results:
        for model_id, outcome, detail, called in outcomes:
            probed += 1
            health = models.get(state_key(name, model_id))
            if health is None:
                continue
            updated, event = record_probe(
                health, outcome, detail, context.now, threshold=context.settings.threshold
            )
            if called is not None:
                updated = record_tools(updated, called=called, now=context.now)
            models[updated.key] = updated
            if event is not None:
                events.append(event)
    return state.model_copy(update={"models": models}), events, probed


def _offer_stage(remaining: int, warn_days: int) -> str:
    """Bucket an offer's remaining days into a notification stage."""
    if remaining < 0:
        return "expired"
    return "expiring" if remaining <= warn_days else "ok"


def _offer_events(
    providers: Sequence[Provider],
    state: HealthState,
    today: date,
    warn_days: int,
) -> tuple[HealthState, list[Event]]:
    """Emit one event per offer whose expiry stage changed since the last cycle."""
    stages = dict(state.offers)
    events: list[Event] = []
    for provider in providers:
        ends_on = provider.offer.ends_on
        if ends_on is None:
            _ = stages.pop(provider.provider_id, None)
            continue
        stage = _offer_stage((ends_on - today).days, warn_days)
        if stages.get(provider.provider_id) == stage:
            continue
        stages[provider.provider_id] = stage
        note = provider.offer.note or provider.name_zh
        if stage == "expired":
            events.append(
                Event(
                    kind=EventKind.OFFER_EXPIRED,
                    provider=provider.provider_id,
                    detail=f"限时活动已于 {ends_on} 结束：{note}",
                )
            )
        elif stage == "expiring":
            events.append(
                Event(
                    kind=EventKind.OFFER_EXPIRING,
                    provider=provider.provider_id,
                    detail=f"限时活动 {ends_on} 到期（剩 {(ends_on - today).days} 天）：{note}",
                )
            )
    return state.model_copy(update={"offers": stages}), events


def _fetch_reference(policy: Policy) -> dict[str, int] | None:
    """Download the reference catalog: the largest context any listing gives a model."""
    response = request(REFERENCE_URL)
    if not response.ok:
        return None
    windows: dict[str, int] = {}
    for item in as_items(dig(response.json(), "data")):
        model_id = as_text(dig(item, "id"))
        length = dig(item, "context_length")
        if (
            model_id is None
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
        ):
            continue
        if "tools" not in as_strings(dig(item, "supported_parameters")):
            continue
        alias = policy.canonical(canonical_alias(model_id))
        windows[alias] = max(windows.get(alias, 0), length)
    return windows


def _reference_windows(value: object) -> dict[str, int] | None:
    """Validate the entire cache snapshot before applying any of its entries."""
    if not isinstance(value, dict):
        return None
    windows: dict[str, int] = {}
    for alias, length in cast("dict[object, object]", value).items():
        if (
            not isinstance(alias, str)
            or not alias
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
        ):
            return None
        windows[alias] = length
    return windows


def _save_reference(path: Path, windows: Mapping[str, int]) -> None:
    """Replace the cache atomically; a failed write leaves the prior file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            _ = stream.write(json.dumps(windows, indent=1, sort_keys=True))
        _ = temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _apply_reference(
    state: HealthState, settings: Settings, policy: Policy, now: datetime
) -> HealthState:
    """Apply validated reference windows, recovering corrupt caches immediately.

    Valid snapshots retain daily refresh and missing-alias clearing semantics.
    Bad snapshots never replace last-known validated health values or freshness.
    """
    try:
        windows = _reference_windows(
            JSON_ADAPTER.validate_json(settings.reference_path.read_bytes())
        )
    except (OSError, ValidationError):
        windows = None
    fetched = state.reference_fetched
    stale = fetched is None or now - fetched >= REFERENCE_MAX_AGE
    if stale or windows is None:
        replacement = _reference_windows(_fetch_reference(policy))
        if replacement:
            try:
                _save_reference(settings.reference_path, replacement)
            except OSError:
                LOGGER.warning("reference cache could not be saved; will retry next cycle")
            else:
                windows = replacement
                state = state.model_copy(update={"reference_fetched": now})
                LOGGER.info("reference catalog refreshed: %d models", len(windows))
        else:
            LOGGER.warning("reference refresh unavailable or invalid; will retry next cycle")
    if windows is None:
        LOGGER.warning("invalid or missing reference cache; retaining last-known contexts")
        return state
    models = dict(state.models)
    for key, health in state.models.items():
        ref = windows.get(policy.canonical(canonical_alias(health.model_id)))
        if ref != health.ref_context:
            models[key] = health.model_copy(update={"ref_context": ref})
    return state.model_copy(update={"models": models})


def _annotate_drops(
    state: HealthState, providers: Sequence[Provider], policy: Policy
) -> HealthState:
    """Write each deployment's rejection reason into the state for operators and the lint."""
    by_id = {provider.provider_id: provider for provider in providers}
    models = dict(state.models)
    for key, health in state.models.items():
        provider = by_id.get(health.provider)
        if provider is None:
            continue
        canonical = policy.canonical(canonical_alias(health.model_id))
        reason = policy.rejection(health, provider, canonical)
        if reason != health.drop_reason:
            models[key] = health.model_copy(update={"drop_reason": reason})
    return state.model_copy(update={"models": models})


def run_cycle(settings: Settings, env: Mapping[str, str] | None = None) -> CycleReport:
    """Run one full discover, probe, reconcile and notify cycle."""
    source = os.environ if env is None else env
    now = utcnow()
    providers = load_providers(settings.providers_dir)
    routable = [provider for provider in providers if provider.routable]
    gateway = Gateway(base_url=settings.gateway_url, master_key=settings.master_key)

    policy = load_policy(settings.policy_dir)
    state = forget_expired_rejections(load_state(settings.state_path), now)
    state, events = _ingest(state, _discover_all(routable, source, settings.workers), now)
    state = _apply_reference(state, settings, policy, now)

    context = CycleContext(now=now, settings=settings, policy=policy)
    first_added, first_deleted, first_errors = apply_plan(
        gateway, providers, state, source, context
    )

    state, traffic_events, verified = observe_traffic(gateway, providers, state, source, context)
    events.extend(traffic_events)

    probing = CycleContext(now=now, settings=settings, verified=verified, policy=policy)
    state, probe_events, probed = _probe_all(gateway, routable, state, probing)
    events.extend(probe_events)
    final_added, final_deleted, final_errors = apply_plan(
        gateway, providers, state, source, context
    )
    state = _annotate_drops(state, providers, policy)

    state, offer_events = _offer_events(providers, state, now.date(), settings.offer_warn_days)
    events.extend(offer_events)

    save_state(state, settings.state_path, now)
    append_events(settings.changelog_path, events, now)

    notifiable = [event for event in events if event.kind in NOTIFIABLE]
    failure = send(settings.webhook, "FreeRouter 免费模型池变更", notifiable)
    errors = [*first_errors, *final_errors, *([failure] if failure else [])]

    return CycleReport(
        discovered=len(state.models),
        healthy=sum(1 for h in state.models.values() if h.status is HealthStatus.HEALTHY),
        quarantined=sum(1 for h in state.models.values() if h.status is HealthStatus.QUARANTINED),
        probed=probed,
        verified_by_traffic=len(verified),
        added=first_added + final_added,
        deleted=first_deleted + final_deleted,
        events=tuple(events),
        errors=tuple(errors),
    )
