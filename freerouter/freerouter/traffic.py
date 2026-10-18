from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from freerouter.gateway import PROBE_TAG
from freerouter.probe import outcome_for
from freerouter.state import ProbeOutcome
from freerouter.transport import as_strings, as_text, dig

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from freerouter.transport import JsonValue

MAX_DETAIL: Final = 200
SUCCESS_STATUS: Final = "success"
SHRINK: Final = 0.75
UNKNOWN_CEILING: Final = 128_000
LIMIT_PATTERNS: Final = (
    re.compile(r"maximum context length is (\d+)"),
    re.compile(r"context (?:length|window)(?: of| is)? (\d{4,})"),
    re.compile(r"limit[: ]+(\d{4,})"),  # groq TPM: "Limit 8000, Requested 21000"
    re.compile(r"max(?:imum)?[_ ](?:input[_ ])?tokens?\D{0,12}(\d{4,})"),
)


@dataclass(frozen=True, slots=True)
class Observation:
    deployment_id: str
    at: datetime
    outcome: ProbeOutcome
    detail: str


def _parse_time(value: str | None) -> datetime | None:
    """Parse a LiteLLM log timestamp, treating a naive value as UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _observation(entry: JsonValue) -> Observation | None:
    """Turn one spend-log record into a health observation, or None if unusable."""
    deployment_id = as_text(dig(entry, "model_id"))
    at = _parse_time(as_text(dig(entry, "startTime")))
    if deployment_id is None or at is None:
        return None
    if PROBE_TAG in as_strings(dig(entry, "request_tags")):
        return None

    if as_text(dig(entry, "status")) == SUCCESS_STATUS:
        return Observation(
            deployment_id=deployment_id, at=at, outcome=ProbeOutcome.OK, detail="真实调用成功"
        )

    error = dig(entry, "metadata.error_information")
    code = as_text(dig(error, "error_code")) or ""
    message = " ".join((as_text(dig(error, "error_message")) or "").split())[:MAX_DETAIL]
    status = int(code) if code.isdigit() else 0
    return Observation(
        deployment_id=deployment_id,
        at=at,
        outcome=outcome_for(status, message),
        detail=f"真实调用失败 HTTP {status or '?'}: {message}",
    )


def newest_entry_time(entries: Iterable[JsonValue]) -> datetime | None:
    """Return the latest ``startTime`` in the call log, or None when it has none.

    This is the traffic watermark: everything up to it has been read. The cycle's
    start time is not, because requests logged between that moment and the read
    would be read again on the next cycle.
    """
    times = (_parse_time(as_text(dig(entry, "startTime"))) for entry in entries)
    return max(filter(None, times), default=None)


def latest_observations(
    entries: Iterable[JsonValue],
    routes: Mapping[str, str],
    since: datetime | None,
) -> dict[str, Observation]:
    """Fold the call log into at most one observation per managed model.

    ``routes`` maps a deployment id to a state key. Entries at or before ``since``
    are ignored so a failure recorded once is never counted again on a later cycle.
    """
    newest: dict[str, Observation] = {}
    for entry in entries:
        observed = _observation(entry)
        if observed is None or (since is not None and observed.at <= since):
            continue
        key = routes.get(observed.deployment_id)
        if key is None:
            continue
        current = newest.get(key)
        if current is None or observed.at > current.at:
            newest[key] = observed
    return newest


@dataclass(frozen=True, slots=True)
class ContextSignal:
    proven: int | None = None  # largest prompt that succeeded
    stated_limit: int | None = None  # limit quoted by a "too large" error
    too_large: bool = False


def stated_limit(message: str) -> int | None:
    """Pull the token limit out of a "prompt too large" error, when it states one."""
    lowered = message.lower()
    for pattern in LIMIT_PATTERNS:
        match = pattern.search(lowered)
        if match:
            return int(match.group(1))
    return None


def context_signals(
    entries: Iterable[JsonValue],
    routes: Mapping[str, str],
    since: datetime | None,
) -> dict[str, ContextSignal]:
    """Summarise, per deployment, what real traffic showed about prompt size.

    Unlike health, which keeps only the newest observation, size evidence is a
    maximum over the whole window: one 90k prompt that succeeded proves the route
    takes 90k even if the last request was small.
    """
    signals: dict[str, ContextSignal] = {}
    for entry in entries:
        observed = _observation(entry)
        if observed is None or (since is not None and observed.at <= since):
            continue
        key = routes.get(observed.deployment_id)
        if key is None:
            continue
        current = signals.get(key, ContextSignal())
        if observed.outcome is ProbeOutcome.OK:
            tokens = dig(entry, "prompt_tokens")
            if isinstance(tokens, int) and tokens > (current.proven or 0):
                current = ContextSignal(tokens, current.stated_limit, current.too_large)
        elif observed.outcome is ProbeOutcome.TOO_LARGE:
            limit = stated_limit(observed.detail)
            lowest = min(filter(None, (limit, current.stated_limit)), default=None)
            current = ContextSignal(current.proven, lowest, too_large=True)
        signals[key] = current
    return signals


def learned_ceiling(signal: ContextSignal, current: int | None) -> int | None:
    """Return the input ceiling a "too large" signal implies, or None without one."""
    if not signal.too_large:
        return None
    if signal.stated_limit is not None:
        return signal.stated_limit
    return int((current or UNKNOWN_CEILING) * SHRINK)
