"""Pool policy: which deployments are usable, what is the same model, which roles exist.

Local addition (litellm-proxy, patches/pool-organization.patch). The bar is the one
the heaviest client sets: Hermes refuses a model with less than 64k context
(``MINIMUM_CONTEXT_LENGTH = 64_000`` in hermes-agent) and needs tool calling. A
route that Hermes cannot use is not worth a slot anywhere, so the gate is applied
to every deployment, not just to role members.

The unit of judgement is one deployment (``provider::model``), never a model name:
the same weights can arrive with a 1M window from one platform and an 8k cap from
another, and only the routes that really accept a large prompt may pool.

Policy is *optimistic about the unknown and strict about the proven*: a deployment
with no context or tool data still pools, and real traffic or a probe that proves
it inadequate removes it (and re-tests it later). Pooling only what is fully
verified would empty the pool on every new catalog entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import yaml

from freerouter.errors import PolicyError
from freerouter.registry import matches_any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from freerouter.registry import Provider
    from freerouter.state import ModelHealth

MIN_CONTEXT: Final = 64_000
MIN_PARAMS_B: Final = 20.0
UNKNOWN_PARAMS_RANK: Final = 30.0
DEFAULT_POLICY_DIR: Final = Path("/app/policy")

DENY: Final = (
    "*tts*",
    "*whisper*",
    "*orpheus*",
    "*voxtral*",
    "*transcribe*",
    "*speech*",
    "*audio*",
    "*embed*",
    "*rerank*",
    "*guard*",
    "*content-safety*",
    "*moderation*",
    "*-fim*",
    "*fim-*",
    "*translate*",
    "*-ocr*",
    "*ocr-*",
    "*image*",
    "*gemini-omni*",
    "*micro*",
    "*tiny*",
    "*-r7b*",
)
"""Never a chat-agent route: speech, embeddings, safety classifiers, fill-in-the-middle,
translation-only, OCR, image generation, and the explicitly tiny families whose id
carries no parameter count (``granite-…-micro``, ``tiny-aya``, ``command-r7b``)."""

_SIZE: Final = re.compile(r"(?<![a-z0-9.])e?(\d+(?:\.\d+)?)b(?![a-z0-9])")
"""A parameter-count token: ``-70b``, ``:20b``, ``-0.3b``, ``-e4b``. The look-behind
rejects MoE active-parameter tokens (``a3b``, ``a12b``) and version digits."""


_EXPERTS: Final = re.compile(r"\d+b-\d+e(?![a-z0-9])")
"""``17b-16e``: active parameters times experts (llama-4-scout is 109B total), so the
stated number is not the model size and must not trip the floor."""


def parse_params(model_id: str) -> float | None:
    """Return the total parameter count in billions stated by the id, if any."""
    lowered = model_id.lower()
    if _EXPERTS.search(lowered):
        return None
    sizes = [float(match[1]) for match in _SIZE.finditer(lowered)]
    return max(sizes) if sizes else None


def hard_rejection(model_id: str) -> str | None:
    """Reject by name alone: a denied family or a stated size below the floor."""
    # Publisher namespaces are not model-family names.
    if matches_any(model_id.rsplit("/", 1)[-1], DENY):
        return "not a chat-agent model"
    params = parse_params(model_id)
    if params is not None and params < MIN_PARAMS_B:
        return f"{params:g}B < {MIN_PARAMS_B:g}B"
    return None


def rank_key(model_id: str) -> tuple[float, str]:
    """Order for the per-provider cap: biggest first, unknown size in the middle."""
    params = parse_params(model_id)
    return (-(params if params is not None else UNKNOWN_PARAMS_RANK), model_id)


@dataclass(frozen=True, slots=True)
class Override:
    context: int | None = None
    tools: bool | None = None
    keep: bool | None = None


@dataclass(frozen=True, slots=True)
class Role:
    include: tuple[str, ...] = ()
    min_context: int | None = None
    # Ordered preference: tiers[0] is tried first, tiers[1] only when every deployment
    # of tier 1 is cooled down or has failed, and so on (LiteLLM's ``order``). Within a
    # tier the router still picks the fastest. A role with tiers ignores ``include``.
    tiers: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class Policy:
    merge: Mapping[str, str] = field(default_factory=dict[str, str])
    exclude: frozenset[str] = frozenset()
    roles: Mapping[str, Role] = field(default_factory=dict[str, Role])
    overrides: Mapping[str, Override] = field(default_factory=dict[str, Override])

    def canonical(self, alias: str) -> str:
        """Map a heuristic canonical alias onto the explicit merge target."""
        return self.merge.get(alias, alias)

    def override_for(self, health: ModelHealth, canonical: str) -> Override:
        """Find the override for one deployment: exact ``provider::model`` beats canonical."""
        return self.overrides.get(health.key) or self.overrides.get(canonical) or Override()

    def effective_context(
        self, health: ModelHealth, provider: Provider, canonical: str
    ) -> int | None:
        """Smallest context any source says this deployment accepts, or None if unknown.

        A prompt the deployment demonstrably served raises the floor: if real traffic
        already carried more than the declared numbers, the declaration is wrong.
        """
        override = self.override_for(health, canonical)
        if override.context is not None:
            return override.context
        declared = health.context if health.context is not None else health.ref_context
        limits = [
            value
            for value in (
                declared,
                provider.limits.max_input_tokens,
                provider.limits.tpm,
                health.max_input,
            )
            if value is not None
        ]
        ceiling = min(limits) if limits else None
        proven = health.proven_input
        # Proven traffic is a LOWER bound: it can raise a declared number, never stand
        # in for a missing one (a 14-token success says nothing about the window).
        if proven is None:
            return ceiling
        if ceiling is None:
            return proven if proven >= MIN_CONTEXT else None
        return max(ceiling, proven)

    def rejection(self, health: ModelHealth, provider: Provider, canonical: str) -> str | None:
        """Why this deployment must not be routed at all, or None when it is usable."""
        override = self.override_for(health, canonical)
        if override.keep is not None:
            return None if override.keep else "dropped by policy override"
        if canonical in self.exclude:
            return "excluded (meta-router)"
        name = hard_rejection(health.model_id)
        if name is not None:
            return name
        tools = override.tools if override.tools is not None else health.tools
        if tools is False:
            return "no tool calling"
        context = self.effective_context(health, provider, canonical)
        if context is not None and context < MIN_CONTEXT:
            return f"context {context} < {MIN_CONTEXT}"
        return None

    def roles_for(self, canonical: str, context: int | None) -> tuple[tuple[str, int | None], ...]:
        """Role aliases one usable deployment joins, each with its tier (None = unordered)."""
        joined: list[tuple[str, int | None]] = []
        for name, role in sorted(self.roles.items()):
            tier = next((i for i, names in enumerate(role.tiers, 1) if canonical in names), None)
            if role.tiers:
                if tier is not None:
                    joined.append((name, tier))
            elif canonical in role.include or (
                role.min_context is not None and context is not None and context >= role.min_context
            ):
                joined.append((name, None))
        return tuple(joined)


def _mapping(value: object, path: Path, field: str) -> dict[str, object]:
    """Validate a string-keyed mapping without coercing routing names."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyError(path=str(path), reason=f"{field}: expected a mapping")
    result: dict[str, object] = {}
    for key, item in cast("dict[object, object]", value).items():
        if not isinstance(key, str):
            raise PolicyError(path=str(path), reason=f"{field}: expected string keys")
        result[key] = item
    return result


def _strings(value: object, path: Path, field: str) -> tuple[str, ...]:
    """Validate an optional list of routing names, preserving its order."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise PolicyError(path=str(path), reason=f"{field}: expected a list of strings")
    result: list[str] = []
    for item in cast("list[object]", value):
        if not isinstance(item, str):
            raise PolicyError(path=str(path), reason=f"{field}: expected string entries")
        result.append(item)
    return tuple(result)


def _context(value: object, path: Path, field: str) -> int | None:
    """Accept positive integer token counts, never bools or lossy coercions."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyError(path=str(path), reason=f"{field}: expected a positive integer")
    return value


def _boolean(value: object, path: Path, field: str) -> bool | None:
    """Accept actual optional boolean capability flags."""
    if value is not None and not isinstance(value, bool):
        raise PolicyError(path=str(path), reason=f"{field}: expected a boolean")
    return value


def _read(path: Path) -> dict[str, object]:
    """Read one optional YAML mapping; all decoding failures are recoverable."""
    try:
        loaded = cast("object", yaml.safe_load(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PolicyError(path=str(path), reason="unreadable or malformed YAML") from error
    return _mapping(loaded, path, "document")


def _roles(path: Path) -> dict[str, Role]:
    """Validate roles, retaining tiers-over-include precedence and metadata."""
    roles: dict[str, Role] = {}
    for name, body in _mapping(_read(path).get("roles"), path, "roles").items():
        field = f"roles.{name}"
        spec = _mapping(body, path, field)
        raw_tiers = spec.get("tiers")
        tiers: list[tuple[str, ...]] = []
        if raw_tiers is not None:
            if not isinstance(raw_tiers, list):
                raise PolicyError(path=str(path), reason=f"{field}.tiers: expected nested lists")
            for index, tier in enumerate(cast("list[object]", raw_tiers)):
                # A null section is optional, but each supplied tier must be a list.
                if tier is None:
                    raise PolicyError(
                        path=str(path), reason=f"{field}.tiers[{index}]: expected a list"
                    )
                tiers.append(_strings(tier, path, f"{field}.tiers[{index}]"))
        roles[name] = Role(
            include=_strings(spec.get("include"), path, f"{field}.include"),
            min_context=_context(spec.get("min_context"), path, f"{field}.min_context"),
            tiers=tuple(tiers),
        )
    return roles


def _overrides(path: Path) -> dict[str, Override]:
    """Validate recognized capabilities, leaving extension metadata uninterpreted."""
    overrides: dict[str, Override] = {}
    for key, body in _mapping(_read(path).get("overrides"), path, "overrides").items():
        field = f"overrides.{key}"
        spec = _mapping(body, path, field)
        overrides[key] = Override(
            context=_context(spec.get("context"), path, f"{field}.context"),
            tools=_boolean(spec.get("tools"), path, f"{field}.tools"),
            keep=_boolean(spec.get("keep"), path, f"{field}.keep"),
        )
    return overrides


def load_policy(directory: Path = DEFAULT_POLICY_DIR) -> Policy:
    """Load policy atomically; a bad edit must not become an empty routing policy."""
    path = directory / "aliases.yaml"
    aliases = _read(path)
    merge: dict[str, str] = {}
    for key, value in _mapping(aliases.get("merge"), path, "merge").items():
        if not isinstance(value, str):
            raise PolicyError(path=str(path), reason=f"merge.{key}: expected a string")
        merge[key] = value
    return Policy(
        merge=merge,
        exclude=frozenset(_strings(aliases.get("exclude"), path, "exclude")),
        roles=_roles(directory / "roles.yaml"),
        overrides=_overrides(directory / "capabilities.yaml"),
    )
