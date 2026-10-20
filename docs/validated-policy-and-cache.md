# Validated policy and reference-cache recovery

## Policy input contract

The three optional policy files are `freerouter/policy/aliases.yaml`,
`roles.yaml`, and `capabilities.yaml` (or the mounted policy directory).
Missing files, empty/null documents, and missing/null sections stay optional.
Routing name mappings use string keys and string values; include/exclude fields
use lists of strings. Tiers are lists of lists of strings, in preference order.
Role and override bodies may be null (empty), but supplied bodies must be mappings.

```yaml
# aliases.yaml
merge: {alternative: canonical}
exclude: [meta-router]
# roles.yaml
roles:
  demo:
    include: [canonical]
    tiers: [[canonical], [fallback]]
    min_context: 64000
# capabilities.yaml
overrides:
  canonical:
    context: 131072
    tools: true
    keep: false
    why: "Operator metadata, not a routing rule"
```

Tiers retain precedence over include; exact deployment overrides retain precedence
over canonical overrides. Context counts must be positive integers, not booleans,
floats or numeric strings. Flags must be actual YAML booleans. Metadata/extensions
such as `distinct` and `why` remain ignored by this loader.

This deliberately stops accidental malformed-input acceptance: `include: alpha`
no longer becomes five character aliases; `min_context: '64000'`, `64000.9`,
`true`, `0`, and `.inf` are not token counts. Use `include: [alpha]` and
`min_context: 64000`. False/zero/scalar substitutes for collections are rejected.
Recognized invalid fields reject the entire policy with a file/field-specific
`PolicyError`. The refresh daemon logs failed cycles and retries; a failed load
does not reconcile gateway deployments or persist a partial routing policy.
Unreadable/malformed files are also recoverable errors, not silent empty policy.

## Reference catalog

A valid cache is a JSON mapping from nonempty canonical alias strings to positive,
non-boolean integer contexts. Validation is atomic across the snapshot. Valid
snapshots keep the existing 24-hour refresh interval. A valid cache with an alias
missing clears that alias's reference context; an empty mapping clears all.

Malformed JSON, scalar/list/null roots, and invalid values trigger an immediate
refresh regardless of cache age. Only validated nonempty fetch results replace
the cache, using a temporary file plus atomic rename. Downloaded entries with
invalid/nonpositive/boolean context lengths are not admitted. If refresh or
persistence fails, a warning is logged, freshness is not advanced, and the next
cycle retries. With an invalid cache, validated last-known health contexts are
kept; a good old cache remains usable after a failed daily refresh. No live
provider credentials are needed to run the offline regression tests.

Strict type checking and lint settings/dependencies are unchanged. Narrow casts
at the YAML/dictionary boundary describe `object` containers only after runtime
shape checks; all keys and values are then validated before use, not cast into a
trusted routing structure.
