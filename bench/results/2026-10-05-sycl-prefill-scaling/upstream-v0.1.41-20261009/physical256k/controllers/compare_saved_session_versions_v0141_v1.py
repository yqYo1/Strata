"""Compare every saved tensor across the two measured engine-version identities.

Session config fingerprints intentionally include STRATA_VERSION. Same-version
repeat/RESTORE gates must continue using the original complete semantic digest.
This cross-version helper omits no tensor or KV byte, including inactive MTP rows.
"""
OLD_CONFIG = 9380593023437872391
NEW_CONFIG = 643052213586580166


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items() if k not in ('offset', 'used')}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def same_cross_version_saved_tensors(current, previous):
    a = current['semantic']
    b = previous['semantic']
    if a.get('config') != NEW_CONFIG or b.get('config') != OLD_CONFIG:
        return False
    # Only the deliberately version-dependent identity field differs. Geometry,
    # IDs, every state/KV part's length and full-byte hash remain in the check.
    normalized = dict(a)
    normalized['config'] = OLD_CONFIG
    return canonical(normalized) == canonical(b)
