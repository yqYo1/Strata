"""Separate expert and cycle bounds in the owned adaptive diagnostic.

Main-authored; repository LICENSE. Pure source projection only: preserve the
running429b inputs and recover both originals from HEADER_SOURCE before use.
The successor still needs committed host, native positive and refusal checks.
"""
import hashlib

from adaptive_residency_sources import HEADER, HEADER_SHA, HEADER_SOURCE, one

FIXTURE = 'tests/core/adaptive_residency_audit_test.cpp'
FIXTURE_SHA = 'd65aec5814e765b1c4947df1cc06b78aa2ad9666e791b1edb6b4aced9e052ec5'
EXPERT_BYTES = 64 << 20
CYCLE_BYTES = 256 << 20
MODEL_MAX_BYTES = 2662400


def header(original):
    if hashlib.sha256(original).hexdigest() != HEADER_SHA:
        raise RuntimeError('recover original299218 audit header;do not refresh its pin')
    value = original.decode('utf-8')
    value = one(value,
        '    static constexpr std::size_t max_cycle_payload_bytes = 64u << 20;\n',
        '    static constexpr std::size_t max_expert_payload_bytes = 64u << 20;\n'
        '    static constexpr std::size_t max_cycle_payload_bytes = 256u << 20;\n'
        '    static_assert(max_expert_payload_bytes <= max_cycle_payload_bytes);\n')
    value = one(value, 'if (!bytes || bytes > max_cycle_payload_bytes)',
                       'if (!bytes || bytes > max_expert_payload_bytes)')
    return value.encode('utf-8')


def legacy_fixture(original):
    """Keep all41 old cases; replace only the obsolete80MiB overflow case."""
    if hashlib.sha256(original).hexdigest() != FIXTURE_SHA:
        raise RuntimeError('recover exact original41-case host fixture')
    before = '''    Fixture f;
    Audit a({40u << 20, 40u << 20}, 4, 4, 2);
    std::string err;
    require(!f.begin(a, err) && f.victim_checks == 0, "payload budget was checked after callbacks");
    ++cases;
'''
    after = '''    {
        // Five separately valid64MiB extents exceed the new256MiB cycle cap.
        Map map(10, -1);
        std::vector<Swap> changes;
        for (int32_t i = 0; i < 5; ++i) {
            map[std::size_t(i)] = i;
            changes.push_back({std::size_t(i + 5), std::size_t(i), i});
        }
        Audit a({Audit::max_expert_payload_bytes}, 10, 5, 5);
        std::string err;
        std::size_t victims = 0;
        require(!a.prepare(map, map, changes, [&](auto row) { return map.at(row); },
            [&](const Swap&, auto, std::string&) { ++victims; return true; }, err) &&
            victims == 0 && a.failed() && a.completed() == 0 &&
            err == "adaptive cache audit: replacement payload budget",
            "payload budget was checked after callbacks");
        ++cases;
    }
'''
    return one(original.decode('utf-8'), before, after).encode('utf-8')


def projection(before):
    if HEADER not in before:
        raise RuntimeError('successor requires the owned positive-swap source composition')
    values = dict(before)
    values[HEADER] = header(before[HEADER])
    if {n for n in values if values[n] != before[n]} != {HEADER}:
        raise RuntimeError('only the diagnostic audit header may change')
    return values
