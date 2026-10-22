"""Main-authored, component-only fixed two-head iteration discriminator.

WG128 and required SG16 give eight subgroups. Each subgroup owns exactly two
of the sixteen padded head rows. Keep math/order/barriers/mapping unchanged.
No engine dispatch or decode source is modified by this generator.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import types

BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
ORIGIN = '760d22dc3178903cdeb0316d349bce06c09a0ffc'
ORIGIN_PLAN = '1ebecb46a'
PRIOR_GENERATOR = 'tools/sycl/qsa_page4_sources.py'
PREFIX = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('source transformation is not unique: ' + before[:90])
    return text.replace(before, after, 1)


def generated(project):
    project = Path(project)
    def git(revision, path):
        return subprocess.check_output(['git', '-C', str(project), 'show', revision + ':' + path], timeout=20)
    raw = (project / PREFIX).read_bytes()
    if raw != git(BASE, PREFIX):
        raise ValueError('production QSA source differs from selected engine')
    source = raw.decode()
    wrapper = source[source.index('bool ' + EXPORT + '('):source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    original = source[:source.index('bool qsa_bf16_prefix_batch(')] + wrapper + '} // namespace strata::kernels\n'
    control = replace_once(original, 'namespace strata::kernels {', 'namespace strata::kernels::control {')
    candidate = replace_once(original, 'namespace strata::kernels {', 'namespace strata::kernels::fixed_heads {')
    candidate = replace_once(candidate, '      for (int head = subgroup; head < 16; head += 8) {',
        '      for (int part = 0; part < 2; ++part) {\n        const int head = subgroup + part * 8;')
    candidate = replace_once(candidate, '          for (int h = subgroup; h < 16; h += 8) {',
        '          for (int part = 0; part < 2; ++part) {\n            const int h = subgroup + part * 8;')
    # A single-caller host component counter proves timed wrapper selection;
    # traced qualification must also verify the actual executed kernel name.
    counter = '''namespace {
uint64_t specialized_routes = 0;
}
std::array<uint64_t,2> route_counts() { return {specialized_routes, 0}; }
'''
    candidate = replace_once(candidate, 'namespace strata::kernels::fixed_heads {',
        'namespace strata::kernels::fixed_heads {\n' + counter)
    marker = 'void *stream) {\n  return prefix_batch_impl<Cells + 1, false, false, true, true, true, true>'
    candidate = '#include <array>\n' + replace_once(candidate, marker,
        marker.replace('\n  return', '\n  ++specialized_routes;\n  return', 1))
    recovered = candidate.removeprefix('#include <array>\n').replace(counter, '', 1)
    recovered = recovered.replace('namespace strata::kernels::fixed_heads {\n', 'namespace strata::kernels {', 1)
    recovered = recovered.replace('      for (int part = 0; part < 2; ++part) {\n        const int head = subgroup + part * 8;',
        '      for (int head = subgroup; head < 16; head += 8) {', 1)
    recovered = recovered.replace('          for (int part = 0; part < 2; ++part) {\n            const int h = subgroup + part * 8;',
        '          for (int h = subgroup; h < 16; h += 8) {', 1).replace('  ++specialized_routes;\n', '', 1)
    if recovered != original:
        raise ValueError('overlay changed another source byte')
    # Reuse only this fork's Main-authored variable-page oracle correction and
    # rotated three-arm timing fixture, restoring its immutable Git source.
    old = git(ORIGIN, PRIOR_GENERATOR)
    plan = json.loads(git(ORIGIN_PLAN, 'bench/results/2026-10-15-pr6-recovery-audit/qsa-page4-current.json'))
    if hashlib.sha256(old).hexdigest() != plan['input_sha256'][PRIOR_GENERATOR]:
        raise ValueError('original fixture generator identity changed')
    module = types.ModuleType('main_authored_historical_fixture_generator')
    exec(compile(old, ORIGIN + ':' + PRIOR_GENERATOR, 'exec'), module.__dict__)
    prior = module.generated(project)
    header = prior['qsa-page4-variants.hpp'].decode().replace('page4', 'fixed_heads')
    timing = prior['timing.cpp'].decode().replace('page4', 'fixed_heads').replace('Page4', 'fixed-head')
    return {'control.cpp': control.encode(), 'fixed-heads.cpp': candidate.encode(),
        'qsa-fixed_heads-variants.hpp': header.encode(), 'timing.cpp': timing.encode(),
        'numerical-reference.inc': prior['numerical-reference.inc']}
