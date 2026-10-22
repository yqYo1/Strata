"""Main-authored QK B-layout discriminator; repository LICENSE applies.

K is stored by logical cell/dimension and loaded as column-major B. PV keeps
the original row-major B. This is an isolated prefill component, not a model
route. The compiler/backend contract still requires independent qualification.
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
        raise ValueError('production QSA differs from the selected native source')
    source = raw.decode()
    wrapper = source[source.index('bool ' + EXPORT + '('):source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    original = source[:source.index('bool qsa_bf16_prefix_batch(')] + wrapper + '} // namespace strata::kernels\n'
    control = replace_once(original, 'namespace strata::kernels {', 'namespace strata::kernels::control {')
    changes = [
        ('namespace strata::kernels {', 'namespace strata::kernels::colmajor {'),
        ('      Acc accumulated[4];',
         '      using RightK = mx::joint_matrix<sycl::sub_group, BFloat, mx::use::b, 16, 16,\n'
         '                                       mx::layout::col_major>;\n      Acc accumulated[4];'),
        ('        if constexpr (VectorK4) {\n'
         '          for (int i = tid; i < Cells * (Dim / 4); i += Workgroup) {\n'
         '            const int cell = i % Cells, d = (i / Cells) * 4;',
         '        if constexpr (VectorK4) {\n'
         '          for (int i = tid; i < Cells * (Dim / 4); i += Workgroup) {\n'
         '            const int cell = i / (Dim / 4), d = (i % (Dim / 4)) * 4;'),
        ('            // Retain parallel rows and contiguous cell stores within a\n'
         '            // subgroup. Each of the 8192 local destinations has one writer.\n'
         '            for (int j = 0; j < 4; ++j)\n'
         '              kv[(d + j) * Cells + cell] = sycl::bit_cast<BFloat>(uint16_t(words[j]));',
         '            // Read and store adjacent dimensions of one logical K row.\n'
         '            // Column-major B interprets that row as a matrix column.\n'
         '            for (int j = 0; j < 4; ++j)\n'
         '              kv[cell * Dim + d + j] = sycl::bit_cast<BFloat>(uint16_t(words[j]));'),
        ('          Acc product; Left a; Right b;', '          Acc product; Left a; RightK b;'),
        ('              mx::joint_matrix_load(sg, b, kv + d * Cells + cell, Cells);',
         '              mx::joint_matrix_load(sg, b, kv + cell * Dim + d, Dim);'),
    ]
    candidate = original
    for before, after in changes:
        candidate = replace_once(candidate, before, after)
    counter = '''namespace {
uint64_t specialized_routes = 0;
}
std::array<uint64_t,2> route_counts() { return {specialized_routes, 0}; }
'''
    candidate = replace_once(candidate, 'namespace strata::kernels::colmajor {',
        'namespace strata::kernels::colmajor {\n' + counter)
    marker = 'void *stream) {\n  return prefix_batch_impl<Cells + 1, false, false, true, true, true, true>'
    candidate = '#include <array>\n' + replace_once(candidate, marker,
        marker.replace('\n  return', '\n  ++specialized_routes;\n  return', 1))
    recovered = candidate.removeprefix('#include <array>\n').replace(counter, '', 1)
    recovered = recovered.replace('namespace strata::kernels::colmajor {\n', 'namespace strata::kernels::colmajor {', 1)
    recovered = recovered.replace('  ++specialized_routes;\n', '', 1)
    for before, after in reversed(changes):
        recovered = replace_once(recovered, after, before)
    if recovered != original:
        raise ValueError('layout overlay changed another source byte')
    old = git(ORIGIN, PRIOR_GENERATOR)
    plan = json.loads(git(ORIGIN_PLAN, 'bench/results/2026-10-15-pr6-recovery-audit/qsa-page4-current.json'))
    if hashlib.sha256(old).hexdigest() != plan['input_sha256'][PRIOR_GENERATOR]:
        raise ValueError('original fixture generator changed')
    module = types.ModuleType('main_authored_historical_fixture_generator')
    exec(compile(old, ORIGIN + ':' + PRIOR_GENERATOR, 'exec'), module.__dict__)
    prior = module.generated(project)
    return {'control.cpp': control.encode(), 'colmajor.cpp': candidate.encode(),
        'qsa-colmajor-variants.hpp': prior['qsa-page4-variants.hpp'].decode().replace('page4', 'colmajor').encode(),
        'timing.cpp': prior['timing.cpp'].decode().replace('page4', 'colmajor').replace('Page4', 'Column-major').encode(),
        'numerical-reference.inc': prior['numerical-reference.inc']}
