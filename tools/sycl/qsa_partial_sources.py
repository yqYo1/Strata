"""Main-authored partial-Q-cache experiment; original production body stays fixed."""
from pathlib import Path

NATIVE = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
          'src/kernels/sycl/prompt_attention.cpp', 'src/kernels/sycl/qsa_bf16_prefix.cpp')
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX = NATIVE[-1]
BENCH = 'tests/sycl/qsa_bf16_prefix_layout_bench.cpp'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('source edit is not unique: '+before[:90])
    return text.replace(before, after, 1)


def generated(project):
    source = (project/PREFIX).read_text()
    wrapper = source[source.index('bool '+EXPORT+'('):source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    prototype = wrapper[:wrapper.index(' {')]+';\n'
    result = {}
    for variant in ('control', 'partialq8'):
        # Both isolated arms contain only the current ordered-exp entrypoint.
        original = source[:source.index('bool qsa_bf16_prefix_batch(')]+wrapper+'} // namespace strata::kernels\n'
        namespace = f'namespace strata::kernels::{variant} {{'
        body = replace_once(original, 'namespace strata::kernels {', namespace)
        edits = []
        if variant == 'partialq8':
            # Q[0:128] stays in eight noncopyable matrix fragments; the full
            # Q arena stays disjoint from K/V for the other eight panels.
            edits = [
                ('bool OrderedExp = false, bool CaptureState = false>',
                 'bool OrderedExp = false, bool CaptureState = false, int CachedPanels = Dim / 16>'),
                ('  static_assert(Dim % 4 == 0);',
                 '  static_assert(Dim % 4 == 0);\n  static_assert(CachedPanels > 0 && CachedPanels <= Dim / 16);\n  constexpr bool ReuseQueryArena = CacheQueries && CachedPanels == Dim / 16;'),
                ('LocalPayload<ScoreStride, CacheQueries>', 'LocalPayload<ScoreStride, ReuseQueryArena>'),
                ('MatrixTile<CacheQueries ? 2 : 16 * Dim>', 'MatrixTile<ReuseQueryArena ? 2 : 16 * Dim>'),
                ('if constexpr (CacheQueries) qtile = kv;', 'if constexpr (ReuseQueryArena) qtile = kv;'),
                ('Left cached_query[CacheQueries ? Dim / 16 : 1];', 'Left cached_query[CacheQueries ? CachedPanels : 1];'),
                ('for (int d = 0; d < Dim; d += 16)\n            mx::joint_matrix_load(sg, cached_query[d / 16]',
                 'for (int d = 0; d < CachedPanels * 16; d += 16)\n            mx::joint_matrix_load(sg, cached_query[d / 16]'),
                ('            if constexpr (CacheQueries)\n              mx::joint_matrix_mad(sg, product, cached_query[d / 16], b, product);\n            else mx::joint_matrix_mad(sg, product, a, b, product);',
                 '            if constexpr (CacheQueries) {\n              // d is uniform within each subgroup; every lane executes each\n              // collective. Accumulation keeps the original ascending order.\n              if (d < CachedPanels * 16)\n                mx::joint_matrix_mad(sg, product, cached_query[d / 16], b, product);\n              else {\n                mx::joint_matrix_load(sg, a, qtile + head * Dim + d, Dim);\n                mx::joint_matrix_mad(sg, product, a, b, product);\n              }\n            } else mx::joint_matrix_mad(sg, product, a, b, product);'),
                ('prefix_batch_impl<Cells + 1, false, false, true, true, true, true>(query,',
                 'prefix_batch_impl<Cells + 1, false, false, true, true, true, true, false, 8>(query,')]
            for before, after in edits:
                body = replace_once(body, before, after)
        # Reverse the bounded overlay to prove that no other body byte changes.
        restored = body
        for before, after in reversed(edits):
            restored = replace_once(restored, after, before)
        restored = replace_once(restored, namespace, 'namespace strata::kernels {')
        if restored != original:
            raise ValueError('partial-Q-cache overlay changed another body byte')
        result[variant+'.cpp'] = body
    header = '// Main-authored experiment declarations; repository LICENSE applies.\n#pragma once\n'
    header += '#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include <array>\n#include <algorithm>\n#include <limits>\n'
    for variant in ('control','partialq8'):
        header += f'namespace strata::kernels::{variant} {{\n'+prototype+'}\n'
    header += 'inline auto qsa_partial_functions() {\n  using namespace strata::kernels;\n'
    header += f'  return std::array<decltype(&{EXPORT}),3>{{{EXPORT},control::{EXPORT},partialq8::{EXPORT}}};\n}}\n'
    result['qsa-partial-variants.hpp'] = header

    timing = (project/BENCH).read_text()
    start = timing.index('#if defined(STRATA_QSA_BENCH_ORDERED_EXP)')
    end = timing.index('#endif', start)+len('#endif')
    timing = timing[:start]+'''const auto attention_variants=qsa_partial_functions();
const std::array<const char*,3> variant_names{"default","control","partialq8"};
long long timer_floor_ns=0;'''+timing[end:]
    timing = '#include "qsa-partial-variants.hpp"\n'+timing
    edits = [
        ('(variant ? candidate_attention : baseline_attention)', 'attention_variants[size_t(variant)]'),
        ('const std::array<int,2> variants{0,1};', 'const std::array<int,3> variants{0,1,2};'),
        ('index<2', 'index<3'),
        ('(process_id+round+index)%2', '(process_id+round+index)%3'),
        ('(variant ? candidate_name : baseline_name)', 'variant_names[size_t(variant)]'),
        ('<<" samples=12 full_bits=PASS', '<<" samples=18 full_bits=PASS'),
        ('const double ms=invoke(variant);', '''const double ms=invoke(variant);
    if (queries>=8192) require(ms>=.1 && ms*1e6>=1000.*timer_floor_ns,"timing batch below declared floor");'''),
        ('std::cout.setf(std::ios::unitbuf);', '''std::cout.setf(std::ios::unitbuf);
    timer_floor_ns=std::numeric_limits<long long>::max();
    for (int i=0;i<128;++i) {
      const auto a=std::chrono::steady_clock::now(),b=std::chrono::steady_clock::now();
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
      if (ns>0) timer_floor_ns=std::min(timer_floor_ns,static_cast<long long>(ns));
    }
    require(timer_floor_ns<std::numeric_limits<long long>::max(),"clock calibration");
    std::cout<<"timer_floor_ns="<<timer_floor_ns<<'\\n';'''),
        ('else { run(78453,8192,33,false);run(78453,8192,2051,false);run(78453,8192,2051,true); }',
         'else { run(32768,8192,2051,false);run(117488,8192,2051,false);run(117488,8192,2051,true); }')]
    for before,after in edits:
        timing=replace_once(timing,before,after)
    result['timing.cpp']=timing
    return {name:body.encode() for name,body in result.items()}
