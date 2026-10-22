"""Main-authored static-indexing experiment; original production body stays fixed."""
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
    for variant in ('control', 'unroll'):
        # Both isolated arms contain only the current ordered-exp entrypoint.
        original = source[:source.index('bool qsa_bf16_prefix_batch(')]+wrapper+'} // namespace strata::kernels\n'
        namespace = f'namespace strata::kernels::{variant} {{'
        body = replace_once(original, 'namespace strata::kernels {', namespace)
        if variant == 'unroll':
            for before in (
                '          for (int d = 0; d < Dim; d += 16)\n            mx::joint_matrix_load(sg, cached_query[d / 16]',
                '          for (int d = 0; d < Dim; d += 16) {\n            if constexpr (TilePacked)'):
                after = '          #pragma clang loop unroll(full)\n'+before
                body = replace_once(body, before, after)
        # Namespace plus two loop hints are the entire overlay. No reduction,
        # state, lifetime, queue, register-property or producer edit is allowed.
        restored=body.replace('          #pragma clang loop unroll(full)\n','').replace(namespace,'namespace strata::kernels {',1)
        if restored != original or body.count('#pragma clang loop unroll(full)') != (2 if variant=='unroll' else 0):
            raise ValueError('static-unroll overlay changed another body byte')
        result[variant+'.cpp'] = body
    header = '// Main-authored experiment declarations; repository LICENSE applies.\n#pragma once\n'
    header += '#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include <array>\n#include <algorithm>\n#include <limits>\n'
    for variant in ('control','unroll'):
        header += f'namespace strata::kernels::{variant} {{\n'+prototype+'}\n'
    header += 'inline auto qsa_unroll_functions() {\n  using namespace strata::kernels;\n'
    header += f'  return std::array<decltype(&{EXPORT}),3>{{{EXPORT},control::{EXPORT},unroll::{EXPORT}}};\n}}\n'
    result['qsa-unroll-variants.hpp'] = header

    timing = (project/BENCH).read_text()
    start = timing.index('#if defined(STRATA_QSA_BENCH_ORDERED_EXP)')
    end = timing.index('#endif', start)+len('#endif')
    timing = timing[:start]+'''const auto attention_variants=qsa_unroll_functions();
const std::array<const char*,3> variant_names{"default","control","unroll"};
long long timer_floor_ns=0;'''+timing[end:]
    timing = '#include "qsa-unroll-variants.hpp"\n'+timing
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
