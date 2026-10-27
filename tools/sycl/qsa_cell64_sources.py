"""Own MIT6e40 QSA: isolated32/64-cell online-attention proposal.

Main-authored overlay, following this fork's17bc scaffold; no XeStrata source.
Cell64 changes the online update/BF16 rounding boundaries. It is a numerical
candidate, not an exact-bit rewrite. Source-only; no production/decode change.
The matrix shape, WG128/SG16, default GRF and source-lifetime contract stay fixed.
"""
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('nonunique cell64 overlay: ' + before[:90])
    return text.replace(before, after, 1)


def generated(project):
    source = (project / PREFIX).read_text()
    body = source[:source.index('bool qsa_bf16_prefix_batch(')]
    wrapper = source[source.index('bool ' + EXPORT + '('):
                     source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    header = ('// Main-authored component declarations; repository LICENSE.\n'
              '#pragma once\n#include "strata/kernels/qsa_bf16_prefix.hpp"\n'
              '#include <sycl/sycl.hpp>\n')
    outputs = {}
    for cells in (32, 64):
        name = 'cell' + str(cells)
        namespace = 'namespace strata::kernels::' + name + ' {'
        edits = [
            ('namespace strata::kernels {', namespace),
            ('QsaPrefixStateProbe probe = {}) {',
             'QsaPrefixStateProbe probe = {}, sycl::event *submitted = nullptr) {'),
            ('  sycl_backend::finish(stream, event);\n  return true;',
             '  if (submitted) *submitted = event;\n'
             '  sycl_backend::finish(stream, event);\n  return true;')]
        if cells == 64:
            edits += [
                ('Cells = 32', 'Cells = 64'),
                ('LocalPayload<Cells> == 28160', 'LocalPayload<Cells> == 47872'),
                ('LocalPayload<Cells + 1> == 28224', 'LocalPayload<Cells + 1> == 47936'),
                ('LocalPayload<Cells + 1, true> == 20036',
                 'LocalPayload<Cells + 1, true> == 39748'),
                ('static_assert(Dim % 4 == 0);',
                 'static_assert(Dim % 4 == 0);\n'
                 '  static_assert(OrderedExp && CacheQueries && VectorK4 && VectorV4);'),
                ('const int head = (subgroup / 2) * 8, cell = (subgroup % 2) * 16;',
                 'const int head = (subgroup / 2) * 8;\n'
                 '          for (int cell = (subgroup % 2) * 16; cell < Cells; cell += 32) {'),
                ('scores[(head + r) * ScoreStride + cell + c] = value;\n'
                 '          });\n        }',
                 'scores[(head + r) * ScoreStride + cell + c] = value;\n'
                 '          });\n          }\n        }'),
                ('const int c0 = lane, c1 = lane + 16;',
                 'const int c0 = lane, c1 = lane + 16, c2 = lane + 32, c3 = lane + 48;'),
                ('const float s1 = scores[h * ScoreStride + c1];',
                 'const float s1 = scores[h * ScoreStride + c1];\n'
                 '            const float s2 = scores[h * ScoreStride + c2];\n'
                 '            const float s3 = scores[h * ScoreStride + c3];'),
                ('const bool row0 = rows[c0] >= 0, row1 = rows[c1] >= 0;',
                 'const bool row0 = rows[c0] >= 0, row1 = rows[c1] >= 0;\n'
                 '            const bool row2 = rows[c2] >= 0, row3 = rows[c3] >= 0;'),
                ('const bool valid1 = row1 && sycl::isfinite(s1);',
                 'const bool valid1 = row1 && sycl::isfinite(s1);\n'
                 '            const bool valid2 = row2 && sycl::isfinite(s2);\n'
                 '            const bool valid3 = row3 && sycl::isfinite(s3);'),
                ('const int bad = (row0 && !valid0) || (row1 && !valid1);',
                 'const int bad = (row0 && !valid0) || (row1 && !valid1) ||\n'
                 '                            (row2 && !valid2) || (row3 && !valid3);'),
                ('const float lane_max = sycl::fmax(valid0 ? s0 : -INFINITY,\n'
                 '                                             valid1 ? s1 : -INFINITY);',
                 'const float lane_max = sycl::fmax(\n'
                 '                sycl::fmax(valid0 ? s0 : -INFINITY, valid1 ? s1 : -INFINITY),\n'
                 '                sycl::fmax(valid2 ? s2 : -INFINITY, valid3 ? s3 : -INFINITY));'),
                ('const float p1 = valid1 ? sycl::exp2(s1 - m) : 0.f;',
                 'const float p1 = valid1 ? sycl::exp2(s1 - m) : 0.f;\n'
                 '            const float p2 = valid2 ? sycl::exp2(s2 - m) : 0.f;\n'
                 '            const float p3 = valid3 ? sycl::exp2(s3 - m) : 0.f;'),
                ('c < 16 ? p0 : p1, c % 16',
                 'c < 16 ? p0 : c < 32 ? p1 : c < 48 ? p2 : p3, c % 16'),
                ('Lane0 alone adds c=0..31', 'Lane0 alone adds c=0..63'),
                ('probability[h * Cells + c1] = BFloat(p1);',
                 'probability[h * Cells + c1] = BFloat(p1);\n'
                 '            probability[h * Cells + c2] = BFloat(p2);\n'
                 '            probability[h * Cells + c3] = BFloat(p3);')]
        value = body
        for before, after in edits:
            value = once(value, before, after)
        restored = value
        for before, after in reversed(edits):
            restored = once(restored, after, before)
        if restored != body:
            raise ValueError('cell64 overlay changed an undeclared source byte')
        wrapper_edits = [
            ('void *stream) {', 'void *stream, sycl::event *submitted) {'),
            ('output, queries, bounds, live_cells, scratch, stream);',
             'output, queries, bounds, live_cells, scratch, stream, {}, submitted);')]
        arm = wrapper
        for before, after in wrapper_edits:
            arm = once(arm, before, after)
        restored = arm
        for before, after in reversed(wrapper_edits):
            restored = once(restored, after, before)
        if restored != wrapper:
            raise ValueError('cell64 wrapper changed an undeclared source byte')
        outputs[name + '.cpp'] = (value + arm + '}\n').encode()
        header += namespace + '\n' + arm[:arm.index(' {')] + ';\n}\n'
    outputs['qsa-cell64.hpp'] = header.encode()
    return outputs
