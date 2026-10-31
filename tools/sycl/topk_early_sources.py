"""Isolate one exact-set radix stopping rule from this fork's selected source.

Main-authored; repository LICENSE. No production selection or GPU execution.
The common counter variant observes rounds outside timing; normal instantiations
compile those stores out. The next component plan must qualify both variants.
"""
import hashlib
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parents[2]
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
SOURCE = 'src/kernels/sycl/qsa_select.cpp'
SOURCE_SHA = '463a9ec02ed28125a5cd38b7b5d28784b849a4fbb199cb017042cd99078efee1'
STOP = '''            // Higher digits are already selected. If the remaining rank
            // equals this whole bucket, every cell in it is selected: lower
            // digits cannot change the selected set or ascending tie order.
            // rank and histogram are replicated WG reductions, so this break
            // is workgroup-uniform, including zero-weight final blocks.
            if (rank == histogram[digit]) break;
            // Every completed block weighs4; only the final block can weigh
            // 1..3. A nonempty bucket weighing<=4 therefore contains exactly
            // one positive-weight block. Fetch its full key before stopping:
            // using only the prefix would overselect a partial block budget.
            if (histogram[digit] <= 4) {
              uint32_t unique_key = 0;
              for (int64_t b = lo; b < hi; ++b) {
                const uint32_t key = order(sc[b]);
                if (weight(b) > 0 && (key & mask) == threshold)
                  unique_key = key;
              }
              threshold = sycl::reduce_over_group(it.get_group(), unique_key,
                                                    sycl::maximum<uint32_t>());
              break;
            }
'''


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('expected one exact source seam: ' + old[:90])
    return text.replace(old, new)


def source():
    raw = subprocess.check_output(['git', 'show', BASE + ':' + SOURCE], cwd=P)
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA or (P / SOURCE).read_bytes() != raw:
        raise ValueError('require the unchanged selected selector source')
    return raw.decode()


def section(text, first, last):
    return text[text.index(first):text.index(last)]


def parts():
    text = source()
    return (
        section(text, 'int64_t validate(', 'uint32_t order('),
        section(text, 'uint32_t order(', '// Independently qualified'),
        section(text, 'template <int Mapping>\nsycl::event submit_topk(', '\nint64_t validate_topk('),
        section(text, 'int64_t validate_topk(', '\nint prefill_selection_mapping()'),
    )


COMMON = (
    ('template <int Mapping>', 'template <int Mapping, bool Capture>'),
    ('int64_t cap, int32_t *ids, int64_t launched) {',
     'int64_t cap, int32_t *ids, int64_t launched, int32_t *rounds) {'),
    ('        const int64_t qi = it.get_group_linear_id();',
     '        const int64_t qi = it.get_group_linear_id();\n'
     '        if constexpr (Capture)\n'
     '          if (it.get_local_linear_id() == 0) rounds[qi] = 0;'),
    ('          uint32_t mask = 0;', '          uint32_t mask = 0;\n          int round_count = 0;'),
    ('          for (int shift = 28; shift >= 0; shift -= 4) {',
     '          for (int shift = 28; shift >= 0; shift -= 4) {\n'
     '            if constexpr (Capture) ++round_count;'),
    ('            mask |= 15u << shift;\n          }\n        } else {',
     '            mask |= 15u << shift;\n          }\n'
     '          if constexpr (Capture)\n'
     '            if (t == 0) rounds[qi] = round_count;\n        } else {'),
)


def body(candidate):
    original = parts()[2]
    result = original
    for old, new in COMMON:
        result = once(result, old, new)
    if candidate:
        result = once(result, '            mask |= 15u << shift;\n',
                      '            mask |= 15u << shift;\n' + STOP)
    restored = once(result, STOP, '') if candidate else result
    for old, new in reversed(COMMON):
        restored = once(restored, new, old)
    if restored != original:
        raise ValueError('normal/counter constructor changed another source region')
    return result


SIGNATURE = '''sycl::event launch(const float *scores, const int32_t *steps, int64_t nq,
    int64_t blocks, int64_t cap, const QsaShapes &shape, int32_t *ids,
    sycl::queue &queue, int64_t active, int32_t *rounds, bool capture)'''


def generated():
    validate, order, _, spans = parts()
    header = '#pragma once\n#include "strata/kernels/qsa_select.hpp"\n#include <sycl/sycl.hpp>\n'
    result = {}
    for arm, candidate in [('original', False), ('candidate', True)]:
        namespace = 'strata::kernels::topk_early_' + arm
        header += 'namespace ' + namespace + ' {\n' + SIGNATURE + ';\n}\n'
        wrapper = SIGNATURE + ''' {
    const int64_t launched = validate_topk(scores, steps, nq, blocks, cap, shape, ids, active);
    if (capture) sycl_backend::validate_spans(
        {{ids, sycl_backend::checked_count(nq, cap) * 4},
         {rounds, sycl_backend::checked_count(nq, 1) * 4}},
        {{scores, sycl_backend::checked_count(nq, blocks) * 4},
         {steps, sycl_backend::checked_count(nq, kStepCount) * 4}});
    const auto event = capture
        ? submit_topk<1, true>(queue, scores, steps, nq, blocks, cap, ids, launched, rounds)
        : submit_topk<1, false>(queue, scores, steps, nq, blocks, cap, ids, launched, rounds);
    sycl_backend::finish(&queue, event);
    return event;
}
'''
        text = '#include "topk-early.hpp"\n#include "strata/sycl/launch.hpp"\n#include <climits>\n'
        text += 'namespace ' + namespace + ' {\nusing namespace sycl_backend;\n'
        text += validate + order + body(candidate) + spans + wrapper + '}\n'
        result[arm + '.cpp'] = text.encode()
    result['topk-early.hpp'] = header.encode()
    return result


def proof():
    return dict(original_source_commit=BASE, original_source=SOURCE,
        original_source_sha256=SOURCE_SHA,
        reversible_selection_edits=1, reversible_common_counter_edits=len(COMMON),
        production_or_decode_changed=False,
        component_scope='Actual Mapping1 body and validation;private wrappers isolate the kernel mechanism,not production startup dispatch. Same WG256,scores,weighted rank,ties and ascending emission.',
        mechanism='Stop at complete selected bucket,or recover the full unique positive-weight block key when bucket weight<=4. The latter preserves partial-cell budgets. Both branches are WG-uniform.',
        counter_scope='Capture=true records0for uniform early returns,otherwise actual1..8passes. Normal instantiations have no counter stores;capture/normal exact outputs need device checks.',
        generated_sha256={k: hashlib.sha256(v).hexdigest() for k, v in generated().items()})
