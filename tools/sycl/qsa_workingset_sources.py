"""Main-authored prepacked diagnostic overlay of own MIT selected QSA source.

No model dispatch/export is changed. Reversal proves the original kernel body,
admission, collectives, arithmetic, FP32 state and local payload are unchanged.
"""
NATIVE = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
          'src/kernels/sycl/prompt_attention.cpp', 'src/kernels/sycl/qsa_bf16_prefix.cpp')
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX = NATIVE[-1]
FIXTURE = 'bench/sycl/qsa_workingset.cpp'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('nonunique overlay: ' + before[:90])
    return text.replace(before, after, 1)


def generated(project):
    source = (project/PREFIX).read_text()
    wrapper = source[source.index('bool '+EXPORT+'('):source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    prototype = wrapper[:wrapper.index(' {')]+';\n'
    original = source[:source.index('bool qsa_bf16_prefix_batch(')]+wrapper+'} // namespace strata::kernels\n'
    namespace = 'namespace strata::kernels::working_set {'
    edits = [
        ('namespace strata::kernels {', namespace),
        ('bool OrderedExp = false, bool CaptureState = false>',
         'bool OrderedExp = false, bool CaptureState = false, bool PreparePrefix = true>'),
        ('  auto prepared = queue.parallel_for(', '  if constexpr (PreparePrefix) {\n  auto prepared = queue.parallel_for('),
        ('  sycl_backend::finish(stream, prepared);\n  auto event = queue.submit(',
         '  sycl_backend::finish(stream, prepared);\n  } // Prepared and drained once by this diagnostic caller.\n  auto event = queue.submit('),
        ('prefix_batch_impl<Cells + 1, false, false, true, true, true, true>(query,',
         'prefix_batch_impl<Cells + 1, false, false, true, true, true, true, false, false>(query,')]
    body = original
    for before, after in edits: body = once(body,before,after)
    restored = body
    for before, after in reversed(edits): restored = once(restored,after,before)
    if restored != original: raise ValueError('overlay changed another source byte')
    header = '// Main-authored diagnostic declarations; repository LICENSE applies.\n#pragma once\n'
    header += '#include "strata/kernels/qsa_bf16_prefix.hpp"\n'
    header += namespace+'\n'+prototype+'}\n'
    return {'prepacked.cpp':body.encode(),'qsa-workingset.hpp':header.encode()}
