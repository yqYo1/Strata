"""Main-authored same-math QSA synchronization proposal; repository LICENSE.

Only a private component is generated. Production/decode dispatch is untouched.
Reversal checks establish the exact source delta, not race freedom or speed.
"""
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
NATIVE = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp',
          'src/sycl/command_profile.cpp', 'src/kernels/sycl/prompt_attention.cpp',
          'src/kernels/sycl/qsa_bf16_prefix.cpp')
PREFIX = NATIVE[-1]
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('nonunique QSA synchronization overlay: ' + before[:90])
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
    for name, merge in (('barrier_control', False), ('barrier_merged', True)):
        namespace = 'namespace strata::kernels::' + name + ' {'
        edits = [
            ('namespace strata::kernels {', namespace),
            ('QsaPrefixStateProbe probe = {}) {',
             'QsaPrefixStateProbe probe = {}, sycl::event *submitted = nullptr) {'),
            ('  sycl_backend::finish(stream, event);\n  return true;',
             '  if (submitted) *submitted = event;\n'
             '  sycl_backend::finish(stream, event);\n  return true;')]
        if merge:
            # PV reads kv/probability/alpha, not rows. Next publish_rows writes
            # only rows/private flags. Its existing following barrier completes
            # previous PV reads before the next K write. Preserve a final fence
            # before group reductions/output, including the zero-width case.
            edits.append((
                '        item.barrier(sycl::access::fence_space::local_space);\n'
                '      }\n'
                '      const bool invalid_scales =',
                '      }\n'
                '      item.barrier(sycl::access::fence_space::local_space);\n'
                '      const bool invalid_scales ='))
        value = body
        for before, after in edits:
            value = once(value, before, after)
        restored = value
        for before, after in reversed(edits):
            restored = once(restored, after, before)
        if restored != body:
            raise ValueError('QSA synchronization overlay changed another byte')
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
            raise ValueError('QSA component wrapper changed another byte')
        outputs[name + '.cpp'] = (value + arm + '}\n').encode()
        header += namespace + '\n' + arm[:arm.index(' {')] + ';\n}\n'
    outputs['qsa-barrier.hpp'] = header.encode()
    return outputs
