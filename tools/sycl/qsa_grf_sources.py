"""Own MIT6e40 QSA proposal: isolate kernel-local GRF128/256 properties.

Scaffold derives only from this fork's17bc703d9 generator, not XeStrata.
Source only: B570 legality, kernel work-group limits and actual lowering must
be qualified before GPU execution. Production and decode remain unchanged.
"""
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('nonunique QSA GRF overlay: ' + before[:90])
    return text.replace(before, after, 1)


def generated(project):
    source = (project / PREFIX).read_text()
    body = source[:source.index('bool qsa_bf16_prefix_batch(')]
    wrapper = source[source.index('bool ' + EXPORT + '('):
                     source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    header = ('// Main-authored component declarations; repository LICENSE.\n'
              '#pragma once\n#include "strata/kernels/qsa_bf16_prefix.hpp"\n'
              '#include <sycl/sycl.hpp>\n')
    launch = ('    handler.parallel_for(sycl::nd_range<1>('
              'size_t(queries) * 2 * Workgroup, Workgroup),\n')
    outputs = {}
    for size in (128, 256):
        name = 'grf_' + str(size)
        namespace = 'namespace strata::kernels::' + name + ' {'
        edits = [
            ('#include "strata/kernels/qsa_bf16_prefix.hpp"',
             '#include "strata/kernels/qsa_bf16_prefix.hpp"\n'
             '#include <sycl/ext/intel/experimental/grf_size_properties.hpp>\n'
             '#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>'),
            ('namespace strata::kernels {', namespace),
            ('QsaPrefixStateProbe probe = {}) {',
             'QsaPrefixStateProbe probe = {}, sycl::event *submitted = nullptr) {'),
            ('  sycl_backend::finish(stream, event);\n  return true;',
             '  if (submitted) *submitted = event;\n'
             '  sycl_backend::finish(stream, event);\n  return true;'),
            (launch,
             '    sycl::ext::oneapi::experimental::nd_launch(handler,\n'
             '        sycl::ext::oneapi::experimental::launch_config{\n'
             '            sycl::nd_range<1>(size_t(queries) * 2 * Workgroup, Workgroup),\n'
             '            sycl::ext::oneapi::experimental::properties{\n'
             '                sycl::ext::intel::experimental::grf_size<' + str(size) + '>}},\n')]
        value = body
        for before, after in edits:
            value = once(value, before, after)
        restored = value
        for before, after in reversed(edits):
            restored = once(restored, after, before)
        if restored != body:
            raise ValueError('GRF overlay changed another source byte')
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
            raise ValueError('GRF wrapper changed another source byte')
        outputs[name + '.cpp'] = (value + arm + '}\n').encode()
        header += namespace + '\n' + arm[:arm.index(' {')] + ';\n}\n'
    outputs['qsa-grf.hpp'] = header.encode()
    return outputs
