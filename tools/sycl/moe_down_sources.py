#!/usr/bin/env python3
"""Selected split model plus qualified Down kernels; repository LICENSE.

No XeStrata implementation input. Preserve the original selected decode and
all non-Down inference bytes; the model must pass fresh qualification.
"""
import re

import moe_split_sources as split

BEFORE_SELECTOR = '7a2bd7743145608452794541b1c36dfea8751262'
PROTOTYPE = 'bf330a8f9741f392cd83b71eac6e6628a7460a18'
NATIVE = ['src/prefill/sycl/vllm_xe2/native16.cpp',
          'src/prefill/sycl/vllm_xe2/native16.hpp']


def without_observer(raw, selected):
    """Keep the selected model's observer refusal guards, not an unselected hook."""
    include = b'#include "strata/sycl/command_profile.hpp"\n'
    assert raw.count(include) == 1
    raw = raw.replace(include, b'')
    start = b'    const char *command_profile = '
    a, b = selected.index(start), selected.index(b'    std::fprintf', selected.index(start))
    c, d = raw.index(start), raw.index(b'    std::fprintf', raw.index(start))
    raw = raw[:c] + selected[a:b] + raw[d:]
    raw, count = re.subn(rb'  auto profile_tag = sycl_backend::command_profile::fused_tag\(\n.*?;\n',
                         b'', raw, flags=re.S)
    assert count == 3 and b'fused_tag' not in raw
    return raw


def overlays(revision):
    baseline = split.overlays()
    result = dict(baseline)
    selected = result[split.GEMM]
    assert without_observer(split.blob(BEFORE_SELECTOR, split.GEMM), selected) == selected
    current = split.blob(revision, split.GEMM)
    result[split.GEMM] = without_observer(current, selected)
    assert b'STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL' in result[split.GEMM]
    assert b'STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE' not in result[split.GEMM]
    prefix = b'enum class Backend '
    math = b'void validate_vllm_aot_device('
    method = b'bool Gemm::try_down_scatter16_split('
    proposed = result[split.GEMM]
    assert proposed[:proposed.index(prefix)] == selected[:selected.index(prefix)]
    assert proposed[proposed.index(math):proposed.index(method)] == selected[
        selected.index(math):selected.index(method)]
    assert b'SplitDownKernel split_down = SplitDownKernel::original;' in proposed
    for name in NATIVE:
        result[name] = split.blob(PROTOTYPE, name)

    fixture = split.blob(PROTOTYPE, split.FIXTURE)
    start = fixture.index(b'  void check_profile() {')
    end = fixture.index(b'  void separate(', start)
    fixture = fixture[:start] + fixture[end:]
    start = fixture.index(b'      } else if (mode == "--profile-engine-hook") {')
    end = fixture.index(b'      } else if (mode == "--smoke") {', start)
    fixture = fixture[:start] + fixture[end:]
    fixture = fixture.replace(b'#include "strata/sycl/command_profile.hpp"\n', b'')
    assert all(v not in fixture for v in (b'check_profile', b'--profile-engine-hook', b'cp::'))
    assert b'--split-row64-tiny' in fixture and b'--split-route-smoke' in fixture
    result[split.FIXTURE] = fixture
    assert len(result) == 12
    assert {name for name in result if result[name] != baseline[name]} == {
        split.GEMM, split.FIXTURE, *NATIVE}
    return result


def proof(revision):
    values = overlays(revision)
    return dict(base=split.BASE, split_source=split.AFTER, prototype_source=PROTOTYPE,
        selector_source=revision, paths={name: dict(
            before_sha256=split.sha(split.blob(split.BASE, name)), after_sha256=split.sha(raw))
            for name, raw in sorted(values.items())},
        selected_model_preservation='Same twelve split overlay paths;only four change from the measured '
            'split candidate:Down native source/header,explicit prefill selector and same-build fixture. '
            'Other selected source,GU math,non-split/short-input processing,decode and entrypoint unchanged.',
        precision='Preserve BF16 Down rounding and FP32 expert-order FMA,checked split addresses and queue/lifetime contract.',
        qualification='Component performance only;new model build,production-route tests,PP/TTFT,decode/handoff,'
            'task quality and physical262144 remain independent required gates.')
