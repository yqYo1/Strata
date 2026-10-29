#!/usr/bin/env python3
"""Exact selected-source split-MoE overlay; repository LICENSE applies.

No XeStrata implementation input. Existing adapter/TLA notices stay intact.
The active tree contains other unselected diagnostics; exclude those explicitly.
"""
import hashlib
from pathlib import Path
import subprocess
import tempfile

P = Path(__file__).resolve().parents[2]
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
BEFORE = '190555603'
AFTER = '8a34a9ed6'
DIRECT = [
    'include/strata/prefill/moe_accum_plan.hpp',
    'include/strata/prefill/moe_window_routes.hpp',
    'include/strata/prefill/native16.hpp',
    'include/strata/prefill/gemm.hpp',
    'src/prefill/sycl/native16.cpp',
    'src/prefill/sycl/vllm_xe2/native16.cpp',
    'src/prefill/sycl/vllm_xe2/native16.hpp',
]
ORCHESTRATION = ['src/prefill/prefill.cpp', 'src/prefill/layer_major.inc', 'src/prefill/moe_window.inc']
GEMM = 'src/prefill/sycl/gemm16.cpp'
FIXTURE = 'bench/sycl/moe_down_scatter.cpp'


def blob(revision, name):
    return subprocess.check_output(['git', 'show', revision + ':' + name], cwd=P)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def orchestration_input(name, revision):
    raw = blob(revision, name)
    if revision != BASE:
        # Adjacent old additions otherwise form a single merge hunk with the
        # new split parser/final guard. Exclude only these exact old blocks.
        if name == 'src/prefill/prefill.cpp':
            begin = raw.index(b'bool expert_upload_staging_requested() {')
            end = raw.index(b'\n}\n', begin) + 3
            raw = raw[:begin] + raw[end:]
        elif name == 'src/prefill/layer_major.inc':
            begin = raw.index(b'    if (window_context && window_context->expert_upload_staging) {')
            end = raw.index(b'\n    }\n', begin) + 7
            raw = raw[:begin] + raw[end:]
    elif name == 'src/prefill/layer_major.inc':
        # Preserve this independent maximum-allocation safety fix explicitly.
        # It predates split work, but is required for honest capacity admission.
        guard = (b'    if (prefix_bytes > max_alloc) {\n'
            b'        err = "prefill BF16 prefix and metadata exceed maximum single allocation"; return false;\n'
            b'    }\n')
        anchor = b'    const auto bank_plan = detail::ExpertBankPlan::make('
        assert raw.count(anchor) == 1 and guard not in raw
        for version in (BEFORE, AFTER):
            assert blob(version, name).count(guard) == 1
        raw = raw.replace(anchor, guard + anchor)
    return raw


def overlays():
    result = {}
    for name in DIRECT:
        # These paths had no pre-existing difference from the selected source
        # before the split work began. New fixture/controller changes have no
        # bearing on their native content.
        pre_split = blob('0239c34ca', name)
        assert pre_split == blob(BASE, name), 'unselected direct source:' + name
        result[name] = blob(AFTER, name)
    original = blob(BASE, GEMM)
    current = blob(AFTER, GEMM)
    begin = current.index(b'bool Gemm::try_down_scatter16_split(')
    end = current.index(b'} // namespace strata::prefill', begin)
    method = current[begin:end]
    observer = (b'  auto profile_tag = sycl_backend::command_profile::fused_tag(\n'
        b'      sycl_backend::command_profile::Kind::down, sycl_backend::command_profile::Kind::down_scatter);\n')
    assert method.count(observer) == 1
    method = method.replace(observer, b'')
    anchor = b'} // namespace strata::prefill'
    assert original.count(anchor) == 1 and b'try_down_scatter16_split' not in original
    result[GEMM] = original.replace(anchor, method + anchor)
    assert b'fused_tag' not in result[GEMM] and b'STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE' not in result[GEMM]

    # The added profile-engine-hook belongs to a different qualified observer,
    # not this selected-source component/model. Retain only the split fixture.
    fixture = blob(AFTER, FIXTURE)
    start = fixture.index(b'  void check_profile() {')
    end = fixture.index(b'  void separate(', start)
    fixture = fixture[:start] + fixture[end:]
    start = fixture.index(b'      } else if (mode == "--profile-engine-hook") {')
    end = fixture.index(b'      } else if (mode == "--smoke") {', start)
    fixture = fixture[:start] + fixture[end:]
    fixture = fixture.replace(b'#include "strata/sycl/command_profile.hpp"\n', b'')
    assert all(v not in fixture for v in [b'check_profile', b'--profile-engine-hook', b'cp::'])
    result[FIXTURE] = fixture

    # Three-way composition removes the pre-existing unselected observers and
    # upload-staging implementation while applying only the newly authored
    # scheduling/owner edits. A conflict is a preparation failure, never an
    # invitation to accept marker-containing source or refresh a run's hashes.
    with tempfile.TemporaryDirectory(prefix='strata-split-merge-') as folder:
        root = Path(folder)
        for name in ORCHESTRATION:
            paths = []
            for role, revision in [('selected', BASE), ('before', BEFORE), ('after', AFTER)]:
                path = root / role
                path.write_bytes(orchestration_input(name, revision))
                paths.append(str(path))
            command = ['/usr/bin/git', 'merge-file', '-p', '-L', BASE, '-L', BEFORE, '-L', AFTER, *paths]
            merged = subprocess.run(command, cwd=P, capture_output=True, timeout=30)
            if merged.returncode:
                raise RuntimeError('split source composition conflict:' + name + ':' + merged.stderr.decode())
            assert b'<<<<<<<' not in merged.stdout
            result[name] = merged.stdout
    combined = b'\n'.join(result.values())
    assert b'ExpertUploadStaging' not in combined and b'expert_upload_staging_requested' not in combined
    assert b'STRATA_SYCL_FULL_CONTEXT_DIAGNOSTIC' not in result['src/prefill/prefill.cpp']
    assert b'STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR' in result['src/prefill/prefill.cpp']
    assert len(result) == 12
    return result


def proof():
    values = overlays()
    return dict(base=BASE, before=BEFORE, after=AFTER, paths={name: dict(
        before_sha256=sha(blob(BASE, name)), after_sha256=sha(raw)) for name, raw in sorted(values.items())},
        independent_safety_fix='Retain pre-existing prefix_bytes <= max_mem_alloc_size guard in layer_major.inc.',
        excluded='Unselected fused-command observers,fullcontext diagnostic,upload staging and all other active-tree native edits. Decode/program entrypoint remains exact selected6e40.')
