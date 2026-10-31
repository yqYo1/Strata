#!/usr/bin/env python3
"""Compose one official-SDK model candidate from pinned, selected sources.

Main-authored; repository LICENSE. No XeStrata input. Never take the whole
working tree: it also contains experiments that have not passed adoption gates.
This constructor does not compile, execute a GPU kernel or select inference.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import build_engine as build
import moe_down_sources as down
import moe_split_sources as split

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ORIGIN = B / 'main-native-gdn-engine/record.json'
ORIGIN_SHA = 'a38f477670fafcd1a931be2e20d90d6d9718df75b61954b421f361af9f6881b2'
DOWN = '9a79cd3520d8f39b7ad026cd64b51a32b505f2a2'
TOPK = 'dc19a1ecb4953e1c9f1c6cf9715a823bc3b68291'
FIXTURE = 'd5d767eb16a1857e98bef32deb74ceab892c4ac7'
BASE_AGGREGATE = 'ae8b2082ab2f7284221f7805de10afceb7e6291e9bceb91272ac3b227d740f83'
CONFIGURATION = ['tools/sycl/' + name + '.cpp' for name in (
    'host_transfer_probe', 'kv_copy_groups_host', 'kv_copy_groups_bench',
    'gemm_orientation_bench', 'projection_bench', 'moe_bench', 'gr_bench',
    'expert_contract', 'gemm_device_probe')]
TOPK_FILES = ('src/kernels/sycl/qsa_topk_prefill_early.cpp',
              'include/strata/kernels/qsa_topk_prefill_early.hpp')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def aggregate(values):
    return hashlib.sha256(json.dumps(
        {n: hashlib.sha256(raw).hexdigest() for n, raw in values.items()},
        sort_keys=True).encode()).hexdigest()


def original():
    require(build.digest(ORIGIN) == ORIGIN_SHA, 'original selected receipt changed')
    full = json.loads(ORIGIN.read_text())
    require(full['source_commit'] == split.BASE and full['project']['count'] == 542,
            'unexpected selected source identity')
    expected = full['project']['source_sha256']
    values = dict(build.blobs(P, split.BASE, [*expected, *CONFIGURATION]))
    require(all(hashlib.sha256(values[n]).hexdigest() == sha for n, sha in expected.items()),
            'selected Git sources differ from original build inputs')
    values.update(down.overlays(DOWN))
    require(len(values) == 551 and aggregate(values) == BASE_AGGREGATE,
            'qualified split/Down construction changed')
    return full, values


def merge_delta(selected, name, before, after):
    """Apply one pinned source change, with no conflict/fuzz acceptance."""
    with tempfile.TemporaryDirectory(prefix='strata-official-compose-') as folder:
        paths = [Path(folder) / role for role in ('selected', 'before', 'after')]
        for path, raw in zip(paths, (selected, split.blob(before, name), split.blob(after, name))):
            path.write_bytes(raw)
        out = subprocess.run(['/usr/bin/git', 'merge-file', '-p', *map(str, paths)],
                             capture_output=True, timeout=30, cwd=P)
    require(out.returncode == 0 and b'<<<<<<<' not in out.stdout,
            'source composition conflict: ' + name)
    return out.stdout


def once(raw, old, new):
    require(raw.count(old) == 1, 'ambiguous source seam: ' + old[:100].decode())
    return raw.replace(old, new)


def sources():
    full, values = original()
    prefill = 'src/prefill/prefill.cpp'
    original_prefill = values[prefill]
    values[prefill] = merge_delta(original_prefill, prefill, TOPK + '^', TOPK)
    require(merge_delta(values[prefill], prefill, TOPK, TOPK + '^') == original_prefill,
            'Top-k route composition changes another prefill region')
    for name in TOPK_FILES:
        values[name] = split.blob(TOPK, name)
    name = 'tests/sycl/qsa_topk_prefill_entry.cpp'
    values[name] = split.blob(FIXTURE, name)

    cmake = 'cmake/sycl_backend.cmake'
    original_cmake = values[cmake]
    values[cmake] = merge_delta(original_cmake, cmake, TOPK + '^', TOPK)
    require(merge_delta(values[cmake], cmake, TOPK, TOPK + '^') == original_cmake,
            'Top-k CMake composition changes another region')
    # Only the maintained entry fixture's target is added from its later move.
    target = split.blob(FIXTURE, cmake)
    first = target.index(b'  # Supervised manual fixture:')
    last = target.index(b'  add_executable(sycl_qsa_score_prefill_test', first)
    anchor = b'  add_executable(sycl_qsa_score_prefill_test'
    values[cmake] = once(values[cmake], anchor, target[first:last] + anchor)

    # Official Intel installs the bundler below bin/compiler. Query the chosen
    # compiler, preserving the existing separate native image/host link model.
    name = 'cmake/sycl_vllm_xe2_native16.cmake'
    later = split.blob(FIXTURE, name)
    first = later.index(b'# Installed Intel SDKs place the bundler')
    last = later.index(b'execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -dumpmachine', first)
    values[name] = once(values[name],
        b'get_filename_component(_strata_sycl_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)\n',
        later[first:last])
    values[name] = once(values[name], b'"${_strata_sycl_bin}/clang-offload-bundler"',
                        b'"${_strata_offload_bundler}"')
    require(values['src/program/generate.cpp'] == split.blob(split.BASE, 'src/program/generate.cpp'),
            'shared phase controller must remain the selected implementation')
    require(values['src/kernels/sycl/qsa_select.cpp'] ==
            split.blob(split.BASE, 'src/kernels/sycl/qsa_select.cpp'),
            'existing decode/short-prefill selector changed')
    require(b'ExpertUploadStaging' not in values[prefill] and
            b'STRATA_SYCL_FULL_CONTEXT_DIAGNOSTIC' not in values[prefill],
            'unselected experiments entered official model source')
    require(len(values) == 554, 'unexpected model source count')
    return full, values


if __name__ == '__main__':
    _, values = sources()
    print(json.dumps({'source_files': len(values), 'aggregate_sha256': aggregate(values),
                      'build_or_GPU_executed': False, 'inference_adopted': False}))
