#!/usr/bin/env python3
"""Read-only source/ref audit; write a compact ledger in this report directory."""
import hashlib
import json
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
D = Path(__file__).parent
BASE = '4fa478f50f8da396e651e3ae862f8b1849d49b54'
MAIN = 'beddab2fd6b61631d73b39b7484bc9a585ebe1de'
Z = '28a7b9ae7ca5eba28eecd65f6453a7479c04fce8'
SKILL = '97f8b40ca99ec298c24d7ecc7b270e530d43194d'


def git(*args):
    return subprocess.check_output(['git', '-C', str(P), *args], text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    head = git('rev-parse', 'HEAD')
    paths = ['CMakeLists.txt', 'cmake/sycl_backend.cmake', 'src/core/session.cpp',
        'src/core/expert_cache.cpp', 'src/core/expert_source.cpp', 'src/core/verify.cpp',
        'src/prefill/prefill.cpp', 'src/prefill/sycl/gemm.cpp',
        'src/prefill/sycl/moe_mmq.cpp', 'src/prefill/sycl/moe_fused.cpp',
        'src/kernels/cpu/native_expert.cpp', 'src/kernels/cpu/pool.cpp',
        'src/kernels/sycl/native_mmq_xmx.cpp', 'src/program/generate.cpp']
    revisions = dict(primary=head, primary_inference_base=BASE, fork_default=MAIN,
        retired_Xe_reference=Z, retired_wrong_base_skill=SKILL,
        Niko_base_v0138=git('rev-parse', 'v0.1.38'), Niko_main_v0139=git('rev-parse', 'v0.1.39'))
    blobs = {}
    for label, rev in revisions.items():
        blobs[label] = {}
        for path in paths:
            check = subprocess.run(['git', '-C', str(P), 'cat-file', '-e', f'{rev}:{path}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            blobs[label][path] = git('rev-parse', f'{rev}:{path}') if check.returncode == 0 else None
    archive = []
    for rev in ['8927eb6991f01f8dcca34aa2f0b31fb2549923e9',
                '4526800257fe49c0db02a414f21b38dac2586e54']:
        parents = git('show', '-s', '--format=%P', rev).split()
        source_diff = git('diff', '--name-only', parents[0], rev, '--',
            'src', 'include', 'cmake', 'tests', 'tools', 'setup.py', 'serve', 'CMakeLists.txt')
        archive.append(dict(commit=rev, first_parent=parents[0], parent_count=len(parents),
            product_diff_against_first_parent=source_diff.splitlines(), code_adoption=bool(source_diff)))
        assert not source_diff
    identity = json.loads(subprocess.check_output(['gh', 'api', 'repos/yqYo1/Strata', '--jq',
        '{full_name,default_branch,parent:.parent.full_name,source:.source.full_name}'], text=True))
    remote = json.loads(subprocess.check_output(['gh', 'api', '--paginate', 'repos/yqYo1/Strata/branches'], text=True))
    record = dict(scope='source/Git/remote identity audit, not performance or latest Niko integration',
        sole_upstream='Niko1221/Strata', personal_hardware_fork=True,
        upstream_contribution_planned=False, CUDA_HIP_inference_in_scope=False,
        Xe_role='optional benchmark and component reference', revisions=revisions,
        trees={label: git('rev-parse', rev + '^{tree}') for label, rev in revisions.items()},
        representative_blobs=blobs, archive_merges=archive,
        local_refs=git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/heads', 'refs/remotes').splitlines(),
        worktrees=git('worktree', 'list', '--porcelain'), GitHub_identity=identity,
        GitHub_branches=[dict(name=x['name'], sha=x['commit']['sha']) for x in remote],
        HEAD_status=git('status', '--short', '--branch'),
        working_source_sha256={path: sha(P / path) for path in paths},
        HEAD_inference_delta_from_base=git('diff', '--name-status', BASE, head, '--',
            'src', 'include', 'cmake', 'CMakeLists.txt').splitlines(),
        required_function_gaps=['active-Arc dynamic VRAM lease/resize adaptation',
            'shared Niko CLI/API functionality ledger and native capability contracts',
            'install/update and model continuation/RESTORE/context boundary qualification'],
        optional_unadopted_Xe_candidates=['layer-major scheduler with owned bulk upload',
            'paired IQ3_S GU activation-load sharing', 'additional route/service diagnostics'],
        fresh_primary_performance_qualified=False, physical_262144_qualified=False,
        default_promotion_complete=False, fresh_Niko_update_evaluation_complete=False)
    assert identity['parent'] == identity['source'] == record['sole_upstream']
    assert {x['name'] for x in remote} == {'main', 'feature/sycl'}
    assert len(git('for-each-ref', '--format=%(refname)', 'refs/heads').splitlines()) == 2
    (D / 'primary-audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(identity=identity, branches=2, product_archive_diffs_empty=True)), flush=True)


if __name__ == '__main__':
    main()
