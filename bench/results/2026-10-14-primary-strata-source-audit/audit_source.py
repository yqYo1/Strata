#!/usr/bin/env python3
"""Record pinned source families; no engine, refs, network or GPU changes."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
G = Path('/home/yayoi/ghq/github.com/yqYo1/Strata')
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
R = B / 'research-20261009'
O = P / 'bench/results/2026-10-14-primary-strata-source-audit'

def git(*args, cwd=G):
    return subprocess.check_output(['git', '-C', str(cwd), *args], text=True)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write(name, obj):
    path = O / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def main():
    O.mkdir(parents=True, exist_ok=True)
    worktrees = {}
    for block in git('worktree', 'list', '--porcelain').strip().split('\n\n'):
        row = dict(line.split(' ', 1) for line in block.splitlines() if ' ' in line)
        worktrees[row['branch']] = row
    rows = []
    for line in git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/heads').splitlines():
        ref, sha = line.split()
        wt = worktrees[ref]['worktree']
        cmake = git('show', f'{sha}:CMakeLists.txt')
        version = re.findall(r'project\([^\n]+', cmake)[:1]
        source_map = git('ls-tree', '-r', sha, '--', 'CMakeLists.txt', 'cmake', 'include', 'src', 'serve', 'setup.py')
        status = git('status', '--porcelain', '--untracked-files=all', cwd=wt)
        rows.append({'ref': ref, 'head': sha, 'worktree': wt,
                     'version_marker_only': version,
                     'source_tree_listing_sha256': digest(source_map.encode()),
                     'dirty_files': status.splitlines(),
                     'role': 'primary' if ref == 'refs/heads/feature/sycl' else
                             'fork_default' if ref == 'refs/heads/main' else
                             'Xe-derived reference or unadopted diagnostic proposal'})
    preview = subprocess.run(['git', '-C', str(G), 'merge-tree', '--write-tree', 'feature/sycl', 'main'], text=True, capture_output=True)
    assert preview.returncode == 1
    conflicts = sorted(set(re.findall(r'^CONFLICT .*?Merge conflict in (.+)$', preview.stdout, re.M)))
    write('branches.json', {'schema': 1, 'date': '2026-10-14',
        'sole_upstream': 'Niko1221/Strata', 'reference_only': 'MistVVK/XeStrata',
        'target': ['Intel Arc B570 10 GiB', 'Ryzen 5 5600X', '128 GiB RAM'],
        'primary': 'feature/sycl', 'default': 'main', 'local_branches': rows,
        'remote_tracking_refs_snapshot': git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/remotes/origin').splitlines(),
        'remote_latest_not_checked': True,
        'main_reconciliation_preview': {'exit_code': preview.returncode, 'conflict_paths': conflicts,
            'raw_sha256': digest(preview.stdout.encode()), 'common_ancestor': git('merge-base', 'feature/sycl', 'main').strip(),
            'unique_commit_counts_feature_then_main': git('rev-list', '--left-right', '--count', 'feature/sycl...main').strip(),
            'tree_not_adopted': True}})
    primary = git('rev-parse', 'feature/sycl').strip()
    write('decisions.json', {'primary_source': primary, 'primary_engine_modified': False,
        'personal_optimization_fork': True, 'upstream_contribution_planned': False,
        'CUDA_HIP_implementation_or_measurement_in_scope': False,
        'whole_Xe_tree_adoption': 'rejected; not component-by-component adoption',
        'primary_current_performance': 'not freshly measured; recent Xe-derived samples cannot represent this branch',
        'Niko_non_inference_functions': 'all required; current main and newer pins still need explicit adaptation and tests',
        'inference_policy': 'select each component and prefill/decode role independently; preserve faster qualified fork parts',
        'safe_primary_execution': 'do not run known failure revision before source/runtime/lifetime correction and bounded checks',
        'component_candidates': [
            {'part': 'checked owned-layer bulk transfer', 'origin': '85cb6b1968bd5c2156eed3b10dbb332f92ee95ac', 'adopted': False, 'missing': ['primary-side lease/admission adaptation', 'matched primary on/off comparison', 'full relevant context/state/RESTORE qualification']},
            {'part': 'layer-major and phase4k workspace/cache leases', 'origin': 'db088ff24e67e6a4566566dfc4762c63c55bf343', 'adopted': False, 'missing': ['primary-side ownership integration', 'physical 262144 and near-boundary state execution', 'RESTORE', 'matched model-level validation']},
            {'part': 'IQ3_S CPU paired gate/up', 'adopted': False, 'evidence': 'synthetic effects differ for NT1 and NT8; no whole-model benefit'},
            {'part': 'decode route/service and route-union diagnostics', 'adopted': False, 'qualification': 'host fixtures and reference full builds only', 'missing': ['primary port', 'default-OFF overhead', 'model/formatter execution']}],
        'prior_research_corrections': [
            'R448 reviewed the older bulk README, not the later closed stock-repeat report; quiet reference measurements exist below.',
            'R448 statement that W has no cited measurements is a source-reading limit, not absence of Main-owned closed W measurements.',
            'Historical layer-major measurements must retain their original hardware/recipe; no cross-hardware assertion is established here.',
            'R446 read-only planning confirmation suggestions do not override already-authorized branch/default work.'],
        'next_order': ['preserve unique source/report commits and retire completed experiments',
                       'reconcile fork main functions with retained primary inference and validate',
                       'update fork default through its own PR',
                       'then check current Niko updates under the corrected skill']})
    write('reference-measurements.json', {'source_family': 'XeStrata-derived experiments; not primary feature/sycl',
        'stock_client': {'repository': 'Niko1221/Strata', 'revision': '7695859aa13afb248bdb4d36f9b2e95707db079d',
                         'command': './bench.sh --url http://127.0.0.1:18095 --long --runs 3',
                         'input_tokens': 32049, 'output_tokens': 24, 'reuse': 0,
                         'strict_32768_input_qualified': False},
        'source_report': {'commit': '28a7b9ae7ca5eba28eecd65f6453a7479c04fce8', 'path': 'bench/results/2026-10-14-layer-bulk-stock-repeat'},
        'samples': [
            {'arm': 'bulk', 'source': '85cb6b1968bd5c2156eed3b10dbb332f92ee95ac', 'prefill_tps': [693.5, 691.1, 701.1], 'decode_tps': [10.1, 7.9, 8.1]},
            {'arm': 'phase4k initial nonbulk', 'source': 'db088ff24e67e6a4566566dfc4762c63c55bf343', 'prefill_tps': [654.7, 667.6, 669.8], 'decode_tps': [8.7, 10.4, 11.1]},
            {'arm': 'phase4k repeat nonbulk', 'source': 'db088ff24e67e6a4566566dfc4762c63c55bf343', 'prefill_tps': [666.5, 666.2, 668.0], 'decode_tps': [7.4, 9.5, 8.5]}],
        'faults': 0, 'ordinary_QUIT': 'passed', 'pre_and_post_health': 'passed',
        'adoption': 'held; control decode also fell; no primary matched A/B',
        'full_context_physical_store_RESTORE_whole_model_quality': 'not qualified'})
    research = []
    for round_no in range(441, 449):
        matches = list(R.glob(f'*-round{round_no}-v1.txt'))
        assert len(matches) == 1, (round_no, matches)
        src = matches[0]; raw = src.read_bytes()
        dst = 'research/' + src.name + '.json'
        write(dst, {'source_path': str(src), 'sha256': digest(raw), 'exact_text': raw.decode(), 'role': 'read-only proposal; Main decisions are in decisions.json'})
        research.append({'round': round_no, 'path': dst, 'sha256': digest(raw)})
    write('research-index.json', {'previous_registry_consumer': '28a7b9ae7ca5eba28eecd65f6453a7479c04fce8:bench/results/2026-10-14-layer-bulk-stock-repeat',
                                  'reports': research, 'reports_are_not_tests': True})
    for name in ['main-decode-service-full-build-root-v1', 'main-decode-service-host-qualification-root-v1', 'main-route-union-host-qualification-root-v1', 'main-primary-branch-policy-publication-root-v1']:
        path = B / name / 'record.json'; raw = path.read_bytes(); d = json.loads(raw)
        commands = []
        for c in d.get('commands', []):
            keep = ['label', 'argv', 'started_utc', 'finished_utc', 'exit_code', 'elapsed_seconds',
                    'ownership_status', 'direct_identity', 'observation_complete', 'direct_child_reaped',
                    'session_empty', 'session_anchored', 'survivors', 'errors', 'logs']
            cc = {k: c[k] for k in keep if k in c}
            if 'owners' in c:
                cc['owner_count'] = len(c['owners'])
                cc['owners_sha256'] = digest(json.dumps(c['owners'], sort_keys=True).encode())
            cc['environment_sha256'] = digest(json.dumps(c.get('environment'), sort_keys=True).encode())
            if c.get('environment') != d.get('environment'):
                cc['environment'] = c.get('environment')
            commands.append(cc)
        compact = {k: v for k, v in d.items() if k not in ['commands', 'evidence', 'traceback'] and not isinstance(v, (list, dict))}
        compact.update({'original_record_path': str(path), 'original_sha256': digest(raw), 'original_bytes': len(raw),
                        'commands_without_duplicate_full_output': commands, 'environment': d.get('environment'),
                        'source_pins': d.get('source_pins'), 'qualification_transfers_to_primary': False})
        for k in ['host_fixture_proof', 'free', 'contrib-llvm', 'phase-host', 'phase-host-asan',
                  'host', 'host-asan-ubsan', 'decode_service_budget-host',
                  'decode_service_budget-host-asan-ubsan', 'decode_route_ledger-host',
                  'decode_route_ledger-host-asan-ubsan']:
            if k in d:
                compact[k] = d[k]
        write('receipts/' + name + '.json', compact)
    def options(sha):
        text = git('show', f'{sha}:src/program/generate.cpp')
        result = {}
        for lineno, line in enumerate(text.splitlines(), 1):
            for option in re.findall(r'"(--[A-Za-z0-9_-]+)"', line):
                result.setdefault(option, []).append(lineno)
        return result
    fo, mo = options(primary), options('main')
    write('cli-option-inventory.json', {
        'method': 'exact quoted --option literals and source lines, not a claim of semantic parity',
        'primary_revision': primary, 'main_revision': git('rev-parse', 'main').strip(),
        'primary_options': fo, 'main_options': mo,
        'main_only_literals': sorted(set(mo) - set(fo)),
        'primary_only_literals': sorted(set(fo) - set(mo)),
        'status': 'map each non-inference behavior and adapt engine-facing dependencies before adoption'})
    parts = {
        'runtime_and_lifetime': ['src/sycl/runtime.cpp', 'src/sycl/device.cpp', 'include/strata/sycl/runtime.hpp'],
        'prefill_scheduler': ['src/prefill/prefill.cpp', 'src/prefill/sycl/moe_mmq.cpp', 'src/prefill/sycl/moe_fused.cpp'],
        'expert_source_cache': ['src/core/expert_source.cpp', 'src/core/expert_cache.cpp', 'include/strata/core/expert_source.hpp', 'include/strata/core/expert_cache.hpp'],
        'CPU_experts': ['src/kernels/cpu/native_expert.cpp', 'src/kernels/cpu/iq_single_avx2.cpp', 'src/kernels/cpu/pool.cpp'],
        'SYCL_MoE': ['src/kernels/sycl/native_iq_esimd.cpp', 'src/kernels/sycl/native_mmq_xmx.cpp', 'src/kernels/sycl/expert_routing.cpp'],
        'decode_state_attention': ['src/core/layer.cpp', 'src/core/session.cpp', 'src/kernels/sycl/decode_attention.cpp', 'src/kernels/sycl/native_qsa_indexer.cpp'],
        'engine_CLI_protocol': ['src/program/generate.cpp'],
        'server_setup_UI': ['serve/server.py', 'serve/frontend.py', 'setup.py', 'serve/web/app.js']}
    pinned = {}
    for group, paths in parts.items():
        pinned[group] = []
        for path in paths:
            data = subprocess.check_output(['git', '-C', str(G), 'show', f'{primary}:{path}'])
            assert (P / path).read_bytes() == data, f'Unexpected source change: {path}'
            pinned[group].append({'path': path, 'sha256': digest(data), 'bytes': len(data)})
    write('primary-component-source-pins.json', {'revision': primary, 'groups': pinned,
        'all_listed_source_unchanged': True,
        'R448_path_correction': 'primary SYCL prefill files are src/prefill/sycl/*.cpp; no src/prefill/moe_mmq.cpp exists here'})
    (O / 'audit_source.py').write_bytes(Path(__file__).read_bytes())
    write('controller.json', {'path': str(Path(__file__)), 'committed_source': 'audit_source.py', 'sha256': digest(Path(__file__).read_bytes()),
        'command': ['python3', str(Path(__file__))], 'network_GPU_build_test_ref_mutation': False})

if __name__ == '__main__':
    main()
