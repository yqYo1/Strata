"""Finalize an already executed deletion plan; never retry file deletion."""
from pathlib import Path
import ast
import datetime
import fcntl
import hashlib
import json
import os
import stat
import subprocess

B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A = W / 'bench/results/2026-10-09-sycl-layer-major-lease-capacity'
plan_path = A / 'closed-T-artifact-retirement-plan-v1.json'
result_path = A / 'closed-T-artifact-retirement-result-v2.json'
lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not result_path.exists()
plan = json.loads(plan_path.read_text())
original = B / 'retire_closed_T_artifacts_v1.py'
assert hashlib.sha256(original.read_bytes()).hexdigest() == plan['script_sha256']
# Reuse the original read-only identity/result guards; do not run its deletion body.
tree = ast.parse(original.read_text())
constants = {'B', 'W', 'A', 'T', 'G', 'T_SHA', 'G_SHA'}
helpers = {'digest', 'same_process', 'guards'}
nodes = [node for node in tree.body
         if isinstance(node, (ast.Import, ast.ImportFrom))
         or (isinstance(node, ast.Assign) and len(node.targets) == 1
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants)
         or (isinstance(node, ast.FunctionDef) and node.name in helpers)]
namespace = {'__file__': str(original)}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(original), 'exec'), namespace)
namespace['guards']()
for row in plan['files']:
    assert not os.path.lexists(row['path']), row['path']
for row in [*plan['retained'], {'path': str(plan_path), 'sha256': namespace['digest'](plan_path)}]:
    p = Path(row['path'])
    data = subprocess.check_output(['git', '-C', str(W), 'show', 'HEAD:' + str(p.relative_to(W))])
    assert hashlib.sha256(data).hexdigest() == row['sha256'] == namespace['digest'](p)
experiment = B / 'gprofng-software-protocol-smoke-v1/result.er'
entries = list(experiment.iterdir())
assert len(entries) == 1 and entries[0].name == 'archives'
empty = entries[0]
assert stat.S_ISDIR(empty.lstat().st_mode) and not list(empty.iterdir())
empty.rmdir()
experiment.rmdir()
assert (B / 'owned-upstream-v0141-integrated-full256k-diagnostic-r2/control32k.session.bin').is_file()
assert (B / 'owned-layer-major-streamed-kv-v0141-full256k-streamed-lease-diagnostic-r1/numerical-failure.session.bin').is_file()
space = os.statvfs(B)
result = {'completed': True, 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'plan_sha256': namespace['digest'](plan_path), 'finalizer_sha256': namespace['digest'](Path(__file__)),
          'original_execution': {'script_sha256': plan['script_sha256'], 'session': 81144,
                                 'exit_code': 1, 'error': 'OSError ENOTEMPTY at result.er.rmdir after every planned file unlink; gprofng created an empty archives subdirectory.',
                                 'all_planned_files_removed_before_error': True},
          'recovery': 'Verified absent planned paths and original read-only guards; removed only the known empty archives and result.er directories. No file deletion retried.',
          'removed_files': len(plan['files']), 'removed_logical_bytes': plan['logical_bytes'],
          'removed_file_allocation_bytes': plan['allocated_bytes'],
          'filesystem_available_bytes_before': None,
          'filesystem_available_bytes_after': space.f_bavail * space.f_frsize,
          'available_space_delta_measured': False,
          'all_planned_paths_absent': True, 'original_receipts_unchanged': True,
          'canonical_baseline_and_C_failure_captures_retained': True,
          'gpu_executed': False, 'model_opened': False, 'source_models_service_GPU_state_changed': False,
          'note': 'Before-space counter was not persisted by the failed wrapper. Removed file allocation is not a measured filesystem free-space gain; snapshots may retain extents.'}
result_path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
