"""Exercise new capacity supervision with a real CPU child/GDB/PTY only."""
from pathlib import Path
import ast
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = base / 'full-layer-trace-serve-cpu-final-four'
out.mkdir(mode=0o700)
controller = base / 'run_full_layer_trace_serve.py'
previous = root / 'bench/results/2026-10-05-sycl-prefill-scaling/residual-inplace/check_full_context_controller.py'
tree = ast.parse(previous.read_text())
stub = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'STUB' for t in n.targets))
stub = stub.replace('STRATA_CONTROLLER_STUB_MODE', 'CAPACITY_STUB_MODE')
needle = '    windows = [(n - 1, 1)]\n'
assert stub.count(needle) == 1
stub = stub.replace(needle, '    print("strata mtp decode release: CPU stub, verified=1", file=sys.stderr, flush=True)\n'
                         '    print("strata mtp decode restore: CPU stub, verified=1", file=sys.stderr, flush=True)\n' + needle)
fake_helper = '''#!/bin/sh
python3 - <<'PY'
import re,os
FAULT=re.compile('fault|reset|hang',re.I)
class Runner:
 def __init__(self,path):self.calls=[]
 def run(self,label,argv,**kwargs):
  self.calls.append({'label':label,'argv':argv,'scope':'CPU fake host operation'})
  return 'ActiveState=inactive' if label=='embedding-state' else ''
def diagnostic_environment(env):return dict(env,UR_ENABLE_LAYERS='UR_LAYER_TRACING')
def journal_cursor(runner,label):
 runner.run(label,['CPU fake cursor'])
 return 'cpu-only-cursor'
PY
'''
record = {'scope': 'CPU stub/GDB/PTY only; no model, GPU, kernel-journal or service operation',
          'controller_sha256': hashlib.sha256(controller.read_bytes()).hexdigest(),
          'stub_sha256': hashlib.sha256(stub.encode()).hexdigest(), 'checks': [], 'completed': False}
with tempfile.TemporaryDirectory(prefix='strata-capacity-cpu-') as scratch:
    scratch = Path(scratch)
    fake_root = scratch / 'root'
    tools = fake_root / 'sycl/tools'
    tools.mkdir(parents=True)
    (tools / 'recover-xe.sh').write_text(fake_helper)
    (tools / 'owned_gdb.py').symlink_to(root / 'sycl/tools/owned_gdb.py')
    for name in ['program/generate.cpp', 'prefill/prefill.cpp']:
        path = fake_root / 'sycl/src' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('CPU test source sentinel\n')
    # The first six cases passed in fixed-fixture/record.json. Repeat the
    # oversized QUIT case with sufficient drain time and run the three pending
    # bounded-cleanup cases, keeping the earlier failed receipt intact.
    record['preceding_checks'] = str(base / 'full-layer-trace-serve-cpu-fixed-fixture/record.json')
    modes = [('quit-writes', True), ('ignores-quit', False), ('no-ready', False), ('does-not-read', False)]
    for mode, expected in modes:
        case = scratch / mode
        case.mkdir()
        exe = case / 'strata-prefill-layer-trace-candidate'
        exe.write_text(stub)
        exe.chmod(0o700)
        cli = case / 'full-context-layer-trace-csr'
        cli.mkdir()
        (cli / 'record.json').write_text(json.dumps({
            'scope': 'CPU fake prerequisite', 'completed': True, 'healthy': True, 'active': False,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'inferior': {'pid': 1000000000}, 'debugger': {'pid': 1000000001},
            'cleanup': {'forced': False, 'inferior_survived': False, 'gdb_survived': False},
            'new_fault_messages': [], 'last_executed_kv_cell': 262143,
            'verify_windows': [[262141, 1], [262142, 2]],
            'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
            'head_sha256': hashlib.sha256(bytes(248320*4)).hexdigest(), 'ids': [100,101],
            'steps': [{'label': 'kernel-after', 'argv': ['CPU fake', '--after-cursor', 'cpu-only-cursor']}]}))
        fixtures = case / 'full-context-copy-off'
        fixtures.mkdir()
        (fixtures / 'reference.json').write_text(json.dumps({'runs': [{'args': ['--pack','unused','--native','unused','--expert-profile','unused']}]}))
        (fixtures / 'coding-context-256k-tokens.txt').write_text('123 ' * 262144)
        (fixtures / 'environment.json').write_text(json.dumps({'PATH': '/usr/bin:/bin', 'CAPACITY_STUB_MODE': mode}))
        summary = fixtures / 'cli-262144-strata-residency-candidate-draft-lease-verified'
        summary.mkdir()
        (summary / 'summary.json').write_text(json.dumps({'env': {'STRATA_TRACE': '1'}}))
        (case / 'read-csr.gdb').write_text('# CPU fixture; no GPU CSR exists\n')
        code = controller.read_text()
        old_root = "root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')"
        assert code.count(old_root) == 1
        code = code.replace(old_root, 'root = Path(' + repr(str(fake_root)) + ')')
        code = code.replace("'scope': 'Actual 262144-cell", "'scope': 'CPU stub 262144-cell")
        reply_deadline = 1 if mode == 'no-ready' else 5
        write_deadline = .5 if mode == 'does-not-read' else 5
        code = code.replace("'protocol_timeout_seconds': 3600", "'protocol_timeout_seconds': " + str(reply_deadline))
        code = code.replace("'write_timeout_seconds': 30", "'write_timeout_seconds': " + str(write_deadline))
        shutdown_deadline = 10 if mode == 'quit-writes' else .5
        code = code.replace("'shutdown_timeout_seconds': 30", "'shutdown_timeout_seconds': " + str(shutdown_deadline))
        target = case / 'controller.py'
        target.write_text(code)
        run = subprocess.run(['/usr/bin/python3', str(target)], capture_output=True, timeout=60)
        result = json.loads((case / 'full-context-layer-trace-serve/record.json').read_text())
        observed = run.returncode == 0 and result['completed'] and result['healthy']
        receipt = out / mode
        shutil.copytree(case / 'full-context-layer-trace-serve', receipt)
        for head in receipt.glob('*.bin'):
            head.unlink()
        (receipt / 'controller.stdout').write_bytes(run.stdout)
        (receipt / 'controller.stderr').write_bytes(run.stderr)
        (receipt / 'controller.py').write_text(code)
        assert observed == expected, (mode,run.returncode,result.get('error'),run.stderr[-1000:])
        assert not result.get('cleanup',{}).get('inferior_survived')
        assert not result.get('cleanup',{}).get('gdb_survived')
        if expected:
            assert len(result['requests']) == 5
            assert result['requests'][1]['verify_windows'] == [[262141,1],[262142,2]]
            assert result['requests'][1]['last_executed_kv_cell'] == 262143
            assert result['requests'][-1]['name'] == 'works-after-refusal'
            if mode == 'quit-writes':
                lines = (case / 'full-context-layer-trace-serve/protocol.stdout.raw').read_bytes().splitlines()
                assert lines[-8:] == [b'x'*262144]*8, 'QUIT tail was truncated'
        record['checks'].append({'mode': mode, 'expected': expected, 'observed': observed,
                                 'exit_code': run.returncode, 'error': result.get('error'),
                                 'cleanup': result.get('cleanup')})
        (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
record['completed'] = True
(out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
