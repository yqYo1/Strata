import ast
import fcntl
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C = B / 'xe-source-llvm-profile32k-v1'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8')

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-source-llvm-profile32k-cpu-root-v1'
    m.require(not out.exists(), 'immutable output')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(), controller_sha256=m.sha(__file__), owner_sha256=m.sha(O), cpu_only=True, gpu_executed=False, model_executed=False, scope='Actual pure protocol/hash helper selfchecks and AST of new32K profiling controller; no Runtime, GPU or process-fault injection qualification')

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()

    def run(label, args):
        e, so, se = owner.run(label, args, ENV, C, wall=10, cpu=10, text_cap=1 << 20, total_cap=2 << 20, rss_cap=2 << 30)
        m.require(m.closed(e) and e['exit_code'] == 0, 'normal closed CPU ' + label)
        return so.read_text()

    try:
        names = ['run_root.py', 'hash_inputs.py', 'protocol.py']
        r['source_pins'] = {name: m.sha(C / name) for name in names}
        for name in names:
            ast.parse((C / name).read_text(), filename=str(C / name))
        m.require('PENDING_MAIN_QUALIFIED_COMMIT' not in (C / 'run_root.py').read_text(), 'actual source commit bound before helper qualification')
        r['ast_parse_passed'] = True
        r['protocol_result'] = ast.literal_eval(run('protocol', ['/usr/bin/python3', str(C / 'protocol.py')]).strip())
        hashes = run('hash', ['/usr/bin/python3', str(C / 'hash_inputs.py'), str(C / 'protocol.py'), str(C / 'run_root.py')])
        rows = [line[len('IDENTITIES '):] for line in hashes.splitlines() if line.startswith('IDENTITIES ')]
        m.require(len(rows) == 1, 'one completed full identity result')
        r['hash_result'] = json.loads(rows[0])
        m.require(len(r['hash_result']) == 2, 'two actual full hashed files')
        for row in r['hash_result']:
            m.require(row['sha256'] == m.sha(Path(row['path'])), 'independent helper full hash matches')
        r['source_stable'] = all(m.sha(C / name) == h for name, h in r['source_pins'].items())
        m.require(r['source_stable'], 'helper/controller source stable')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ['passed', 'error', 'protocol_result', 'ast_parse_passed']}))
    raise SystemExit(0 if r['passed'] else 1)
