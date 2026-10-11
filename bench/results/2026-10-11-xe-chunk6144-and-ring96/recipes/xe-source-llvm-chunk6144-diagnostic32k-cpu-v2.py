import ast
import fcntl
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C = B / 'xe-source-llvm-chunk6144-diagnostic32k-v2'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-source-llvm-chunk6144-diagnostic32k-cpu-root-v2'
    m.require(not out.exists(), 'immutable output')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             cpu_only=True, gpu_executed=False, model_executed=False, controller_sha256=m.sha(__file__),
             scope='AST and actual --help import of new diagnostic controller; unchanged component fixtures reused by exact bytes and closed actual receipts')
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    try:
        names = ['run_root.py', 'hash_inputs.py', 'protocol.py', 'runtime.py',
                 'transfer_ledger.py', 'timer_overhead.py', 'selfcheck_driver.py']
        r['source_pins'] = {n:m.sha(C/n) for n in names}
        for n in names:
            ast.parse((C/n).read_text(), filename=str(C/n))
        r['ast_parse_passed'] = True
        for label, p in [('original', B/'xe-source-llvm-quiet32k-cpu-root-v2/record.json'),
                         ('diagnostic', B/'xe-prefill-diagnostic-observer-cpu-root-v1/record.json')]:
            v = json.loads(p.read_text())
            m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']), 'closed actual component fixtures')
            for n,h in v['source_pins'].items():
                if label == 'original' and n == 'run_root.py':
                    continue
                m.require(m.sha(C/n) == h, 'unchanged qualified helper ' + n)
            r[label+'_qualification_sha256'] = m.sha(p)
        e, so, se = owner.run('controller-help', ['/usr/bin/python3', '-B', str(C/'run_root.py'), '--help'],
                             ENV, C, wall=10, cpu=10, text_cap=1 << 20, total_cap=2 << 20, rss_cap=2 << 30)
        m.require(m.closed(e) and e['exit_code'] == 0 and '--kind {counts,timer}' in so.read_text(), 'normal closed CPU controller import/help')
        m.require(all(m.sha(C/n) == h for n,h in r['source_pins'].items()), 'source stable')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','ast_parse_passed')}))
    raise SystemExit(0 if r['passed'] else 1)
