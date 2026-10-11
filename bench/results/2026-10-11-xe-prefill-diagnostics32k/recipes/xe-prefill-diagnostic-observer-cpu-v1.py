import ast
import fcntl
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C = B / 'xe-prefill-diagnostic-observer-source-v1'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-prefill-diagnostic-observer-cpu-root-v1'
    m.require(not out.exists(), 'immutable output')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], cpu_only=True,
             gpu_executed=False, model_executed=False, started_utc=m.utc(),
             controller_sha256=m.sha(__file__), owner_sha256=m.sha(O),
             scope='AST and finite diagnostic grammar fixtures only; no Runtime/kernel/model qualification')
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    try:
        review = B / 'xe-prefill-diagnostic-observer-main-review-v1.json'
        v = json.loads(review.read_text())
        m.require(v['main_reviewed'] and v['full_source_report_manifest_read'], 'full Main review')
        r['review_sha256'] = m.sha(review)
        r['source_pins'] = v['source_pins']
        for name, digest in r['source_pins'].items():
            m.require(m.sha(C / name) == digest, 'unchanged reviewed parser')
            ast.parse((C / name).read_text(), filename=str(C / name))
        r['ast_parse_passed'] = True
        e, so, se = owner.run('finite-parser-fixtures', ['/usr/bin/python3', '-B', str(C / 'selfcheck_driver.py')],
                             ENV, C, wall=10, cpu=10, text_cap=1 << 20, total_cap=2 << 20, rss_cap=2 << 30)
        m.require(m.closed(e) and e['exit_code'] == 0, 'normal closed fixtures')
        r['fixture_result'] = json.loads(so.read_text())
        m.require(r['fixture_result']['synthetic_parser_positive_cases'] > 0 and
                  r['fixture_result']['synthetic_parser_negative_cases'] > 0, 'completed positive/rejection cases')
        m.require(all(m.sha(C / n) == h for n, h in r['source_pins'].items()), 'source stable')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'fixture_result')}))
    raise SystemExit(0 if r['passed'] else 1)
