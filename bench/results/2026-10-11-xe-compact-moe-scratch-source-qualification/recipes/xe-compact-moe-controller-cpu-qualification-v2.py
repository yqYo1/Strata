"""Main: qualify the reviewed source-v2 controller without model/GPU execution."""
import fcntl
import hashlib
import importlib.util
import json
import time
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
H = B / 'xe-compact-moe-model-controller-source-v2'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
with O.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('cpu_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-compact-moe-model-controller-cpu-root-v2'
    m.require(not out.exists(), 'immutable CPU controller output')
    out.mkdir()
    review_path = B / 'xe-compact-moe-model-controller-main-review-v1.json'
    review = json.loads(review_path.read_text())
    pins = {p.name:m.sha(p) for p in sorted(H.glob('*.py'))}
    m.require(review['main_reviewed'] and review['source_pins'] == pins and len(pins) == 11, 'Main exact controller source review')
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(), source_pins=pins,
             owner_sha256=m.sha(O), main_review_sha256=m.sha(review_path), recipe_sha256=m.sha(__file__),
             original_failed_qualification=dict(path=str(B/'xe-compact-moe-model-controller-cpu-root-v1/record.json'), sha256=m.sha(B/'xe-compact-moe-model-controller-cpu-root-v1/record.json'), original_status='FAILED', correction='Outer RLIMIT_FSIZE now64MiB matches immutable nested owner defaults; raw log8MiB and total artifact16MiB budgets unchanged; controller bytes unchanged'),
             gpu_executed=False, model_executed=False, full262144_qualified=False, restore_qualified=False,
             broad_quality_qualified=False, performance_eligible=False, adopted=False, raw_log_budget_bytes=8 << 20,
             scope='Actual source syntax/helper selftests, exported prerequisite/sequence/output/QUIT fixtures and real CLI refusals only')

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    begin = time.monotonic()

    def run(label, argv):
        left = 180 - (time.monotonic() - begin)
        m.require(left > 0, 'whole CPU qualification deadline')
        c, so, se = owner.run(label, argv, ENV, H, wall=min(90, left), cpu=180,
                              text_cap=8 << 20, total_cap=16 << 20, rss_cap=1 << 30, file_cap=64 << 20)
        r.setdefault('command_evidence', {})[label] = dict(stdout=so.read_text(), stderr=se.read_text())
        save()
        m.require(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] == 0, 'normal anchored CPU closure ' + label)
        return so.read_text()

    save()
    try:
        syntax = run('syntax-and-helper-selftests', ['/usr/bin/python3', '-c',
            "from pathlib import Path\nimport protocol,runtime\nfiles=sorted(Path('.').glob('*.py'))\nassert len(files)==11\nfor p in files:compile(p.read_text(),str(p),'exec')\nprotocol.selftest();runtime.selftest()\nprint('11 sources compile; unchanged protocol/runtime selftests PASS')\n"])
        fixture = run('exported-controller-fixtures', ['/usr/bin/python3', str(H / 'controller_cpu_fixtures.py'),
            '--output', str(out / 'exported-fixtures'), '--lint-receipt', str(B / 'xe-compact-moe-lint-qualification-root-v2/record.json')])
        value = json.loads(fixture)
        m.require(value['passed'] and value['source_pins'] == pins and value['owner_sha256'] == m.sha(O) and
                  not value['gpu_executed'] and not value['model_executed'], 'actual same-source exported CPU fixture output')
        m.require(len(value['owned_quit_cases']) == 4 and all(m.closed(x['command']) and
                  x['command'].get('session_anchored') is True and x['command']['exit_code'] == 0 for x in value['owned_quit_cases']), 'actual four CPU protocol children normally closed')
        r['exported_fixture_result'] = value
        r['prerequisite_fixture_receipt_pins'] = value['prerequisite_fixture_receipt_pins']
        cli_recipe = out / 'negative_cli.py'
        cli_recipe.write_text('''import hashlib,importlib.util,json
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
H=B/'xe-compact-moe-model-controller-source-v2'
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
with O.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
s=importlib.util.spec_from_file_location('cli_owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
owner=m.Owner(Path(__file__).parent/'negative-cli-commands')
env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
cases=[['--input-tokens','64','--group','16','--compact','on'],
       ['--input-tokens','64','--group','16','--compact','off','--reference','/tmp/unused-no-open'],
       ['--input-tokens','65','--group','16','--compact','off'],
       ['--input-tokens','64','--group','8','--compact','off']]
for i,args in enumerate(cases):
 c,so,se=owner.run('invalid-cli-'+str(i),['/usr/bin/python3',str(H/'run_root.py'),*args],env,H,wall=10,cpu=20,text_cap=65536,total_cap=1<<20,rss_cap=128<<20)
 assert m.closed(c) and c.get('session_anchored') is True and c['exit_code']==2
 assert not so.read_text() and 'error:' in se.read_text()
p=H/'qualification.pending.json'
c,so,se=owner.run('pending-refused',['/usr/bin/python3',str(H/'run_root.py'),'--input-tokens','64','--group','16','--compact','off','--qualification',str(p),'--qualification-sha256',m.sha(p)],env,H,wall=10,cpu=20,text_cap=65536,total_cap=1<<20,rss_cap=128<<20)
assert m.closed(c) and c.get('session_anchored') is True and c['exit_code']==1
assert not so.read_text() and 'actual Main qualification, no pending launch' in se.read_text()
assert not (B/'xe-compact-moe-model-root-v1').exists()
print(json.dumps(dict(passed=True,gpu_executed=False,model_executed=False,actual_cli_commands=owner.commands)))
''')
        r['negative_cli_recipe_sha256'] = m.sha(cli_recipe)
        cli = json.loads(run('actual-negative-cli-fixtures', ['/usr/bin/python3', str(cli_recipe)]))
        m.require(cli['passed'] and len(cli['actual_cli_commands']) == 5 and all(m.closed(c) and
                  c.get('session_anchored') is True for c in cli['actual_cli_commands']), 'actual narrow CLI refusals and closure')
        r['actual_cli_fixture_result'] = cli
        m.require(all(m.sha(H / name) == h for name,h in pins.items()) and
                  all(m.sha(Path(path)) == h for path,h in r['prerequisite_fixture_receipt_pins'].items()), 'all actual source/prerequisite bytes stable')
        r['source_stable'] = True
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','source_stable')}))
    raise SystemExit(0 if r['passed'] else 1)
