"""Main: actual serial CPU qualification of reviewed three-case controller."""
import fcntl
import hashlib
import importlib.util
import json
import time
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
H = B / 'xe-compact-moe-chunk8192-controller-source-v1'
OLD = B / 'xe-compact-moe-model-controller-source-v2'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('chunk_cpu_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
SHARED = ('batch_census.py', 'fixtures.py', 'hash_inputs.py', 'protocol.py', 'route_census.py',
          'runtime.py', 'timer_overhead.py', 'transfer_ledger.py')

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-compact-moe-chunk8192-controller-cpu-root-v1'
    review_path = B / 'xe-compact-moe-chunk8192-controller-main-review-v1.json'
    m.require(not out.exists() and not review_path.exists(), 'immutable new CPU evidence and Main review')
    out.mkdir()
    pins = {p.name: m.sha(p) for p in sorted(H.glob('*.py'))}
    all_pins = {p.name: m.sha(p) for p in sorted(H.iterdir()) if p.is_file()}
    m.require(len(pins) == 12 and len(all_pins) == 15 and
              pins['run_root.py'] == 'e56dfb1196b81e7fdc4b8c07b5cedad03fae47eff80bbb6d59dd28175e297b2a' and
              all(m.sha(H / name) == m.sha(OLD / name) for name in SHARED), 'exact returned source and eight unchanged helpers')
    report = B / 'research-20261009/implementation-xe-compact-moe-chunk8192-controller-source-v1.txt'
    m.require(m.sha(report) == 'b03826467f05c590d8336f18d52d5c25e7258a00326e8a329db969fc6288dbd0', 'exact original implementation report')
    review = dict(main_reviewed=True, review_scope='Full changed admission/CPU/window sources and complete run_root diff read; unchanged qualified eight helpers reused by exact hash.',
        source_pins=pins, all_fifteen_file_pins=all_pins, implementation_report=dict(path=str(report), sha256=m.sha(report)),
        source_publication_revision='39e395a103af36323262cdc0dbaea27cba81f3e4', source_only_review=True,
        controller_cpu_qualified=False, model_executed=False, gpu_executed=False, performance_eligible=False,
        original_source_modified=False, full262144_qualified=False, restore_qualified=False, broad_quality_qualified=False,
        findings=['Exact three-case ON sequence and fixed262144 context preserved.',
            'Separate twelve-code-file qualification and all-six exact closed6144 prerequisites mandatory.',
            'READY actual free reserve and normal QUIT before strict output acceptance retained.',
            'Changed-chunk deltas are recorded before any strict-continuation refusal.',
            'No new kernel, allocation lifetime, arithmetic precision or automatic retry introduced.'])
    review_path.write_text(json.dumps(review, indent=2) + '\n')
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             source_pins=pins, owner_sha256=m.sha(O), main_review_sha256=m.sha(review_path),
             recipe_sha256=m.sha(__file__), gpu_executed=False, model_executed=False,
             full262144_qualified=False, restore_qualified=False, broad_quality_qualified=False,
             performance_eligible=False, adopted=False, source_stable=False,
             scope='Actual syntax, exported prerequisite/three-case/window/delta fixtures and anchored CPU-child QUIT; real CLI refusals only.')

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    begin = time.monotonic()

    def run(label, argv):
        left = 180 - (time.monotonic() - begin)
        m.require(left > 0, 'whole CPU deadline')
        c, so, se = owner.run(label, argv, ENV, H, wall=min(90, left), cpu=180,
            text_cap=8 << 20, total_cap=16 << 20, rss_cap=1 << 30, file_cap=64 << 20)
        r.setdefault('command_evidence', {})[label] = dict(stdout=so.read_text(), stderr=se.read_text())
        save()
        m.require(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] == 0, 'normal anchored CPU closure ' + label)
        return so.read_text()

    save()
    try:
        run('syntax-and-helper-selftests', ['/usr/bin/python3', '-c',
            "from pathlib import Path\nimport protocol,runtime\nfiles=sorted(Path('.').glob('*.py'))\nassert len(files)==12\nfor p in files:compile(p.read_text(),str(p),'exec')\nprotocol.selftest();runtime.selftest()\nprint('12 sources compile; unchanged qualified helpers PASS')\n"])
        matrix = {}
        for n in (64, 2049, 32768):
            for variant in ('off', 'on'):
                p = B / f'xe-compact-moe-model-root-v1/n{n}-g16-{variant}/record.json'
                v = json.loads(p.read_text())
                m.require(v['passed'] and v['complete'] and not v['active'] and v['model_normal_close'] and
                          v['model_close']['normal'] and not v['model_close']['forced'] and v['model_close']['exit_code'] == 0,
                          'Main selected actual closed original matrix')
                matrix[f'n{n}-{variant}'] = dict(path=str(p), sha256=m.sha(p))
        manifest = out / 'matrix.closed.json'
        manifest.write_text(json.dumps(dict(six_case_receipts=matrix, original_results_modified=False), indent=2) + '\n')
        r['matrix_manifest'] = dict(path=str(manifest), sha256=m.sha(manifest))
        value = json.loads(run('exported-controller-fixtures', ['/usr/bin/python3', str(H / 'controller_cpu_fixtures.py'),
            '--output', str(out / 'exported-fixtures'), '--lint-receipt', str(B / 'xe-compact-moe-lint-qualification-root-v2/record.json'),
            '--matrix-manifest', str(manifest)]))
        m.require(value['passed'] and value['source_pins'] == pins and value['owner_sha256'] == m.sha(O) and
                  not value['gpu_executed'] and not value['model_executed'], 'actual same-source production fixture output')
        m.require(len(value['owned_quit_cases']) == 4 and all(m.closed(x['command']) and
            x['command'].get('session_anchored') is True and x['command']['exit_code'] == 0 for x in value['owned_quit_cases']),
            'four actual CPU protocol children normally closed')
        r['exported_fixture_result'] = value
        r['prerequisite_fixture_receipt_pins'] = value['prerequisite_fixture_receipt_pins']
        cli_recipe = out / 'negative_cli.py'
        cli_recipe.write_text('''import hashlib,importlib.util,json
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
H=B/'xe-compact-moe-chunk8192-controller-source-v1'
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest()=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
s=importlib.util.spec_from_file_location('cli_owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
owner=m.Owner(Path(__file__).parent/'negative-cli-commands')
env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
cases=[['--input-tokens','32768','--chunk','8192','--compact','on'],
       ['--input-tokens','65536','--chunk','6144','--compact','on','--reference','/tmp/unused-no-open'],
       ['--input-tokens','32768','--chunk','6144','--compact','on'],
       ['--input-tokens','32768','--chunk','8192','--compact','off','--reference','/tmp/unused-no-open'],
       ['--input-tokens','65537','--chunk','8192','--compact','on','--reference','/tmp/unused-no-open'],
       ['--input-tokens','32768','--chunk','8192','--group','8','--compact','on','--reference','/tmp/unused-no-open']]
for i,args in enumerate(cases):
 c,so,se=owner.run('invalid-cli-'+str(i),['/usr/bin/python3',str(H/'run_root.py'),*args],env,H,wall=10,cpu=20,text_cap=65536,total_cap=2<<20,rss_cap=128<<20)
 assert m.closed(c) and c.get('session_anchored') is True and c['exit_code']==2
 assert not so.read_text() and 'error:' in se.read_text()
p=H/'qualification.pending.json'
c,so,se=owner.run('pending-refused',['/usr/bin/python3',str(H/'run_root.py'),'--input-tokens','65536','--chunk','6144','--compact','on','--qualification',str(p),'--qualification-sha256',m.sha(p)],env,H,wall=10,cpu=20,text_cap=65536,total_cap=2<<20,rss_cap=128<<20)
assert m.closed(c) and c.get('session_anchored') is True and c['exit_code']==1
assert not so.read_text() and 'actual Main qualification, no pending launch' in se.read_text()
assert not (B/'xe-compact-moe-chunk8192-root-v1').exists()
print(json.dumps(dict(passed=True,gpu_executed=False,model_executed=False,actual_cli_commands=owner.commands)))
''')
        r['negative_cli_recipe_sha256'] = m.sha(cli_recipe)
        cli = json.loads(run('actual-negative-cli-fixtures', ['/usr/bin/python3', str(cli_recipe)]))
        m.require(cli['passed'] and len(cli['actual_cli_commands']) == 7 and
                  all(m.closed(c) and c.get('session_anchored') is True for c in cli['actual_cli_commands']), 'actual CLI refusals and closure')
        r['actual_cli_fixture_result'] = cli
        m.require(all(m.sha(H / name) == h for name, h in all_pins.items()) and
                  all(m.sha(Path(path)) == h for path, h in r['prerequisite_fixture_receipt_pins'].items()), 'source and prerequisites stable')
        r['source_stable'] = True
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'source_stable')}))
    raise SystemExit(0 if r['passed'] else 1)
