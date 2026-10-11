"""Main: actual serial CPU qualification of reviewed two-case coarse phase controller."""
import fcntl
import hashlib
import importlib.util
import json
import time
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
H = B / 'xe-compact-moe-phase-observer-controller-source-v1'
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
    out = B / 'xe-compact-moe-phase-observer-controller-cpu-root-v1'
    review_path = B / 'xe-compact-moe-phase-observer-controller-main-review-v1.json'
    m.require(not out.exists() and not review_path.exists(), 'immutable new CPU evidence and Main review')
    out.mkdir()
    pins = {p.name: m.sha(p) for p in sorted(H.glob('*.py'))}
    all_pins = {p.name: m.sha(p) for p in sorted(H.iterdir()) if p.is_file()}
    m.require(len(pins) == 14 and len(all_pins) == 19 and
              pins['run_root.py'] == '2ffc095faa201da13ccc54b6c27ba155f984821a18cc050a1c90159b4569e84c' and
              all(m.sha(H / name) == m.sha(OLD / name) for name in SHARED) and
              all(m.sha(H / name) == m.sha(B / 'xe-compact-moe-chunk8192-controller-source-v1' / name)
                  for name in ('window_protocol.py','source-review.expected.json')) and
              all(m.sha(H/name) == m.sha(B/'xe-compact-moe-quiet64k-phase-controller-source-v1'/name)
                  for name in ('phase_observer.py','phase.recipe.pending.json')),
              'exact returned source and unchanged qualified helpers/window/source review')
    report = B / 'research-20261009/implementation-xe-compact-moe-phase-observer-controller-source-v1.txt'
    m.require(m.sha(report) == '5b720fd094413fdabce82ddee5586114c3bd93f611ebbacb1b0ef7b52304e356', 'exact original implementation report')
    review = dict(main_reviewed=True, review_scope='Full changed admission/CPU/phase parser, all templates and complete run_root diff read; eight helpers, window protocol and applied source review reused by exact hash.',
        source_pins=pins, all_nineteen_file_pins=all_pins, implementation_report=dict(path=str(report), sha256=m.sha(report)),
        source_publication_revision='39e395a103af36323262cdc0dbaea27cba81f3e4', source_only_review=True,
        controller_cpu_qualified=False, model_executed=False, gpu_executed=False, performance_eligible=False,
        original_source_modified=False, full262144_qualified=False, restore_qualified=False, broad_quality_qualified=False,
        findings=['N64 timestamp sanity thenN32768 diagnostic, fixed8192 and262144 context compactON preserved.',
            'Fourteen-code-file qualification, all-six6144 and three current closed diagnostics mandatory.',
            'READY actual free reserve and normal QUIT before strict output acceptance retained.',
            'Changed-chunk deltas are recorded before any strict-continuation refusal.',
            'No new kernel, allocation lifetime, arithmetic precision or automatic retry introduced.'])
    review_path.write_text(json.dumps(review, indent=2) + '\n')
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             source_pins=pins, owner_sha256=m.sha(O), main_review_sha256=m.sha(review_path),
             recipe_sha256=m.sha(__file__), gpu_executed=False, model_executed=False,
             full262144_qualified=False, restore_qualified=False, broad_quality_qualified=False,
             performance_eligible=False, adopted=False, source_stable=False,
             scope='Actual syntax, six+three prerequisites, phase prefix/environment/argv/protocol/parser and anchored CPU QUIT fixtures; real CLI refusals only.')

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
            "from pathlib import Path\nimport protocol,runtime\nfiles=sorted(Path('.').glob('*.py'))\nassert len(files)==14\nfor p in files:compile(p.read_text(),str(p),'exec')\nprotocol.selftest();runtime.selftest()\nprint('14 sources compile; unchanged qualified helpers PASS')\n"])
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
        triple = {}
        for n, chunk in ((32768,8192),(65536,6144),(65536,8192)):
            p = B / f'xe-compact-moe-chunk8192-root-v1/n{n}-t{chunk}-g16-on/record.json'
            v = json.loads(p.read_text())
            m.require(v['passed'] and v['complete'] and not v['active'] and v['model_normal_close'] and
                v['model_close']['normal'] and not v['model_close']['forced'] and v['model_close']['exit_code'] == 0 and
                v['kernel_gate_passed'] and v['boot_unchanged'] and v['validation']['generated_tokens'] == 16,
                'Main selected actual three closed GPU diagnostics')
            triple[f'n{n}-t{chunk}'] = dict(path=str(p), sha256=m.sha(p))
        triple_manifest = out / 'three-cases.closed.json'
        triple_manifest.write_text(json.dumps(dict(three_case_receipts=triple, original_results_modified=False), indent=2) + '\n')
        r['three_case_manifest'] = dict(path=str(triple_manifest), sha256=m.sha(triple_manifest))
        value = json.loads(run('exported-controller-fixtures', ['/usr/bin/python3', str(H / 'controller_cpu_fixtures.py'),
            '--output', str(out / 'exported-fixtures'), '--lint-receipt', str(B / 'xe-compact-moe-lint-qualification-root-v2/record.json'),
            '--matrix-manifest', str(manifest), '--three-case-manifest', str(triple_manifest)]))
        m.require(value['passed'] and value['source_pins'] == pins and value['owner_sha256'] == m.sha(O) and
                  not value['gpu_executed'] and not value['model_executed'], 'actual same-source production fixture output')
        m.require(len(value['owned_quit_cases']) == 4 and all(m.closed(x['command']) and
            x['command'].get('session_anchored') is True and x['command']['exit_code'] == 0 for x in value['owned_quit_cases']),
            'four actual CPU protocol children normally closed')
        r['exported_fixture_result'] = value
        r['prerequisite_fixture_receipt_pins'] = value['prerequisite_fixture_receipt_pins']
        cli_cases = value['cli_rejection_cases']
        m.require(len(cli_cases) == 7 and all(m.closed(x['command']) and
            x['command'].get('session_anchored') is True and x['command']['exit_code'] == (2 if i < 6 else 1)
            for i,x in enumerate(cli_cases)), 'actual seven exact phase CLI refusals')
        m.require(not (B/'xe-compact-moe-phase-observer-root-v1').exists(), 'phase GPU namespace remains unlaunched')
        r['actual_cli_fixture_result'] = cli_cases
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
