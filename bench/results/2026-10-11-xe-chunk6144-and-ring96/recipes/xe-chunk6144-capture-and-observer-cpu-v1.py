import ast
import difflib
import fcntl
import importlib.util
import json
import re
import sys
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
C = B / 'xe-source-llvm-chunk6144-diagnostic32k-v3'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
sys.path.insert(0, str(C))
from protocol import validate
from runtime import validate_runtime
from transfer_ledger import validate as transfers
from timer_overhead import validate as overhead

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-chunk6144-capture-and-observer-cpu-root-v1'
    m.require(not out.exists(), 'immutable result')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), cpu_only=True, gpu_executed=False,
             model_executed=False, performance_eligible=False)
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    try:
        cpu_path = B / 'xe-source-llvm-chunk6144-diagnostic32k-cpu-root-v3/record.json'
        cpu = json.loads(cpu_path.read_text())
        m.require(cpu['passed'] and cpu['complete'] and not cpu['active'] and all(m.closed(c) for c in cpu['commands']), 'qualified6144 protocol/helper fixtures')
        m.require(all(m.sha(C/n) == h for n,h in cpu['source_pins'].items()), 'exact qualified helper bytes')
        r['helper_qualification_sha256'] = m.sha(cpu_path)
        r['helper_pins'] = cpu['source_pins']
        p = B / 'xe-source-llvm-chunk6144-diagnostic32k-root-v2/counts/record.json'
        v = json.loads(p.read_text())
        r['original_receipt'] = dict(path=str(p), sha256=m.sha(p), passed=v['passed'], complete=v['complete'], error=v['error'])
        m.require(not v['passed'] and not v['complete'] and not v['active'] and v['model_normal_close'] and
                  v['error'] == "ValueError('exact eight chunks')" and v['kernel_gate_passed'] and not v['new_fault_messages'] and
                  all(m.closed(c) for c in v['commands']), 'original observer-only failure preserved')
        for cmd in v['commands']:
            for ident in cmd['owners']:
                now = m.identity(ident['pid'])
                m.require(not now or now['start_ticks'] != ident['start_ticks'], 'original owned identities absent')
        model = next(c for c in v['commands'] if c['label'] == 'model')
        stdout, stderr = (p.parent/'commands'/name for name in ('model.stdout','model.stderr'))
        for name, f in [('model.stdout', stdout), ('model.stderr', stderr)]:
            m.require(f.stat().st_size < 2 << 20 and m.sha(f) == model['logs'][name]['sha256'] and
                      f.stat().st_size == model['logs'][name]['bytes'], 'complete actual capture hash')
        m.require(stdout.read_text().splitlines() == v['startup'] + v['request_protocol'], 'full actual stdout coverage')
        r['capture_validation'] = validate(v['request_protocol'])
        capture = stderr.read_text()
        layout_path = B/'xe-prefill-native-transfer-layout-v1.json'
        m.require(m.sha(layout_path) == 'c9b40d122d19f277280560a495ad0a1c6ce98376ef0660c1f75e9b3026bd642b', 'independent full metadata')
        layout = json.loads(layout_path.read_text())
        pins = {x['layer']:{k:x[k] for k in ('blob_bytes','gu_type','down_type')} for x in layout['expected_layers']}
        r['transfer_ledger'] = transfers(capture, expected_device=0, expected_stage=[0,48], expected_pos0=0,
            expected_n=32767, expected_chunk_limit=6144,
            expected_mode={0:'full_nonresident_stream',6144:'full_nonresident_stream',12288:'full_nonresident_stream',
                           18432:'full_nonresident_stream',24576:'full_nonresident_stream',30720:'routed'},
            expected_issuer='inline', expected_format='native_gguf', expected_layers=pins)
        m.require(all(row['calls'] == 509 if row['offset'] < 30720 else row['calls'] <= 509
                      for row in r['transfer_ledger']['rows']), 'full-stream per-layer actual quota and routed-tail bound')
        r['timer_overhead'] = overhead(capture, timing_enabled=False)
        m.require('strata prefill timing:' not in capture and 'strata trace:' not in capture, 'timer/trace off')
        m.require(not re.search(r'UR_RESULT_ERROR_[A-Z_]+|ZE_RESULT_ERROR_[A-Z_]+|\[(?:ERROR|FATAL)\]|GPU completion unconfirmed|terminate called|Assertion .*failed|CAT error|fault response|out of resources', capture, re.I), 'no severe runtime errors')
        health = json.loads((B/'xe-source-llvm-small-health-root-v3/record.json').read_text())
        r['runtime_mapping_validation'] = validate_runtime(v['observed_libraries'], health['loaded_libraries'])
        quiet_path = B/'xe-source-llvm-quiet32k-root-v2/1/record.json'
        quiet = json.loads(quiet_path.read_text())
        output = lambda value:[line for line in value['request_protocol'] if line.startswith(('T ','LP '))]
        r['selected16_ids_and_lp_top5_equal_quiet'] = output(v) == output(quiet)
        m.require(quiet['passed'] and quiet['input'] == v['input'] and r['selected16_ids_and_lp_top5_equal_quiet'], 'same-input complete selected-output equality')
        r['original_posthealth_not_reached'] = True
        r['scope'] = 'CPU-only review of closed original6144 capture; original overallFAILED preserved and excluded from qualified performance. Local logical H2D and selected16 output evidence only. No retrospective fresh-health/global-quality qualification.'
        r['original_overall_status_unchanged'] = m.sha(p) == r['original_receipt']['sha256']
        m.require(r['original_overall_status_unchanged'], 'original receipt unchanged')

        a = B/'xe-prefill-route-branch-census-source-v1'
        manifest = json.loads((a/'manifest.json').read_text())
        m.require(m.sha(W/'src/prefill/prefill.cpp') == manifest['base_source_sha256'], 'current reviewed census base')
        diff = ''.join(difflib.unified_diff((W/'src/prefill/prefill.cpp').read_text().splitlines(True),
            (a/'src/prefill/prefill.cpp').read_text().splitlines(True), fromfile='a/src/prefill/prefill.cpp',tofile='b/src/prefill/prefill.cpp'))
        m.require(diff == (a/'change.patch').read_text(), 'full replacement equals fully read exact diff')
        for x in manifest['artifacts']:
            m.require(m.sha(Path(x['path'])) == x['sha256'] and Path(x['path']).stat().st_size == x['bytes'], 'census artifacts pinned')
        r['census_main_review'] = dict(main_reviewed=True, artifact_manifest_sha256=m.sha(a/'manifest.json'),
            artifact_pins={x['path']:x['sha256'] for x in manifest['artifacts']}, applied=False,
            review='Full patch/report read, exact replacement diff verified against unchanged fully reviewed base. Consumer-only census array before issuer guard, scans/counters enabled only, join/drains before final row grammar; no arithmetic/queue policy changes.',
            build_lint_runtime_qualified=False)
        a = B/'xe-mmq-group-gather-byte-source-v1'
        probe_pins = {}
        for line in (a/'SHA256SUMS').read_text().splitlines():
            h,path = line.split(maxsplit=1)
            m.require(m.sha(Path(path)) == h, 'probe artifact pin')
            probe_pins[path] = h
        r['gather_probe_main_review'] = dict(main_reviewed=True, artifact_pins=probe_pins,
            review='Full source/report/build plan read. Actual Runtime compute queue, production entrypoints, independent pattern oracle and all padding/guard bytes. Checked terminal drains precede release; uncertain GPU submission/wait exits before unwind.8 sequential arms,40 cases,112 refusals.',
            device_payload_max_bytes=85199744, host_payload_max_bytes=127798784,
            build_lint_runtime_qualified=False, individual_misaligned_memcpy_qualified=False)
        a = B/'xe-prefill-route-census-observer-source-v1'
        manifest = json.loads((a/'manifest.json').read_text())
        for x in manifest['artifacts']:
            m.require(m.sha(Path(x['path'])) == x['sha256'] and Path(x['path']).stat().st_size == x['bytes'], 'observer artifact pins')
        for name in ('route_census.py','selfcheck.py'):
            ast.parse((a/name).read_text(), filename=str(a/name))
        e,so,se = owner.run('route-observer-selfcheck', ['/usr/bin/python3','-B',str(a/'selfcheck.py'),
            '--transfer-ledger',str(C/'transfer_ledger.py')], ENV, a, wall=15,cpu=15,
            text_cap=2 << 20,total_cap=4 << 20,rss_cap=2 << 30)
        m.require(m.closed(e) and e['exit_code'] == 0, 'closed actual observer CPU fixtures')
        fixtures = json.loads(so.read_text())
        m.require(fixtures == dict(positive_cases=10,negative_cases=39,actual=None), 'finite actual10 positive39 rejection fixtures')
        r['observer_main_review'] = dict(main_reviewed=True, artifact_pins={x['path']:x['sha256'] for x in manifest['artifacts']},
            full_source_report_manifest_read=True, ast_parse_passed=True, fixtures=fixtures,
            actual_gpu_capture_qualified=False, gpu_completion_inferred=False,
            review='Same-capture independent qualified H2D validation plus exact complete census rows; explicit model and nonresident population; disabled absence; global allocation fallback distinguished from per-layer execution.')
        m.require(all(m.sha(Path(x['path'])) == x['sha256'] for x in manifest['artifacts']), 'observer stable')
        for key,name in [('census_main_review','xe-prefill-route-branch-census-main-review-v1.json'),
                         ('gather_probe_main_review','xe-mmq-group-gather-byte-main-review-v1.json'),
                         ('observer_main_review','xe-prefill-route-census-observer-main-review-v1.json')]:
            q = B/name
            m.require(not q.exists(), 'immutable Main review')
            q.write_text(json.dumps(r[key], indent=2)+'\n')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','original_overall_status_unchanged')}))
    raise SystemExit(0 if r['passed'] else 1)
