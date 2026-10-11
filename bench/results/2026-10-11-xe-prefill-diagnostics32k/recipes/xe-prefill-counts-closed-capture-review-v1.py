import fcntl
import importlib.util
import json
import re
import sys
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C = B / 'xe-source-llvm-prefill-diagnostic32k-v2'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner')
sys.path.insert(0, str(C))
from transfer_ledger import validate as validate_transfers
from timer_overhead import validate as validate_overhead

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-prefill-counts-closed-capture-review-root-v1'
    m.require(not out.exists(), 'immutable result')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), cpu_only=True, gpu_executed=False, model_executed=False,
             performance_eligible=False,
             scope='CPU review of complete closed actual counts capture with source-derived full-stream expectation; original overall FAILED unchanged. Logical request extents only, not physical bandwidth or retrospective full run qualification.')
    try:
        review = B / 'xe-prefill-diagnostic32k-controller-main-review-v2.json'
        a = json.loads(review.read_text())
        m.require(a['main_reviewed'] and all(m.sha(C/n) == h for n,h in a['source_pins'].items()), 'exact Main-reviewed v2 controller/helpers')
        r['controller_review_sha256'] = m.sha(review)
        observer_path = B / 'xe-prefill-diagnostic-observer-cpu-root-v1/record.json'
        observer = json.loads(observer_path.read_text())
        m.require(observer['passed'] and observer['complete'] and not observer['active'] and
                  all(m.closed(c) for c in observer['commands']) and
                  all(m.sha(C/n) == h for n,h in observer['source_pins'].items()), 'CPU-qualified parser bytes')
        p = B / 'xe-source-llvm-prefill-diagnostic32k-root-v1/counts/record.json'
        v = json.loads(p.read_text())
        r['original_receipt'] = dict(path=str(p), sha256=m.sha(p), passed=v['passed'], complete=v['complete'], error=v['error'])
        m.require(not v['passed'] and not v['complete'] and not v['active'] and
                  v['error'] == "ValidationError('line 38: wrong mode, issuer or format')" and
                  v['model_normal_close'] and v['kernel_gate_passed'] and not v['new_fault_messages'] and
                  all(m.closed(c) for c in v['commands']), 'original observer-only failure with normal closed native model/no new faults')
        m.require(v['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'same actual boot')
        for cmd in v['commands']:
            for ident in cmd['owners']:
                now = m.identity(ident['pid'])
                m.require(not now or now['start_ticks'] != ident['start_ticks'], 'closed owned identities absent')
        model = next(c for c in v['commands'] if c['label'] == 'model')
        stdout = p.parent / 'commands/model.stdout'
        stderr = p.parent / 'commands/model.stderr'
        for name, f in [('model.stdout', stdout), ('model.stderr', stderr)]:
            m.require(f.stat().st_size < 2 << 20 and m.sha(f) == model['logs'][name]['sha256'] and
                      f.stat().st_size == model['logs'][name]['bytes'], 'exact complete native capture')
        m.require(stdout.read_text().splitlines() == v['startup'] + v['request_protocol'], 'full protocol retained exactly')
        capture = stderr.read_text()
        r['capture'] = dict(path=str(stderr), sha256=m.sha(stderr), bytes=stderr.stat().st_size)
        layout_path = B / 'xe-prefill-native-transfer-layout-v1.json'
        m.require(m.sha(layout_path) == 'c9b40d122d19f277280560a495ad0a1c6ce98376ef0660c1f75e9b3026bd642b', 'independent all-layer metadata')
        layout = json.loads(layout_path.read_text())
        m.require(m.sha(Path(layout['layout_path'])) == layout['layout_sha256'], 'actual model metadata stable')
        pins = {x['layer']: {k:x[k] for k in ('blob_bytes','gu_type','down_type')} for x in layout['expected_layers']}
        r['transfer_ledger'] = validate_transfers(capture, expected_device=0, expected_stage=[0,48],
            expected_pos0=0, expected_n=32767, expected_chunk_limit=4096, expected_mode='full_nonresident_stream',
            expected_issuer='inline', expected_format='native_gguf', expected_layers=pins)
        r['timer_overhead'] = validate_overhead(capture, timing_enabled=False)
        m.require('strata prefill timing:' not in capture and 'strata trace:' not in capture, 'actual phase/trace off')
        m.require(not re.search(r'UR_RESULT_ERROR_[A-Z_]+|ZE_RESULT_ERROR_[A-Z_]+|\[(?:ERROR|FATAL)\]|GPU completion unconfirmed|terminate called|Assertion .*failed|CAT error|fault response|out of resources', capture, re.I), 'no severe model runtime error')
        quiet_path = B / 'xe-source-llvm-quiet32k-root-v2/1/record.json'
        quiet = json.loads(quiet_path.read_text())
        m.require(quiet['passed'] and quiet['complete'] and not quiet['active'] and
                  quiet['input']['sha256'] == v['input']['sha256'], 'qualified unchanged-input quiet reference')
        output = lambda x: [line for line in x['request_protocol'] if line.startswith(('T ', 'LP '))]
        r['selected_outputs_equal_quiet'] = output(quiet) == output(v)
        m.require(r['selected_outputs_equal_quiet'], 'selected16 outputs/LP equal quiet')
        rows = r['transfer_ledger']['rows']
        m.require(all(x['calls'] == 509 for x in rows), 'actual all509 nonresident experts per layer/chunk')
        r['layer_calls_each_chunk'] = 509
        r['per_chunk_submitted_bytes'] = [sum(x['bytes'] for x in rows if x['offset'] == offset) for offset in range(0,32767,4096)]
        m.require(r['per_chunk_submitted_bytes'] == [sum(x['blob_bytes'] for x in pins.values()) * 509] * 8, 'independent layer extent/quota byte crosscheck')
        r['original_posthealth_not_reached'] = True
        r['original_overall_status_unchanged'] = m.sha(p) == r['original_receipt']['sha256']
        m.require(r['original_overall_status_unchanged'], 'original failure unchanged')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = False
        r['passed'] = bool(r['complete'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(dict(passed=r['passed'], error=r.get('error'), total=r.get('transfer_ledger',{}).get('total'))))
    raise SystemExit(0 if r['passed'] else 1)
