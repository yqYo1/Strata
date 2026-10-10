"""Independent closed-run evidence audit. Never changes original failed status."""
from pathlib import Path
import ast, datetime, fcntl, hashlib, json, math, struct, subprocess

B = Path(__file__).parent
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def pin(p):
    return {'path': str(p), 'bytes': Path(p).stat().st_size, 'sha256': sha(p)}
def read(p):
    return json.loads(Path(p).read_text())

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'prefill-phase-v2-independent-evidence-audit-v1'
    assert not out.exists()
    out.mkdir(mode=0o700)
    original = B / 'owned-prefill-phase-validity-32769-v2/record.json'
    original_pin = pin(original)
    r = read(original)
    baseline_path = B / 'owned-decode-pool-phase-32769-v1-baseline/record.json'
    reference = read(baseline_path)
    build_path = B / 'prefill-phase-validity-cpu-build-v1/record.json'
    build = read(build_path)
    assert not r['active'] and not r['completed'] and not r['healthy'] and not r['math_gate_passed']
    assert r['error'] == 'AssertionError()' and r['exit_code'] == 0 and r['exit_signal'] is None
    assert not any(r['cleanup'].values()) and not r['new_fault_messages']
    assert not r['source_status_before'] and not r['source_status_after']
    assert reference['completed'] and reference['healthy'] and reference['math_gate_passed'] and not reference['active']
    assert build['passed'] and build['complete'] and not build['active']
    assert sha(build_path) == r['prefill_validity_build_receipt_sha256']
    assert sha(build['binary']) == r['binary_sha256'] == r['actual_executable_identity']['sha256']
    assert r['commit'] == build['commit'] == '185fa78098bee4be01ac81ad4f791673d24246f3'
    root = Path(r['controller_review_root'])
    git = lambda *args: subprocess.check_output(['git', '-C', str(root), *args], text=True, timeout=10).strip()
    assert git('rev-parse', 'HEAD') == r['commit'] and not git('status', '--porcelain')
    for name, expected in r['source_sha256'].items():
        assert sha(name) == expected, name
    assert sha(root / 'sycl/src/prefill/prefill.cpp') == build['source_sha256']
    controller = B / 'run_owned_prefill_phase_validity_32769_v2.py'
    source = controller.read_text()
    ast.parse(source)
    frozen = "review_head='7d0105f2a942ac72ca20fef69d4c2e7960653de3'"
    capture = 'commits={mode:review_head}'
    override = "review_head=prefill_validity_build['commit']"
    final = "assert not record['source_status_after'] and git('rev-parse', 'HEAD') == commits[mode]"
    assert source.index(frozen) < source.index(capture) < source.index(override) < source.index(final)
    assert source.count('commits') == 2
    assert source.index(final) < source.index("record['completed'] = True")
    # The original got as far as source_status_after, then the single assertion
    # before completed=True necessarily compares the frozen old head.
    expected_old = '7d0105f2a942ac72ca20fef69d4c2e7960653de3'
    assert expected_old != r['commit']
    process_evidence = []
    for key in ['inferior', 'debugger']:
        p = r[key]
        proc = Path(f"/proc/{p['pid']}/stat")
        current_ticks = None
        if proc.exists():
            current_ticks = int(proc.read_text().split(') ', 1)[1].split()[19])
        assert current_ticks != p['start_ticks'], (key, p)
        process_evidence.append({'role': key, 'original': p, 'current_start_ticks': current_ticks, 'original_identity_alive': False})
    assert len(r['requests']) == len(reference['requests']) == 3
    requests = []
    for q, ref in zip(r['requests'], reference['requests']):
        assert q['name'] == ref['name'] and q['math_gate_passed'] and ref['math_gate_passed']
        assert all(q['validation'].values())
        assert len(q['ids']) == len(q['logprobs']) == 64
        for key in ['ids', 'logprobs', 'mtp_counts', 'finish_reason', 'resume_tokens']:
            assert q[key] == ref[key], (q['name'], key)
        for lp in q['logprobs']:
            assert math.isfinite(float(lp.split()[1]))
            assert all(math.isfinite(float(v.split(':')[1])) for v in lp.split()[2:])
        for head in [q['first_head'], ref['first_head']]:
            assert sha(head['file']) == head['sha256']
            data = Path(head['file']).read_bytes()
            assert len(data) == head['floats'] * 4
            assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', data))
        assert q['first_head']['sha256'] == ref['first_head']['sha256']
        requests.append({'name': q['name'], 'exact_output_and_first_head': True, 'measurement': q['measurement']})
    q, ref = r['requests'][0], reference['requests'][0]
    assert q['prefill_state']['parts'] == ref['prefill_state']['parts']
    assert len(q['prefill_state']['parts']) == 66
    assert not q['live_prefill_comparison']['different_live_parts']
    state_pins = []
    for state in [q['prefill_state'], ref['prefill_state']]:
        p = Path(state['file'])
        assert p.stat().st_size == state['bytes']
        with p.open('rb') as f:
            for part in state['parts']:
                f.seek(part['offset'])
                data = f.read(part['bytes'])
                assert len(data) == part['bytes'] and hashlib.sha256(data).hexdigest() == part['sha256']
        state_pins.append(pin(p))
    assert state_pins[0]['sha256'] == state_pins[1]['sha256']
    log = original.parent / 'debugger/inferior.stderr'
    assert sha(log) == r['engine_log_sha256'] and log.stat().st_size == r['engine_log_bytes']
    lines = log.read_text().splitlines()
    validity_rows = [json.loads(l.split(': ', 1)[1]) for l in lines if l.startswith('strata prefill phase validity: ')]
    phase_rows = [json.loads(l.split(': ', 1)[1]) for l in lines if l.startswith('strata prefill phases: ')]
    assert len(validity_rows) == len(phase_rows) == 1
    v, phase = validity_rows[0], phase_rows[0]
    assert v['status'] == 'valid' and v['reason'] == 'none' and v['queue_admitted']
    assert v['marker_attempts'] == v['markers_submitted'] == 469556
    assert v['intervals_attempted'] == v['intervals_valid'] == v['markers_submitted'] - 1
    assert v['query_attempts'] == v['query_successes'] == 2 * v['intervals_valid']
    assert all(v[k] == 0 for k in ['query_failures', 'nonmonotonic', 'incomplete'])
    assert v['retained_markers'] == 1 and v['raw_end_ns'] >= v['raw_begin_ns']
    assert all(math.isfinite(x) and x >= 0 for x in phase['phase_ms'].values())
    assert math.isclose(sum(phase['phase_ms'].values()), phase['gpu_timeline_ms'], abs_tol=0.00001)
    for k in ['host_setup_ms', 'host_chunk_wait_ms', 'host_after_chunk_ms', 'ple_gather_wall_ms', 'ple_host_wait_ms']:
        assert math.isfinite(phase[k]) and phase[k] >= 0
    boundary = r['stderr_request_boundaries'][0]
    raw = log.read_bytes()[boundary['stderr_begin']:boundary['stderr_end']].decode()
    assert 'strata prefill phases: ' in raw and 'strata prefill phase validity: ' in raw
    assert r['environment']['STRATA_PREFILL_TIMING'] == '1'
    assert 'STRATA_PREFILL_SYNC' not in r['environment'] and 'STRATA_EXPERT_TRANSFER_TIMING' not in r['environment']
    result = {
        'audit_passed': True, 'active': False, 'complete': True,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'original_failed_receipt': original_pin, 'original_controller_whole_pass': False,
        'original_status_preserved': True, 'audit_controller': pin(__file__),
        'baseline_receipt': pin(baseline_path), 'build_receipt': pin(build_path),
        'original_failure_explanation': {'frozen_expected_commit': expected_old, 'actual_pinned_commit': r['commit'], 'source_status_after': r['source_status_after'], 'assertion': final, 'stage': 'after QUIT/normal exit and individual request math checks, before aggregate completed flag'},
        'numerical_evidence_independently_verified': requests,
        'state_parts_independently_hashed': 66, 'state_files': state_pins,
        'original_processes_closed': process_evidence,
        'engine_log': pin(log), 'phase_validity': v, 'phase_observation': phase,
        'independent_phase_observation_admitted': True,
        'performance_eligible': False, 'adopted': False, 'full_256K_capacity_proven': False,
        'scope': 'Valid opt-in marker intervals include queued work, dependency waits and host-induced queue idle; they are not exclusive kernel or DMA service. No speed/adoption/retry claim.'
    }
    assert pin(original) == original_pin
    (out / 'record.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'audit_passed': True, 'original_controller_whole_pass': False, 'independent_phase_observation_admitted': True, 'record': str(out / 'record.json')}))
