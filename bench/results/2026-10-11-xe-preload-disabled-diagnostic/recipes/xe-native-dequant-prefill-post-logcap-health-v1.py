import fcntl, importlib.util, json, re, time, types
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
H = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh')
DUMP = Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner identity')
m.require(m.sha(H) == '51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457', 'diagnostic helper identity')
helper = types.ModuleType('read_only_helper')
exec(compile(H.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(H), 'exec'), helper.__dict__)
BASE = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    failure_path = B / 'xe-native-dequant-prefill-model-root-v4/n64-g16-off/record.json'
    failure = json.loads(failure_path.read_text())
    m.require(not failure['active'] and not failure['passed'] and failure['gen_sent'] and
              failure['primary_model_failure'] == "RuntimeError('aggregate owner log cap')",
              'closed exact owned log-budget failure after GEN')
    model = next(c for c in failure['commands'] if c['label'] == 'model')
    m.require(model['direct_child_reaped'] and model['session_empty'] and not model['survivors'],
              'failed model process absent')
    m.require(all(c['direct_child_reaped'] and c['session_empty'] and not c['survivors'] for c in failure['commands']), 'all failed-job owned sessions absent')
    for c in failure['commands']:
        now = m.identity(c['pid'])
        m.require(not now or now['start_ticks'] != c['direct_identity']['start_ticks'], 'original failed-job identities absent')
        if c['label'] not in ('model','kernel-final'):
            m.require(m.closed(c), 'original preflight owner normally closed')
    final = next(c for c in failure['commands'] if c['label'] == 'kernel-final')
    m.require(final['exit_code'] == 0 and final['error'] == "RuntimeError('aggregate owner log cap')" and not final['cleanup'], 'exact final-journal observer budget failure retained')
    m.require(failure['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'same boot')
    proof_path = B / 'xe-source-llvm-small-health-root-v3/record.json'
    proof = json.loads(proof_path.read_text())
    m.require(proof['passed'] and proof['complete'] and not proof['active'], 'qualified safe integer probe')
    binary = Path(failure['health_probe']['path'])
    m.require(m.sha(binary) == failure['health_probe']['sha256'] == proof['binary']['sha256'], 'probe identity')
    for p in proof['loaded_libraries']:
        m.require(m.sha(Path(p['path'])) == p['sha256'], 'runtime identity')
    prehealth = next(c for c in failure['commands'] if c['label'] == 'health-before')
    env = prehealth['environment']
    expected_env = dict(proof['environment'], STRATA_EVENT_MARK='barrier',
                        STRATA_GPU_PCI='0000:05:00.0', XDG_RUNTIME_DIR='/run/user/1000')
    m.require(env == helper.diagnostic_environment(expected_env), 'exact diagnostic environment')
    out = B / 'xe-native-dequant-prefill-post-logcap-health-root-v1'
    m.require(not out.exists(), 'immutable output')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), owner_sha256=m.sha(O), helper_sha256=m.sha(H),
             failure_receipt_sha256=m.sha(failure_path), probe_receipt_sha256=m.sha(proof_path),
             boot_id=failure['boot_id'], binary=dict(failure['health_probe']), environment=env,
             reset_executed=False, recovery_executed=False, retry_count=0, model_launched=False,
             scope='One post-owned-logcap-stop integer H2D/kernel/D2H check; no model or failed matrix compilation retry',
             raw_text_log_budget_bytes=4 << 20, total_artifact_budget_bytes=8 << 20,
             whole_wall_budget_seconds=90)
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    start = time.monotonic()
    cursor = None
    save()
    def command(label, args, environment=BASE, wall=20):
        m.require(time.monotonic() - start < 90, 'whole deadline')
        c, so, se = owner.run(label, args, environment, W, wall=wall, cpu=wall,
                             text_cap=4 << 20, total_cap=8 << 20, rss_cap=2 << 30)
        m.require(m.closed(c) and c['exit_code'] == 0, 'normal closed ' + label)
        return so.read_text(), se.read_text()
    def relevant(text):
        rows = [json.loads(v) for v in text.splitlines() if v.startswith('{')]
        return [v for v in rows if '0000:05:00.0' in v.get('MESSAGE', '') or re.search(r'\bxe\b', v.get('MESSAGE', ''))]
    try:
        m.require(not DUMP.exists(), 'no preexisting GPU dump')
        before, _ = command('kernel-before', ['/usr/bin/journalctl', '-k', '-b', '--no-pager', '-o', 'json'])
        r['kernel_preflight_entries'] = relevant(before)
        m.require(not any(helper.FAULT.search(v['MESSAGE']) for v in r['kernel_preflight_entries']), 'no boot GPU fault')
        text, _ = command('kernel-cursor', ['/usr/bin/journalctl', '-k', '-n', '0', '--show-cursor', '--no-pager'])
        match = re.search(r'^-- cursor: (.+)$', text, re.M)
        m.require(match is not None, 'fresh cursor')
        cursor = match[1]
        r['kernel_cursor_before'] = cursor
        r['gpu_submission_attempted'] = True
        save()
        text, err = command('integer-health', [str(binary), '0000:05:00.0'], environment=env, wall=30)
        lines = text.splitlines()
        expected = ['STATUS 0000:05:00.0 0', 'QUEUE 0000:05:00.0', 'H2D 65536 complete',
                    'KERNEL 16384 complete', 'D2H 65536 complete',
                    'PASS 0000:05:00.0: 1 round, 16384 exact words']
        r['protocol'] = [v for v in lines if not v.startswith('LIBRARY ')]
        m.require(r['protocol'] == expected, 'exact integer completion')
        m.require(not re.search(r'UR_RESULT_ERROR_[A-Z_]+|ZE_RESULT_ERROR_[A-Z_]+|\[(?:ERROR|FATAL)\]', err),
                  'no runtime errors')
        paths = sorted(set(v[8:] for v in lines if v.startswith('LIBRARY ')))
        r['observed_libraries'] = [dict(path=p, sha256=m.sha(Path(p))) for p in paths]
        m.require(r['observed_libraries'] == sorted(proof['loaded_libraries'], key=lambda v: v['path']),
                  'exact qualified loaded libraries')
        r['exact_checked_words'] = 16384
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        if cursor and owner.active is None:
            try:
                text, _ = command('kernel-after', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                r['kernel_device_entries'] = relevant(text)
                r['new_fault_messages'] = [v['MESSAGE'] for v in r['kernel_device_entries'] if helper.FAULT.search(v['MESSAGE'])]
                m.require(not r['new_fault_messages'] and not DUMP.exists(), 'no fresh GPU fault/dump')
                r['kernel_gate_passed'] = True
            except BaseException as exc:
                r['kernel_gate_error'] = repr(exc)
        r['boot_unchanged'] = r['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and r.get('kernel_gate_passed') and r['boot_unchanged']
                           and not r['active'] and not r.get('error') and not r.get('kernel_gate_error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ['passed', 'error', 'protocol', 'new_fault_messages']}))
    raise SystemExit(0 if r['passed'] else 1)
