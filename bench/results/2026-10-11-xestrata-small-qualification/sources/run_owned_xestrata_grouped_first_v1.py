"""Finite root-owned first grouped XMX numeric check; no model/timing."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, re, types

B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/eval-b570-20261010')
OUT = B / 'xestrata-grouped-GU-1-first-diagnostic-root-v1'
PARENT = B / 'run_gdn_gate_factor_probe_v2.py'
HEALTH = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/tools/recover-xe.sh')

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def ident(p):
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))

def allowed(env):
    prefixes = ('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_')
    names = {'PATH', 'HOME', 'TMPDIR', 'LANG', 'LC_CTYPE', 'LC_ALL', 'LD_LIBRARY_PATH', 'NEOReadDebugKeys', 'EnableDirectSubmission', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS'}
    return {k: v for k, v in env.items() if k.startswith(prefixes) or k in names}

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not OUT.exists()
    assert sha(PARENT) == '7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
    assert sha(HEALTH) == '2f90005f79d2309747d917cff51dc53d8ee6192d68cb1feca074f84af8b0796c'
    build_path = B / 'xestrata-grouped-gemm-build-root-v2/record.json'
    build = json.loads(build_path.read_text())
    assert build['passed'] and not build['active']
    binary = Path(build['binary']['path'])
    assert ident(binary) == build['binary']
    assert not Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
    OUT.mkdir(mode=0o700)
    (OUT / 'probes').mkdir()
    m = types.ModuleType('qualified_small_owner')
    owner_source = PARENT.read_text().split('\ndef parse_probe(', 1)[0]
    exec(compile(owner_source, str(PARENT), 'exec'), m.__dict__)
    m.W = W
    owner = m.Owner(OUT)
    health = types.ModuleType('qualified_read_only_health')
    exec(compile(HEALTH.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(HEALTH), 'exec'), health.__dict__)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    record = dict(active=True, complete=False, passed=False, gpu_executed=False, model_executed=False,
                  scope='one group1/GU row1 synthetic XMX numeric qualifier; no model quality or timing; frozen FP64 mathematical gate',
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  controller=ident(Path(__file__)), owner_source=ident(PARENT), health_source=ident(HEALTH),
                  binary=ident(binary), build_receipt=ident(build_path), commands=owner.commands,
                  log_budget_bytes=64 << 20, boot_before=boot)
    def save():
        p = OUT / 'record.json.tmp'; p.write_text(json.dumps(record, indent=2) + '\n'); p.replace(OUT / 'record.json')
    owner.persist = save
    class Runner:
        def run(self, label, argv, seconds=20, env=None):
            assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and p.suffix in {'.stdout', '.stderr'}) < 64 << 20
            execution_env = allowed(os.environ if env is None else env)
            e, stdout, stderr = owner.run(label, argv, execution_env, wall=seconds, text_cap=64 << 20, file_cap=64 << 20)
            assert m.completed(e) and e['exit_code'] == 0, label + ' did not complete normally'
            assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and p.suffix in {'.stdout', '.stderr'}) < 64 << 20
            return stdout.read_text(errors='replace')
    runner = Runner(); runner.output = OUT / 'probes'; runner.calls = owner.commands
    cursor = None
    try:
        os.environ.update(NEOReadDebugKeys='1', EnableDirectSubmission='0')
        assert runner.run('embedding-state', ['/usr/bin/systemctl', '--user', 'show', 'llama-server-qwen3embed.service', '-p', 'ActiveState'], seconds=5).strip() == 'ActiveState=inactive'
        cursor = health.journal_cursor(runner, 'kernel-before')
        health.check_health(types.SimpleNamespace(bdf='0000:05:00.0'), runner, Path('/home/yayoi/.local/bin/strata-xe-health'), cursor=cursor)
        env = allowed(health.diagnostic_environment(health.health_environment()))
        env.update(NEOReadDebugKeys='1', EnableDirectSubmission='0', STRATA_GPU_PCI='0000:05:00.0')
        env['LD_LIBRARY_PATH'] += ':/opt/intel/oneapi/mkl/2026.1/lib'
        record['environment'] = env; record['gpu_executed'] = True; save()
        stdout = runner.run('grouped-GU-1', [str(binary), '--role', 'GU', '--case', '1'], seconds=120, env=env)
        record['qualifier_stdout'] = stdout
        results = [json.loads(x) for x in stdout.splitlines() if x.startswith('{')]
        record['qualifier_results'] = results
        cap = next(x for x in results if x.get('kind') == 'capability')
        value = next(x for x in results if x.get('kind') == 'qualification')
        assert cap['reported_path'] == 'XMX' and cap['xmx_available'] and cap['subgroup16']
        assert value['passed'] and value['xmx_qualified'] and value['checked'] == 1280 and value['failed'] == value['nonfinite'] == 0
        assert not value['model_quality_qualified'] and not value['performance_eligible']
        record['complete'] = True
    except BaseException as e:
        record['error'] = type(e).__name__ + ': ' + str(e)
    finally:
        if cursor and owner.active is None:
            try:
                text = runner.run('kernel-after', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor, '--no-pager', '-o', 'json'], seconds=5)
                rows = [json.loads(x) for x in text.splitlines() if x.startswith('{')]
                def is_fault(x):
                    message = x.get('MESSAGE', '')
                    hardware = '0000:05:00.0' in message or re.search(r'\bxe\b', message)
                    return hardware and health.FAULT.search(message)
                record['new_fault_messages'] = [x['MESSAGE'] for x in rows if is_fault(x)]
            except BaseException as e: record['kernel_gate_error'] = repr(e)
        record['active'] = owner.active is not None
        record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
        record['passed'] = bool(record['complete'] and not record['active'] and not record.get('error') and not record.get('kernel_gate_error') and record['boot_unchanged'] and 'new_fault_messages' in record and not record['new_fault_messages'] and all(m.completed(x) and x['exit_code'] == 0 for x in owner.commands))
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
        print(json.dumps({k: record.get(k) for k in ['active', 'complete', 'passed', 'error', 'gpu_executed', 'new_fault_messages']}))
    if not record['passed']: raise SystemExit(1)
