#!/usr/bin/env python3
"""Run one GPU diagnostic, preserving its first failure and kernel context."""
import argparse, datetime, hashlib, json, os, pathlib, signal, subprocess, time

p = argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('--runtime', choices=['stock', 'custom'], default='stock')
p.add_argument('--adapter', choices=['auto', 'legacy', 'v2'], default='auto')
p.add_argument('--disable-copy-offload', action='store_true')
p.add_argument('--no-persistent-cache', action='store_true')
p.add_argument('--env', action='append', default=[])
p.add_argument('--timeout', type=float, default=45)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parent
old = root.parent / '20261006-012240'
env = os.environ.copy()
for k in list(env):
    if k.startswith(('STRATA_', 'UR_', 'SYCL_', 'ZE_')):
        env.pop(k)
env.update(json.loads((old / 'health-build.json').read_text())['env'])
if a.runtime == 'stock':
    env['LD_LIBRARY_PATH'] = ':'.join(x for x in env['LD_LIBRARY_PATH'].split(':')
                                   if '/strata-compute-runtime-' not in x)
if a.adapter != 'auto':
    suffix = '_v2' if a.adapter == 'v2' else ''
    env['UR_ADAPTERS_FORCE_LOAD'] = '/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero' + suffix + '.so.0'
if a.disable_copy_offload:
    env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] = '1'
if a.no_persistent_cache:
    env['SYCL_CACHE_PERSISTENT'] = '0'
for assignment in a.env:
    key, value = assignment.split('=', 1)
    if not key.startswith(('STRATA_', 'GGML_')):
        raise SystemExit('Only diagnostic STRATA_/GGML_ overrides are allowed')
    env[key] = value
command = a.command
if command and command[0] == '--':
    command = command[1:]
if not command:
    command = [str(old / 'health')]
prefix = root / a.name
if prefix.with_suffix('.json').exists():
    raise SystemExit('Refusing to overwrite an earlier diagnostic')
start = datetime.datetime.now().astimezone()
record = dict(start=start.isoformat(), boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              command=command, runtime=a.runtime, adapter=a.adapter,
              env={k: v for k, v in env.items() if k.startswith(('UR_', 'SYCL_', 'ONEAPI_', 'STRATA_', 'GGML_')) or k == 'LD_LIBRARY_PATH'})
binary = pathlib.Path(command[0]).resolve()
if '--args' in command:
    binary = pathlib.Path(command[command.index('--args') + 1]).resolve()
if binary.is_file():
    record['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    record['binary_path'] = str(binary)
    record['binary_mtime_ns'] = binary.stat().st_mtime_ns
def save():
    prefix.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
save()
maps = ''
with prefix.with_suffix('.stdout').open('w') as so, prefix.with_suffix('.stderr').open('w') as se:
    proc = subprocess.Popen(command, env=env, stdout=so, stderr=se, start_new_session=True)
    record['pid'] = proc.pid
    save()
    timer = time.monotonic()
    while proc.poll() is None and time.monotonic() - timer < a.timeout:
        try:
            maps = pathlib.Path(f'/proc/{proc.pid}/maps').read_text()
        except OSError:
            pass
        time.sleep(.05)
    if proc.poll() is None:
        record['timeout'] = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            record['still_live'] = True
    record.update(exit_code=proc.poll(), elapsed_s=time.monotonic() - timer,
                  end=datetime.datetime.now().astimezone().isoformat())
prefix.with_suffix('.maps').write_text(maps)
kernel = subprocess.run(['journalctl', '-k', '-b', '--no-pager', '-o', 'short-iso', '--since',
                         start.strftime('%Y-%m-%d %H:%M:%S')], text=True, capture_output=True)
prefix.with_suffix('.kernel.log').write_text(kernel.stdout)
record['kernel_journal_exit'] = kernel.returncode
if binary.is_file():
    record['binary_sha256_after'] = hashlib.sha256(binary.read_bytes()).hexdigest()
save()
print(json.dumps({k: v for k, v in record.items() if k != 'env'}, indent=2))
print(prefix.with_suffix('.stdout').read_text()[-12000:])
print(prefix.with_suffix('.stderr').read_text()[-6000:])
print('Kernel events:')
print(kernel.stdout[-8000:])
