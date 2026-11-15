#!/usr/bin/env python3
"""Main-only discriminating stale-tag attention gate, before model execution."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_dflash2_ring as ring

build, source, guards, ownership = ring.build, ring.source, ring.guards, ring.ownership
P, B, STATUS = ring.P, ring.B, ring.STATUS
ROOT = B / 'dflash2-mask-discriminator-v2'
INPUTS = [*ring.INPUTS, 'tools/sycl/check_dflash2_mask.py']
require, write = guards.require, build.write


def compact(row):
    out = {k: v for k, v in row.items() if k not in ['native_build', 'runtime_files', 'controller', 'commands']}
    out['controller_source'] = dict(commit=row['controller']['revision'], sha256=row['controller']['aggregate_sha256'])
    out['native_build'] = {k: row['native_build'][k] for k in ['source_commit', 'project_source_sha256', 'record']}
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'), error=c.get('error'),
        ownership_closed=ownership.closed(c), errors=c.get('errors', []), cleanup=c.get('cleanup', [])) for c in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    status = json.loads(STATUS.read_text()); status['current_sync']['dflash2_mask_check'] = compact(row)
    write(STATUS, status)


def verify(row):
    public = json.loads(source.git(P, 'show', 'HEAD:' + str(STATUS.relative_to(P))))['current_sync']
    require(public['dflash2_mask_check'] == compact(json.loads((ROOT / 'record.json').read_text())), 'commit mask plan/outcome')
    pin = row['controller']
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in source.blobs(P, pin['revision'], pin['source_sha256'])}
    require(recovered == pin['source_sha256'] and build.signature(recovered) == pin['aggregate_sha256'] and
            all(source.digest(P / p) == h for p, h in recovered.items()), 'controller changed or unrecoverable')
    native = json.loads((build.ROOT / 'record.json').read_text())
    require(build.compact(native) == row['native_build'] == public['native_feature_build'] and
            native['passed'] and native['ownership_closed'], 'native build changed')
    build.check_inputs(native)
    for binary in native['binaries'].values():
        require(Path(binary['path']).stat().st_size == binary['bytes'] and source.digest(binary['path']) == binary['sha256'], 'fixture/engine changed')
    prior = json.loads((ring.ROOT / 'record.json').read_text())
    require(source.digest(ring.ROOT / 'record.json') == row['ring_reference']['sha256'] and
            public['dflash2_ring_check'] == ring.compact(prior) and prior['passed'] and prior['ownership_closed'] and
            all(ownership.closed(c) for c in prior['commands']), 'ring evidence changed or unclosed')
    for name, binary in prior['native_build']['binaries'].items():
        require(binary == native['binaries'][name], 'prior state fixture bytes/origin changed')
    require(source.digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
            json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'], 'runtime map changed')
    for p, h in row['runtime_files'].items(): require(source.digest(p) == h, 'runtime ELF changed: ' + p)
    require(guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    guards.no_model_owners([])


def prepare():
    require(not ROOT.exists(), 'one finite discriminator;no automatic retry')
    native = json.loads((build.ROOT / 'record.json').read_text())
    status = json.loads(source.git(P, 'show', 'HEAD:' + str(STATUS.relative_to(P))))['current_sync']
    require(status['native_feature_build'] == build.compact(native) and native['passed'] and native['ownership_closed'], 'commit mask build first')
    prior = json.loads((ring.ROOT / 'record.json').read_text())
    require(status['dflash2_ring_check'] == ring.compact(prior) and prior['stage'] == 'full_passed' and prior['passed'], 'closed ring gates first')
    # No engine or kernel source is changed by this fixture addition. Check both
    # retained exact executables above and the new fixture's native source base.
    changes = source.git(P, 'diff', '--name-only', prior['native_build']['source_commit'], native['project']['revision'], '--', 'src', 'include')
    require(not changes.strip(), 'changed native implementation needs new state qualification')
    env = dict(prior['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    row = dict(stage='prepared', controller=source.pin(P, INPUTS), native_build=build.compact(native),
        ring_reference=dict(path=str(ring.ROOT / 'record.json'), sha256=source.digest(ring.ROOT / 'record.json')),
        runtime_manifest=prior['runtime_manifest'], runtime_files=prior['runtime_files'], environment=env,
        driver=prior['driver'], boot=prior['boot'], budgets=dict(wall=180, cpu=180, text_cap=64 << 20,
            file_cap=64 << 20, total_cap=128 << 20, rss_cap=4 << 30),
        oracle='Q=K=0;130 context plus8 block keys,only V[row64]=138. A live tag must give dequantized1 in all32768 elements;stale epoch or wrong absolute position must give0. Absolute error<2e-5,finite nonnegative scales and output guards. Four explicit variant/chunk combinations,12cases. No target model,text identity or speed claim.',
        original_ring_limit='The original missing_context_mask case proves execution/tag state only,not exclusion semantics.Original26case V1 PASS stays unchanged;corrected-output V2 repeats28state cases and this12case discriminator.',
        commands=[], passed=False, ownership_closed=True)
    require(row['controller']['count'] == len(INPUTS), 'commit controller closure')
    ROOT.mkdir(mode=0o700)
    for name in ['tmp', 'cache']: (ROOT / name).mkdir(mode=0o700)
    save(row); print('Prepared12case mask discriminator;commit before GPU launch.', flush=True)


def run():
    row = json.loads((ROOT / 'record.json').read_text()); require(row['stage'] == 'prepared', 'no duplicate/retry')
    verify(row)
    before = ring.device.kernel_log('-n', '0', '--show-cursor')
    cursor = next(line.split(': ', 1)[1] for line in before.splitlines() if line.startswith('-- cursor: '))
    monitor = None
    def persist():
        write(ROOT / 'active.json', dict(active=bool(monitor and monitor.active), commands=monitor.commands if monitor else []))
    try:
        monitor = ownership.Owner(ROOT / 'owned', persist)
        fixture = row['native_build']['binaries']['sycl_dflash2_mask_test']
        def inspect(io, entry):
            lines = []
            while True:
                line = io.line(seconds=165); lines.append(line); require(len(lines) <= 20, 'stdout cap')
                if line == 'DF2_MASK_WAIT': break
            expected = [f'dflash2 mask variant={v} chunk={c} {m} PASS' for v in [1, 2] for c in [64, 128]
                        for m in ['stale_epoch', 'wrong_position', 'live_control']]
            require(lines == expected + ['dflash2 mask cases=12 PASS', 'DF2_MASK_WAIT'], 'mask case sequence incomplete')
            actual = set()
            for line in Path('/proc', str(entry['pid']), 'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and 'x' in fields[1] and fields[5].startswith('/'):
                    require(not fields[5].endswith(' (deleted)'), 'mapped code removed'); actual.add(str(Path(fields[5]).resolve()))
            allowed = {str(Path(p).resolve()): h for p, h in row['runtime_files'].items()}
            allowed[str(Path(fixture['path']).resolve())] = fixture['sha256']
            required = {str(Path(p).resolve()) for p in [row['environment']['UR_ADAPTERS_FORCE_LOAD'],
                row['environment']['ZE_ENABLE_ALT_DRIVERS'], '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0']}
            require(len(actual) >= 8 and actual <= allowed.keys() and required <= actual, 'unqualified executable mappings')
            require(all(source.digest(p) == allowed[p] for p in actual), 'mapped ELF changed')
            row['mapping'] = dict(files=len(actual), sha256=build.signature({p: allowed[p] for p in sorted(actual)}))
            io.send(b'QUIT\n'); io.exit(25)
        entry, stdout, stderr = monitor.run('mask', [fixture['path'], '--supervised'], row['environment'], P,
            interaction=inspect, **row['budgets'])
        require(ownership.closed(entry) and entry['exit_code'] == 0, 'native closure failed: ' + str(entry.get('error')))
        text = stderr.read_text(errors='replace')
        require('urEnqueueKernelLaunch' in text and not re.search(r'UR_RESULT_ERROR_[A-Z_]+', text), 'UR error or missing trace')
        verify(row); row.update(passed=True, cases_passed=12)
    except BaseException as error: row['error'] = repr(error)
    finally:
        row['commands'] = monitor.commands if monitor else []
        row['ownership_closed'] = not monitor or monitor.active is None and all(ownership.closed(c) for c in row['commands'])
        try:
            after = ring.device.kernel_log('--after-cursor', cursor, '-n', '2048')
            require(len(after.splitlines()) < 2048, 'incomplete kernel journal')
            row['fresh_faults'] = [s for s in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', s)]
            require(not row['fresh_faults'] and guards.driver_guard() == row['driver'] and
                    Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'fresh fault/boot/driver change')
        except BaseException as error: row.update(passed=False, fault_error=repr(error))
        row['passed'] = bool(row['passed'] and row['ownership_closed']); row['stage'] = 'passed' if row['passed'] else 'failed'
        save(row)
    require(row['passed'], row.get('error', row.get('fault_error', 'mask gate failed')))
    print('Mask discriminator passed12cases;normal closure,no fresh fault.No target-model/speed qualification.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'run'], 'use prepare or run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prepare() if sys.argv[1] == 'prepare' else run())
