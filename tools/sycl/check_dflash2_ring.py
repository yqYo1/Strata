#!/usr/bin/env python3
"""Main-only finite DFlash2 state gates, before target-model/speed work.

First repeat the unchanged sixteen-case fixture, then small stock-drafter
execution with tracing. Only a committed normal small pass permits ring wrap,
rewind/parked-branch replay and a masked missing row in a separate process.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import build_upstream_features as build
import build_engine as source
import check_dflash2 as device
import measurement_guards as guards
import measurement_owner as ownership

P, B, STATUS = build.P, build.B, build.STATUS
ROOT = B / 'dflash2-ring-contract-v2'
PRIOR = B / 'dflash2-kernel-contract-v4'
REFERENCE = device.REFERENCE
PACK = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/dflash2/qwen3.8-flash-next-dflash2.q4_0.pack')
PACK_SHA = 'cda40d62d0eb8f37ed6e272142641a1d2f0de4a32f9e14d0416e7a71534849f1'
INPUTS = ['tools/sycl/check_dflash2_ring.py', 'tools/sycl/check_dflash2.py',
          'tools/sycl/build_upstream_features.py', 'tools/sycl/build_engine.py',
          'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_owner.py']
require, write = guards.require, build.write


def public():
    return json.loads(source.git(P, 'show', 'HEAD:' + str(STATUS.relative_to(P))))['current_sync']['dflash2_ring_check']


def compact(row):
    out = {k: v for k, v in row.items() if k not in ['native_build', 'runtime_files', 'controller', 'commands']}
    out['controller_source'] = dict(commit=row['controller']['revision'], sha256=row['controller']['aggregate_sha256'])
    out['native_build'] = {k: row['native_build'][k] for k in ['source_commit', 'project_source_sha256', 'record']}
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'), error=c.get('error'),
        ownership_closed=ownership.closed(c), errors=c.get('errors', []), cleanup=c.get('cleanup', []),
        direct_child_reaped=c.get('direct_child_reaped'), session_empty=c.get('session_empty'))
        for c in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    status = json.loads(STATUS.read_text())
    status['current_sync']['dflash2_ring_check'] = compact(row)
    write(STATUS, status)


def verify(row):
    require(public() == compact(json.loads((ROOT / 'record.json').read_text())), 'commit exact state-gate plan/result')
    pin = row['controller']
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in source.blobs(P, pin['revision'], pin['source_sha256'])}
    require(recovered == pin['source_sha256'] and build.signature(recovered) == pin['aggregate_sha256'] and
            all(source.digest(P / p) == h for p, h in recovered.items()), 'controller not recoverable or changed')
    native = json.loads((build.ROOT / 'record.json').read_text())
    require(build.compact(native) == row['native_build'] and native['passed'] and native['ownership_closed'], 'native build changed')
    build.check_inputs(native)
    for value in native['binaries'].values():
        require(Path(value['path']).stat().st_size == value['bytes'] and source.digest(value['path']) == value['sha256'], 'binary changed')
    require(source.digest(row['prior']['path']) == row['prior']['sha256'], 'original sixteen-case result changed')
    require(source.digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
            json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'], 'runtime map changed')
    for p, h in row['runtime_files'].items(): require(source.digest(p) == h, 'runtime ELF changed: ' + p)
    require(PACK.stat().st_size == 522458112 and source.digest(PACK) == PACK_SHA, 'stock pack changed')
    require(guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    guards.no_model_owners([])


def prepare():
    require(not ROOT.exists(), 'one finite state-gate plan;no automatic retry')
    native = json.loads((build.ROOT / 'record.json').read_text())
    prior = json.loads((PRIOR / 'record.json').read_text())
    status = json.loads(source.git(P, 'show', 'HEAD:' + str(STATUS.relative_to(P))))['current_sync']
    require(status['native_feature_build'] == build.compact(native) and native['passed'] and native['ownership_closed'], 'commit successful ring-fixture build first')
    require(status['dflash2_device_check'] == device.compact(prior) and prior['passed'] and prior['ownership_closed'] and
            prior['numerical_cases_passed'] == 16 and all(ownership.closed(c) for c in prior['commands']), 'prior sixteen-case proof not closed/committed')
    require(native['binaries']['sycl_dflash2_test']['sha256'] == prior['fixture']['sha256'], 'preserve sixteen-case executable bytes')
    build.check_inputs(native)
    reference = json.loads((REFERENCE / 'record.json').read_text())
    env = {k: reference['environment_template'][k] for k in device.ENV_KEYS}
    env.update(STRATA_DF2_RING_LOG='1', STRATA_DF2_LOG_MIB='64', STRATA_DF2_RING_STATS='1',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
        UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
        UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
    manifest = dict(path=str(REFERENCE / 'runtime-inputs.json'), sha256=source.digest(REFERENCE / 'runtime-inputs.json'))
    row = dict(stage='prepared', controller=source.pin(P, INPUTS), native_build=build.compact(native),
        prior=dict(path=str(PRIOR / 'record.json'), sha256=source.digest(PRIOR / 'record.json')),
        runtime_manifest=manifest, runtime_files=json.loads(Path(manifest['path']).read_text()), environment=env,
        environment_origin=dict(path=str(REFERENCE / 'record.json'), sha256=source.digest(REFERENCE / 'record.json')),
        pack=dict(path=str(PACK), sha256=PACK_SHA, bytes=522458112),
        driver=guards.driver_guard(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        budgets=dict(small=dict(wall=360, cpu=360, text_cap=64 << 20, file_cap=64 << 20, total_cap=256 << 20, rss_cap=8 << 30),
                     full=dict(wall=600, cpu=600, text_cap=64 << 20, file_cap=64 << 20, total_cap=384 << 20, rss_cap=8 << 30)),
        cases=dict(kernel=16, small=7, full=5), commands=[], phases={}, passed=False, ownership_closed=True,
        meaning='Stock draft weights with synthetic target embedding/Q8zero head;finite state equivalence,failed-owner containment,all13undersized token/probability span refusals before block submission with unchanged ring,all7draft widths with output guards and inactive zeros,optional probabilities.No target model,task quality,benchmark speed,phase preservation or physical full target-context qualification.')
    require(row['controller']['count'] == len(INPUTS), 'commit all controller inputs')
    require(row['driver'] == reference['driver'] and row['boot'] == reference['boot'], 'requalify changed runtime boot/driver first')
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0' and
            env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0' and env['ZES_ENABLE_SYSMAN'] == '0', 'qualified safety settings')
    ROOT.mkdir(mode=0o700)
    for phase in ['small', 'full']:
        for child in ['tmp', 'cache']: (ROOT / phase / child).mkdir(parents=True, mode=0o700)
    save(row)
    print('Prepared fixed sixteen/small/full state sequence;commit before the first small process.', flush=True)


def execute(phase):
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == ('prepared' if phase == 'small' else 'small_passed'), 'no duplicate/retry or full run without small pass')
    verify(row)
    before = device.kernel_log('-n', '0', '--show-cursor')
    cursor = next(line.split(': ', 1)[1] for line in before.splitlines() if line.startswith('-- cursor: '))
    monitor = None
    outcome = dict(passed=False, cases=0, results=[])
    previous_commands = list(row['commands'])
    def persist():
        write(ROOT / 'active.json', dict(active=bool(monitor and monitor.active),
            commands=monitor.commands if monitor else [], phase=phase))
    try:
        monitor = ownership.Owner(ROOT / phase / 'owned', persist)
        env = dict(row['environment'], TMPDIR=str(ROOT / phase / 'tmp'), XDG_CACHE_HOME=str(ROOT / phase / 'cache'))
        if phase == 'full':
            env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
            env['STRATA_TRACE'] = '1' # API path already traced by small;bound larger history to useful stage output
        cases = ['tail_transitions_2046_2047_2048', 'stock_pack_context', 'stock_block_forward',
                 'draft_output_span_admission', 'narrow_draft_output_guards',
                 'physical_drafter_tail', 'failed_owner_containment'] if phase == 'small' else [
                 'wrapped_context', 'prefix_rebuild', 'parked_branch_restore', 'fresh_replay_equivalence', 'missing_context_mask']
        steps = [('ring_' + phase, 'sycl_dflash2_ring_test', ['--' + phase, str(PACK), '--supervised'],
                  [f'dflash2 ring {name} PASS' for name in cases] + [f'dflash2 ring {phase} cases={len(cases)} PASS'], 'DF2_RING_WAIT', len(cases))]
        if phase == 'small':
            steps.insert(0, ('kernel', 'sycl_dflash2_test', ['--supervised'], ['dflash2 fixture cases=16 PASS'], 'DF2_WAIT', 16))
        for label, name, args, markers, wait_line, count in steps:
            fixture = row['native_build']['binaries'][name]
            result = dict(label=label, passed=False)
            outcome['results'].append(result)
            def inspect(io, entry):
                lines = []
                while True:
                    line = io.line(seconds=row['budgets'][phase]['wall'] - 15); lines.append(line)
                    require(len(lines) <= 32, 'fixture stdout cap')
                    if line == wait_line: break
                require(all(m in lines for m in markers), 'fixture case markers incomplete')
                actual = set()
                for line in Path('/proc', str(entry['pid']), 'maps').read_text().splitlines():
                    fields = line.split(maxsplit=5)
                    if len(fields) == 6 and 'x' in fields[1] and fields[5].startswith('/'):
                        require(not fields[5].endswith(' (deleted)'), 'mapped code removed')
                        actual.add(str(Path(fields[5]).resolve()))
                allowed = {str(Path(p).resolve()): h for p, h in row['runtime_files'].items()}
                allowed[str(Path(fixture['path']).resolve())] = fixture['sha256']
                required = {str(Path(p).resolve()) for p in [env['UR_ADAPTERS_FORCE_LOAD'],
                    env['ZE_ENABLE_ALT_DRIVERS'], '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0']}
                result.update(mapped_files=len(actual), unknown=sorted(actual - allowed.keys()), missing=sorted(required - actual))
                require(len(actual) >= 8 and not result['unknown'] and not result['missing'], 'unqualified runtime executable mappings')
                require(all(source.digest(p) == allowed[p] for p in actual), 'mapped ELF changed')
                result['actual_mapping_sha256'] = build.signature({p: allowed[p] for p in sorted(actual)})
                io.send(b'QUIT\n'); io.exit(25)
            entry, stdout, stderr = monitor.run(label, [fixture['path'], *args], env, P,
                interaction=inspect, **row['budgets'][phase])
            require(ownership.closed(entry) and entry['exit_code'] == 0, 'native closure failed: ' + str(entry.get('error')))
            text = stderr.read_text(errors='replace')
            require(not re.search(r'UR_RESULT_ERROR_[A-Z_]+', text) and
                    (phase == 'full' or 'urEnqueueKernelLaunch' in text), 'UR error or missing small API trace')
            result.update(passed=True, cases=count)
            outcome['cases'] += count
            print(json.dumps(dict(label=label, passed=True, cases=count, exit_code=0)), flush=True)
        verify(row)
        outcome['passed'] = True
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        row['commands'] = previous_commands + (monitor.commands if monitor else [])
        row['ownership_closed'] = not monitor or monitor.active is None and all(ownership.closed(c) for c in row['commands'])
        try:
            after = device.kernel_log('--after-cursor', cursor, '-n', '2048')
            require(len(after.splitlines()) < 2048, 'incomplete kernel journal window')
            outcome['fresh_faults'] = [line for line in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', line)]
            outcome['fault_probe_successful'] = True
            outcome['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot']
            require(guards.driver_guard() == row['driver'] and not outcome['fresh_faults'] and outcome['boot_unchanged'], 'fresh fault or boot/driver change')
        except BaseException as error:
            outcome.update(passed=False, fault_error=repr(error))
        outcome['passed'] = bool(outcome['passed'] and row['ownership_closed'])
        row['phases'][phase] = outcome
        row.update(stage=(phase + '_passed') if outcome['passed'] else 'failed', passed=bool(phase == 'full' and outcome['passed']))
        save(row)
    require(outcome['passed'], outcome.get('error', outcome.get('fault_error', 'state gate failed')))
    print(phase + ' state gate passed;commit before advancing. No target-model/speed qualification.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'small', 'full'], 'use prepare,small or full')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1]))
