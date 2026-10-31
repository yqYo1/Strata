#!/usr/bin/env python3
"""Main-owned refusal and small device gates for the private resident-MTP build.

Repository LICENSE. No model, speed sample or production selection. Native
source remains the original selected build plus the two declared diagnostics.
"""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import mtp_resident_control as native
import ur_validation as validation
import generation_engine as generation

P, B = native.P, native.B
ROOT = B / 'mtp-resident-check-v2'
RESULT = native.RESULT.with_name('mtp-resident-check-current.json')
INPUTS = sorted(set([*native.INPUTS, 'tools/sycl/check_mtp_resident_control.py',
    'tools/sycl/ur_validation.py', 'tools/sycl/generation_engine.py',
    'tools/sycl/bench_engine.py']))
REJECTION = ('resident MTP control requires fresh single-GPU serve, spec4, mtp-max-t0/1, '
             'prefill8192,1256 per-layer slots,resident131072; no caches,suffix,coupled/oracle,graphs/split')


def write(path, value):
    native.write(path, value)


def pins(revision):
    result = {}
    for name, raw in native.build.blobs(P, revision, INPUTS):
        native.guards.require((P / name).read_bytes() == raw, 'commit exact check source: ' + name)
        result[name] = native.build.digest(P / name)
    return result


def current_build():
    row = json.loads((native.ROOT / 'record.json').read_text())
    native.admission(row)
    native.guards.require(row['stage'] == 'closed_link' and row['passed'] and row['complete'] and
        row['ownership_closed'] and not row['active'] and row['dependency_closure']['uncovered_files'] == 0,
        'closed actual native compilation required')
    closure = row['actual_compiled_inputs']
    native.guards.require(len(closure) == row['dependency_closure']['files'] and
        native.fingerprint(closure) == row['dependency_closure']['sha256'], 'compiled closure seal changed')
    for path, sha in closure.items():
        native.guards.require(native.build.digest(path) == sha, 'compiled input changed: ' + path)
    for asset in row['assets'].values():
        native.guards.require(native.build.digest(asset['path']) == asset['sha256'], 'native asset changed')
    return row


def cases(binary):
    args = ['--serve', '--mtp', str(ROOT / 'absent-mtp'), '--spec', '4', '--mtp-max-t', '0',
        '--prompt-cache', '0', '--conversation-cache-mib', '0', '--suffix-draft', '0',
        '--prefill', '8192', '--expert-cache', '1256', '--expert-cache-per-layer',
        '--max-context', '131072', '--kv-resident', '131072', '--no-token-graph']
    result = []
    for cap in ('0', '1'):
        base = list(args); base[base.index('--mtp-max-t') + 1] = cap
        # Change one restriction at a time; all other controlled options match.
        for label, flag, value in [('spec', '--spec', '2'), ('cap', '--mtp-max-t', '2'),
            ('prompt-reuse', '--prompt-cache', '1'), ('session-reuse', '--conversation-cache-mib', '1'),
            ('suffix', '--suffix-draft', '1'), ('chunk', '--prefill', '4096'),
            ('auto-chunk', '--prefill', 'auto'), ('slots', '--expert-cache', '1255'),
            ('context', '--max-context', '131071'), ('resident', '--kv-resident', '131071')]:
            changed = list(base); changed[changed.index(flag) + 1] = value
            result.append(dict(name='cap' + cap + '-' + label, argv=[binary, *changed]))
        for label, flag in [('standalone', '--serve'), ('global-slots', '--expert-cache-per-layer')]:
            changed = list(base); changed.remove(flag)
            result.append(dict(name='cap' + cap + '-' + label, argv=[binary, *changed]))
        # SYCL defaults no_token_graph=true and has no CLI enable flag. Removing
        # --no-token-graph cannot exercise that guard; use a real missing owner.
        changed = list(base); offset = changed.index('--mtp'); del changed[offset:offset + 2]
        result.append(dict(name='cap' + cap + '-missing-mtp', argv=[binary, *changed]))
        for label, extra in [('coupled', ['--coupled-draft']), ('remote', ['--expert-cache-device1', '1']),
            ('peer', ['--peer-device', '0'])]:
            result.append(dict(name='cap' + cap + '-' + label, argv=[binary, *base, *extra]))
    return result


def compact(row):
    return {k: v for k, v in row.items() if k not in ('commands', 'cases', 'environment', 'runtime_pins')}


def private_seal(row):
    return native.fingerprint({k: row[k] for k in ('cases', 'environment', 'runtime_pins')})


def prepare():
    native.guards.require(not ROOT.exists(), 'fresh check root only; no retry')
    previous = generation.committed(RESULT)
    native.guards.require(previous['stage'] == 'failed' and not previous['passed'] and
        previous['ownership_closed'] and not previous['active'] and previous['fresh_faults'] == [] and
        len(previous['outcomes']) == 13 and not previous['outcomes'][-1]['passed'] and
        all(v['passed'] for v in previous['outcomes'][:-1]), 'preserve closed original refusal failure')
    full = current_build()
    reference = generation.committed(RESULT.with_name('client-resources-current.json'))
    native.guards.require(reference['passed'] and reference['ownership_closed'] and
        reference['fresh_faults'] == [] and not reference['active'], 'qualified current runtime required')
    generation.runtime_assets_unchanged(reference)
    preload = validation.preload_qualification()
    settings = generation.committed(generation.SETTINGS)
    runtime = {p: reference['runtime_asset_delta'].get(p, {}).get('candidate_sha256', item['sha256'])
        for p, item in settings['runtime_elfs'].items()}
    revision = native.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    source = pins(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    env = dict(full['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        CUDA_MODULE_LOADING='EAGER', ZES_ENABLE_SYSMAN='0',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
        UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
        UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
        LD_PRELOAD=preload['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=preload['loader']['path'])
    row = dict(stage='prepared', source_commit=revision, input_sha256=source,
        prior_failed_operation=dict(commit=revision, path=str(RESULT.relative_to(P)),
            outcome='V1 remains failed: removal of --no-token-graph cannot enable the SYCL default-off token graph.12 native refusals passed;13th returned later model-argument validation. No device fixture/GPU work. New v2 replaces this unreachable case with missing-MTP;no native/source change.'),
        native_build_sha256=native.build.digest(native.RESULT), assets=full['assets'],
        native_source_commit=full['source_commit'], base_native_source=full['base_native_source'],
        preload=preload, runtime_pins=runtime, environment=env,
        runtime_reference_sha256=native.build.digest(RESULT.with_name('client-resources-current.json')),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=native.guards.driver_guard(),
        cases=cases(full['assets']['strata']['path']), case_count=32,
        fixtures=[dict(name='submission', target='sycl_submission_test', args=['--submission']),
                  dict(name='graph', target='sycl_graph_envelope_test', args=['--profile=0'])],
        criteria='All32 invalid requests exit2 at the exact native pre-device guard with no stdout or UR context/queue/enqueue. Each native child proves qualified preload first-init. Then same-build submission passes10 cases and ordinary graph replay passes numerical/guards/alias/checked owner cleanup. Require owned normal exits,empty validation warnings,unchanged boot/driver/runtime and no fresh fault. No model,full-context,speed,ordinary runtime teardown or adoption qualification.',
        budgets=dict(wall=90, cpu=120, rss_cap=2 << 30, text_cap=8 << 20,
                     file_cap=8 << 20, total_cap=64 << 20),
        retention='Keep compact per-case/refusal and two fixture outcomes. Replay before retiring successful full traces/JIT. Native generation remains only for imminent short serve/MTP attribution.',
        active=False, ownership_closed=True, complete=False, passed=False,
        model_executed=False, model_adoption_qualified=False)
    row['private_inputs_sha256'] = private_seal(row)
    write(ROOT / 'record.json', row); write(RESULT, compact(row))
    print('Prepared32 CPU refusals and two bounded traced device fixtures; commit before run.', flush=True)


def admission(row):
    native.guards.require(RESULT.read_bytes() == native.build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))) and
        json.loads(RESULT.read_text()) == compact(row) and pins(row['source_commit']) == row['input_sha256'],
        'check differs from committed plan')
    native.guards.require(private_seal(row) == row['private_inputs_sha256'] and
        native.build.digest(RESULT.with_name('client-resources-current.json')) == row['runtime_reference_sha256'],
        'omitted private inputs or runtime reference changed')
    full = current_build()
    native.guards.require(native.build.digest(native.RESULT) == row['native_build_sha256'] and
        full['assets'] == row['assets'] and validation.preload_qualification() == row['preload'] and
        native.guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'native/runtime qualification changed')
    for p, sha in row['runtime_pins'].items():
        native.guards.require(native.build.digest(p) == sha, 'runtime input changed: ' + p)


def run():
    row = json.loads((ROOT / 'record.json').read_text()); admission(row)
    native.guards.require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'no duplicate/retry')
    original = dict(row); owner = None
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(x.split(': ', 1)[1] for x in cursor_text.splitlines() if x.startswith('-- cursor: '))
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else []))
    try:
        owner = native.supervision.Owner(ROOT / 'owned', persist); row['stage'] = 'running'
        sequence = [dict(c, kind='refusal') for c in row['cases']]
        sequence += [dict(c, kind='device', argv=[row['assets'][c['target']]['path'], *c['args']]) for c in row['fixtures']]
        for case in sequence:
            name = case['name']; validation_log = ROOT / (name + '-validation.log')
            env = dict(row['environment'], UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(validation_log))
            entry, out, err = owner.run(name, case['argv'], env, ROOT, **row['budgets'])
            stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
            ok = native.supervision.closed(entry) and entry['exit_code'] == (2 if case['kind'] == 'refusal' else 0)
            proof = validation.preload_evidence(stderr, entry['pid'])
            if case['kind'] == 'refusal':
                ok = ok and not stdout and REJECTION in stderr and not re.search(r'ur(?:ContextCreate|QueueCreate|Enqueue)', stderr)
            elif name == 'submission':
                ok = ok and len(re.findall(r'^case ', stdout, re.M)) == 10 and 'fixture_status=PASS mode=submission' in stdout
            else:
                ok = ok and 'numerical=PASS guards=PASS aliases=PASS checked_teardown=PASS support=NOT_ENABLED' in stdout
            warnings = validation_log.stat().st_size if validation_log.exists() else 0
            ok = bool(ok and not warnings)
            row.setdefault('outcomes', []).append(dict(name=name, passed=ok, exit_code=entry['exit_code'],
                ownership_closed=native.supervision.closed(entry), preload_activated=proof['first_init'],
                validation_warning_bytes=warnings,
                stdout_sha256=native.build.digest(out), stderr_sha256=native.build.digest(err)))
            native.guards.require(ok, 'native check failed: ' + name + ': ' + stderr[-1024:])
        admission(original)
        row.update(stage='closed', complete=True, host_probe_accepted=True)
    except BaseException as error:
        row.update(stage='failed', complete=False, error=repr(error))
    finally:
        if owner is not None: validation.finish(row, owner, cursor, row['boot'])
        else: row.update(passed=False, ownership_closed=True)
        row['passed'] = bool(row.get('passed') and row.get('driver_after') == row['driver'])
        persist(); write(RESULT, compact(row))
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'complete', 'ownership_closed', 'error')}), flush=True)
    return 0 if row['passed'] else 1


if __name__ == '__main__':
    native.guards.require(sys.argv[1:] in (['prepare'], ['run']), 'prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); native.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: sys.exit(run())
