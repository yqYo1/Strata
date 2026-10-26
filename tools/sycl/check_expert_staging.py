#!/usr/bin/env python3
"""Main-owned expert-bank staging component gate; repository LICENSE applies.

No model execution or engine overwrite. Original source/recipe stays in Git;
source, SDK and runtime checks precede every owned build/device process.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import select
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import build_fused_profile as n

r, P, B, RESULT = n.r, n.P, n.B, n.RESULT
PREVIOUS = B / 'prefill-expert-staging-v3'
ROOT = B / 'prefill-expert-staging-v4'
KEY = 'prefill_upload_staging_v4'
OFFICIAL = Path('/opt/intel/oneapi/compiler/2026.2')
UMF = Path('/opt/intel/oneapi/umf/1.2')
PACKAGES = ['intel-oneapi-dpcpp-cpp-2026.2', 'intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2',
            'intel-oneapi-umf-1.2']
PREFIXES = ['include', 'src', 'tools/sycl', 'tests/sycl/expert_bank.cpp', 'cmake', 'CMakeLists.txt']
require, digest, write = n.require, n.digest, n.write


def scope(container):
    return container['event_reconciliation']['fused_command_profile']


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('project', 'runtime_files', 'commands')}
    out['project'] = {k: row['project'][k] for k in ('revision', 'count', 'aggregate_sha256')}
    return out


def sdk_admit(row):
    manifest = row['sdk']
    require(digest(manifest['path']) == manifest['sha256'], 'official SDK snapshot differs')
    files = json.loads(Path(manifest['path']).read_text())
    require(len(files) == manifest['files'] and all(digest(p) == h for p, h in files.items()),
        'pinned official SDK/host input changed')
    require(subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n', *PACKAGES],
        text=True, timeout=15).splitlines() == manifest['packages'], 'official package versions changed')
    return files


def prepare():
    c = r.validation.committed(RESULT)
    require(not ROOT.exists() and KEY not in scope(c), 'fresh device plan only; inspect ownership before retry')
    prior = json.loads((PREVIOUS / 'record.json').read_text())
    failure = scope(c)['prefill_upload_staging_v3']
    frozen = scope(json.loads(r.validation.git('show', '377ca6194:' + str(RESULT.relative_to(P)))))['prefill_upload_staging_v3']
    for key in ('source_commit', 'compiler', 'sdk', 'compiled_inputs', 'binary', 'build_commands'):
        require(prior[key] == frozen[key], 'original native build identity/recipe differs: ' + key)
    require(prior['stage'] == 'failed' and not prior['passed'] and prior['ownership_closed'] and
        not prior['active'] and prior['GPU_executed'] is False and prior['fresh_faults'] == [] and
        len(prior['commands']) == 1 and r.supervision.closed(prior['commands'][0]) and
        prior['commands'][0]['exit_code'] == 125 and
        'unexpected loader provider' in prior['error'] and failure['stage'] == 'failed',
        'preserve and close original pre-GPU provider failure')
    rt = r.validation.committed(r.RESULT); r.current(rt)
    require(r.selected.baseline()[1] == prior['selected_original'], 'original fallback changed')
    sdk_admit(prior)
    require(digest(prior['binary']['path']) == prior['binary']['sha256'], 'exact original fixture differs')
    manifest = prior['compiled_inputs']
    require(digest(manifest['path']) == manifest['sha256'], 'original compiler manifest differs')
    compiled = json.loads(Path(manifest['path']).read_text()); native = 0
    require(len(compiled) == 724 and all(digest(p) == h for p, h in compiled.items()),
        'actual compiled inputs changed; do not reuse native build')
    for p, h in compiled.items():
        path = Path(p)
        if path.is_relative_to(P):
            raw = r.validation.git('show', prior['source_commit'] + ':' + str(path.relative_to(P)))
            require(hashlib.sha256(raw).hexdigest() == h, 'original native source not recoverable')
            native += 1
    require(native == 19, 'native source closure differs')
    loader = str((OFFICIAL / 'lib/libur_loader.so.0').resolve())
    require(loader != prior['preload']['loader'] and
        digest(loader) == digest(prior['preload']['loader']), 'expected identical-content loader files')
    project = n.build.pin(P, PREFIXES)
    ROOT.mkdir(mode=0o700)
    for name in ('cache', 'tmp'): (ROOT / name).mkdir()
    row = copy.deepcopy(prior)
    for key in ('commands', 'error', 'summary', 'case_outcomes', 'host_probe_accepted'):
        row.pop(key, None)
    row.update(stage='prepared-device', project=project, controller_commit=project['revision'],
        native_build_reused=True, native_source_commit=prior['source_commit'], native_build_passed=True,
        original_native_build_reference='377ca6194:bench/results/2026-10-15-pr6-recovery-audit/'
            'prefill-resources-current.json#event_reconciliation.fused_command_profile.prefill_upload_staging_v3',
        reuse_proof=dict(compiled_inputs_verified=724, project_inputs_recovered=19,
            external_inputs_verified=705, original_final_ELF_verified=True,
            native_source_and_recipe_unchanged=True),
        passed=False, complete=False, active=False, ownership_closed=True, GPU_executed=False,
        model_executed=False, model_adoption_qualified=False, case_outcomes=[], samples=[], fresh_faults=[])
    row['preload']['loader'] = loader
    row['build_environment'].update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    row['environment_scope'] = 'This is a device-only plan;original native build environment/recipe restores '
    row['environment_scope'] += 'from377ca6194.Only private cache/tmp and expected loader path differ;no native rebuild.'
    row['plan'] = 'V4:preserve V3failed. Provider path correction to actual SONAME file(libur_loader.so.0),'
    row['plan'] += 'same bytes as previously expected versioned file. Reuse exact617e1845a native fixture '
    row['plan'] += 'after724-input/19project-source recovery;new controller and fresh device/cache plan. ' + prior['plan']
    row['retention'] = 'V3fixture/724manifest/official10945SDKsnapshot/build receipt serve this V4qualification;'
    row['retention'] += 'retire extra objects/raw and then fixture at V4closure after compact samples/source proof. '
    row['retention'] += 'Original selected13assets and two named GU/Down ceiling fixtures stay intact.'
    write(ROOT / 'record.json', row); scope(c)[KEY] = compact(row); write(RESULT, c)
    print('Prepared NEW committed-source device plan reusing exact native build;commit before device.', flush=True)


def admit(row):
    require(n.build.pin(P, PREFIXES)['source_sha256'] == row['project']['source_sha256'], 'source differs from frozen build')
    sdk_admit(row)
    require(digest(row['compiler']['path']) == row['compiler']['sha256'], 'official compiler differs')
    require(digest(r.RESULT) == row['runtime_sha256'], 'runtime qualification differs')
    rt = r.validation.committed(r.RESULT); r.current(rt)
    require(r.selected.baseline()[1] == row['selected_original'], 'selected fallback differs')
    require(all(digest(p) == h for p, h in row['runtime_files'].items()), 'proposed runtime differs')
    require(r.guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver differs')




def summarize(samples):
    require(len(samples) == 120, 'three complete40sample benchmark processes required')
    require({(v['payload_bytes'], v['process'], v['arm'], v['round']) for v in samples} ==
        {(size, process, arm, round) for size in (1971200, 2662400) for process in range(3)
         for arm in (0, 1) for round in range(-1, 9)}, 'missing/duplicate sample identity')
    require(all(v['custom_queue'] is True and v['eligible'] and v['guards_and_every_consumer'] and
        v['private_map_unchanged'] for v in samples), 'custom queue/first-use correctness/timing gate failed')
    require(all(len(v['empty_control_wall_ms']) == 5 and
        v['empty_control_max_wall_ms'] == max(v['empty_control_wall_ms']) and
        v['marker_setup_wall_ms'] >= 0 for v in samples), 'instrumentation setup/control evidence missing')
    by_payload = []
    for payload in (1971200, 2662400):
        ratios = []
        for process in range(3):
            pair = []
            for arm in (0, 1):
                rows = [v for v in samples if v['payload_bytes'] == payload and v['process'] == process and
                        v['arm'] == arm and v['round'] >= 0]
                require(len(rows) == 9 and all(v['eligible'] and v['guards_and_every_consumer'] and
                    v['private_map_unchanged'] and v['uploads'] == 641 and v['hits'] == 383 and
                    v['consumers'] == 2048 for v in rows), 'timing/correctness/dispatch gate failed')
                pair.append(statistics.median(v['wall_ms'] for v in rows))
            ratios.append(pair[1] / pair[0])
        gm = math.exp(math.fsum(math.log(v) for v in ratios) / 3)
        by_payload.append(dict(payload_bytes=payload, time_ratios_staged_over_ordinary=ratios,
            geometric_mean=gm, component_gain_gate=gm <= .95 and max(ratios) < 1))
    return dict(payloads=by_payload, bank_dependency_gate_passed=all(v['component_gain_gate'] for v in by_payload),
        full_dequant_GEMM_gate_passed=False, model_advancement_allowed=False)


def execute():
    c = r.validation.committed(RESULT); row = json.loads((ROOT / 'record.json').read_text())
    require(scope(c)[KEY] == compact(row) and row['stage'] == 'prepared-device', 'fresh committed phase required')
    initial = copy.deepcopy(row); admit(initial); owner = None; cursor = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else []))
    def monitor(io, entry):
        seen = set()
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            try: maps = Path('/proc', str(io.proc.pid), 'maps').read_text()
            except (FileNotFoundError, ProcessLookupError): maps = ''
            for p in r.executable_mappings(maps):
                require(p in row['runtime_files'] or p in
                    (row['binary']['path'], row['preload']['canary']['path']), 'unplanned executable map: ' + p)
                if p in row['runtime_files']: seen.add(p)
            if select.select([io.proc.stdout], [], [], .01)[0]: io.read(); io.pending.clear()
        io.exit(1); entry['runtime_seen'] = sorted(seen)
    try:
        owner = r.supervision.Owner(ROOT / 'owned-device', persist)
        text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
        cursor = next(x.split(': ', 1)[1] for x in text.splitlines() if x.startswith('-- cursor: '))
        require(digest(row['binary']['path']) == row['binary']['sha256'], 'fixture differs')
        manifest = row['compiled_inputs']; require(digest(manifest['path']) == manifest['sha256'], 'manifest differs')
        require(all(digest(p) == h for p, h in json.loads(Path(manifest['path']).read_text()).items()), 'compiled inputs differ')
        row['samples'] = []; row['case_outcomes'] = []; row['complete'] = False
        for label, *args in row['device_cases']:
            admit(initial)
            trace, warning = ROOT / (label + '.trace'), ROOT / (label + '.validation')
            env = dict(row['build_environment'], LD_PRELOAD=row['preload']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader'],
                UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(warning))
            if label.endswith('smoke'):
                env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                    UR_LOG_TRACING='level:info;flush:info;output:file,' + str(trace))
            if label != 'host-canary': row['GPU_executed'] = True
            persist()
            executable = row['preload']['canary']['path'] if label == 'host-canary' else row['binary']['path']
            e, out, err = owner.run(label, [executable, *args], env, P,
                interaction=monitor, **row['device_limits'])
            require(r.supervision.closed(e) and e['exit_code'] == 0, 'device case failed: ' + err.read_text(errors='replace')[-2500:])
            require(r.validation.preload_evidence(err.read_text(), e['pid'])['loader'] == row['preload']['loader'], 'provider mismatch')
            neo = next(p for p in row['runtime_files'] if 'libze_intel_gpu' in p)
            require(not warning.exists() or warning.stat().st_size == 0, 'parameter validation gate failed')
            text = out.read_text()
            if label == 'host-canary':
                require(json.loads(text) == dict(pid=e['pid'], initialized=0, checked=46, expected=46,
                    version=(1 << 32) - 1, closed=0), 'official loader host canary failed')
            else:
                require(neo in e['runtime_seen'] and {str(OFFICIAL / 'lib/libsycl.so.9.0.0'),
                    str(OFFICIAL / 'lib/libur_adapter_level_zero_v2.so.0.12.0')} <= set(e['runtime_seen']),
                    'actual official SYCL/adapter/NEO provider absent')
                require(('expert_bank_staging_benchmark PASS' if label.startswith('bench')
                    else 'expert_bank_fixture PASS') in text, 'success marker absent')
            if label.endswith('smoke'): require(trace.exists() and 'urEnqueue' in trace.read_text(), 'smoke trace absent')
            if label == 'staging-smoke':
                require('expert_staging_custom_queue destructor_tail_complete=PASS host_copy_owners_live=PASS'
                    in text, 'custom-queue destructor retirement absent')
            rows = [json.loads(v) for v in text.splitlines() if v.startswith('{')]
            if label.startswith('bench'): require(len(rows) == 40, 'benchmark sample count'); row['samples'].extend(rows)
            probe = dict(host_probe_accepted=True); r.validation.finish(probe, owner, cursor, row['boot'])
            require(probe['passed'] and probe['driver_after'] == row['driver'], 'fault/ownership gate failed')
            row['case_outcomes'].append(dict(label=label, passed=True, ownership_closed=True, fresh_faults=[], samples=len(rows)))
            persist(); print(label + ': normal closure,correctness/provider/validation/fault gates PASS', flush=True)
        row['summary'] = summarize(row['samples'])
        row.update(stage='closed', host_probe_accepted=True, complete=True, passed=True)
        admit(initial)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        if owner is not None and cursor:
            r.validation.finish(row, owner, cursor, row['boot'])
            row['passed'] = bool(row['passed'] and row.get('host_probe_accepted') and row['stage'] == 'closed')
        row['active'] = bool(owner and owner.active is not None)
        row['ownership_closed'] = bool(owner and not row['active'] and all(r.supervision.closed(v) for v in owner.commands))
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist(); scope(c)[KEY] = compact(row); write(RESULT, c)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'summary', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['device']), 'prepare|device')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); r.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else sys.exit(execute())
