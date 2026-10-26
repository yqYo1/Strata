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
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import build_fused_profile as n

r, P, B, RESULT = n.r, n.P, n.B, n.RESULT
ROOT = B / 'prefill-expert-staging-v1'
KEY = 'prefill_upload_staging'
PREFIXES = ['include', 'src', 'tools/sycl', 'tests/sycl/expert_bank.cpp', 'cmake', 'CMakeLists.txt']
require, digest, write = n.require, n.digest, n.write


def scope(container):
    return container['event_reconciliation']['fused_command_profile']


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('project', 'runtime_files', 'commands')}
    out['project'] = {k: row['project'][k] for k in ('revision', 'count', 'aggregate_sha256')}
    return out


def prepare():
    c = r.validation.committed(RESULT)
    require(not ROOT.exists() and KEY not in scope(c), 'fresh staging gate only; inspect existing handle before retry')
    rt = r.validation.committed(r.RESULT); r.current(rt)
    baseline = r.selected.baseline()[1]
    compiler = json.loads((n.ROOT / 'record.json').read_text())['tools']['sycl']
    require(digest(compiler['path']) == compiler['sha256'], 'compiler mismatch')
    project = n.build.pin(P, PREFIXES)
    ROOT.mkdir(mode=0o700)
    for name in ('cache', 'tmp'):
        (ROOT / name).mkdir()
    settings = r.validation.committed(r.SETTINGS)
    env = dict(settings['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    sources = ['tests/sycl/expert_bank.cpp', 'src/sycl/runtime.cpp', 'src/sycl/compat.cpp',
               'src/sycl/command_profile.cpp', 'src/core/expert_cache.cpp']
    commands = [[compiler['path'], '-std=c++20', '-O3', '-fsycl', '-fsycl-id-queries-range=size_t',
        '-DSTRATA_ENABLE_SYCL=1', '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat'),
        '-MD', '-MF', str(ROOT / (str(i) + '.d')), '-c', str(P / source),
        '-o', str(ROOT / (str(i) + '.o'))] for i, source in enumerate(sources)]
    commands.append([compiler['path'], '-fsycl', '-fsycl-id-queries-range=size_t',
        *(str(ROOT / (str(i) + '.o')) for i in range(len(sources))), '-o', str(ROOT / 'sycl_expert_staging')])
    row = dict(stage='prepared-build', source_commit=project['revision'], project=project, sdk=n.sdk_inputs(),
        compiler=compiler, sources=sources, build_commands=commands, build_environment=env,
        runtime_sha256=digest(r.RESULT), runtime_files=rt['runtime_files'], preload=rt['preload'],
        boot=rt['boot'], driver=rt['driver'], selected_original=baseline,
        active=False, ownership_closed=True, passed=False, complete=False, GPU_executed=False,
        model_executed=False, model_adoption_qualified=False, native_engine_built=False,
        build_limits=dict(wall=600, cpu=1800, rss_cap=8 << 30, text_cap=8 << 20, file_cap=128 << 20, total_cap=512 << 20),
        device_limits=dict(wall=600, cpu=1200, rss_cap=6 << 30, text_cap=8 << 20, file_cap=16 << 20, total_cap=128 << 20),
        device_cases=[['baseline-smoke', '--packed-smoke'], ['staging-smoke', '--staging-smoke'],
                      ['staging-validate', '--staging-validate'],
                      ['bench0', '--staging-bench', '0'], ['bench1', '--staging-bench', '1'],
                      ['bench2', '--staging-bench', '2']],
        plan='Two traced small smoke cases,then parameter-validated staged512-expert/mixed-stride/unwind validation;'
            'three independent processes,2payloads1.9712/2.6624MB,384slots,512experts,two scans,'
            'one first-use and five rotated subsequent pairs. Actual production private-bank entrypoint both arms;'
            'two full-data queued readers per expert,641uploads+383hits+2048readers per timed batch. '
            'Not the dequant/GEMM dependency path or a model benchmark. No environment mutation/adaptive retry.',
        criteria='Execution:normal owned closure,silent parameter validation,actual pinned provider/runtime,'
            'zero fresh fault/same boot,all source/target/host guards,all readers,private-map and counters pass. '
            'Each timing batch needs wall>=10ms and queue interval>=10ms,timer p99<=10us and '
            'empty-control maximum<=5percent of batch wall;queue intervals are nonexclusive,not GPU service. '
            'For EACH payload,GM of3process subsequent-median staged/ordinary wall ratios<=0.95 and '
            'every process ratio<1. Preserve first-use and all samples. A failed/ineligible screen forbids '
            'gated full-pipeline/model advancement;passing permits the separately planned dequant/GEMM component '
            'check only,not a model trial or adoption.',
        safety_audit='Main-authored ring;no XeStrata source.4host-USM slots,<=16MiB capacity/slot,256byte guards;'
            'same in-order context/device queue and registered device ownership. Reject device sources,overlap,'
            'bad extents/queues. Copy-event wait before CPU host-slot reuse;GPU readers and scratch overwrite '
            'ordered on same queue. Catch drains before unwinding;failed drain fail-stops with owners live. '
            'No new GPU arithmetic in inference;only bounded transfer-preparation memcpy. Default disabled '
            'prefill-only route;decode/short-prefill/cache placement unchanged. No physical256K qualification.',
        retention='Keep compact individual timing/CPU samples,eligibility/correctness/fault/source results. '
            'Review component executable at gate closure or imminent full-pipeline comparison. Original selected '
            'engine+12fixtures stay intact;two earlier GU/Down fixtures retain only their named ceiling consumer. '
            'Retire routine raw traces/build output once extracted;no full API trace during long validation.')
    write(ROOT / 'record.json', row); scope(c)[KEY] = compact(row); write(RESULT, c)
    print('Prepared committed-source build plan;commit generated plan before build.', flush=True)


def admit(row):
    require(n.build.pin(P, PREFIXES)['source_sha256'] == row['project']['source_sha256'], 'source differs from frozen build')
    require(n.sdk_inputs() == row['sdk'] and digest(row['compiler']['path']) == row['compiler']['sha256'], 'SDK/compiler differs')
    require(digest(r.RESULT) == row['runtime_sha256'], 'runtime qualification differs')
    rt = r.validation.committed(r.RESULT); r.current(rt)
    require(r.selected.baseline()[1] == row['selected_original'], 'selected fallback differs')
    require(r.guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver differs')


def dependencies(row):
    expected = {str(P / k): v for k, v in row['project']['source_sha256'].items()}
    expected.update({str(Path(v['path']).resolve()): n.SYSTEM_HEADER_UPDATES.get(v['path'], (None, v['sha256']))[1]
                    for v in json.loads(n.lineage.SDK.read_text())})
    found = set()
    for i in range(len(row['sources'])):
        text = (ROOT / (str(i) + '.d')).read_text().replace('\\\n', ' ')
        found.update(str(Path(p).resolve()) for p in shlex.split(text.split(':', 1)[1]))
    require({str(P / p) for p in row['sources']} <= found and
        str(P / 'include/strata/prefill/expert_upload_staging.hpp') in found, 'native closure incomplete')
    require(all(p in expected and digest(p) == expected[p] for p in found), 'unplanned/changed compiled input')
    write(ROOT / 'compiler-inputs.json', {p: digest(p) for p in sorted(found)})
    return dict(files=len(found), path=str(ROOT / 'compiler-inputs.json'), sha256=digest(ROOT / 'compiler-inputs.json'))


def summarize(samples):
    require(len(samples) == 72, 'three complete24sample benchmark processes required')
    by_payload = []
    for payload in (1971200, 2662400):
        ratios = []
        for process in range(3):
            pair = []
            for arm in (0, 1):
                rows = [v for v in samples if v['payload_bytes'] == payload and v['process'] == process and
                        v['arm'] == arm and v['round'] >= 0]
                require(len(rows) == 5 and all(v['eligible'] and v['guards_and_every_consumer'] and
                    v['private_map_unchanged'] and v['uploads'] == 641 and v['hits'] == 383 and
                    v['consumers'] == 2048 for v in rows), 'timing/correctness/dispatch gate failed')
                pair.append(statistics.median(v['wall_ms'] for v in rows))
            ratios.append(pair[1] / pair[0])
        gm = math.exp(math.fsum(math.log(v) for v in ratios) / 3)
        by_payload.append(dict(payload_bytes=payload, time_ratios_staged_over_ordinary=ratios,
            geometric_mean=gm, component_gain_gate=gm <= .95 and max(ratios) < 1))
    return dict(payloads=by_payload, bank_dependency_gate_passed=all(v['component_gain_gate'] for v in by_payload),
        full_dequant_GEMM_gate_passed=False, model_advancement_allowed=False)


def execute(mode):
    c = r.validation.committed(RESULT); row = json.loads((ROOT / 'record.json').read_text())
    require(scope(c)[KEY] == compact(row) and row['stage'] ==
        ('prepared-build' if mode == 'build' else 'built-not-executed'), 'fresh committed phase required')
    initial = copy.deepcopy(row); admit(initial); owner = None; cursor = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else []))
    def monitor(io, entry):
        seen = set()
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if mode == 'device':
                try: maps = Path('/proc', str(io.proc.pid), 'maps').read_text()
                except (FileNotFoundError, ProcessLookupError): maps = ''
                for p in r.executable_mappings(maps):
                    require(p in row['runtime_files'] or p == row['binary']['path'], 'unplanned executable map: ' + p)
                    if p in row['runtime_files']: seen.add(p)
            if select.select([io.proc.stdout], [], [], .01)[0]: io.read(); io.pending.clear()
        io.exit(1); entry['runtime_seen'] = sorted(seen)
    try:
        owner = r.supervision.Owner(ROOT / ('owned-' + mode), persist)
        if mode == 'build':
            for i, command in enumerate(row['build_commands']):
                e, _, err = owner.run('compile' + str(i), command, row['build_environment'], P,
                    interaction=monitor, **row['build_limits'])
                require(r.supervision.closed(e) and e['exit_code'] == 0, 'build failed: ' + err.read_text(errors='replace')[-3500:])
            row['compiled_inputs'] = dependencies(row); binary = ROOT / 'sycl_expert_staging'
            row['binary'] = dict(path=str(binary), bytes=binary.stat().st_size, sha256=digest(binary))
            row.update(stage='built-not-executed', passed=True, complete=True)
        else:
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
                row['GPU_executed'] = True; persist()
                e, out, err = owner.run(label, [row['binary']['path'], *args], env, P,
                    interaction=monitor, **row['device_limits'])
                require(r.supervision.closed(e) and e['exit_code'] == 0, 'device case failed: ' + err.read_text(errors='replace')[-2500:])
                require(r.validation.preload_evidence(err.read_text(), e['pid'])['loader'] == row['preload']['loader'], 'provider mismatch')
                neo = next(p for p in row['runtime_files'] if 'libze_intel_gpu' in p)
                require(neo in e['runtime_seen'] and (not warning.exists() or warning.stat().st_size == 0), 'runtime/validation gate failed')
                text = out.read_text(); require(('expert_bank_staging_benchmark PASS' if label.startswith('bench')
                    else 'expert_bank_fixture PASS') in text, 'success marker absent')
                if label.endswith('smoke'): require(trace.exists() and 'urEnqueue' in trace.read_text(), 'smoke trace absent')
                rows = [json.loads(v) for v in text.splitlines() if v.startswith('{')]
                if label.startswith('bench'): require(len(rows) == 24, 'benchmark sample count'); row['samples'].extend(rows)
                probe = dict(host_probe_accepted=True); r.validation.finish(probe, owner, cursor, row['boot'])
                require(probe['passed'] and probe['driver_after'] == row['driver'], 'fault/ownership gate failed')
                row['case_outcomes'].append(dict(label=label, passed=True, ownership_closed=True, fresh_faults=[], samples=len(rows)))
                persist(); print(label + ': normal closure,correctness/provider/validation/fault gates PASS', flush=True)
            row['summary'] = summarize(row['samples'])
            row.update(stage='closed', host_probe_accepted=True, complete=True)
        admit(initial)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        if mode == 'device' and owner is not None and cursor:
            r.validation.finish(row, owner, cursor, row['boot'])
            row['passed'] = bool(row['passed'] and row.get('host_probe_accepted') and row['stage'] == 'closed')
        row['active'] = bool(owner and owner.active is not None)
        row['ownership_closed'] = bool(owner and not row['active'] and all(r.supervision.closed(v) for v in owner.commands))
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist(); scope(c)[KEY] = compact(row); write(RESULT, c)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'summary', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['build'], ['device']), 'prepare|build|device')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); r.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else sys.exit(execute(sys.argv[1]))
