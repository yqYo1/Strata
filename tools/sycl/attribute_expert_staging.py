#!/usr/bin/env python3
"""Main-owned transfer attribution and observer-cost gate; repository LICENSE applies.

No model execution or engine overwrite. Original source/recipe stays in Git;
source, SDK and runtime checks precede every owned build/device process.
"""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import build_fused_profile as n

r, P, B, RESULT = n.r, n.P, n.B, n.RESULT
ROOT = B / 'prefill-expert-attribution-v1'
KEY = 'prefill_upload_attribution_v1'
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
    require(scope(c).get('new_build_toolchain', {}).get('independent_build_admitted') is True,
        'SDK origin is unresolved; qualify an independently sourced compiler/runtime before another build')
    require(not ROOT.exists() and KEY not in scope(c), 'fresh staging gate only; inspect existing handle before retry')
    rt = r.validation.committed(r.RESULT); r.current(rt)
    baseline = r.selected.baseline()[1]
    candidate = scope(c)['new_build_toolchain']['official_candidate']
    compiler = dict(path=str(OFFICIAL / 'bin/icpx'), sha256=candidate['compiler_sha256'])
    require(digest(compiler['path']) == compiler['sha256'] and candidate['compiler'] == compiler['path'],
        'independent compiler mismatch')
    verified = subprocess.run(['dpkg', '--verify', *PACKAGES], capture_output=True, text=True, timeout=60)
    require(verified.returncode == 0 and not verified.stdout and not verified.stderr,
        'official package files changed')
    previous = scope(c)['prefill_upload_staging_v5']
    require(previous['ownership_closed'] and not previous['active'] and previous['complete'] and
        previous['summary']['bank_dependency_gate_passed'] is False, 'close prior failed performance screen')
    sdk_meta = copy.deepcopy(previous['sdk'])
    sdk = sdk_admit({'sdk': sdk_meta})
    project = n.build.pin(P, PREFIXES)
    ROOT.mkdir(mode=0o700)
    for name in ('cache', 'tmp'):
        (ROOT / name).mkdir()
    settings = r.validation.committed(r.SETTINGS)
    env = dict(settings['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    old_lib = Path(next(p for p in rt['runtime_files'] if '/libsycl.so.' in p)).parent
    env['LD_LIBRARY_PATH'] = ':'.join([str(OFFICIAL / 'lib'), str(UMF / 'lib'),
        *(v for v in env['LD_LIBRARY_PATH'].split(':') if v != str(old_lib))])
    env['UR_ADAPTERS_FORCE_LOAD'] = str(OFFICIAL / 'lib/libur_adapter_level_zero_v2.so.0.12.0')
    require(b'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD' in Path(env['UR_ADAPTERS_FORCE_LOAD']).read_bytes(),
        'official adapter lacks required copy-offload avoidance key')
    runtime_files = {p: h for p, h in rt['runtime_files'].items() if not Path(p).is_relative_to(old_lib)}
    for p, h in sdk.items():
        path = Path(p)
        if (path.is_relative_to(OFFICIAL) or path.is_relative_to(UMF)) and '.so' in path.name:
            with path.open('rb') as f: elf = f.read(4) == b'\x7fELF'
            if elf: runtime_files[p] = h
    preload = copy.deepcopy(rt['preload'])
    preload['loader'] = str((OFFICIAL / 'lib/libur_loader.so.0').resolve())
    dep_env = dict(os.environ, **env);dep_env.pop('LD_PRELOAD', None)
    for seed in (OFFICIAL / 'lib/libsycl.so.9.0.0', Path(preload['loader']),
                 OFFICIAL / 'lib/libur_adapter_level_zero_v2.so.0.12.0', UMF / 'lib/libumf.so.1.2.0'):
        text = subprocess.check_output(['/usr/bin/ldd', str(seed)], env=dep_env, text=True, timeout=15)
        require('not found' not in text, 'unresolved prospective runtime dependency')
        for line in text.splitlines():
            path = next((v for v in line.split() if v.startswith('/')), None)
            if path:
                path = str(Path(path).resolve());runtime_files[path] = digest(path)
    sources = ['tests/sycl/expert_bank.cpp', 'src/sycl/runtime.cpp', 'src/sycl/compat.cpp',
               'src/sycl/command_profile.cpp', 'src/core/expert_cache.cpp']
    commands = [[compiler['path'], '-std=c++20', '-O3', '-fsycl', '-fsycl-id-queries-range=size_t',
        '-DSTRATA_ENABLE_SYCL=1', '-DSTRATA_SYCL_EXPERT_UPLOAD_ATTRIBUTION=1', '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat'),
        '-MD', '-MF', str(ROOT / (str(i) + '.d')), '-c', str(P / source),
        '-o', str(ROOT / (str(i) + '.o'))] for i, source in enumerate(sources)]
    commands.append([compiler['path'], '-fsycl', '-fsycl-id-queries-range=size_t',
        *(str(ROOT / (str(i) + '.o')) for i in range(len(sources))), '-o', str(ROOT / 'sycl_expert_staging')])
    row = dict(stage='prepared-build', source_commit=project['revision'], project=project, sdk=sdk_meta,
        compiler=compiler, sources=sources, build_commands=commands, build_environment=env,
        runtime_sha256=digest(r.RESULT), runtime_files=runtime_files, preload=preload,
        runtime_scope='Proposed official SDK runtime for this NEW bounded component only;old successful '
            'runtime/engine reference is immutable. No legacy/model/GEMM/runtime qualification inherited.',
        boot=rt['boot'], driver=rt['driver'], selected_original=baseline,
        active=False, ownership_closed=True, passed=False, complete=False, GPU_executed=False,
        model_executed=False, model_adoption_qualified=False, native_engine_built=False,
        build_limits=dict(wall=600, cpu=1800, rss_cap=8 << 30, text_cap=8 << 20, file_cap=128 << 20, total_cap=512 << 20),
        device_limits=dict(wall=600, cpu=1200, rss_cap=6 << 30, text_cap=8 << 20, file_cap=16 << 20, total_cap=128 << 20),
        device_cases=[['host-canary'], ['baseline-smoke', '--packed-custom-smoke'],
                      ['staging-smoke', '--staging-custom-smoke'],
                      ['staging-validate', '--staging-custom-validate'],
                      ['bench-off0', '--staging-attribution-off', '0'],
                      ['bench-on0', '--staging-attribution-on', '0'],
                      ['bench-on1', '--staging-attribution-on', '1'],
                      ['bench-off1', '--staging-attribution-off', '1'],
                      ['bench-off2', '--staging-attribution-off', '2'],
                      ['bench-on2', '--staging-attribution-on', '2']],
        plan='Independent official2026.2 SDK;no model or original engine modification. Fresh diagnostic build '
            'with compile-only transfer subregions,disabled in normal production. Three off/on,on/off,off/on '
            'process pairs;both bank arms in every process,one first-use+nine subsequent rotated pairs per '
            'payload. CPU transfer-preparation memcpy and API submission separated from explicit wait CPU/wall '
            'and unclassified host time. Sample actual upload events plus synthetic full-data reader events '
            'after checked retirement using existing bounded command observer. Samples are selected service, '
            'not complete physical traffic or additive critical-path wall. Both timed arms call production '
            'private-bank entrypoint;641uploads,383hits,2048byte-validation readers per batch. '
            'Cold marker setup and five empty controls separate. Preserve V5failed performance gate; '
            'this experiment qualifies attribution only and cannot advance a model or staging adoption.',
        criteria='Normal owned closure,actual pinned official providers,pre-main/parameter validation, '
            'same boot/no fresh faults,all guards/readers/maps/counts.240timing samples;every batch wall '
            'and nonexclusive queue>=10ms,timer p99<=10us,empty-control max<=5percent wall. '
            'Every on sample needs complete valid disjoint CPU accounting,zero throws,641upload submission '
            'regions,staged641preparation regions versus ordinary0;one status region. '
            'GPU observer exact641eligible copies,2048eligible readers,one status;all selected services '
            'positive/ordered and complete after checked drain. Submit-clock inversion rejects queue '
            'delay only. For each payload AND bank arm,the GM of3process median on/off wall ratios<=1.02 '
            'and every process ratio<=1.05. Failed overhead gates preserve raw attribution as unqualified. '
            'No source change,adaptive retry,performance adoption or model advancement.',
        safety_audit='Main-authored ring;no XeStrata source.4host-USM slots,<=16MiB capacity/slot,256byte guards;'
            'same in-order context/device queue and registered device ownership. Reject device sources,overlap,'
            'bad extents/queues. Copy-event wait before CPU host-slot reuse;GPU readers and scratch overwrite '
            'ordered on same queue. Destructor,guard and catch drains wait the bound queue itself;Runtime.wait '
            'alone does not drain custom queues. Finite65536-step unsigned tail checks destructor completion '
            'without host-gated spin/barrier. Failed drain fail-stops with owners live. '
            'No new GPU arithmetic in inference;only bounded transfer-preparation memcpy. Default disabled '
            'prefill-only route;decode/short-prefill/cache placement unchanged. No physical256K qualification.',
        retention='Keep240individual wall/CPU/queue samples and120on-sample CPU regions/selected GPU services. '
            'Review component executable at gate closure or imminent full-pipeline comparison. Original selected '
            'engine+12fixtures stay intact;V1/V2closed fixtures are superseded and not timed arms. '
            'Two earlier GU/Down fixtures retain only their named ceiling consumer. '
            'Retire routine raw traces/build output once extracted;no full API trace during long validation.')
    write(ROOT / 'record.json', row); scope(c)[KEY] = compact(row); write(RESULT, c)
    print('Prepared committed-source build plan;commit generated plan before build.', flush=True)


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


def dependencies(row):
    expected = {str(P / k): v for k, v in row['project']['source_sha256'].items()}
    expected.update(sdk_admit(row))
    found = set()
    for i in range(len(row['sources'])):
        text = (ROOT / (str(i) + '.d')).read_text().replace('\\\n', ' ')
        found.update(str(Path(p).resolve()) for p in shlex.split(text.split(':', 1)[1]))
    require({str(P / p) for p in row['sources']} <= found and
        str(P / 'include/strata/prefill/expert_upload_staging.hpp') in found, 'native closure incomplete')
    require(all(p in expected and digest(p) == expected[p] for p in found), 'unplanned/changed compiled input')
    write(ROOT / 'compiler-inputs.json', {p: digest(p) for p in sorted(found)})
    return dict(files=len(found), path=str(ROOT / 'compiler-inputs.json'), sha256=digest(ROOT / 'compiler-inputs.json'))


def parsed_fields(line):
    return dict(part.split('=', 1) for part in line.split() if '=' in part)


def attribution(stderr, rows):
    import re
    starts = list(re.finditer(r'^staging attribution sample: (.*)$', stderr, re.M))
    require(len(starts) == len(rows) == 40, 'one attribution scope per batch')
    for i, start in enumerate(starts):
        text = stderr[start.end(): starts[i+1].start() if i+1 < len(starts) else len(stderr)]
        ident = parsed_fields(start.group(1)); sample = rows[i]
        require(all(int(ident[k]) == sample[v] for k, v in
            [('process','process'),('round','round'),('arm','arm'),('payload','payload_bytes')]),
            'attribution scope/sample mismatch')
        groups = {}
        for prefix, key in [('strata prefill cpu wait: ', 'wait'),
                            ('strata prefill cpu host: ', 'host'),
                            ('strata command profile: ', 'gpu')]:
            lines = [v[len(prefix):] for v in text.splitlines() if v.startswith(prefix)]
            require(len(lines) == 1, 'one complete attribution footer: ' + key)
            groups[key] = parsed_fields(lines[0])
        w, h, g = groups['wait'], groups['host'], groups['gpu']
        require(w['available'] == w['complete'] == h['available'] == '1' and
            w['open_waits'] == w['wait_throws'] == h['open_regions'] == '0', 'invalid CPU attribution')
        require(h['upload_submit_count'] == '641' and h['source_copy_count'] ==
            ('641' if sample['arm'] else '0') and h['moe_status_count'] == '1', 'wrong CPU population')
        require(all(v == '0' for k, v in h.items() if k.endswith('_throws')), 'host attribution exception')
        require(g['service_complete'] == '1' and g['windows'] == g['expected'] == '1' and
            g['error'] == 'none' and g['invalid'] == '0', 'invalid GPU service observation')
        service = {}
        for line in text.splitlines():
            if line.startswith('strata command sample: '):
                fields = parsed_fields(line); kind = fields['kind']
                delta = int(fields['end']) - int(fields['start'])
                require(delta > 0, 'nonpositive service')
                service.setdefault(kind, []).append(delta)
        populations = [parsed_fields(v) for v in text.splitlines() if v.startswith('strata command group: ')]
        require({v['kind']:int(v['eligible']) for v in populations} ==
            {'expert_upload':641, 'gather':2048, 'status':1}, 'incomplete GPU populations')
        require(4 <= len(service.get('expert_upload', [])) <= 8 and
            len(service.get('gather', [])) == 8 and len(service.get('status', [])) == 1,
            'unexpected selected service population')
        sample['cpu'] = {k[:-2]+'_ms':float(w[k])*1000 for k in
            ('wall_s','main_cpu_s','wait_cpu_s','wait_wall_s','outside_wait_cpu_s')}
        for kind in ('source_copy','upload_submit','moe_status','unclassified'):
            for clock in ('cpu','wall'):
                sample['cpu'][kind+'_'+clock+'_ms'] = float(h[kind+'_'+clock+'_s'])*1000
        sample['selected_gpu_service_ns'] = service
        sample['queue_delay_qualified'] = g['submit_clock_qualified'] == '1'
    return rows


def summarize(samples):
    require(len(samples) == 240, 'six complete40sample processes required')
    require({(v['payload_bytes'],v['process'],v['arm'],v['round'],v['attribution']) for v in samples} ==
        {(size,p,arm,round,on) for size in (1971200,2662400) for p in range(3)
         for arm in (0,1) for round in range(-1,9) for on in (False,True)}, 'sample identity differs')
    require(all(v['eligible'] and v['custom_queue'] and v['guards_and_every_consumer'] and
        v['private_map_unchanged'] and v['uploads']==641 and v['hits']==383 and v['consumers']==2048 and
        v['timer_p99_ns']<=10000 and len(v['empty_control_wall_ms'])==5 for v in samples),
        'timing/dispatch/correctness gates failed')
    outcomes=[]
    for payload in (1971200,2662400):
        for arm in (0,1):
            ratios=[]
            for process in range(3):
                pair=[statistics.median(v['wall_ms'] for v in samples if v['payload_bytes']==payload and
                    v['arm']==arm and v['process']==process and v['attribution']==on and v['round']>=0)
                    for on in (False,True)]
                ratios.append(pair[1]/pair[0])
            gm=math.exp(math.fsum(math.log(v) for v in ratios)/3)
            outcomes.append(dict(payload_bytes=payload,arm=arm,time_ratios_on_over_off=ratios,
                geometric_mean=gm,observer_gate_passed=gm<=1.02 and max(ratios)<=1.05))
    return dict(observer_checks=outcomes,attribution_qualified=all(v['observer_gate_passed'] for v in outcomes),
        model_advancement_allowed=False,staging_adoption_qualified=False,historical_regression_cause='unattributed')

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
                    require(p in row['runtime_files'] or p in
                        (row['binary']['path'], row['preload']['canary']['path']), 'unplanned executable map: ' + p)
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
                if label.startswith('bench'):
                    require(len(rows) == 40, 'benchmark sample count')
                    if '-on' in label: rows = attribution(err.read_text(), rows)
                    else: require('staging attribution sample:' not in err.read_text(), 'disabled observer emitted')
                    row['samples'].extend(rows)
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
