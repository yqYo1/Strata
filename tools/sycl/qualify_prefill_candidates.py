#!/usr/bin/env python3
"""Bounded prefill candidate checks using the retained official seven-ELF build.

Main-authored under the repository LICENSE. No build, inference adoption,
profiler requirement or XeStrata input. Main serializes every invocation.
"""
import fcntl
import json
from pathlib import Path
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import prefill_resources as reference
import check_moe_split as split
import matrix_shapes

P, B, RESULT = reference.P, reference.B, reference.RESULT
ROOT, KEY = B / 'official-row64-device-v1', 'official_row64_device_v1'
require, digest, write = reference.require, reference.digest, reference.write
build, component = reference.build, reference.entry.component
validation, owner_api = reference.shared.runtime.validation, build.supervision
CASES = [('tiny-row64', ['--split-row64-tiny'], 10),
         ('large-row64', ['--split-row64-smoke'], 24),
         ('route-row64', ['--split-route-smoke', '1'], 6)]


def compact(row):
    return reference.entry.compact(row)


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def prepare():
    revision = reference.shared.gen.git('rev-parse', 'HEAD').decode().strip()
    sources = matrix_shapes.sources(revision, Path(__file__).resolve())
    built, providers = reference.native_reuse(revision)
    data = build.committed(RESULT)
    require(KEY not in data and not ROOT.exists(), 'fresh candidate cohort only')
    gate, model = data[component.KEY], data[reference.KEY]
    require(model['passed'] and model['complete'] and model['ownership_closed'] and
            not model['active'] and model['binary'] == built['binaries']['strata'],
            'closed same-generation model/resource qualification required')
    ROOT.mkdir(mode=0o700)
    write(ROOT / 'runtime-inputs.json', providers)
    cases = []
    for index, (mode, args, count) in enumerate(CASES):
        root = ROOT / mode
        root.mkdir(mode=0o700)
        for part in ('tmp', 'cache'):
            (root / part).mkdir()
        env = dict(gate['environment'], TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
            STRATA_TEST_MAIN_GATES='1', STRATA_SYCL_PREFILL_GU_SWIGLU='large',
            STRATA_SYCL_PREFILL_DOWN_SCATTER='large', STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='row64',
            STRATA_SYCL_PREFILL_TOPK_EARLY='0',
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
        require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0'
                and env['ZES_ENABLE_SYSMAN'] == '0' and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1'
                and env['LD_PRELOAD'] == model['native_validation_probe']['library']['path']
                and env['STRATA_UR_PROBE_EXPECTED_LOADER'] == model['native_validation_probe']['loader']['path'],
                'qualified immutable runtime/validation safety required')
        cases.append(dict(index=index, mode=mode, root=str(root), environment=env,
            command=[built['binaries']['sycl_moe_down_scatter_bench']['path'], *args],
            expected_split_kernel_launches=dict(original=0, row128=0, row64=count)))
    row = dict(stage='prepared', source_commit=revision, sources=sources,
        native_source_commit=built['source_commit'], binaries=built['binaries'],
        native_validation_probe=model['native_validation_probe'], boot=model['boot'], driver=model['driver'],
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'), sha256=digest(ROOT / 'runtime-inputs.json'),
            files=len(providers)), cases=cases, processes=[], passed=False, complete=False,
        active=False, ownership_closed=True, model_qualified=False, inference_adopted=False,
        budgets=dict(wall=180, cpu=600, rss_cap=16 << 30, text_cap=8 << 20,
            file_cap=16 << 20, total_cap=64 << 20), whole_bytes=128 << 20,
        plan=dict(question='Qualify the previously faster row64 component and production dispatch on the current exact native generation.',
            order='Fresh tiny,large,public-route processes;stop first failure;commit each closure before next.',
            coverage='Tiny:edge values,invalid maps1/2/3,dense64bits. Large:63/64/65rows,117488-row '
                'primary/tail/final destinations,regular/irregular maps. Public:2333/2334/2335/8192rows '
                'on both maps,selected/fallback and false-return no-write. Full FP32 consumer oracle,'
                'BF16 capture,guards and immutable inputs;reuse the maintained source-bound readers.',
            acceptance='Same seven ELF/1447compiler inputs/84providers/source recipe;immutable pre-exec '
                'row64 choice with current safety controls. Three native gates,actual code maps/environment,'
                'pre-main validation,normal closure and no fresh fault. Successful UR create-to-launch '
                'identities must show exactly10/24/6row64split kernels and no original/row128split kernels.',
            limitations='Component/dispatch qualification only;no clean speed,TTFT,model quality,sustained '
                'decode,physical262144or adoption follows. No blanket minimum percentage speed gain.',
            retention='Keep compact individual outcomes,counts,source/ELF/provider identities and limits. '
                'Retire successful full UR/JIT/polling output after coverage verification;keep one useful '
                'distinct failure diagnostic. No new native build generation.'))
    save(row)
    print('Prepared3current-build row64 gates;commit plan before execution.', flush=True)


def current():
    row = json.loads((ROOT / 'record.json').read_text())
    require(compact(row) == build.committed(RESULT)[KEY], 'commit exact cohort plan/results first')
    require(matrix_shapes.sources(row['source_commit'], Path(__file__).resolve()) == row['sources'],
            'candidate controller changed')
    built, providers = reference.native_reuse(row['source_commit'])
    require(row['binaries'] == built['binaries'], 'one of seven native executables changed')
    manifest = row['runtime_manifest']
    require(digest(manifest['path']) == manifest['sha256'] and
            json.loads(Path(manifest['path']).read_text()) == providers, 'provider inventory changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot']
            and build.guards.driver_guard() == row['driver'], 'boot/driver changed')
    return row, providers


def run(index):
    row, providers = current()
    require(row['stage'] in ('prepared', 'running') and len(row['processes']) == index and
            all(p['passed'] for p in row['processes']), 'fresh next ordered gate only;no retry')
    case = row['cases'][index]
    root, env = Path(case['root']), case['environment']
    require(not (root / 'owned').exists(), 'existing execution must be inspected')
    cursor = next(s.split(': ', 1)[1] for s in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if s.startswith('-- cursor: '))
    result = dict(mode=case['mode'], complete=False, passed=False, host_probe_accepted=False,
        model_executed=False, inference_adopted=False, gates=[])
    owner, command, maps = None, None, set()
    def persist():
        write(root / 'execution.json', dict(result, commands=owner.commands if owner else [],
            active=bool(owner and owner.active)))
    def observe(pid):
        who = owner_api.identity(pid)
        require(who is not None, 'native identity missing at admission gate')
        identity = {k: who[k] for k in ('pid', 'start_ticks', 'sid')}
        require('native_identity' not in result or result['native_identity'] == identity,
                'native identity changed')
        result['native_identity'] = identity
        actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
        require(all(actual.get(k.encode()) == v.encode() for k, v in env.items()) and
                set(actual)-{k.encode() for k in env} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                'native startup environment differs')
        mapped = component.runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
        allowed = set(providers) | {str(Path(case['command'][0]).resolve())}
        require(mapped <= allowed, 'unplanned native executable/provider mapping')
        maps.update(mapped)
    def monitor(io, entry):
        next_size = 0.
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic() >= next_size:
                require(owner_api.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole-cohort output budget')
                next_size = time.monotonic() + .25
            if select.select([io.proc.stdout], [], [], .01)[0]:
                io.read()
                while b'\n' in io.pending:
                    line, _, tail = io.pending.partition(b'\n'); io.pending[:] = tail
                    if line.startswith(b'SPLIT_READY '):
                        stage = line.decode().split(' ', 1)[1]
                        require(len(result['gates']) < 3 and stage ==
                            ['before-runtime', 'runtime-ready', 'before-exit'][len(result['gates'])],
                            'unexpected native admission gate')
                        observe(entry['pid'])
                        result['gates'].append(stage)
                        persist()
                        io.send(b'X')
        io.exit(1)
    try:
        owner = owner_api.Owner(root / 'owned', persist)
        command, out, err = owner.run('target', case['command'], env, P, interaction=monitor, **row['budgets'])
        require(owner_api.closed(command) and command['exit_code'] == 0, 'native component did not close normally')
        require(result.get('native_identity', {}).get('pid') == command['pid'], 'native identity not observed')
        result['native_exit_code'] = command['exit_code']
        proof = validation.preload_evidence(err.read_text(), command['pid'])
        require(proof['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'] and result['gates'] ==
            ['before-runtime', 'runtime-ready', 'before-exit'], 'validation/phase gates incomplete')
        require({env['STRATA_UR_PROBE_EXPECTED_LOADER'], env['LD_PRELOAD'], env['ZE_ENABLE_ALT_DRIVERS'],
            '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0'} <= maps,
            'actual official provider observation incomplete')
        require('fixture_status=PASS' in out.read_text() and
            (not (root / 'validation.log').exists() or (root / 'validation.log').stat().st_size == 0),
            'fixture footer or quiet parameter validation failed')
        require(not any(t in err.read_text() for t in
            ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'CHECKED_DRAIN_FAILED', 'terminate called')),
            'runtime error or failed drain')
        details = split.parse(out.read_text(), case['mode'])
        counts = split.split_kernel_launches((root / 'ur.trace').read_text())
        require(counts == case['expected_split_kernel_launches'], 'actual selected kernel launches differ')
        require(all(digest(P / n) == sha for n, sha in row['sources'].items()) and
            all(digest(p) == sha for p, sha in providers.items()) and
            all(digest(a['path']) == a['sha256'] for a in row['binaries'].values()),
            'post-run source/provider/executable changed')
        result.update(complete=True, host_probe_accepted=True, details=details,
            actual_split_kernel_launches=counts, pre_main_validation=proof, official_providers_observed=True)
    except BaseException as error:
        result.update(error=repr(error), complete=False, host_probe_accepted=False)
    finally:
        if owner:
            validation.finish(result, owner, cursor, row['boot'])
        result['passed'] = bool(result.get('passed') and result['complete'])
        persist()
        row['processes'].append(result)
        row.update(stage='running' if result['passed'] and index < 2 else 'qualified' if result['passed'] else 'failed',
            passed=bool(result['passed'] and index == 2), complete=bool(result['passed'] and index == 2),
            active=bool(owner and owner.active), ownership_closed=result.get('ownership_closed', False))
        save(row)
    print(json.dumps({k:result.get(k) for k in ('mode', 'passed', 'complete', 'ownership_closed', 'error',
        'actual_split_kernel_launches')}), flush=True)
    require(result['passed'], 'candidate device qualification failed;stop before next process')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']:
            prepare()
        elif len(sys.argv) == 3 and sys.argv[1] == 'run' and sys.argv[2] in ('0', '1', '2'):
            run(int(sys.argv[2]))
        else:
            raise RuntimeError('qualify_prefill_candidates.py prepare|run INDEX(0..2)')
