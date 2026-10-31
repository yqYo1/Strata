#!/usr/bin/env python3
"""Same-binary prefill strategy production checks; repository LICENSE.

Main-authored from this fork's owned exact-count benchmark controls. Keep
frontend, component evidence, model progress and clean throughput distinct.
"""
import copy
import fcntl
import hashlib
import json
import re
import select
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import build_official_model as build
import qualify_prefill_candidate as component
import build_prefill_candidate as candidate
import official_model_entry as original
import compare_official_prefill as comparison
import prefill_exact_checks as shared
import engine_once

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY, MODE, STRATEGY, READ, ROWS, RUNS = None, None, None, None, 0, 0, 2


def configure(mode):
    global ROOT, KEY, MODE, STRATEGY, READ, ROWS
    require(mode in ('ttft','throughput'),'choose ttft|throughput')
    MODE=mode;STRATEGY=mode
    READ=32769;ROWS=READ-1
    ROOT=candidate.ROOT/('model-'+mode+'-v1');KEY='official_prefill_strategy_'+mode+'_model_v1'


def pinned_sources(revision):
    values=shared.pins(revision)
    values.update(build.controls(revision,controller=Path(__file__).resolve()))
    return values
require, digest, write = build.require, build.digest, build.write


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('sources', 'model_assets')}
    out['source_inputs'] = dict(files=len(row['sources']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['sources'], sort_keys=True).encode()).hexdigest())
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def proof():
    gate=build.committed(RESULT)[component.KEY]
    built,_,files=component.current(gate)
    require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and not gate['active']
            and gate['fresh_faults']==[] and len(gate['processes'])==21
            and all(p['passed'] for p in gate['processes']), 'closed same-build CLI/device cohort required')
    return built,gate,files


def python_manifest():
    previous = shared.runtime.validation.committed(shared.REFERENCE,
        ('historical_regression_audit', 'benchmark_python_runtime_v4'))
    contract = previous['plan']['manifest']
    manifest = shared.python_runtime.read_manifest(contract)
    shared.python_runtime.validate_manifest(manifest)
    return contract, manifest


def prepare():
    require(not ROOT.exists() and KEY not in build.committed(RESULT), 'fresh model-entry scope only')
    built, gate, files = proof()
    contract, manifest = python_manifest()
    template = shared.runtime.validation.committed(shared.TEMPLATE)
    old = template['environment_template']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(ROOT)) for k, v in template['environment_template'].items()}
    env.pop('CPLUS_INCLUDE_PATH', None)
    env.update(PATH=str(Path(manifest['selected']['path']).parent) + ':/usr/bin:/bin',
        LD_LIBRARY_PATH=built['environment']['LD_LIBRARY_PATH'],
        UR_ADAPTERS_FORCE_LOAD=built['environment']['UR_ADAPTERS_FORCE_LOAD'],
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPYCACHEPREFIX=str(ROOT / 'cache/python'),
        STRATA_TRACE='1', STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR='1',
        STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original', STRATA_SYCL_PREFILL_TOPK_EARLY='0',
        STRATA_PREFILL_RELEASE_HEADS='1', STRATA_PREFILL_LAYER_MAJOR='1',
        STRATA_SYCL_PREFILL_GDN_BARRIERS3='1' if STRATEGY=='throughput' else '0',
        ZE_ENABLE_ALT_DRIVERS=gate['environment']['ZE_ENABLE_ALT_DRIVERS'],
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'),
        UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
        UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
    require(not env.get('LD_PRELOAD') and not any('/MistVVK/' in v for v in env.values()),
            'native-only preload and official environment required')
    shared.python_runtime.require_environment(env)
    require(shared.python_runtime.selection(env) == manifest['selected'], 'Python selection changed')
    config = copy.deepcopy(template['configuration'])
    args = config['args']
    require('--prefill-strategy' not in args,'unexpected inherited strategy override')
    args.extend(['--prefill-strategy',STRATEGY])
    require(build.basis.split.blob(build.basis.split.BASE, 'data/expert-profile.bin') ==
            (P / 'data/expert-profile.bin').read_bytes(), 'selected original expert profile changed')
    args[args.index('--expert-profile') + 1] = str(P / 'data/expert-profile.bin')
    config.update(log=str(ROOT / 'engine.stderr'), env={})
    require(args[args.index('--spec') + 1] == '4' and args[args.index('--expert-cache') + 1] == '1256' and
            args[args.index('--prefill') + 1] == '8192' and '--no-token-graph' in args,
            'selected decode/placement configuration changed')
    expected = json.loads(shared.gen.git('show', '1e676d554:' + str(shared.TEMPLATE.relative_to(P))))['model_assets']
    assets = shared.models.model_pins(config, expected, full=True)
    loader = gate['environment']['STRATA_UR_PROBE_EXPECTED_LOADER']
    probe = dict(library=dict(path=gate['environment']['LD_PRELOAD'],
        sha256=files[gate['environment']['LD_PRELOAD']]), loader=dict(path=loader, sha256=files[loader]))
    revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    sources = pinned_sources(revision)
    parser_controls = route_controls()
    ROOT.mkdir(mode=0o700)
    for sub in ('tmp', 'cache'):
        (ROOT / sub).mkdir()
    write(ROOT / 'config.json', config)
    write(ROOT / 'plan.json', dict(binary=built['binaries']['strata'], native_validation_probe=probe))
    write(ROOT / 'runtime-inputs.json', files)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    chat = Path(config['tokenizer']) / 'chat_template.jinja'
    row = dict(stage='prepared', source_commit=revision, sources=sources, native_source_commit=built['source_commit'],
        binary=built['binaries']['strata'], native_validation_probe=probe,
        native_component_source=gate['source_commit'], configuration=config, environment=env,
        mode=MODE, strategy=STRATEGY, read_tokens=READ, rows=ROWS, runs=RUNS,
        model_assets=assets, python_contract=contract, python_selected=manifest['selected'],
        template=dict(path=str(chat), sha256=digest(chat)),
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'), sha256=digest(ROOT / 'runtime-inputs.json'), files=len(files)),
        generated={n: digest(ROOT / n) for n in ('config.json', 'plan.json')},
        host_commands=[(role, [manifest['selected']['path'], str(shared.python_runtime.SOURCE),
            'probe', str(P / entry), str(chat)]) for role, entry in
            (('benchmark', 'tools/strata_bench.py'), ('server', 'serve/server.py'))],
        host_budgets=dict(wall=45, cpu=30, rss_cap=512 << 20, text_cap=512 << 10, file_cap=2 << 20, total_cap=8 << 20),
        command=['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
            '--prefill-exact-tokens', str(READ), '--think', '--runs', str(RUNS), '--profile-request-gates',
            '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
            '--out', str(ROOT / 'bench')],
        budgets=dict(wall=1200, cpu=3600, rss_cap=104 << 30, text_cap=16 << 20, file_cap=16 << 20, total_cap=64 << 20),
        boot=built['boot'], driver=built['driver'], parser_controls=parser_controls,
        host_passed=False, passed=False, complete=False,
        active=False, ownership_closed=True, model_executed=False, inference_adopted=False,
        criteria='Two current-source Python frontend probes,then two fresh bench.sh exact-input thinking-on requests '
            'at the pinned read_tokens with1..24outputs,zero reuse,normal stream/native exit,quiet final validation, '
            'immutable sources/assets/environment/providers and no new fault. Observe requested strategy/actual '
            'route and dense/head/MTP owner behavior. The existing short thinking warmup must finish before the '
            'first long-request gate;throughput requires actual GDN3 dispatch and every8-fully-submitted-MoE-job checkpoint;TTFT retains dense/head/MTP owners. '
            'Both preserve1256decode slots/spec4/chunk8192. No clean-speed,quality,sustained TG,physical262144 '
            'or full RESTORE adoption from this diagnostic.',
        retention='Compact host/progress/route/latency and failure evidence only;successful full journals,API/polling '
            'and debug output retire after verification. Runtime manifest replaces the consumed component inventory '
            'for this named model entry;current official reference and this same-ELF strategy candidate remain for imminent comparisons.')
    save(row)
    print('Prepared two frontend probes and two explicit-strategy model entries;commit before execution.', flush=True)


def current(row):
    require(build.committed(RESULT)[KEY] == compact(row), 'commit exact model-entry plan')
    built, gate, _ = proof()
    require(built['binaries']['strata'] == row['binary'] and built['source_commit'] == row['native_source_commit'],
            'official native generation changed')
    require(pinned_sources(row['source_commit']) == row['sources'], 'benchmark/controller source changed')
    require((row['mode'],row['strategy'],row['read_tokens'],row['rows'],row['runs'])==(MODE,STRATEGY,READ,ROWS,RUNS),'profile/workload plan differs')
    require(all(digest(ROOT / n) == sha for n, sha in row['generated'].items()) and
            digest(row['template']['path']) == row['template']['sha256'], 'generated/tokenizer input changed')
    files = json.loads(Path(row['runtime_manifest']['path']).read_text())
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
            all(digest(p) == sha for p, sha in files.items()), 'official provider inventory changed')
    require(build.guards.driver_guard() == row['driver'], 'driver/fault state changed')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    _, manifest = python_manifest()
    return files, manifest


def host():
    row = json.loads((ROOT / 'record.json').read_text())
    _, manifest = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'host').exists(), 'fresh frontend probes only')
    owner = None
    try:
        owner = build.supervision.Owner(ROOT / 'host')
        allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
        reports = []
        for role, command in row['host_commands']:
            entry, out, err = owner.run(role, command, row['environment'], P, **row['host_budgets'])
            require(build.supervision.closed(entry) and entry['exit_code'] == 0 and err.stat().st_size == 0, 'frontend probe failed')
            result = json.loads(out.read_text())
            require(str(Path(result['executable']).resolve()) == row['python_selected']['file']['realpath'], 'host Python differs')
            for name in [*result['modules'], *result['code_maps']]:
                path = Path(name).resolve()
                if path.is_relative_to(P):
                    require(str(path.relative_to(P)) in row['sources'], 'unfrozen project import')
                else:
                    require(str(path) in allowed, 'unqualified Python import/map')
            reports.append(dict(role=role, modules=len(result['modules']), code_maps=len(result['code_maps']), passed=True))
        row.update(stage='frontend-qualified', host_passed=True, host_results=reports)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), ownership_closed=bool(owner and owner.active is None and
            all(build.supervision.closed(e) for e in owner.commands)))
        write(ROOT / 'host-commands.json', owner.commands if owner else [])
        save(row)
    require(row['host_passed'] and row['ownership_closed'], row.get('error', 'host probes failed'))
    print('Two current-source frontend probes passed;commit before model execution.', flush=True)


def requested_route(text):
    lines=[v for v in text.splitlines() if v.startswith('strata prefill policy: ')]
    require(len(lines)==1,'missing/duplicate actual prefill policy record')
    fields=dict(re.findall(r'(\w+)=([^\s;]+)',lines[0]))
    expected='layer-major' if STRATEGY=='throughput' and ROWS>8192 else 'chunked'
    require(fields.get('requested')==STRATEGY and fields.get('route')==expected
            and int(fields['tokens'])==ROWS and 0<int(fields['chunk'])<=8192,
            'requested strategy/actual route/token extent differs')
    return fields


def profile_route(text):
    policy=requested_route(text)
    if STRATEGY=='throughput':
        result=original.progress(text)
        calls=[dict(re.findall(r'(\w+)=([^\s;]+)',v)) for v in text.splitlines()
               if v.startswith('strata GDN dispatch: ')]
        require(len(calls)==36*((ROWS+8191)//8192) and sum(int(v['tokens']) for v in calls)==36*ROWS
                and all(v['schedule']=='barriers3' and v['partitions']=='4' and 0<int(v['tokens'])<=8192
                        for v in calls),'actual GDN3 dispatch differs')
        result.update(strategy=STRATEGY,actual_route=policy['route'],GDN3_calls=len(calls))
        return result
    require(not any(v in text for v in ('strata prefill heads: target detached ',
        'strata prefill heads: draft detached ','strata mtp: prompt expert phase=',
        'strata prefill: layer-major candidate, ','strata GDN dispatch: ')),
        'TTFT entered a long-prefill detach/recurrence route')
    states=[dict(re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)',v)) for v in text.splitlines()
            if v.startswith('strata prefill dense: phase=')]
    require([v['phase'] for v in states]==['retired','cache-loan','restored']
            and all(v['residency']=='Full' and v['activations']=='0' for v in states),
            'TTFT did not retain full dense owners through checked cache borrowing')
    return dict(strategy=STRATEGY,actual_route=policy['route'],rows=ROWS,dense_activations=0,
                dense_head_MTP_weights_retained=True,cache_loan_retired_and_restored=True)


def route_controls():
    expected='layer-major' if STRATEGY=='throughput' and ROWS>8192 else 'chunked'
    good=f'strata prefill policy: requested={STRATEGY} route={expected} tokens={ROWS} chunk=8192\n'
    requested_route(good)
    bad=[good+good,'',good.replace('requested='+STRATEGY,'requested=invalid'),
         good.replace('route='+expected,'route=cache-window'),good.replace('tokens='+str(ROWS),'tokens=0'),
         good.replace('chunk=8192','chunk=0'),good.replace('chunk=8192','chunk=8193')]
    for value in bad:
        try:requested_route(value)
        except (RuntimeError,ValueError,KeyError):pass
        else:raise RuntimeError('invalid actual profile route accepted')
    return dict(passed=True,valid_profile_route=1,invalid_profile_routes=len(bad),GPU_executed=False,
                original_throughput_progress_controls=original.selfcheck() if STRATEGY=='throughput' else None)


def run():
    row = json.loads((ROOT / 'record.json').read_text())
    files, manifest = current(row)
    require(row['stage'] == 'frontend-qualified' and row['host_passed'] and not (ROOT / 'owned').exists(), 'fresh frontend-qualified model only')
    journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(v.split(': ', 1)[1] for v in journal.splitlines() if v.startswith('-- cursor: '))
    outcome, owner = dict(complete=False, host_probe_accepted=False, model_executed=False), None
    def persist():
        write(ROOT / 'execution.json', dict(outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    log_windows = []
    def interact(io, entry):
        for run_index in range(RUNS):
            offsets = {}
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(900)
                    if line.startswith('STRATA_PROFILE_GATE '):
                        break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                        dict(stage=stage, phase='prefill-only', run=run_index), 'unexpected benchmark request gate')
                claim = json.loads((ROOT / 'engine-start.json').read_text())['native']
                actual = build.supervision.identity(claim['pid'])
                require(actual and actual['start_ticks'] == claim['start_ticks'] and actual['sid'] == entry['pid'] and
                        str(Path('/proc', str(actual['pid']), 'exe').resolve()) == row['binary']['path'], 'native ownership/executable differs')
                env = engine_once.native_environment(dict(native_validation_probe=row['native_validation_probe']), row['environment'])
                observed = dict(v.split(b'=', 1) for v in Path('/proc', str(actual['pid']), 'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(observed.get(k.encode()) == v.encode() for k, v in env.items()) and
                        set(observed) - {k.encode() for k in env} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
                mapped = shared.runtime.executable_mappings(Path('/proc', str(actual['pid']), 'maps').read_text())
                require(mapped <= set(files) | {row['binary']['path']} and
                        {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'],
                         row['environment']['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'native provider mapping differs')
                shared.python_runtime.check_running(entry['pid'], P / 'tools/strata_bench.py', manifest['selected'],
                    row['environment'], {v['realpath'] for v in manifest['inventory']['files'].values()}, row['command'][2:])
                require(not (ROOT / 'validation.log').exists() or (ROOT / 'validation.log').stat().st_size == 0, 'parameter warning/error')
                offsets[stage] = (ROOT / 'engine.stderr').stat().st_size
                outcome.update(model_executed=True, native_identity=claim)
                if run_index==0 and stage=='begin':
                    warm=(ROOT/'engine.stderr').read_text()
                    labels=[dict(re.findall(r'(\w+)=([^\s;]+)',v)) for v in warm.splitlines()
                            if v.startswith('strata prefill policy: ')]
                    require(len(labels)==1 and labels[0]['requested']==STRATEGY and labels[0]['route']=='chunked'
                            and 0<int(labels[0]['tokens'])<=int(labels[0]['chunk'])<=8192
                            and 'strata prefill heads: target detached ' not in warm
                            and 'strata prefill dense: phase=bank ' not in warm,
                            'short warmup route/owner admission failed before long request')
                    outcome['warmup_route']=labels[0]
                print(STRATEGY+'32K request ' + str(run_index) + ' ' + stage, flush=True)
                io.send(b'continue\n')
            require(offsets['end'] > offsets['begin'], 'missing native request log window')
            log_windows.append(offsets)
        io.proc.stdin.close()
        deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(time.monotonic() < deadline, 'model teardown deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)
    try:
        owner = build.supervision.Owner(ROOT / 'owned', persist)
        entry, _, _ = owner.run('bench', row['command'], row['environment'], P, interaction=interact, **row['budgets'])
        require(build.supervision.closed(entry) and entry['exit_code'] == 0, 'model process failed')
        outputs = list((ROOT / 'bench').glob('*/results.json'))
        require(len(outputs) == 1, 'missing/duplicate official result')
        result = json.loads(outputs[0].read_text())
        cohort = result['results']['prefill_only']
        require(result['complete'] and not result['results']['failures'] and
                len(cohort['requests']) == len(cohort['count_preflight']) == len(log_windows) == RUNS,
                'incomplete official result')
        for sample, count in zip(cohort['requests'], cohort['count_preflight']):
            require(sample['ok'] and sample['stream_complete'] and sample['thinking_requested'] and sample['reasoning_observed'] and
                    sample['actual_read_tokens'] == sample['actual_prompt_tokens'] == count['counted_input_tokens'] == READ and
                    sample['actual_reused_tokens'] == 0 and sample['request_sha256'] == count['body_sha256'] and
                    1 <= sample['generated_tokens'] <= 24 and sample['prompt_tps_source'] == 'engine' and sample['prompt_ms'] > 0,
                    'model token/workload/clock evidence differs')
        for sample in cohort['requests']:comparison.require_valid_client_ttft(sample)
        comparison.require_quiet_validation(ROOT)
        require(shared.client.verify_request_journal(outputs[0].parent)['valid'], 'request journal invalid')
        end = shared.gen.native_end(ROOT, row, dict(config=row['configuration']), row['boot'], entry['pid'])
        raw_text = (ROOT / 'engine.stderr').read_bytes()
        text = raw_text.decode()
        validation = shared.runtime.validation.preload_evidence(text, end['native']['pid'])
        require(validation['loader'] == row['native_validation_probe']['loader']['path'] and
                not any(s in text for s in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')), 'native runtime error')
        routes = [profile_route(raw_text[w['begin']:w['end']].decode()) for w in log_windows]
        shared.models.model_pins(row['configuration'], row['model_assets'], full=True)
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, routes=routes, validation=validation,
            samples=[{k: sample[k] for k in ('actual_read_tokens', 'actual_reused_tokens', 'prompt_ms', 'prompt_tps',
                'ttft_s', 'total_s', 'generated_tokens', 'reasoning_observed', 'request_sha256', 'timing_frontend_state')} for sample in cohort['requests']],
            clean_speed_qualified=False,
            sustained_decode_qualified=False, physical262144_qualified=False)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if owner:
            shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
        if outcome.get('driver_after')!=row['driver']:outcome.update(passed=False,final_driver_mismatch=True)
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'])
        row.update(stage='closed' if outcome['passed'] else 'failed', passed=outcome['passed'], complete=outcome['complete'],
            model_executed=outcome['model_executed'], outcome=outcome, active=bool(owner and owner.active),
            ownership_closed=outcome.get('ownership_closed', False))
        save(row)
    require(row['passed'], outcome.get('error', 'model-entry check failed'))
    print('Explicit strategy model route and short decode passed;performance/adoption checks remain.', flush=True)


if __name__=='__main__':
    require(len(sys.argv)==3 and sys.argv[1] in ('ttft','throughput') and
            sys.argv[2] in ('prepare','host','run'),'use ttft|throughput prepare|host|run')
    configure(sys.argv[1])
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([])
        {'prepare':prepare,'host':host,'run':run}[sys.argv[2]]()
