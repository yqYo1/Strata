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
import gdn_storage_pipeline as gdn16
import adaptive_cache_model as adaptive
import adaptive_cache_device as adaptive_device
import proc_observer

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY, MODE, STRATEGY, READ, ROWS, RUNS = None, None, None, None, 0, 0, 2
GDN_MODES = {'gdn16-throughput': ('throughput', '1'),
             'gdn32-throughput': ('throughput', '0'), 'gdn16-ttft': ('ttft', '1'),
             'gdn16-mtp16-throughput': ('throughput', '1')}
ADAPTIVE_MODES = {'adaptive-throughput': ('throughput', '1'),
    'adaptive-ttft': ('ttft', '1'), 'adaptive-fault-throughput': ('throughput', '1'),
    'adaptive-positive-throughput': ('throughput', '1')}
GDN_MODES.update(ADAPTIVE_MODES)
MTP16_MODES = {'gdn16-mtp16-throughput', 'adaptive-throughput',
    'adaptive-fault-throughput', 'adaptive-positive-throughput'}


def configure(mode):
    global ROOT, KEY, MODE, STRATEGY, READ, ROWS
    require(mode in ('ttft','throughput',*GDN_MODES),'choose a declared strategy/GDN mode')
    MODE=mode;STRATEGY=GDN_MODES[mode][0] if mode in GDN_MODES else mode
    READ=32769;ROWS=READ-1
    ROOT=(adaptive.ROOT if mode in ADAPTIVE_MODES else gdn16.ROOT if mode in GDN_MODES else candidate.ROOT)/('model-'+mode+'-v1')
    KEY=('adaptive_cache_model_entry_' if mode in ADAPTIVE_MODES else
         'gdn16_model_entry_' if mode in GDN_MODES else 'official_prefill_strategy_')+mode+'_model_v1'


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
    if MODE in ADAPTIVE_MODES:
        gate=json.loads((adaptive_device.ROOT/'record.json').read_text())
        built,providers=adaptive_device.current(gate)
        require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and not gate['active'] and
                not gate['fresh_faults'] and [v['name'] for v in gate['processes']]==['host-canary','submission'] and
                all(v['passed'] and v['native_exit_code']==0 for v in gate['processes']),
                'closed fixed-source fresh-device qualification required')
        audit=shared.runtime.validation.committed(shared.runtime.validation.INDEX,
            ('adaptive_expert_cache_request_boundary_20261102',))
        require(audit['implementation']['built'] and audit['implementation']['native_recipe_commit']==built['source_commit'] and
                audit['fixed_source_review']['reviewed_program_sha256']==built['native_sources'][adaptive.PROGRAM],
                'reviewed actual single-GPU fixed program required')
        if MODE=='adaptive-positive-throughput':
            key='adaptive_cache_model_entry_adaptive-fault-throughput_model_v1'
            previous=build.committed(RESULT)[key]
            require(previous['passed'] and previous['complete'] and previous['ownership_closed'] and not previous['active'] and
                    previous['binary']==built['binaries']['strata'] and previous['outcome']['expected_record_failure_qualified'],
                    'qualify the actual second-record failure before positive-swap lifecycle entry')
        return built,gate,providers
    if MODE in GDN_MODES:
        built=gdn16.built_module.verify();data=build.committed(RESULT)
        rows=[]
        for key,folder in ((gdn16.CHECK,'component-device-v1'),(gdn16.SPEED,'component-speed-v1')):
            row=json.loads((gdn16.ROOT/folder/'record.json').read_text())
            require(data[key]==build.compact(row) and row['passed'] and row['complete'] and
                    row['ownership_closed'] and not row['active'] and not row['fresh_faults'] and
                    row['binary']==built['binaries'][gdn16.built_module.COMPONENT] and
                    row['native_source_commit']==built['source_commit'],
                    'closed same-build production-wrapper component qualification required')
            gdn16.controls(row);rows.append(row)
        require(all(v['follow_up_eligible'] for v in rows[1]['component_decisions']) and
                [v['T'] for v in rows[1]['component_decisions']]==[4096,8192],
                'both prospective production-wrapper speed screens must pass')
        audit=shared.runtime.validation.committed(shared.runtime.validation.INDEX,
            ('gdn16_model_integration_contract_audit_20261101',))
        require(audit['main_source_gate_passed'] and audit['source_commit']==built['source_commit'] and
                all(built['native_sources'].get(n)==sha for n,sha in audit['source_sha256'].items()),
                'reviewed actual projected integration source required')
        return built,rows[0],gdn16.retained.runtime.inventory(rows[0]['runtime_manifest'])
    gate=json.loads((component.ROOT/'record.json').read_text())
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
    if MODE in GDN_MODES:
        env.update(STRATA_SYCL_PREFILL_GDN_FP16=GDN_MODES[MODE][1],
                   STRATA_SYCL_PREFILL_GDN_CHUNK64='0')
    if MODE in MTP16_MODES:
        env.update(STRATA_MTP_BATCH_F16='1', STRATA_SNAPSHOT_VERIFY='1')
    if MODE in ('adaptive-fault-throughput','adaptive-positive-throughput'):
        env['STRATA_TRACE_ADAPT']='1'
    if MODE=='adaptive-fault-throughput':
        env['STRATA_SYCL_TEST_ADAPT_RECORD_FAILURE']='1'
    require(not env.get('LD_PRELOAD') and not any('/MistVVK/' in v for v in env.values()),
            'native-only preload and official environment required')
    shared.python_runtime.require_environment(env)
    require(shared.python_runtime.selection(env) == manifest['selected'], 'Python selection changed')
    config = copy.deepcopy(template['configuration'])
    args = config['args']
    require('--prefill-strategy' not in args,'unexpected inherited strategy override')
    args.extend(['--prefill-strategy',STRATEGY])
    if MODE in ('adaptive-fault-throughput','adaptive-positive-throughput'):
        args[args.index('--adapt-swaps')+1]='8'
        require('--adapt-every' not in args,'unexpected inherited adaptation interval')
        args.extend(['--adapt-every','1'])
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
    if MODE in ADAPTIVE_MODES:
        row['criteria']+=' Fixed program only;all eight unchanged fixtures,nonprogram native source and normalized compiler/link recipe match the qualified generation. Fresh host-canary/submission passes;no new model qualification is inherited. Single GPU,scratch lease enabled,immutable safety settings. Short outputs are not sustained-decode evidence.'
        row['retention']='Keep compact model/lifecycle outcomes and source/binary/runtime identities. Retire successful traces,journals,JIT and expected-error output after evidence verification. Keep selected nine-ELF fallback and fixed nine-ELF candidate only for this imminent qualification/comparison;review and retire the extra generation at closure.'
    elif MODE in GDN_MODES:
        row['criteria']+=' New GDN modes require this generation\'s closed device and favorable repeated-speed cohorts plus reviewed source integration. Verify actual per-request FP16-storage dispatch counts when enabled in throughput and absence in TTFT/off. This is the first bounded real-model entry,not activation-range/task-quality qualification;those remain pending along with full-context and sustained decode. No clean-speed inference from this logged diagnostic.'
    if MODE in MTP16_MODES:
        row['criteria']+=' MTP FP16 changes only Prefill::draft_kv projection selection. Require successful batched mode2 callbacks and nominal chunk-cell conservation. The markers do not expose internal needed rows or batch shapes. Prompt/conversation reuse is disabled; short outputs do not qualify sustained decode or adoption.'
    if MODE=='adaptive-fault-throughput':
        row['criteria']='Expected-failure diagnostic,not a speed or successful-generation run. Use the official bench.sh thinking workload and its ordinary warmup;the injected failure may occur before any long request. Require two actual refill record attempts:ordinal1 succeeds and is committed,ordinal2 returns a nonzero real compatibility-API status after copies were queued. Native and benchmark must exit nonzero normally;no later ADAPT_COMMIT/prefill,DONE or retry. Require the source-qualified return/checked-teardown path,absence of fail-stop/termination/runtime errors,quiet parameter validation,matched native environment/provider/identity,closed owners,no fresh fault and all final model hashes. This does not emulate valid-handle provider failure or directly measure DMA completion.'
    if MODE=='adaptive-positive-throughput':
        row['criteria']+=' Additionally require observed positive submits/commits and an actual final-window pending swap committed between a completed request and the next long GEN prefill. Conserve submit/commit counts and report terminal unpublished copies separately;QUIT may drain without publishing them. Source/runtime progress does not establish byte-exact cache payload or task quality;those remain pending before adoption.'
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


def storage_route(text):
    """Actual typed recurrence markers, rather than an environment-only claim."""
    tokens=[int(v) for v in re.findall(r'^strata GDN storage: fp16;state=fp32;tokens=(\d+)$',text,re.M)]
    enabled=MODE in GDN_MODES and GDN_MODES[MODE]==('throughput','1')
    require((len(tokens)==36*((ROWS+8191)//8192) and sum(tokens)==36*ROWS and
             all(0<t<=8192 for t in tokens)) if enabled else not tokens,
            'actual GDN FP16-storage dispatch differs')
    return dict(GDN_FP16_storage_calls=len(tokens),GDN_FP16_storage_tokens=sum(tokens))


def profile_route(text):
    policy=requested_route(text)
    storage=storage_route(text)
    if STRATEGY=='throughput':
        result=original.progress(text)
        calls=[dict(re.findall(r'(\w+)=([^\s;]+)',v)) for v in text.splitlines()
               if v.startswith('strata GDN dispatch: ')]
        require(len(calls)==36*((ROWS+8191)//8192) and sum(int(v['tokens']) for v in calls)==36*ROWS
                and all(v['schedule']=='barriers3' and v['partitions']=='4' and 0<int(v['tokens'])<=8192
                        for v in calls),'actual GDN3 dispatch differs')
        result.update(strategy=STRATEGY,actual_route=policy['route'],GDN3_calls=len(calls),**storage)
        if MODE in MTP16_MODES:
            result['MTP_prefill'] = mtp_route(text, ROWS)
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
    return dict(strategy=STRATEGY,actual_route=policy['route'],rows=ROWS,dense_activations=0,**storage,
                dense_head_MTP_weights_retained=True,cache_loan_retired_and_restored=True)


def mtp_route(text, rows):
    """Successful callback coverage; early-skipped/internal rows remain unknown."""
    actual = re.findall(r'^strata serve: DRAFT_PREFILL path=(\w+) mode=(\d+) cells=(\d+)$', text, re.M)
    expected = [('batched', '2', str(min(8192, rows - offset)))
                for offset in range(0, rows, 8192)]
    require(actual == expected, 'MTP callback path/mode/chunk-cell coverage differs')
    return dict(path='batched', placement_mode=2, callbacks=len(actual), nominal_cells=rows,
                internal_needed_rows_and_shapes_observed=False)


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
    chunks=[min(8192,ROWS-offset) for offset in range(0,ROWS,8192)]
    typed=''.join(f'strata GDN storage: fp16;state=fp32;tokens={t}\n' for _ in range(36) for t in chunks)
    enabled=MODE in GDN_MODES and GDN_MODES[MODE]==('throughput','1')
    storage_route(typed if enabled else '')
    for value in ((typed.rsplit('\n',2)[0]+'\n',typed+typed,typed.replace('tokens=8192','tokens=8193')) if enabled else (typed,)):
        try:storage_route(value)
        except (RuntimeError,ValueError,KeyError):pass
        else:raise RuntimeError('invalid actual GDN storage route accepted')
    mtp_good=''.join(f'strata serve: DRAFT_PREFILL path=batched mode=2 cells={t}\n' for t in chunks)
    mtp_route(mtp_good, ROWS)
    mtp_bad=('', mtp_good+mtp_good, mtp_good.replace('batched','token'),
             mtp_good.replace('mode=2','mode=0'), mtp_good.replace('cells=8192','cells=8191'))
    for value in mtp_bad:
        try:mtp_route(value, ROWS)
        except RuntimeError:pass
        else:raise RuntimeError('invalid MTP callback route accepted')
    return dict(passed=True,valid_profile_route=1,invalid_profile_routes=len(bad),
                MTP_callback_positive=1,MTP_callback_negative=len(mtp_bad),
                GDN_storage_positive=1,GDN_storage_negative=3 if enabled else 1,GPU_executed=False,
                original_throughput_progress_controls=original.selfcheck() if STRATEGY=='throughput' else None)


def adaptive_progress(text):
    pattern=r'^strata serve: ADAPT_(SUBMIT swaps=(\d+) H2D_requested_bytes=(\d+)|COMMIT swaps=(\d+))$'
    events=list(re.finditer(pattern,text,re.M));pending=0;submits=commits=payload=0
    for event in events:
        if event[4] is not None:
            count=int(event[4]);require(count==pending>0,'adaptive publication does not match pending copies')
            commits+=count;pending=0
        else:
            count,size=int(event[2]),int(event[3])
            require(0<=count<=8 and (size>0)==(count>0),'adaptive submit count/byte evidence differs')
            if count:
                require(pending==0,'overlapping adaptive batches');pending=count
            submits+=count;payload+=size
    require(submits>0 and commits>0 and submits==commits+pending,'positive adaptive work required')
    footers=list(re.finditer(r'^strata serve: prompt \d+ tokens = \d+ reused \+ \d+ read[^\n]*$',text,re.M))
    boundaries=0
    for footer in footers[:-1]:
        previous=[v for v in events if v.end()<footer.start()]
        later=text[footer.end():]
        policy=re.search(r'^strata prefill policy: .*tokens=32768\b',later,re.M)
        if not previous or policy is None:continue
        last=previous[-1]
        if last[2] is not None and int(last[2])>0 and re.search(
                r'^strata serve: ADAPT_COMMIT swaps='+last[2]+r'$',later[:policy.start()],re.M):
            boundaries+=1
    require(boundaries>0,'no observed deferred final-window copy before the next long GEN prefill')
    return dict(submitted_swaps=submits,committed_swaps=commits,terminal_unpublished_swaps=pending,
        H2D_requested_bytes=payload,deferred_next_long_prefill_boundaries=boundaries,
        measured_PCIe_bytes=False,cache_payload_bitwise_verified=False)


def run_fault():
    row=json.loads((ROOT/'record.json').read_text());files,manifest=current(row)
    require(MODE=='adaptive-fault-throughput' and row['stage']=='frontend-qualified' and row['host_passed'] and
            not (ROOT/'owned').exists(),'fresh frontend-qualified expected-failure diagnostic only')
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    owner=None;outcome=dict(complete=False,host_probe_accepted=False,model_executed=False,
        expected_record_failure_qualified=False);mapped=set();claims=[];observer=proc_observer.ProcObserver()
    def persist():write(ROOT/'execution.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    def monitor(io,entry):
        next_sample=0.
        def observe():
            if not (ROOT/'engine-start.json').exists():return
            claim=json.loads((ROOT/'engine-start.json').read_text())['native'];pid=claim['pid']
            identity=build.supervision.identity(pid)
            if identity is None:raise ProcessLookupError('retiring native')
            require(identity['start_ticks']==claim['start_ticks'] and identity['sid']==entry['pid'] and
                    str(Path('/proc',str(pid),'exe').resolve())==row['binary']['path'],'native identity differs')
            expected=engine_once.native_environment(row,row['environment'])
            actual=dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
            require(all(actual.get(k.encode())==v.encode() for k,v in expected.items()) and
                    set(actual)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native environment differs')
            seen=shared.runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
            require(seen<=set(files)|{row['binary']['path']},'unplanned native provider')
            shared.python_runtime.check_running(entry['pid'],P/'tools/strata_bench.py',manifest['selected'],
                row['environment'],{v['realpath'] for v in manifest['inventory']['files'].values()},row['command'][2:])
            mapped.update(seen);claims.append(claim);outcome['model_executed']=True
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic()>=next_sample:
                observer.sample(observe,lambda:bool(io.exit_watch.poll(0)));next_sample=time.monotonic()+.1
            if select.select([io.proc.stdout],[],[],.02)[0]:
                io.read()
                while b'\n' in io.pending:
                    line,_,tail=io.pending.partition(b'\n');io.pending[:]=tail
                    if line.startswith(b'STRATA_PROFILE_GATE '):
                        gate=json.loads(line[len(b'STRATA_PROFILE_GATE '):])
                        require(gate['phase']=='prefill-only' and gate['stage'] in ('begin','end') and
                                gate['run'] in (0,1),'unexpected diagnostic request gate')
                        io.send(b'continue\n')
                if io.eof:io.pending.clear()
        io.proc.stdin.close();io.exit(1)
    try:
        owner=build.supervision.Owner(ROOT/'owned',persist)
        entry,_,_=owner.run('bench',row['command'],row['environment'],P,interaction=monitor,**row['budgets'])
        require(build.supervision.closed(entry) and entry['exit_code']!=0,'expected benchmark failure missing')
        start=json.loads((ROOT/'engine-start.json').read_text());end=json.loads((ROOT/'engine-exit.json').read_text())
        require(set(start)=={'binary','args','boot_id','native','wrapper','native_validation_probe'} and
                start['binary']==row['binary'] and start['args']==['--serve',*row['configuration']['args']] and
                start['boot_id']==row['boot'] and start['native_validation_probe']==row['native_validation_probe'] and
                end==dict(start,exit_code=1),'expected native failure/identity differs')
        require(claims and all(v==start['native'] for v in claims) and
                all(v['sid']==entry['pid'] for v in (start['native'],start['wrapper'])) and
                start['native']['pid']!=start['wrapper']['pid'],'native ownership evidence incomplete')
        text=(ROOT/'engine.stderr').read_text()
        actual=list(re.finditer(r'^strata: ADAPT_RECORD_DIAGNOSTIC ordinal=(\d+) injected=(\d+) status=(\d+)$',text,re.M))
        require(len(actual)==2 and actual[0].groups()==('1','0','0') and (actual[1][1],actual[1][2])==('2','1') and
                int(actual[1][3])!=0,'two actual record attempts and second API failure required')
        between=text[actual[0].end():actual[1].start()];tail=text[actual[1].end():]
        require(re.search(r'^strata serve: ADAPT_SUBMIT swaps=[1-8] H2D_requested_bytes=[1-9]\d*$',between,re.M) and
                re.search(r'^strata serve: ADAPT_COMMIT swaps=[1-8]$',between,re.M) and
                'strata serve: cannot record the adaptive refill event' in tail and
                not re.search(r'^strata (?:serve: ADAPT_COMMIT|prefill policy:|serve: prompt )',tail,re.M),
                'stale-event error propagated after publication/prefill or old completion not established')
        require(not any(s in text for s in ('Strata SYCL cleanup:','UR_RESULT_ERROR_','ZE_RESULT_ERROR_',
                'terminate called','teardown uncertain')),'unexpected failure/fail-stop')
        comparison.require_quiet_validation(ROOT)
        activation=shared.runtime.validation.preload_evidence(text,start['native']['pid'])
        required={row['native_validation_probe']['loader']['path'],row['native_validation_probe']['library']['path'],
            row['environment']['ZE_ENABLE_ALT_DRIVERS'],row['environment']['UR_ADAPTERS_FORCE_LOAD']}
        require(required<=mapped and activation['loader']==row['native_validation_probe']['loader']['path'],
                'actual native validation/provider evidence incomplete')
        current(row);assets=shared.models.model_pins(row['configuration'],row['model_assets'],full=True)
        outcome.update(complete=True,host_probe_accepted=True,expected_record_failure_qualified=True,
            native_exit_code=1,benchmark_exit_code=entry['exit_code'],native_identity=start['native'],validation=activation,
            actual_record_attempts=2,first_record_committed_before_failure=True,no_publication_or_prefill_after_failure=True,
            checked_teardown_evidence='Source-qualified normal return and absent fail-stop diagnostic with closed native/session;no direct DMA timeline or valid-handle provider failure claim.',
            final_model_hashes=len(assets),proc_observation=observer.summary(),observed_provider_count=len(mapped),
            clean_speed_qualified=False,sustained_decode_qualified=False,physical262144_qualified=False)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        if owner:shared.runtime.validation.finish(outcome,owner,cursor,row['boot'])
        if outcome.get('driver_after')!=row['driver']:outcome['passed']=False
        outcome['passed']=bool(outcome.get('passed') and outcome['complete'])
        row.update(stage='closed' if outcome['passed'] else 'failed',passed=outcome['passed'],complete=outcome['complete'],
            model_executed=outcome['model_executed'],outcome=outcome,active=bool(owner and owner.active),ownership_closed=outcome.get('ownership_closed',False))
        save(row)
    require(row['passed'],outcome.get('error','expected-record-failure diagnostic failed'))
    print('Expected second-record failure propagated with closed owners and no fresh fault.',flush=True)


def run():
    if MODE=='adaptive-fault-throughput':return run_fault()
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
        if MODE=='adaptive-positive-throughput':outcome['adaptive_work']=adaptive_progress(text)
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
    require(len(sys.argv)==3 and sys.argv[1] in ('ttft','throughput',*GDN_MODES) and
            sys.argv[2] in ('prepare','host','run'),'use a declared strategy/GDN mode and prepare|host|run')
    configure(sys.argv[1])
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([])
        {'prepare':prepare,'host':host,'run':run}[sys.argv[2]]()
