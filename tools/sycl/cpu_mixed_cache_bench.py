#!/usr/bin/env python3
"""Owned repeated mixed-cache model screen; repository LICENSE.

The common census/trace envelope verifies the timed routes. These are charged
measurements, not clean speed or adoption. No native rebuild or XeStrata input.
"""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import cpu_mixed_cache_model as model
import coherent_decode_bench as previous

P, B, RESULT = model.P, model.B, model.RESULT
ROOT, KEY = B/'cpu-mixed-cache-speed-v1', 'cpu_mixed_cache_speed_v1'
THIS = 'tools/sycl/cpu_mixed_cache_bench.py'
ARMS = ('reference', 'candidate', 'candidate', 'reference', 'reference', 'candidate')
WORKLOADS = ('regular-code', 'varied-prose')
WORKLOAD_ID = 'cpu_mixed_cache_20261113'
FIXTURE = model.ROOT/'owned/model.stderr'
FIXTURE_SHA = '868d15a51dd8492ed38c6ee6cc0a5ba5001bbb41357577100b39e933f13df8db'
core, runner, census = previous.core, previous.runner, model.census
require, sha, signature = model.require, model.sha, previous.signature
PYTHON = previous.PYTHON


def save(row):
    core.write(ROOT/'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = core.compact(row); core.write(RESULT, data)


def basis(expected=None):
    n = model.native(expected, observers={THIS})
    row = json.loads((model.ROOT/'record.json').read_text())
    public = model.base.original.validation.committed(RESULT, (model.KEY,))
    require(public == model.base.original.previous.compact(row) and row['stage'] == 'closed'
        and all(row[k] for k in ('passed', 'complete', 'ownership_closed', 'model_executed'))
        and not row['active'] and row['native_build']['binaries'] == n['binaries'],
        'closed same-build model source/cache/state admission first')
    proof = copy.deepcopy(n['model_observer_admission'])
    source = model.base.original.device.common.source
    after = source.pin(P, model.base.original.device.common.PREFIXES)['source_sha256']
    require(THIS in proof['declared_observers'] and THIS in after, 'declare benchmark observer')
    del proof['declared_observers'][THIS]; del after[THIS]; proof['observer_source_files'] -= 1
    proof['observer_source_sha256'] = model.base.original.device.common.signature(after)
    proof['additional_observers'] = []
    require(proof == row['native_observer_admission'], 'recover original qualified model controls')
    before = row['controller']['source_sha256']
    after = source.pin(P, ['tools/sycl', 'tools/strata_tokenizer.py',
        'data/expert-profile.bin'])['source_sha256']
    require(THIS in after, 'benchmark controller input missing'); del after[THIS]
    recovered = {p: sha_ for p, sha_ in before.items()}
    import hashlib
    require({p: hashlib.sha256(raw).hexdigest() for p, raw in source.blobs(P,
        row['controller']['revision'], before)} == recovered
        and set(after)-set(before) == set() and not set(before)-set(after)
        and all(after[p] == h for p, h in before.items()), 'only this observer extends model inputs')
    result, = row['results']
    cache = result['routes']['CPU_mixed_cache_census']
    require(result['native_exit_code'] == 0 and result['ownership_closed']
        and not result['fresh_faults'] and result['boot_unchanged']
        and cache['cache_setup']['entries'] == 8704
        and [x['bound_nt1_jobs'] for x in cache['requests']] == [2844, 3157, 2844]
        and result['routes']['cache_binding_and_final_drains'], 'actual full-cache handoff required')
    providers = row['runtime_files']
    require(sha(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256']
        and all(sha(p) == h for p, h in providers.items())
        and previous.build.guards.driver_guard() == row['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'],
        'runtime/driver/boot changed')
    return n, row, providers


def configuration(row, plan):
    value = copy.deepcopy(row['configuration']); value['log'] = str(Path(plan['root'])/'engine.stderr')
    args = value['args']
    require(args[args.index('--pcie-frac')+1] == args[args.index('--adapt-swaps')+1] == '0',
        'static qualified placement in both arms')
    return value


def environment(row, plan):
    env = previous.environment(row, plan)
    env.update(STRATA_CPU_COMPACT_GU_POLICY='mixed', STRATA_CPU_DECODE_WHOLE='0',
        STRATA_CPU_COMPACT_GU_MIB='16384' if plan['kernel'] == 'candidate' else '0',
        STRATA_SYCL_VERIFY_ARENA_COPY='0', STRATA_CPU_WORK_CENSUS='1',
        STRATA_TRACE='1', STRATA_DECODE_TIMING='1', STRATA_CPU_SERVICE='0')
    return env


def invariant(row, plan):
    value = dict(configuration=configuration(row, plan), environment=environment(row, plan))
    value['environment']['STRATA_CPU_COMPACT_GU_MIB'] = '<arm>'
    return signature(json.dumps(value, sort_keys=True).replace(plan['root'], '<process>'))


def current(row):
    require(previous.reuse.build.validation.committed(RESULT, (KEY,)) == core.compact(row)
        and core.pins(row['source_commit']) == row['sources'], 'commit exact comparison inputs/state')
    n, prior, providers = basis(row['native_observer_admission'])
    require(row['binary'] == n['binaries']['strata'] and row['native_source_commit'] == n['native_source_commit']
        and row['qualified_model_sha256'] == signature(model.base.original.previous.compact(prior))
        and row['native_validation_probe'] == prior['probe'], 'qualified build/model changed')
    contract, manifest = runner.model.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected']
        and sha(row['template']['path']) == row['template']['sha256']
        and sha(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256']
        and json.loads(Path(row['runtime_manifest']['path']).read_text()) == providers,
        'benchmark/provider/template inputs changed')
    require(previous.build.supervision.artifact_bytes(ROOT) <= row['whole_bytes']
        and len(row['plans']) == len(ARMS), 'finite campaign/order changed')
    for i, plan in enumerate(row['plans']):
        root = Path(plan['root'])
        require(plan['index'] == i and plan['kernel'] == ARMS[i] and plan['adapt_swaps'] == 0
            and plan['heads'] == 1 and invariant(row, plan) == row['invariant_signature']
            and json.loads((root/'config.json').read_text()) == configuration(row, plan)
            and json.loads((root/'plan.json').read_text()) == dict(binary=row['binary'],
                native_validation_probe=row['native_validation_probe'])
            and all(sha(root/p) == h for p, h in plan['generated'].items())
            and runner.comparison.fingerprint(environment(row, plan)) == plan['environment_sha256'],
            'actual generated arm/configuration/environment changed')
    previous.shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    return providers, manifest


def observe(text, samples, arm):
    """Check every callback, including official warmups with zero decode windows."""
    require(arm in ('reference', 'candidate') and len(text) <= 64 << 20, 'bounded planned arm')
    formats = census.manifest_formats(model.device.candidate.cpu.inputs.METADATA)
    profile = P/'data/expert-profile.bin'; membership = census.expected_membership(profile, 1256)
    setup = None
    if arm == 'candidate': setup = model.startup(text, formats, 16384)
    else:
        require(not re.search(r'^CPU_COMPACT_GU_(BEGIN|PREPARED|READY|OFF)\b', text, re.M),
            'reference must not prepare optional cache')
    require(re.findall(r'^CPU_DECODE_SCHEDULE whole_mixed=(\d+) down=42 jobs=24,36 '
        r'nt=mixed_1_to_4 workers=5 host=1$', text, re.M) == ['0'], 'unchanged whole selector')
    frames, current = [], None
    for line in text.splitlines():
        if line.startswith(census.PREFIX):
            kind, fields = census.rows(line+'\n')[0]
            if kind == 'begin':
                require(current is None and len(frames) < len(samples)
                    and fields['phase'] == 'decode' and int(fields['request']) == len(frames)+1
                    and int(fields['prompt_tokens']) == samples[len(frames)]['read_tokens'], 'request binding')
                current = dict(windows=[], calls=0, jobs=0, entries=0, bindings=0, jobs_NT=[0]*8)
            elif kind == 'end':
                require(current is not None and current['calls'] == 48*len(current['windows'])
                    and int(fields['callbacks']) == current['calls']
                    and int(fields['windows']) == len(current['windows']), 'complete native windows')
                require(current['windows'] or samples[len(frames)]['generated_tokens'] == 1,
                    'only a one-token official warmup may omit decode')
                frames.append(current); current = None
            continue
        match = re.fullmatch(r'strata trace: window (\d+) (\d+)', line)
        if match and current is not None:
            pos, width = map(int, match.groups())
            require(1 <= width <= 4 and current['calls'] == 48*len(current['windows']), 'window order/width')
            if current['windows']:
                old, extent = current['windows'][-1]; require(1 <= pos-old <= extent, 'target advancement')
            current['windows'].append((pos, width))
        if not line.startswith('strata cpu schedule:'): continue
        match = model.JOINT.fullmatch(line)
        require(current is not None and match and current['windows'], 'actual callback syntax/framing')
        layer, gu, down, jobs, *tail = map(int, match.groups()[:10])
        nt, other, whole = tail[:4], tail[4], tail[5]
        width = current['windows'][-1][1]; rows = sum((i+1)*v for i, v in enumerate(nt))
        eligible = arm == 'candidate' and gu == 18 and down in (20, 42) and 0 < jobs <= 40
        require(layer == current['calls'] % 48 and current['calls']//48 == len(current['windows'])-1
            and formats[layer][:2] == (gu, down) and sum(nt) == jobs and jobs <= 10*width
            and rows <= 10*width and not any(nt[width:]) and other == whole == 0
            and match[11] == ('mixed' if eligible else 'small')
            and int(match[12]) == (nt[0] if eligible else 0), 'actual source/NT/cache route')
        current['calls'] += 1; current['jobs'] += jobs; current['entries'] += rows
        current['bindings'] += int(match[12])
        for i, count in enumerate(nt): current['jobs_NT'][i] += count
    require(current is None and len(frames) == len(samples), 'complete official requests')
    clocks = iter(re.findall(r'^strata decode timing: (\d+) windows,[^\n]* CPU ([0-9.]+)\]', text, re.M))
    bindings = []
    for frame, sample in zip(frames, samples):
        windows = len(frame['windows']); pool_ms = 0
        if windows:
            clock = next(clocks, None); require(clock is not None and int(clock[0]) == windows, 'pool clock join')
            pool_ms = float(clock[1])*windows
        bindings.append(dict(prompt_tokens=sample['read_tokens'], windows=windows,
            rows=sum(t for _, t in frame['windows']), cpu_jobs=frame['jobs'],
            cpu_entries=frame['entries'], pool_ms=pool_ms))
    require(next(clocks, None) is None, 'no unbound pool clocks')
    work = census.extract(text, bindings, formats, membership)
    require(all(f['jobs_NT'] == w['jobs_NT'] for f, w in zip(frames, work)), 'independent NT counts')
    require(arm == 'reference' or all(f['bindings'] > 0 for f, s in zip(frames, samples)
        if s['read_tokens'] >= 32768 and s['generated_tokens'] >= 1024), 'timed long requests use NT1cache')
    return dict(cache_setup=setup, requests=[dict(request=w['request'], read_tokens=s['read_tokens'],
        generated_tokens=s['generated_tokens'], windows=w['windows'], rows=w['rows'],
        CPU_jobs=w['cpu_jobs'], CPU_entries=w['cpu_entries'], GPU_entries=w['vram_entries'],
        jobs_NT=w['jobs_NT'], NT1_bindings=f['bindings'], pool_ms=w['pool_ms'],
        logical_job_blob_bytes=w['logical_job_blob_bytes']) for f, w, s in zip(frames, work, samples)],
        actual_timed_binding_checked=True, physical_traffic_or_saturation_qualified=False)


def controls():
    require(sha(FIXTURE) == FIXTURE_SHA, 'original successful collector fixture changed')
    text = FIXTURE.read_text(); samples = [dict(read_tokens=n, generated_tokens=32) for n in (32769,82,32769)]
    require([r['NT1_bindings'] for r in observe(text, samples, 'candidate')['requests']] == [2844,3157,2844],
        'actual trace positive control')
    reference = '\n'.join(s for s in text.splitlines() if not s.startswith('CPU_COMPACT_GU_'))+'\n'
    reference = re.sub(r'cache_policy=mixed bound_nt1=\d+', 'cache_policy=small bound_nt1=0', reference)
    require(all(r['NT1_bindings'] == 0 for r in observe(reference, samples, 'reference')['requests']),
        'synthetic cache0 grammar control,not a measured reference')
    records = census.rows(text); zero = []
    for kind, fields in records:
        fields = dict(fields, request='4')
        if kind == 'begin': fields['prompt_tokens'] = '74'
        if kind == 'bin':
            for k in ('calls','pool_ns','CPU','VRAM','PCIe','peer'): fields[k] = '0'
            fields['jobs_NT'] = '0,0,0,0,0,0,0,0'
        if kind == 'end':
            for k in ('windows','rows','callbacks','cpu_jobs','CPU','VRAM','PCIe','peer','pcie_jobs'): fields[k] = '0'
        zero.append(census.PREFIX+kind+' '+' '.join(k+'='+v for k,v in fields.items()))
        if kind == 'end': break
    mixed = text+'\n'+'\n'.join(zero)+'\n'; with_zero = samples+[dict(read_tokens=74, generated_tokens=1)]
    require(observe(mixed, with_zero, 'candidate')['requests'][-1]['windows'] == 0,
        'synthetic single-output warmup positive')
    wrong = [(text, samples, 'reference'), (reference, samples, 'candidate'),
        (text.replace('cache_policy=mixed', 'cache_policy=small',1), samples, 'candidate'),
        (text.replace('bound_nt1=0','bound_nt1=1',1), samples, 'candidate'),
        (text.replace('callbacks=480','callbacks=479',1), samples, 'candidate'),
        (text.replace('strata cpu schedule: layer=0 ','strata cpu schedule: layer=1 ',1), samples, 'candidate'),
        (text.replace('CPU_COMPACT_GU_READY','MISSING_READY',1), samples, 'candidate'),
        (text, [dict(read_tokens=32768,generated_tokens=32)]+samples[1:], 'candidate'),
        (mixed, samples+[dict(read_tokens=74,generated_tokens=2)], 'candidate')]
    for raw, expected, arm in wrong:
        try: observe(raw, expected, arm)
        except RuntimeError: continue
        raise RuntimeError('counted benchmark negative control escaped')
    return dict(positive=3, rejected=len(wrong), GPU_executed=False,
        fixture_sha256=FIXTURE_SHA, synthetic_cache0_and_zero_window_not_model_evidence=True)


def route(text, read):
    value = previous.prefill.context().request_route(text, dict(read=read, heads=1))
    runner.generation.gdn.native16_route(text, read)
    runner.generation.gdn.gdn_selection(text, 'candidate')
    return value


def clean_observers(text):
    require(not any(x in text for x in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called',
        'teardown uncertain', 'strata serve: ADAPT_', 'strata prefill cpu ')), 'runtime/disabled observer failure')


def official(root, result):
    suite = result['results']['long_generation']; rows = previous.shared.gen.collect_samples(suite)
    runner_context().generation_contract(suite, rows)
    require(result['complete'] and not result['results']['failures'] and result['params']['runs'] == 1
        and result['params']['think'] is True and result['params']['long_generation_only'] is True,
        'official benchmark incomplete')
    journal, actual = previous.shared.gen.journal_evidence(root)
    require(journal['valid'] and journal['complete'] and actual ==
        [{k:v for k,v in r.items() if k not in ('controller_eligible','reasoning_tokens')} for r in rows],
        'official result/journal mismatch')
    path, = (root/'bench').glob('*/requests.jsonl'); events = [json.loads(s) for s in path.read_text().splitlines()]
    warm = previous.shared.gen.warmup_reconciliation(events, WORKLOADS)
    identity, binding = events[0], result['comparison_identity']
    from strata_bench import client_fingerprint
    workload = dict(workload_id=WORKLOAD_ID,nonce=WORKLOAD_ID,nonce_policy='fixed-id')
    require(identity['mode'] == binding['mode'] == 'suite' and identity['requested_runs'] == 1
        and identity['thinking'] is binding['thinking'] is True
        and identity['source'] == binding['client'] == client_fingerprint()
        and identity['context'] == binding['context'] == result['machine']['context'] == 262144
        and identity['workload'] == binding['workload'] == result['params']['workload'] == workload
        and all(identity[k] == binding[k] == result['machine'][k] for k in ('model','quant')), 'body/client binding')
    samples = [e['sample'] for e in events if e['kind'] == 'result']
    require(len(samples) == 6 and all(s['reused_tokens'] == 0 for s in samples), 'four warmups/two fresh long requests')
    text = (root/'engine.stderr').read_text()
    measured = observe(text, samples, ARMS[INDEX])
    return rows, events, dict(warm, valid=True, journal_sha256=sha(path), CPU_cache_census=measured,
        timing_scope='Common trace/census charged;not clean speed,task quality or adoption.')


def decision(processes):
    value = runner_context().original_decision(processes)
    value.update(variable='Only pre-exec STRATA_CPU_COMPACT_GU_MIB=0/16384;requested mixed in both.',
        clean_speed_qualified=False, inference_adopted=False,
        next='Judge the counted mechanism separately per workload. If useful,qualify a low-output observer '
            'for clean speed;independent prefill strategies,agent/task/full-context/RESTORE/teardown remain open.')
    return value


def runner_context():
    spec = importlib.util.spec_from_file_location('mixed_cache_counted_bench', runner.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx); ctx.select_mode('cohort')
    ctx.ROOT,ctx.KEY,ctx.WORKLOAD_ID = ROOT,KEY,WORKLOAD_ID;ctx.ARMS=tuple((a,0) for a in ARMS)
    ctx.original_decision=ctx.decision;ctx.shared=SimpleNamespace(**vars(previous.shared))
    ctx.shared.gen=SimpleNamespace(**vars(previous.shared.gen));ctx.shared.gen.request_evidence=previous.request_evidence
    ctx.current,ctx.environment,ctx.save,ctx.compact=current,environment,save,core.compact
    ctx.route,ctx.official,ctx.decision,ctx.clean_observers=route,official,decision,clean_observers
    return ctx


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh frozen cohort only')
    checked=controls();n,prior,providers=basis()
    old=json.loads((previous.ROOT/'record.json').read_text())
    contract,manifest=runner.model.python_manifest()
    assets=previous.shared.models.model_pins(old['configuration'],old['model_assets'],full=False)
    require(assets == prior['models'], 'reuse final15full model hashes only with exact current identities')
    row={k:copy.deepcopy(old[k]) for k in ('configuration','environment_template','template','boot','driver')}
    row.update(stage='prepared',source_commit=previous.shared.gen.git('rev-parse','HEAD').decode().strip(),
        binary=n['binaries']['strata'],native_source_commit=n['native_source_commit'],native_validation_probe=prior['probe'],
        native_observer_admission=n['model_observer_admission'],qualified_model_sha256=signature(model.base.original.previous.compact(prior)),
        sources=core.pins(previous.shared.gen.git('rev-parse','HEAD').decode().strip()),model_assets=assets,
        python_contract=contract,python_selected=manifest['selected'],runtime_manifest=prior['runtime_manifest'],
        controls=checked,plans=[],processes=[],host_passed=False,passed=False,complete=False,active=False,
        ownership_closed=True,whole_bytes=768<<20,
        budgets=dict(wall=2700,cpu=16200,rss_cap=104<<30,text_cap=64<<20,file_cap=64<<20,total_cap=128<<20),
        plan=dict(question='Does the measured selective NT1cache gain survive actual mixed-NT model work?',
            order='Six fresh processes R,C,C,R,R,C;four official warmups then code/prose once each. Commit each closure.',
            constants='Same393d88/e8a1full build,13ELFs,static1256slots,pcie0,adapt0,spec4,5workers+host,262144/20480INT8,8192BF16throughput,largeGU,head2/dense1,MTP16 and safety. Only CPUcache budget0/16384MiB differs.',
            observers='Common STRATA_TRACE=1,CPU_WORK_CENSUS=1,DECODE_TIMING=1 and nominal snapshot marker. Actual timed callback/NT1bindings,pool-clock/membership/NT/entry conservation;charged diagnostics,not clean speed.',
            acceptance='All6processes/12long requests:>=32768fresh input,>=1024generated with thinking,>=512answer chars,>=256drafts,valid SSE/clientTTFT/journal/body/GPUroute/state/source/build/13ELFs/assets/providers,normal0/closed ownership/quiet validation/no fresh fault. Check all6actual CPUframes including zero-window single-token warmups;candidate timed NT1bindings positive,reference0.',
            decision='Retain3paired TG/prompt/TTFT ratios and spread per workload;gain requires all3TG ratios>1,no minimum floor. Strong code all acceptance>=.60,weak prose all<=.40;classification failure does not stop collection. No adoption/clean-speed/target claim.',
            stopping='Stop first source/ownership/input/provider/clock/route/fault/budget/eligibility failure;exclusive durable attempt before preflight,no retry,hash refresh or changed criteria. Final15full hashes after process5.',
            retention='Keep compact individual samples/cache/setup/CPUshape/binding/conservation and decision. Retire successful raw logs/JIT after verification;no new engine generation. Native source14508 remains clean speed reference.'))
    ROOT.mkdir(mode=0o700)
    for i,arm in enumerate(ARMS):
        root=ROOT/str(i);root.mkdir()
        for name in ('tmp','cache'):(root/name).mkdir()
        plan=dict(index=i,kernel=arm,adapt_swaps=0,heads=1,root=str(root))
        core.write(root/'config.json',configuration(row,plan));core.write(root/'plan.json',
            dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan.update(generated={p:sha(root/p) for p in ('config.json','plan.json')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--long-generation-only','--think','--runs','1','--profile-request-gates',
                '--workload-id',WORKLOAD_ID,'--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')])
        plan['environment_sha256']=runner.comparison.fingerprint(environment(row,plan));row['plans'].append(plan)
    row['invariant_signature']=invariant(row,row['plans'][0])
    require(all(invariant(row,p)==row['invariant_signature'] for p in row['plans']), 'undeclared arm difference')
    save(row);print('Prepared6counted bench.sh processes;commit before host/model.',flush=True)


INDEX=None
def execute(action):
    global INDEX
    INDEX=int(action) if action.isdigit() else None
    attempt=ROOT/('host-attempt.json' if INDEX is None else f'{INDEX}/attempt.json')
    with attempt.open('x') as stream:json.dump(dict(action=action,key=KEY),stream)
    row=json.loads((ROOT/'record.json').read_text())
    try:
        if INDEX is None:
            current(row);checked=controls();drafts=previous.draft_controls()
            runner.comparison.host(runner_context())
            row=json.loads((ROOT/'record.json').read_text());row['collector_controls']=checked
            row['draft_controls']=drafts;save(row)
        else:runner_context().run(INDEX)
    except BaseException as error:
        row=json.loads((ROOT/'record.json').read_text())
        if row['stage']!='failed':row.update(stage='failed',passed=False,complete=False,error=repr(error),failed_action=action);save(row)
        raise


if __name__=='__main__':
    require(sys.executable==PYTHON and len(sys.argv)==2 and sys.argv[1] in
        ('prepare','host','controls',*map(str,range(6))), 'qualified Python;prepare|host|controls|0..5')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);previous.build.guards.no_model_owners([])
        if sys.argv[1]=='controls':print(json.dumps(controls()))
        elif sys.argv[1]=='prepare':prepare()
        else:execute(sys.argv[1])
