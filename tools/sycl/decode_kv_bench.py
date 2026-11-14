#!/usr/bin/env python3
"""Same-binary KV planner comparison with API/native continuation accounting.

Main-authored from this fork's owned measurement controls and native/server
protocols. Repository LICENSE applies; no XeStrata source or new native build.
"""
import ast
import copy
import fcntl
import json
import math
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import check_decode_kv as device
import coherent_decode_bench as base
import decode_kv_model as model
import decode_footer

P, B, RESULT = base.P, base.B, base.RESULT
ROOT, KEY = B / 'decode-kv-speed-v1', 'decode_kv_speed_v1'
THIS = 'tools/sycl/decode_kv_bench.py'
PYTHON, WORKLOADS = base.PYTHON, base.WORKLOADS
WORKLOAD_ID = 'decode_kv_20261114'
ARMS = ('reference', 'candidate', 'candidate', 'reference', 'reference', 'candidate')
require, sha, signature = base.require, base.sha, base.signature
core, shared, build, runner = base.core, base.shared, base.build, base.runner
FRONTEND = dict(build_evidence_commit='6c05d9a4dd6d813627672d9c9badd61daa173b90', changes={
    'tools/sycl/generation_engine.py': [
        'eb5bd20c70dbeca6dad253675c76735f9e058507f54668a44074760ae7a5d5d9',
        '741e854a1954269b9c5419a5007e223ef70b30d3c9b291382462d30996485d43']})


def compact(row):
    return core.compact(row)


def save(row):
    core.write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); core.write(RESULT, data)


def basis(expected=None):
    native = device.native(expected, observers={device.THIS, model.THIS, THIS}, retained_frontend=FRONTEND)
    for key in (device.KEY, model.KEY):
        evidence = device.common.validation.committed(RESULT, (key,))
        require(evidence['stage'] == 'closed' and not evidence['active'] and all(evidence[k]
            for k in ('passed', 'complete', 'ownership_closed'))
            and evidence['native_source_commit'] == native['native_source_commit'],
            'closed same-build native/model handoff required')
        if key == device.KEY:
            require(evidence['binaries'] == native['binaries'] and len(evidence['results']) == 11
                and all(v['passed'] and v['ownership_closed'] and v['native_exit_code'] == 0
                    for v in evidence['results']), 'all eleven same-build stages required')
        else:
            require(evidence['native_build']['binaries'] == native['binaries']
                and len(evidence['results']) == 1 and evidence['results'][0]['native_exit_code'] == 0
                and not evidence['results'][0]['fresh_faults'], 'actual cooperative handoff required')
    prior = json.loads((base.prefill.ROOT / 'record.json').read_text())
    require(device.common.validation.committed(RESULT, (base.prefill.KEY,)) == core.compact(prior)
        and prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and not prior['active'], 'closed clean configuration required')
    providers = json.loads(Path(prior['runtime_manifest']['path']).read_text())
    require(sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
        and all(sha(p) == h for p, h in providers.items())
        and build.guards.driver_guard() == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
        'qualified runtime, boot or driver changed')
    return native, prior, providers


def insertion_recipe(config):
    """Compute allowed token sequences independently of observed generations."""
    sys.path.insert(0, str(P / 'tools'))
    from strata_tokenizer import Tokenizer
    root = Path(config['tokenizer'])
    vocab = json.loads((root / 'vocab.json').read_text()); tokens = [None] * len(vocab)
    for token, index in vocab.items(): tokens[index] = token
    settings = json.loads((root / 'tokenizer.json').read_text())
    tokenizer = Tokenizer(tokens, (root / 'merges.txt').read_text().split('\n'),
        json.loads((root / 'token_type.json').read_text()), settings.get('pre') or 'qwen35',
        settings.get('special_ids') or None)
    constants = {}
    for node in ast.parse((P / 'serve/server.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id in ('REASONING_CLOSE', 'REASONING_WRAP_UP') for t in node.targets):
            for target in node.targets:
                if isinstance(target, ast.Name): constants[target.id] = ast.literal_eval(node.value)
    require(set(constants) == {'REASONING_CLOSE', 'REASONING_WRAP_UP'}, 'pinned insertion constants missing')
    return {name: dict(tokens=len(ids), token_ids_sha256=signature(ids))
        for name, text in constants.items() for ids in [tokenizer.encode(text, parse_special=True)]}


def configuration(row, plan):
    value = copy.deepcopy(row['configuration']); value['log'] = str(Path(plan['root']) / 'engine.stderr')
    args = value['args']
    require(args[args.index('--spec')+1] == '4' and '--mtp-max-t' not in args
        and args[args.index('--pcie-frac')+1] == args[args.index('--adapt-swaps')+1] == '0',
        'preserve MTP window and static CPU/GPU placement')
    return value


def environment(row, plan):
    env = base.environment(row, plan)
    env.update(STRATA_SYCL_DECODE_KV_COALESCED=str(int(plan['kernel'] == 'candidate')),
        STRATA_SYCL_KV_COPY_GROUPS='0')
    return env


def invariant(row, plan):
    value = dict(config=configuration(row, plan), environment=environment(row, plan))
    value['environment']['STRATA_SYCL_DECODE_KV_COALESCED'] = '<arm>'
    return signature(json.dumps(value, sort_keys=True).replace(plan['root'], '<process>'))


def current(row):
    require(device.common.validation.committed(RESULT, (KEY,)) == compact(row)
        and core.pins(row['source_commit']) == row['sources']
        and all(sha(P / p) == h for p, h in row['policies'].items()), 'freeze exact committed comparison inputs')
    native, prior, providers = basis(row['native_observer_admission'])
    require(row['binary'] == native['binaries']['strata'] and all(row[k] == prior[k] for k in
        ('configuration', 'environment_template', 'template', 'boot', 'driver', 'runtime_manifest', 'model_assets')),
        'frozen native/configuration/runtime differs')
    contract, manifest = runner.model.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected']
        and sha(row['template']['path']) == row['template']['sha256']
        and build.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'Python/template/output budget differs')
    require(len(row['plans']) == len(ARMS), 'frozen process count differs')
    for i, plan in enumerate(row['plans']):
        root = Path(plan['root'])
        require(plan['index'] == i and plan['kernel'] == ARMS[i] and plan['heads'] == 1
            and invariant(row, plan) == row['invariant_signature']
            and json.loads((root / 'config.json').read_text()) == configuration(row, plan)
            and json.loads((root / 'plan.json').read_text()) == dict(binary=row['binary'],
                native_validation_probe=row['native_validation_probe'])
            and all(sha(root / p) == h for p, h in plan['generated'].items())
            and runner.comparison.fingerprint(environment(row, plan)) == plan['environment_sha256'],
            'generated arm or immutable environment differs')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    return providers, manifest


def native_windows(text):
    headers = list(re.finditer(r'^strata serve: prompt .+$', text, re.M))
    copies = list(re.finditer(decode_footer.COPY, text, re.M))
    caches = list(re.finditer(r'^strata serve: decode expert cache hit rate: .+$', text, re.M))
    require(len(headers) == len(copies) == len(caches) > 0, 'native footer/copy/cache coverage differs')
    require(text.count('strata serve: SYCL_EXPERT_COPY ') == len(copies), 'malformed/extra copy footer')
    previous, windows = 0, []
    for i, (header, copy_footer, cache) in enumerate(zip(headers, copies, caches)):
        require(header.end() < copy_footer.start() < cache.start()
            and (i+1 == len(headers) or cache.end() < headers[i+1].start()), 'native footer order differs')
        windows.append(text[previous:cache.end()]); previous = cache.end()
    return windows


def kv_captures(text, enabled):
    rows = []
    for line in text.splitlines():
        if not line.startswith('strata verify KV_PLAN:'): continue
        match = re.fullmatch(r'strata verify KV_PLAN: phase=(prompt|decode) tokens=([1-4]) '
            r'enabled=([01]) scalar_calls=(\d+) coalesced_calls=(\d+)', line)
        require(match is not None, 'malformed KV dispatch capture')
        phase, width, flag, scalar, cooperative = match.groups()
        width, flag, scalar, cooperative = map(int, (width, flag, scalar, cooperative))
        selected = enabled and phase == 'decode' and width in (2, 4)
        require(flag == enabled and (scalar, cooperative) == ((0, 12) if selected else (12, 0)),
            'actual phase/shape KV dispatch differs')
        rows.append(dict(phase=phase, width=width, scalar_calls=scalar, coalesced_calls=cooperative))
    return rows


def api_window(text, sample, row, arm):
    proof = shared.gen.api_segment_evidence(text, sample,
        insertion_token_counts=tuple(v['tokens'] for v in row['insertions'].values()))
    windows = native_windows(text)
    require(len(windows) == len(proof['segments']) and sample['read_tokens'] >= 32768
        and sample['reused_tokens'] == 0 and proof['native_generated_tokens'] >= 1024,
        'long API/native work extent differs')
    routes = []
    for segment, body in zip(proof['segments'], windows):
        copied, = re.findall(decode_footer.COPY, body, re.M)
        counters = dict(zip(decode_footer.COPY_KEYS, map(int, copied)))
        require(all(counters[k] == 0 for k in decode_footer.COPY_KEYS[:6])
            and 0 <= counters['cpu_unique_misses'] <= counters['cpu_entries'], 'common pcie0 copy/work differs')
        callbacks = shared.gen.draft_rows(body); cells = max(0, segment['read_tokens']-1)
        require(all(c['mode'] == 2 and 0 < c['cells'] <= 8192 for c in callbacks)
            and sum(c['cells'] for c in callbacks) <= cells, 'native draft callback bounds differ')
        route = dict(read_tokens=segment['read_tokens'], reused_tokens=segment['reused_tokens'],
            CPU_entries=counters['cpu_entries'], CPU_unique_jobs=counters['cpu_unique_misses'],
            draft_cells=sum(c['cells'] for c in callbacks), cache=runner.cache_footer(body))
        if segment['read_tokens'] >= 32768:
            expected = [8192]*(cells//8192) + ([cells % 8192] if cells % 8192 else [])
            require([c['cells'] for c in callbacks] == expected
                and all(c['path'] == 'batched' for c in callbacks), 'long native draft chunks differ')
            route['prefill'] = base.prefill.context().request_route(body, dict(read=segment['read_tokens'], heads=1))
        routes.append(route)
    captures = kv_captures(text, int(arm == 'candidate'))
    require(any(c['phase'] == 'decode' and c['width'] == 4 for c in captures),
        'timed long request must actually recapture decode4')
    base.clean_observers(text)
    return dict(proof, routes=routes, KV_captures=captures)


def gated_events(events, done, pending=None, finished=False):
    """Reconcile completed official requests and the sole live gate separately."""
    plain = [e for e in events if e['kind'] not in ('identity', 'gate')]
    if pending is not None and not finished:
        start = plain.pop()
        require(start['kind'] == 'start' and start['phase'] == 'long-generation/'+pending
            and start['run'] == 0 and start['turn'] is None and start['max_tokens'] == 16384
            and start['thinking'] is True and start['gate_required'] is True, 'pending API start differs')
    names = (*done, pending) if pending is not None and finished else done
    shared.gen.warmup_reconciliation(plain, names)
    starts = [e for e in plain if e['kind'] == 'start' and e['phase'].startswith('long-generation/')]
    require(all(e.get('gate_required') is True for e in starts), 'long request lacks gate requirement')
    expected = starts[:-1] if pending is not None and finished else starts
    gates = [e for e in events if e['kind'] == 'gate']
    require(len(gates) == len(expected) and all(g['status'] == 'passed'
        and g['request_index'] == s['index'] and all(g[k] == s[k] for k in ('phase', 'run', 'turn'))
        for g, s in zip(gates, expected)), 'missing/failed/misbound request gate')
    return [e['sample'] for e in plain if e['kind'] == 'result']


def boundary(root, minimum):
    deadline = time.monotonic()+5
    while True:
        raw = (root/'engine.stderr').read_bytes(); text = raw.decode()
        count = len(re.findall(r'^strata serve: prompt .+$', text, re.M))
        if count >= minimum and count == text.count('strata serve: SYCL_EXPERT_COPY ') == text.count('strata serve: decode expert cache hit rate: '):
            return raw, runner.cache_history(text, count)
        require(time.monotonic() < deadline, 'complete native footer boundary deadline')
        time.sleep(.01)


def read_journal(root):
    sys.path.insert(0, str(P/'tools'))
    from strata_bench import verify_request_journal, client_fingerprint
    path, = (root/'bench').glob('*/requests.jsonl')
    proof = verify_request_journal(path.parent); require(proof['valid'], 'invalid durable journal')
    events = [json.loads(v) for v in path.read_text().splitlines()]
    identity = events[0]
    require(identity['kind'] == 'identity' and identity['source'] == client_fingerprint()
        and identity['context'] == 262144 and identity['requested_runs'] == 1
        and identity['thinking'] is True and identity['workload'] == dict(
            workload_id=WORKLOAD_ID, nonce=WORKLOAD_ID, nonce_policy='fixed-id'),
        'actual official client/context/workload identity differs')
    return proof, events, path


def decision(processes):
    out = {}
    for name in WORKLOADS:
        pairs = []
        for i in (0, 2, 4):
            pair = {p['kernel']: p['samples'][WORKLOADS.index(name)] for p in processes[i:i+2]}
            pairs.append(dict(TG=pair['candidate']['exact_native_decode_tps']/pair['reference']['exact_native_decode_tps'],
                prompt_time=pair['candidate']['prompt_ms']/pair['reference']['prompt_ms'],
                TTFT=pair['candidate']['ttft_s']/pair['reference']['ttft_s']))
        reference = [p['samples'][WORKLOADS.index(name)]['acceptance_rate'] for p in processes if p['kernel']=='reference']
        out[name] = dict(paired_ratios=pairs, TG_geometric_mean=math.exp(statistics.mean(math.log(p['TG']) for p in pairs)),
            reproducible_TG_gain=all(p['TG']>1 for p in pairs), reference_acceptance=reference,
            reference_MTP_class_confirmed=all(v >= .60 for v in reference) if name=='regular-code' else all(v <= .40 for v in reference))
    return dict(workloads=out, inference_adopted=False, dedicated_prefill_preservation_pending=True,
        task_agent_physical262144_RESTORE_and_ordinary_teardown_pending=True, physical_saturation_qualified=False)


def run(index):
    row = json.loads((ROOT/'record.json').read_text()); files, manifest = current(row)
    require(row['host_passed'] and row['stage'] in ('ready','running') and len(row['processes']) == index
        and all(p['passed'] for p in row['processes']), 'fresh ordered process only; no retry')
    plan = row['plans'][index]; root = Path(plan['root']); env = environment(row, plan)
    cursor = next(v.split(': ',1)[1] for v in subprocess.check_output(['journalctl','-k','-b','-n','0',
        '--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    outcome, owner = dict(index=index, kernel=plan['kernel'], passed=False, complete=False,
        host_probe_accepted=False, model_executed=False, samples=[]), None
    offsets, bundles, identity = [], [], None
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def persist():
        core.write(root/'execution.json', dict(outcome, active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    def interact(io, entry):
        nonlocal identity
        last_count = 4
        for i, name in enumerate(WORKLOADS):
            window = {}
            for stage in ('begin','end'):
                while True:
                    line = io.line(2400)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage,phase='long-generation/'+name,run=0), 'actual API gate differs')
                delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
                require(min(delta)>=0 and abs(delta[0]-delta[1])<=50000000, 'clock/suspend discontinuity')
                claim = json.loads((root/'engine-start.json').read_text())['native']
                actual = build.supervision.identity(claim['pid'])
                require(actual and actual['start_ticks']==claim['start_ticks'] and actual['sid']==entry['pid']
                    and str(Path('/proc',str(claim['pid']),'exe').resolve())==row['binary']['path']
                    and (identity is None or identity==claim), 'native ownership/executable differs')
                identity = claim
                outcome.update(model_executed=True, native_identity=claim)
                expected = runner.model.engine_once.native_environment(row,env)
                actual_env = dict(v.split(b'=',1) for v in Path('/proc',str(claim['pid']),'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(actual_env.get(k.encode())==v.encode() for k,v in expected.items())
                    and set(actual_env)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'}, 'native environment differs')
                mapped = shared.runtime.executable_mappings(Path('/proc',str(claim['pid']),'maps').read_text())
                require(mapped <= set(files)|{row['binary']['path']} and
                    {row['native_validation_probe']['loader']['path'],row['native_validation_probe']['library']['path'],env['ZE_ENABLE_ALT_DRIVERS']} <= mapped,
                    'native providers differ')
                shared.python_runtime.check_running(entry['pid'],P/'tools/strata_bench.py',manifest['selected'],env,
                    {v['realpath'] for v in manifest['inventory']['files'].values()},plan['command'][2:])
                runner.comparison.require_quiet_validation(root)
                raw, history = boundary(root,last_count); window[stage] = len(raw)
                journal, events, _ = read_journal(root)
                samples = gated_events(events,WORKLOADS[:i],name,stage=='end')
                require(len(journal['pending_gates'])==1 and not journal['failed_gates']
                    and len(journal['pending'])==(stage=='begin'), 'live journal/gate state differs')
                if stage=='begin':
                    require(len(history)==last_count, 'unbound native work before API request')
                    if i==0:
                        base.copy_records(raw.decode(),samples,arm='reference'); base.request_evidence(raw.decode(),samples)
                else:
                    sample = samples[-1]
                    require(shared.gen.sample_eligible(sample), 'ineligible completed long API sample')
                    runner.comparison.require_valid_client_ttft(sample)
                    bundle = api_window(raw[window['begin']:].decode(),sample,row,plan['kernel'])
                    require(len(history)==last_count+len(bundle['segments']), 'API/native segment coverage differs')
                    bundles.append(bundle); last_count=len(history)
                    fields = ('prompt_tokens','reused_tokens','read_tokens','generated_tokens','prompt_ms','prompt_tps',
                        'decode_ms','decode_tps','ttft_s','total_s','drafts_offered','drafts_accepted','acceptance_rate',
                        'finish_reason','output_chars','reasoning_chars','request_sha256','output_sha256','reasoning_sha256')
                    outcome['samples'].append(dict({k:sample[k] for k in fields},native_generated_tokens=bundle['native_generated_tokens'],
                        server_inserted_tokens=bundle['server_inserted_tokens'],exact_native_decode_tps=bundle['exact_native_decode_tps']))
                    outcome['model_executed']=True
                persist(); print(f'KV process{index} {plan["kernel"]} {name} {stage}',flush=True)
                io.send(b'continue\n')
            offsets.append(window)
        io.proc.stdin.close(); deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic()<deadline,'teardown deadline')
            if select.select([io.proc.stdout],[],[],.02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner=build.supervision.Owner(root/'owned',persist)
        entry,_,_=owner.run('bench',plan['command'],env,P,interaction=interact,**row['budgets'])
        require(build.supervision.closed(entry) and entry['exit_code']==0,'owned benchmark failed: '+entry.get('error','exit/closure'))
        path,=(root/'bench').glob('*/results.json'); result=json.loads(path.read_text())
        actual=shared.gen.collect_samples(result['results']['long_generation']); runner_context=base.runner_context()
        runner_context.generation_contract(result['results']['long_generation'],actual)
        journal,events,journal_path=read_journal(root); gated_events(events,WORKLOADS)
        require(result['complete'] and not result['results']['failures'] and journal['complete']
            and result['params']['runs'] == 1 and result['params']['think'] is True
            and result['params']['long_generation_only'] is True
            and len(actual)==len(outcome['samples'])==len(bundles)==2,'incomplete official suite')
        require(all(all(sample.get(k)==api.get(k) for k in ('request_sha256','generated_tokens','prompt_tokens','read_tokens',
            'reused_tokens','decode_ms','prompt_ms','drafts_offered','drafts_accepted'))
            for sample,api in zip(outcome['samples'],actual)),'official/durable API samples differ')
        hashes=[s['request_sha256'] for s in outcome['samples']]
        require(not row['processes'] or hashes==row['processes'][0]['request_sha256'],'paired request bodies differ')
        end=shared.gen.native_end(root,row,dict(config=configuration(row,plan)),row['boot'],entry['pid'])
        text=(root/'engine.stderr').read_text(); base.clean_observers(text)
        runner.generation.gdn.validate_model_text(text,plan)
        validation=shared.runtime.validation.preload_evidence(text,end['native']['pid'])
        require(validation['loader']==row['native_validation_probe']['loader']['path'],'validation loader differs')
        current(row)
        if index==len(ARMS)-1: shared.models.model_pins(row['configuration'],row['model_assets'],full=True)
        outcome.update(complete=True,host_probe_accepted=True,native_exit_code=0,native_identity=identity,
            validation=validation,API_native_binding=bundles,request_sha256=hashes,journal_sha256=sha(journal_path),
            journal_complete=True)
    except BaseException as error:
        outcome['error']=repr(error)
    finally:
        if owner: shared.runtime.validation.finish(outcome,owner,cursor,row['boot'])
        if outcome.get('driver_after')!=row['driver']: outcome.update(passed=False,final_driver_mismatch=True)
        outcome['passed']=bool(outcome.get('passed') and outcome['complete'])
        row['processes'].append(outcome); done=index==len(ARMS)-1
        row.update(stage='closed' if outcome['passed'] and done else 'running' if outcome['passed'] else 'failed',
            passed=bool(outcome['passed'] and done),complete=bool(outcome['passed'] and done),
            active=bool(owner and owner.active),ownership_closed=outcome.get('ownership_closed',False))
        if row['passed']: row['decision']=decision(row['processes'])
        save(row)
    require(outcome['passed'],outcome.get('error','KV benchmark failed'))
    print(json.dumps({k:outcome.get(k) for k in ('index','kernel','passed','complete','error')}),flush=True)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh committed comparison only')
    native,prior,_=basis(); contract,manifest=runner.model.python_manifest()
    row={k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','template','boot','driver','runtime_manifest')}
    revision=shared.gen.git('rev-parse','HEAD').decode().strip()
    row.update(stage='prepared',source_commit=revision,sources=core.pins(revision),
        policies={p:sha(P/p) for p in ('AGENTS.md','docs/ARTIFACT_RETENTION.md')},
        binary=native['binaries']['strata'],native_source_commit=native['native_source_commit'],
        native_validation_probe=native['native_validation_probe'],native_observer_admission=native['observer_admission'],
        python_contract=contract,python_selected=manifest['selected'],
        model_assets=shared.models.model_pins(prior['configuration'],prior['model_assets'],full=True),
        insertions=insertion_recipe(prior['configuration']),plans=[],processes=[],host_passed=False,
        passed=False,complete=False,active=False,ownership_closed=True,whole_bytes=448<<20,
        budgets=dict(wall=2700,cpu=16200,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20),
        plan=dict(question='Does the component-qualified cooperative KV planner improve full-entrypoint sustained decode?',
            evidence='Six isolated INT8 T2/4 cells improve in all three pairs;actual same-build decode4 long/short/long handoff passes. Prior failed-overhead service observations motivate this test,not hardware saturation.',
            order='Six fresh processes R,C,C,R,R,C. Each has four official warmups then regular-code and varied-prose once,thinking enabled,output cap16384. Commit each closed process before advancing. No historical speed sample is reused.',
            constants='Engine330405 from1eec28,13same-build ELFs;262144logical/20480resident INT8KV,8192throughput BF16 ordered-exp/bulk1/staging0/largeGU,1256static expert slots,5workers,MTPspec4,pcie0/adapt0,packedhead2/dense1;qualified safety,quiet validation and nominal footers unchanged. Only pre-exec KV_COALESCED0/1 differs;copy-groups0 in both.',
            accounting='Each API interval binds one or more ordered native generations. Original prompt/read/reuse,per-segment read+reuse,all native outputs/drafts and summed rounded clocks must conserve. Continuation insertions must match independently tokenizer-derived server lengths. Native outputs alone form exact TG;API usage and inserted counts remain separate. API sample fsynced before end gate;failed/pending gate never qualifies.',
            acceptance='All12long requests need>=32768fresh initial input,>=1024native outputs including thinking,>=512answer chars,>=256drafts,normal SSE/journal,valid clientTTFT,identical bodies,actual largeGU/typed long-prefill and batched8192callbacks,actual phase-specific12-QSA KVcaptures including decode4. Bind every native footer/copy/cache;pcie0 copies0. Exact source/build/runtime/owner identities,normal0,reaped/empty owner,quiet validation,no freshfault,unchangedboot/driver;all15full asset hashes before/aftercohort.',
            decision='Retain all individual samples and three adjacent C/R TG,prompt-time andTTFT ratios per workload. Reproducible gain requires allthree TG ratios>1,no minimum floor. Reference code strong>=.60/prose weak<=.40 in allthree processes;report candidate acceptance separately. Any prefill impact is a separation issue,not an accepted cost. Dedicated32K/64K preservation,agent/task,physical262144/RESTORE,ordinaryteardown and default adoption remain separate.',
            stopping='First source/provider/owner/clock/token/route/validation/fault/budget/eligibility failure stops. Exclusive attempt before preflight;no retry,pin refresh,speed-based stopping,posthoc exclusions or changed criteria.',
            retention='No new build generation. Keep compact per-process API/native segments,samples,source/configuration and decisions;successful raw output/cache ends at verified closure. Retained14508baseline,330405candidate and1e776fixture remain through this immediate comparison;three legacy provenance holds remain separate.'))
    ROOT.mkdir(mode=0o700)
    for i,arm in enumerate(ARMS):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for name in ('tmp','cache'): (root/name).mkdir()
        plan=dict(index=i,kernel=arm,heads=1,adapt_swaps=0,root=str(root))
        core.write(root/'config.json',configuration(row,plan))
        core.write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan.update(generated={n:sha(root/n) for n in ('config.json','plan.json')},command=['/bin/sh',str(P/'bench.sh'),
            '--config',str(root/'config.json'),'--port',str(port),'--long-generation-only','--think','--runs','1',
            '--profile-request-gates','--workload-id',WORKLOAD_ID,'--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')])
        plan['environment_sha256']=runner.comparison.fingerprint(environment(row,plan));row['plans'].append(plan)
    row['invariant_signature']=invariant(row,row['plans'][0])
    require(all(invariant(row,p)==row['invariant_signature'] for p in row['plans']),'undeclared arm difference')
    save(row);print('Prepared6same-binary KV0/1processes;commit before host/native execution.',flush=True)


if __name__=='__main__':
    require(sys.executable==PYTHON,'use qualified benchmark interpreter')
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','host',*map(str,range(6))),'prepare|host|0..5')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='prepare': prepare()
        else:
            attempt=ROOT/('host-attempt.json' if action=='host' else action+'/attempt.json')
            with attempt.open('x') as stream: json.dump(dict(action=action,result_key=KEY),stream)
            try:
                runner.comparison.host(sys.modules[__name__]) if action=='host' else run(int(action))
            except BaseException as error:
                row=json.loads((ROOT/'record.json').read_text())
                if row['stage']!='failed':row.update(stage='failed',passed=False,complete=False,error=repr(error),failed_action=action);save(row)
                raise
