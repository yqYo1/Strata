#!/usr/bin/env python3
"""Main-owned MTP attribution on the official weak-MTP thinking workload."""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import shutil
import socket
import sys
import time

sys.dont_write_bytecode = True
import cpu_capacity as assets
import cpu_census_engine as faults
import generation_compare as comparison
import generation_engine as gen
import iq2_pair_model as recipe
import measurement_guards as guards
import measurement_owner as supervision
import query_profile as q
from build_engine import archived_model_observation
from check_expert_bank import check_packed_cases

P, B, A = gen.P, gen.B, gen.A
ROOT = B/'mtp-model-attribution-v2'
RESULT = A/'mtp-model-observer-current.json'
MODES = ('graph',)
WORKLOADS = ('varied-prose',)
DECODE_ARCHIVE = 'a8af3d7fa96a7713f2e516d2da3d7410956be015'
DECODE_ARCHIVE_SHA256 = '7dda2bcefa6f7e54d568e6aa3d5226d5e0a7383f0b41efd7584093854d9547ca'
GRAPH_ORIGIN = '1ccdfd6d4'
GRAPH_ORIGIN_SHA256 = 'b63a49b6cfac9df92d5064f47c5ceafe944ce7e0cfbe080361051baab8c1a2a2'
GRAPH_BUILD_SHA256 = '6ac902826311b0c8c7f334f480dd787114968713d5f33d5f9917eaf63a1736a6'
GRAPH_REUSE_SOURCES = ('05223b55965c98fc82be660dbde3a5215027b5d9',
                       '6e40afc1beddd059ee518617f6c1efce1da14aca')


def write(path, value): path.write_text(json.dumps(value,indent=2)+'\n')


def observation(text, mode):
    prefixes = ('strata MTP_OBSERVER ', 'strata MTP_OBSERVER_PHASE ', 'strata MTP_GRAPH_CLOCK ')
    lines = [line for line in text.splitlines() if line.startswith(prefixes)]
    if mode == '0':
        guards.require(not lines,'disabled MTP observer emitted records'); return None
    records = [[],[],[]]
    for line in lines:
        kind = next(i for i,p in enumerate(prefixes) if line.startswith(p))
        words = line[len(prefixes[kind]):].split(); row = {}
        for word in words:
            guards.require(word.count('=') == 1,'malformed MTP record')
            key,value = word.split('=')
            guards.require(key not in row,'duplicate MTP field')
            if re.fullmatch(r'-?\d+',value): value = int(value)
            elif key.endswith('_s'): value = float(value); guards.require(math.isfinite(value),'nonfinite MTP clock')
            row[key] = value
        records[kind].append(row)
    headers,phases,raw = records
    guards.require(len(headers) == 1 and len(phases) == 8,'MTP profile framing')
    h = headers[0]
    guards.require(all(h[k] == v for k,v in dict(v=1,request=5,profile=int(mode=='graph'),request_complete=1,
        sample_complete=1,accounting_valid=1,skip=8,limit=64,windows=64,completed=64,failed=0,
        rows=256,T_mask=16,observer_charged=1,scope='draft_entry_through_return',
        clocks='HOST_DISJOINT_DEVICE_NONADDITIVE',arithmetic_service='UNKNOWN',hardware_saturation='UNKNOWN').items())
        and h['seen'] >= 72,'MTP sample incomplete or wrong scope')
    expected = [(kind,phase) for kind in ('round','chain') for phase in ('capture','submit','drain','query')]
    guards.require([(r['kind'],r['phase']) for r in phases] == expected,'MTP phase order/uniqueness')
    for row in phases:
        guards.require(row['v'] == 1 and row['request'] == 5 and row['throws'] == 0 and
            row['calls'] >= 0 and row['wall_s'] >= 0 and row['thread_cpu_s'] >= 0,'MTP phase failure')
        guards.require(row['calls'] > 0 or row['wall_s'] == row['thread_cpu_s'] == 0,'MTP time without calls')
    by = {(r['kind'],r['phase']):r for r in phases}
    replayed = 0
    for kind in ('round','chain'):
        counts = [by[kind,phase]['calls'] for phase in ('capture','submit','drain')]
        guards.require(len(set(counts)) == 1 and (counts[0] == 64 if kind == 'round' else 0 <= counts[0] <= 128),
            'MTP capture/replay/drain work differs')
        guards.require(by[kind,'query']['calls'] == (counts[0] if mode == 'graph' else 0),'MTP query work differs')
        replayed += counts[0]
    guards.require(h['replays'] == replayed and h['raw_retained'] == len(raw),'MTP replay/raw conservation')
    for total,other,key in (('wall_s','other_wall_s','wall_s'),('thread_cpu_s','other_cpu_s','thread_cpu_s')):
        guards.require(h[total] > 0 and h[other] >= 0 and
            abs(h[total]-h[other]-sum(r[key] for r in phases)) <= 1e-6,'MTP host accounting does not close')
    if mode == 'graph':
        guards.require(h['valid_graph_clocks'] == replayed and len(raw) == min(32,replayed) and
            h['raw_omitted'] == replayed-len(raw) and h['round_clocks'] == 64 and
            h['round_clocks']+h['chain_clocks'] == replayed and
            h['graph_envelope_ns'] == h['round_envelope_ns']+h['chain_envelope_ns'] > 0,'MTP graph clock conservation')
        last = 0
        for row in raw:
            guards.require(row['v'] == 1 and row['request'] == 5 and row['T'] == 4 and row['coupled'] == 0 and
                8 <= row['window'] < 72 and row['status'] == 0 and row['available'] == 7 and
                row['envelope_valid'] == 1 and row['resolution_ns'] > 0 and row['start'] >= last and
                row['end']-row['start'] >= row['resolution_ns'] and
                row['queue_delay_valid'] == int(row['start'] >= row['submit']),'MTP raw clock invalid')
            last = row['end']
    else:
        guards.require(not raw and all(h[k] == 0 for k in ('valid_graph_clocks','valid_queue_clocks',
            'graph_envelope_ns','round_clocks','round_envelope_ns','chain_clocks','chain_envelope_ns','raw_omitted')),
            'host-only observer exposed device clocks')
    return dict(summary=h,phases=phases,raw=raw)


def plan():
    return dict(question='On the exact restored6e40 IQ2-decode route, where does the weak-MTP draft wrapper spend64 selected windows: host capture/submission/drain/query/other versus graph execution envelope?',
        modes=list(MODES),workload='varied-prose',selector=5,skip=8,windows=64,
        execution='One fresh graph-observed process, four official warmups then official v4 weak-prose thinking-on >=32K fresh input,>=1024 generated tokens and a substantive answer. Same retained6e40 binary,1256 slots and MTP4. Explicit IQ2 decode1/prefill reference; no inference implementation change,environment mutation or retry.',
        observers='Parameter validation,stage trace,existing CPU work census,IQ2 dispatch audit and decode timing. MTP graph observer adds its checked completion-event wrapper in the selected64 windows and requests profiled graph executables. These alter diagnostic work; native source and ordinary default-off route stay unchanged.',
        interpretation='One diagnostic process only: no observer-overhead bound,clean speed,performance comparison or adoption. Host intervals are disjoint calling-thread wall/CPU; worker arithmetic,RAM and PCIe traffic remain unmeasured. Graph envelopes are nonadditive first-to-last node intervals,may contain idle/dependencies,and are not arithmetic service or hardware saturation.',
        gate='Verify exact original sources,recipe,all13 artifacts,18-group engine qualification and identical tiny graph fixture continuity before preparation. Fresh same-build packed-smoke/validate precede the model. Require owned native/environment/four warmups,full journal/request/answer/native closure,weak acceptance<=.40,actual GPU BF16/mode10 restoration,current IQ2 dispatch/census conservation,complete64-window host/graph accounting,unchanged15 model assets and no fresh fault. Stop on first failure or budget; no replaced samples.',
        scope='Weak-MTP diagnostic only; strong-MTP,agent/task quality,physical262144/session/RESTORE,model adoption,observer overhead and70tok/s remain open.',
        retention='Keep one individual request,compact64-window host/graph/census/dispatch summaries and exact source/config/environment/fault decisions. Raw32 graph tuples/logs/JIT stay only through attribution and verification; retire after closure. Original zero-process v1 plan remains closed in Git.')


def expand_root(value, root):
    if isinstance(value,str): return value.replace('${RUN_ROOT}',str(root))
    if isinstance(value,list): return [expand_root(v,root) for v in value]
    if isinstance(value,dict): return {k:expand_root(v,root) for k,v in value.items()}
    return value


def original_setup():
    """Recover the closed retained arm without reopening its retired run files."""
    compact_path=A/'canonical-restore-decode-current.json'
    closed=gen.committed(compact_path)
    raw=gen.git('show',DECODE_ARCHIVE+':'+str(compact_path.relative_to(P)))
    guards.require(hashlib.sha256(raw).hexdigest()==DECODE_ARCHIVE_SHA256 and
        closed['schema']=='strata-canonical-decode-closed-v1' and closed['stage']=='closed' and
        closed['archive']['commit']==DECODE_ARCHIVE and closed['full_model_hashes_matched_after_cohort']==15,
        'closed canonical reference/archive required')
    archive=json.loads(raw); selected=archive['plans'][0]
    guards.require(len(archive['processes'])==6 and selected['arm']==0 and selected['cohort']==0 and
        all(r['passed'] and r['complete'] and r['ownership_closed'] and not r['fresh_faults']
            and r['boot_unchanged'] and r['fault_probe_successful'] for r in archive['processes']),
        'complete closed reference cohort required; failed performance decision remains unchanged')
    config=expand_root(closed['configuration'],selected['root'])
    guards.require(hashlib.sha256((json.dumps(config,indent=2)+'\n').encode()).hexdigest()==
        selected['inputs'][str(Path(selected['root'])/'config.json')], 'normalized reference configuration differs')
    build=gen.committed(q.BUILD); model_reference=build['retention_model_reference']
    recovery=archived_model_observation(model_reference)
    guards.require(build['source_commit']==closed['native_sources'][0] and
        all(build['binary'][k]==archive['binaries'][0][k] for k in ('bytes','sha256')) and
        recovery['plan']['native_source']==build['source_commit'] and len(recovery['models'])==15,
        'selected native/reference/model identities differ')
    return dict(config=config,environment=selected['environment'],model_assets=recovery['models'],
        runtime_asset_delta=gen.committed(q.COMPONENT)['runtime_asset_delta'],
        reference=dict(commit=DECODE_ARCHIVE,path=str(compact_path.relative_to(P)),sha256=DECODE_ARCHIVE_SHA256,
            compact_sha256=gen.common.digest(compact_path),model_reference=model_reference))


def graph_support_evidence(full, support):
    """Reuse the exact tested fixture across the reviewed CPU-phase-only change.

    The engine is still admitted separately by current_build. This establishes
    tiny graph-support continuity, never model profiling or observer overhead.
    Other native revisions require their own support qualification.
    """
    guards.require(support['passed'] and support['complete'] and support['ownership_closed'] and
        not support['active'] and support['boot_unchanged'] and support['fresh_faults'] == [] and
        support['fixture'] == full['fixtures']['sycl_graph_envelope_test'] and
        [(c['profile'],c['passed'],c['exit_code']) for c in support['checks']] ==
            [(0,True,0),(1,True,0)], 'matching passed off/on graph fixture required')
    if support['source_commit'] == full['source_commit']:
        guards.require(support['build_record_sha256'] == gen.common.digest(q.BUILD_RECORD),
            'same-source graph support build receipt differs')
        return dict(kind='same-build', native_source=full['source_commit'])
    guards.require((support['source_commit'],full['source_commit']) == GRAPH_REUSE_SOURCES,
        'graph support reuse is limited to the reviewed CPU-phase source pair')
    path=str((A/'mtp-graph-observer-current.json').relative_to(P))
    raw=gen.git('show',GRAPH_ORIGIN+':'+path)
    build_raw=gen.git('show',GRAPH_ORIGIN+':'+str(q.BUILD.relative_to(P)))
    guards.require(hashlib.sha256(raw).hexdigest()==GRAPH_ORIGIN_SHA256 and
        hashlib.sha256(build_raw).hexdigest()==GRAPH_BUILD_SHA256 and json.loads(raw)==support,
        'original graph qualification/archive changed')
    old=json.loads(build_raw)
    guards.require(old['source_commit']==support['source_commit'] and
        old['fixtures']['sycl_graph_envelope_test']==support['fixture'] and
        all(old[k]==full[k] for k in ('compiler','configure_argv','recipe_reference','fixture_external_headers')) and
        old['inputs'][1:]==[{k:s[k] for k in ('root','revision','count','aggregate_sha256')}
                            for s in full['dependencies']], 'graph compiler/recipe/dependencies changed')
    # The only native deltas in this exact pair implement CPU GU phase selection
    # and its host fixture. These graph/MTP/device implementation inputs are exact.
    guards.require(not gen.git('diff','--name-only',*GRAPH_REUSE_SOURCES,'--',
        'src/core/mtp.cpp','include/strata/core/mtp.hpp','src/sycl','include/strata/sycl',
        'src/kernels/sycl','include/strata/kernels/sycl','tests/sycl/graph_envelope.cpp'),
        'MTP/graph implementation changed across support reuse')
    return dict(kind='identical-fixture-reviewed-native-pair', original_commit=GRAPH_ORIGIN,
        original_native_source=support['source_commit'], native_source=full['source_commit'],
        fixture_sha256=support['fixture']['sha256'], graph_implementation_unchanged=True,
        new_GPU_validation=False, model_profile_qualified=False)


def qualification():
    full,count = q.current_build()
    support = gen.committed(A/'mtp-graph-observer-current.json')
    graph_proof=graph_support_evidence(full,support)
    return full,dict(original_inputs_recovered=count,build_sha256=gen.common.digest(q.BUILD_RECORD),
        component_sha256=gen.common.digest(q.COMPONENT),graph_support_sha256=gen.common.digest(A/'mtp-graph-observer-current.json'),
        graph_support_reuse=graph_proof)


def environment(setup, root, mode):
    env = recipe.environment(setup,root,1)
    env.update(STRATA_CPU_WORK_CENSUS='1',STRATA_DECODE_TIMING='1',STRATA_IQ2_PAIR_AUDIT='1',
        STRATA_TRACE='1',STRATA_SNAPSHOT_VERIFY='1',STRATA_SYCL_CACHE_LOAN_AUDIT='1',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
    for key in ('UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):
        env[key]='level:warning;flush:warning;output:stderr'
    env.pop('UR_LOG_TRACING',None)
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[key]==gen.committed(gen.SETTINGS)['environment'][key], 'qualified safety setting changed')
    env.update(STRATA_SYCL_MTP_OBSERVER=mode,STRATA_SYCL_MTP_OBSERVER_REQUEST='5',
        STRATA_SYCL_MTP_OBSERVER_SKIP='8',STRATA_SYCL_MTP_OBSERVER_WINDOWS='64')
    return env


def current(record, full_models=False):
    full,proof = qualification()
    guards.require(record['plan'] == plan() and record['binary'] == full['binary'] and record['qualification'] == proof,
        'MTP native qualification/plan changed')
    guards.require(assets.fingerprint(recipe.source_pins(record['source_commit'])) == record['source_fingerprint'] and
        gen.common.digest(gen.SETTINGS) == record['settings_sha256'] and assets.machine_identity() == record['machine'],
        'MTP source/settings/platform changed')
    setup = original_setup()
    guards.require(record['setup_reference']==setup['reference'],'restored route reference changed')
    guards.require(assets.model_pins(setup['config'],record['models'],full=full_models) == record['models'],'model changed')
    for row in record['process_plans']:
        root = Path(row['root'])
        guards.require(json.loads((root/'config.json').read_text()) == recipe.config(setup,root) and
            row['environment'] == environment(setup,root,row['mode']),'MTP config/environment changed')
        for name,sha in row['inputs'].items(): guards.require(gen.common.digest(name) == sha,'generated input changed')
    gen.runtime_assets_unchanged(dict(mode='engine-compare',runtime_asset_delta=setup['runtime_asset_delta']))
    return full


def prepare():
    guards.require(not ROOT.exists(),'fresh MTP cohort only')
    previous=gen.committed(RESULT)
    guards.require(previous['stage']=='closed' and previous['processes']==[] and
        bool(previous.get('closure_reason')),'only the closed zero-model v1 plan may be replaced')
    previous_reference=dict(commit=gen.git('log','-1','--format=%H','--',str(RESULT.relative_to(P))).decode().strip(),
        path=str(RESULT.relative_to(P)),sha256=gen.common.digest(RESULT))
    full,proof = qualification(); setup = original_setup()
    source = gen.git('rev-parse','HEAD').decode().strip(); pins = recipe.source_pins(source)
    models = assets.model_pins(setup['config'],setup['model_assets'],full=True)
    ROOT.mkdir(mode=0o700); plans = []
    for mode in MODES:
        root = ROOT/mode; root.mkdir(mode=0o700)
        for name in ('cache','tmp'): (root/name).mkdir()
        shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
        write(root/'config.json',recipe.config(setup,root)); write(root/'plan.json',dict(binary=full['binary'],native_validation_probe=None))
        with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
        command = recipe.command(root,port)
        command[command.index('--think'):command.index('--think')] = ['--long-generation-workload','varied-prose']
        plans.append(dict(mode=mode,root=str(root),command=command,environment=environment(setup,root,mode),
            inputs={str(root/n):gen.common.digest(root/n) for n in ('config.json','plan.json','expert-profile.bin')}))
    record = dict(source_commit=source,binary=full['binary'],qualification=proof,plan=plan(),process_plans=plans,
        models=models,setup_reference=setup['reference'],previous_result_reference=previous_reference,source_fingerprint=assets.fingerprint(pins),machine=assets.machine_identity(),
        settings_sha256=gen.common.digest(gen.SETTINGS),stage='prepared',processes=[],bank=None)
    current(record); write(ROOT/'frozen.json',record); write(RESULT,record)


def execute(action):
    record = gen.committed(RESULT); full = current(record)
    frozen = json.loads((ROOT/'frozen.json').read_text()); mutable = {'stage','processes','bank'}
    guards.require({k:v for k,v in record.items() if k not in mutable} == {k:v for k,v in frozen.items() if k not in mutable},
        'frozen MTP plan changed')
    bank = action == 'qualify'
    guards.require(record['stage'] == ('prepared' if bank else 'collecting') and
        (bank or (record['bank']['passed'] and len(record['processes']) < len(MODES) and all(r['passed'] for r in record['processes']))),
        'wrong MTP stage; never retry a failed or completed attempt')
    selected = None if bank else record['process_plans'][len(record['processes'])]
    root = ROOT/'bank' if bank else Path(selected['root'])
    guards.require(not (root/'record.json').exists(),'existing MTP attempt; inspect owner')
    if bank:
        root.mkdir()
        for name in ('cache','tmp'): (root/name).mkdir()
    boot,cursor = faults.start_fault_scope(); owner = None; row = dict(passed=False,complete=False)
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(root/'record.json',dict(row,commands=owner.commands if owner else []))
    def interact(io,leader):
        for stage in ('begin','end'):
            while True:
                line = io.line(3300)
                if line.startswith('STRATA_PROFILE_GATE '): break
            guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage,phase='long-generation/varied-prose',run=0),'MTP request gate differs')
            if stage == 'begin':
                native = json.loads((root/'engine-start.json').read_text())['native']; identity = supervision.identity(native['pid'])
                guards.require(identity and identity['sid'] == leader['pid'] and identity['start_ticks'] == native['start_ticks'] and
                    Path('/proc',str(native['pid']),'exe').resolve() == Path(record['binary']['path']).resolve(),'MTP native owner differs')
                actual = dict(x.split(b'=',1) for x in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in x)
                guards.require(all(actual.get(k.encode()) == v.encode() for k,v in selected['environment'].items()) and
                    b'LD_PRELOAD' not in actual,'MTP immutable native environment differs')
                path, = (root/'bench').glob('*/requests.jsonl'); events = [json.loads(x) for x in path.read_text().splitlines()]
                guards.require(len(events) == 10 and events[-1]['kind'] == 'start','MTP warmup journal extent')
                gen.warmup_reconciliation(events[:-1],()); text = (root/'engine.stderr').read_text(errors='replace')
                gen.request_evidence(text,[e['sample'] for e in events if e['kind']=='result'])
                guards.require(not re.search(comparison.FAILURE,text),'MTP warmup native failure')
                before_long = faults.fault_evidence(cursor,boot)
                guards.require(before_long['fault_probe_successful'] and before_long['boot_unchanged'] and
                    before_long['fresh_faults'] == [],'fresh fault before MTP long request')
                row['native_identity'] = native; persist()
            io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); guards.require(time.monotonic() < deadline,'MTP teardown budget')
            if select.select([io.proc.stdout],[],[],.05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    owner = supervision.Owner(root/'owned',persist)
    try:
        if bank:
            row['checks'] = []
            for mode in ('packed-smoke','packed-validate'):
                env = dict(gen.committed(gen.SETTINGS)['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
                if mode.endswith('validate'):
                    env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'; env.pop('UR_LOG_TRACING',None)
                fixture = full['fixtures']['sycl_expert_bank_test']
                entry,out,err = owner.run(mode,[fixture['path'],'--'+mode],env,P,wall=180,cpu=600,
                    rss_cap=8<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
                guards.require(supervision.closed(entry) and entry['exit_code']==0 and not re.search(comparison.FAILURE,err.read_text()),
                    'MTP packed fixture failed')
                row['checks'].append(dict(mode=mode,cases=check_packed_cases(out.read_text(),mode),fixture=fixture,passed=True))
        else:
            entry,_,_ = owner.run('bench',selected['command'],selected['environment'],P,interaction=interact,
                wall=3600,cpu=108000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
            paths = list((root/'bench').glob('*/results.json'))
            guards.require(supervision.closed(entry) and entry['exit_code']==0 and len(paths)==1,'MTP official benchmark failed')
            samples,journal,events = comparison.official(root,json.loads(paths[0].read_text()),workloads=WORKLOADS)
            native = gen.native_end(root,dict(binary=record['binary'],native_validation_probe=None),
                dict(config=json.loads((root/'config.json').read_text())),boot,entry['pid'])
            guards.require(native['native'] == row['native_identity'] and samples[0]['acceptance_rate'] <= .40,'MTP native exit/class differs')
            text = (root/'engine.stderr').read_text(errors='replace')
            guards.require(not re.search(comparison.FAILURE,text),'MTP model error')
            event_samples=[e['sample'] for e in events if e['kind']=='result']
            audit,work,timing=recipe.work_evidence(text,event_samples,json.loads((root/'config.json').read_text()),1,root)
            row.update(mode=selected['mode'],requests=samples,warmups=journal,native_exit_code=native['exit_code'],
                routes=comparison.native_routes(text,events,1,workloads=WORKLOADS),
                timing=timing[-1],iq2_dispatch=audit[-1],census=recipe.compact_work(work[-1:])[0],
                observation=observation(text,selected['mode']))
        current(record,full_models=not bank); row.update(passed=True,complete=True)
    except BaseException as error: row['error'] = str(error)
    finally:
        row['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        try: row.update(faults.fault_evidence(cursor,boot))
        except Exception as error: row['fault_probe_error'] = str(error)
        row['passed'] = bool(row['passed'] and row['ownership_closed'] and row.get('fault_probe_successful') and
            row.get('boot_unchanged') and row.get('fresh_faults') == [])
        persist()
        if bank: record['bank'] = row
        else: record['processes'].append(row)
        record['stage'] = 'collecting' if row['passed'] and (bank or len(record['processes'])<len(MODES)) else 'closed'
        write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ('mode','passed','complete','ownership_closed','error')}))
    return 0 if row['passed'] else 1


def main():
    guards.require(sys.argv[1:] in (['prepare'],['qualify'],['run']),'use prepare|qualify|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        return prepare() if sys.argv[1]=='prepare' else execute(sys.argv[1])


if __name__ == '__main__': sys.exit(main())
