#!/usr/bin/env python3
"""Reproduce the historical 117K prefill workload before changing its path.

Current decode code, placement and algorithm flags are held fixed. This baseline
does not claim restoration, sustained decode nonregression or adoption.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import cache_long_compare as history
import iq2_pair_model as setup_reader
import cpu_census_engine as faults
import generation_compare as generation
import packed_bank
from check_expert_bank import check_packed_cases

gen, assets, guards, supervision = history.gen, history.assets, history.guards, history.supervision
P, B, A = gen.P, gen.B, gen.A
ROOT = B/'prefill-recovery-baseline-v1'
RESULT = A/'prefill-recovery-current.json'
HISTORY = A/'cache-long-comparison-current.json'
MODES = ('checked', 'clean-0', 'clean-1', 'clean-2')
DROP = {'STRATA_TRACE', 'UR_ENABLE_LAYERS', 'LD_PRELOAD', 'STRATA_SNAPSHOT_VERIFY',
    'STRATA_SYCL_CACHE_LOAN_AUDIT', 'STRATA_IQ2_PAIR_AUDIT', 'STRATA_CPU_WORK_CENSUS',
    'STRATA_DECODE_TIMING'}


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def environment(setup, root, checked):
    old = setup['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old,str(root)) for k,v in setup['environment'].items()
        if k not in DROP and not k.startswith('UR_LOG_')}
    # Keep current algorithm choices, including IQ2 decode and packed prefill.
    guards.require(env['STRATA_IQ2_PAIR'] == '1' and env['STRATA_IQ_SINGLE_EXACT'] == '0' and
        env['STRATA_MTP_BATCH_F16'] == '0' and env['STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED'] == '1',
        'current algorithm settings changed')
    env['CUDA_MODULE_LOADING'] = 'EAGER'
    env['STRATA_SYCL_MTP_OBSERVER'] = '0'
    if checked:
        env.update(STRATA_TRACE='1', STRATA_SNAPSHOT_VERIFY='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1',
            UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
        for k in ('UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):
            env[k] = 'level:warning;flush:warning;output:stderr'
    safety = gen.committed(gen.SETTINGS)['environment']
    for k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys',
              'EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[k] == safety[k], 'qualified safety setting changed: '+k)
    guards.require('LD_PRELOAD' not in env, 'undeclared preload')
    return env


def plan():
    return dict(question='Does the present engine reproduce the fastest recorded117K prefill workload with current decode unchanged?',
        native_source='6e40afc1beddd059ee518617f6c1efce1da14aca', order=list(MODES),
        execution='Same current binary and1256-slot decode placement. Packed bank1,IQ2 pair1,Q8 MTP,8192 prefill,131072 resident INT8 context,5 workers+host. First qualify packed smoke/validate, then one parameter-validated117489-input request, then three fresh clean processes with four117489-input requests each through bench.sh --think. One short warmup/process.',
        decision='Baseline diagnosis before any restoration change. Preserve first-use and all subsequent samples; use each clean process subsequent median. Compare request hashes to the old396d owned128 record. Historical old/current figures are context, not a contemporary causal A/B. No speed-based early stopping, noise dismissal or adoption gate.',
        stopping='Stop on first source,configuration,model,runtime,owner,fault,budget,request,route or restoration-proof failure. No retries or replaced samples. Commit each closed outcome before the next process.',
        limits='Checked timing is observer-charged. Short24-token thinking output does not qualify sustained decode,task quality,agent sessions or physical262144/session/RESTORE. Restoration still requires the selected prefill change and repeated current-decode checks; a decode impact is a separation defect.',
        retention='Keep individual samples,source/configuration,actual route,restore and fault evidence. Retire successful logs/JIT after verified committed closure; keep frozen inputs only through this baseline diagnosis.')


def qualification():
    full, count = history.q.current_build()
    guards.require(full['source_commit'] == plan()['native_source'] and
        len(full['cpu_phase_host_contract']) == 2 and
        all(r['passed'] and r['cases'] == 12 for r in full['cpu_phase_host_contract']),
        'phase-scoped build and24 host cases required')
    return full, dict(original_inputs_recovered=count,
        build_sha256=gen.common.digest(history.q.BUILD_RECORD),
        component_sha256=gen.common.digest(history.q.COMPONENT))


def source_pins(revision):
    pins = gen.benchmark_pins(revision)
    for name in ('data/expert-profile.bin','tests/core/test_prefill_recovery.py'):
        raw = gen.git('show',revision+':'+name)
        guards.require(raw == (P/name).read_bytes(), 'uncommitted recovery input: '+name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def current(record, full_models=False):
    full, proof = qualification()
    guards.require(record['plan'] == plan() and record['binary'] == full['binary'] and
        record['qualification'] == proof and record['history_sha256'] == gen.common.digest(HISTORY) and
        record['source_fingerprint'] == assets.fingerprint(source_pins(record['source_commit'])) and
        record['settings_sha256'] == gen.common.digest(gen.SETTINGS) and
        record['machine'] == assets.machine_identity(), 'recovery qualification/source drift')
    setup = setup_reader.pilot()
    guards.require(assets.model_pins(setup['config'],record['models'],full=full_models) == record['models'], 'model assets changed')
    for row in record['process_plans']:
        root = Path(row['root'])
        guards.require(json.loads((root/'config.json').read_text()) == setup_reader.config(setup,root) and
            row['environment'] == environment(setup,root,row['mode']=='checked'), 'current decode/configuration drift')
        for path,sha in row['inputs'].items():
            guards.require(gen.common.digest(path) == sha, 'generated recovery input drift')
    gen.runtime_assets_unchanged(dict(mode='engine-compare',runtime_asset_delta=setup['runtime_asset_delta']))
    return full


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'fresh single-use recovery baseline only')
    full, proof = qualification(); setup = setup_reader.pilot()
    previous = gen.committed(HISTORY)
    guards.require(previous['comparison_closed'] and previous['passed'] and previous['ownership_closed'], 'closed historical reference required')
    source = gen.git('rev-parse','HEAD').decode().strip(); pins = source_pins(source)
    models = assets.model_pins(setup['config'],setup['model_assets'],full=True)
    ROOT.mkdir(mode=0o700); rows = []; ports = set()
    for mode in MODES:
        root = ROOT/mode; root.mkdir()
        for name in ('tmp','cache'): (root/name).mkdir()
        shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
        old_profile = setup['config']['args'][setup['config']['args'].index('--expert-profile')+1]
        guards.require(gen.common.digest(root/'expert-profile.bin') == setup['generated_inputs'][old_profile], 'profile changed')
        write(root/'config.json',setup_reader.config(setup,root))
        write(root/'plan.json',dict(binary=full['binary'],native_validation_probe=None))
        with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
        guards.require(port not in ports,'duplicate port; do not retry'); ports.add(port)
        rows.append(dict(mode=mode,root=str(root),environment=environment(setup,root,mode=='checked'),
            inputs={str(root/n):gen.common.digest(root/n) for n in ('config.json','plan.json','expert-profile.bin')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--prefill-only','--prefill-tokens','98304','--think','--runs','1' if mode=='checked' else '4',
                '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]))
    record = dict(mode='engine-compare',source_commit=source,binary=full['binary'],qualification=proof,
        plan=plan(),process_plans=rows,history_sha256=gen.common.digest(HISTORY),
        models=models,source_fingerprint=assets.fingerprint(pins),machine=assets.machine_identity(),
        settings_sha256=gen.common.digest(gen.SETTINGS),stage='prepared',bank=None,processes=[])
    current(record); write(ROOT/'frozen.json',record); write(RESULT,record)


def execute(action):
    record = gen.committed(RESULT); full = current(record)
    frozen = json.loads((ROOT/'frozen.json').read_text()); mutable = {'stage','bank','processes'}
    guards.require({k:v for k,v in record.items() if k not in mutable} ==
        {k:v for k,v in frozen.items() if k not in mutable}, 'frozen recovery baseline changed')
    bank = action == 'qualify'
    guards.require(record['stage'] == ('prepared' if bank else 'collecting') and
        (bank or record['bank']['passed'] and len(record['processes']) < len(MODES) and
         all(r['passed'] for r in record['processes'])), 'closed/failed recovery baseline; never retry')
    selected = None if bank else record['process_plans'][len(record['processes'])]
    root = ROOT/'bank' if bank else Path(selected['root'])
    guards.require(not (root/'record.json').exists(), 'existing attempt; inspect its owner')
    if bank:
        root.mkdir()
        for name in ('tmp','cache'): (root/name).mkdir()
    boot,cursor = faults.start_fault_scope(); owner = None; row = dict(passed=False,complete=False)
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(root/'record.json',dict(row,commands=owner.commands if owner else []))
    owner = supervision.Owner(root/'owned',persist)
    try:
        if bank:
            row['checks'] = []
            for mode in ('packed-smoke','packed-validate'):
                env = dict(gen.committed(gen.SETTINGS)['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
                if mode.endswith('validate'):
                    env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION'; env.pop('UR_LOG_TRACING',None)
                fixture = full['fixtures']['sycl_expert_bank_test']
                entry,out,err = owner.run(mode,[fixture['path'],'--'+mode],env,P,
                    wall=180,cpu=600,rss_cap=8<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
                guards.require(supervision.closed(entry) and entry['exit_code']==0 and
                    not re.search(generation.FAILURE,err.read_text()),'packed prerequisite failed')
                row['checks'].append(dict(mode=mode,cases=check_packed_cases(out.read_text(),mode),passed=True))
        else:
            checked = selected['mode']=='checked'
            entry,_,_ = owner.run('bench',selected['command'],selected['environment'],P,
                wall=2400,cpu=72000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
            paths = list((root/'bench').glob('*/results.json'))
            if len(paths)==1:
                # Preserve individual samples before any later validator can raise.
                data = json.loads(paths[0].read_text())
                row['requests'] = data.get('results',{}).get('prefill_only',{}).get('requests',[])
            guards.require(supervision.closed(entry) and entry['exit_code']==0 and len(paths)==1, 'official benchmark failed')
            samples,journal = history.official(root,data,1 if checked else 4)
            row.update(requests=samples,journal=journal)
            native = gen.native_end(root,dict(binary=record['binary'],native_validation_probe=None),
                dict(config=json.loads((root/'config.json').read_text())),boot,entry['pid'])
            text = (root/'engine.stderr').read_text(errors='replace'); assets.verify_pool(text,5)
            guards.require(not re.search(generation.FAILURE,text), 'native runtime failure')
            placement = history.native_evidence(text,samples,1,False,history.admitted_loan(),native['native']['pid'],None) if not checked else None
            if checked:
                placement = history.admission.mixed_evidence(text,samples[0])
                row['packed_geometry'] = packed_bank.geometry(
                    text[text.index('strata serve: prompt '):],1)
            old = gen.committed(HISTORY)['processes'][0]['requests']
            guards.require([s['request_sha256'] for s in samples] ==
                [s['request_sha256'] for s in old[:len(samples)]], 'historical workload bytes changed')
            row.update(native_identity=native['native'],native_exit_code=0,route=placement,
                prompt_tok_s=[s['actual_read_tokens']*1000/s['prompt_ms'] for s in samples])
            if not checked:
                row['subsequent_median_tok_s']=statistics.median(row['prompt_tok_s'][1:])
        current(record,full_models=not bank and (checked or selected['mode']==MODES[-1]))
        row.update(passed=True,complete=True)
    except BaseException as error:
        row['error']=str(error)
        if (root/'engine.stderr').exists(): row['error_excerpt']=(root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row['ownership_closed']=owner.active is None and all(supervision.closed(e) for e in owner.commands)
        row.update(fault_probe_successful=False,boot_unchanged=False,fresh_faults=None)
        try:
            row.update(faults.fault_evidence(cursor,boot))
        except Exception as error:
            row['fault_probe_error']=str(error)
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row['fault_probe_successful'] and
            row['fresh_faults']==[] and row['boot_unchanged'])
        persist()
        if bank: record['bank']=row
        else: record['processes'].append(dict(mode=selected['mode'],**row))
        record['stage']='collecting' if row['passed'] and (bank or len(record['processes'])<len(MODES)) else 'closed'
        write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ('passed','complete','ownership_closed','error')}))
    return 0 if row['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','qualify','run')); args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.action=='prepare' else execute(args.action)


if __name__=='__main__': sys.exit(main())
