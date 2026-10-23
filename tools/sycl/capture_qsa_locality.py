#!/usr/bin/env python3
"""Main-owned actual-mask capture; repository LICENSE applies.

Derived only from own checked controller41579947c. Unchanged official bench.sh,
one short warmup and one98304-minimum thinking request on the separate checked
diagnostic build. Readbacks/logging make these clocks ineligible as clean speed.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import select
import shutil
import socket
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode=True
import check_qsa_order_replay as checked
import generation_engine as gen
import resource_engine as resource
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
import packed_bank as bank
from measure_down_scatter import postchecks

P,B,A=gen.P,gen.B,gen.A
ROOT=checked.replay.diagnostic.ROOT/'capture-v1'
RESULT=A/'qsa-current-tail-capture.json'
SCOPE='actual_qsa_tail_capture_controller_audit_20261023'
BASE=ROOT/'selected'


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def review():
    row=gen.committed(A/'research-index.json')[SCOPE]
    guards.require(row.get('source_gate_passed') is True,'actual capture controller/source review required')
    for item in row['review'].values():guards.require(gen.common.digest(item['path'])==item['sha256'],'source review changed')
    return row['review']


def qualification():
    replay=checked.native();diagnostic,_=checked.replay.diagnostic_admission()
    readback=gen.committed(checked.replay.READBACK_RESULT);smoke=gen.committed(checked.paths('smoke')[1])
    guards.require(readback['passed'] and readback['complete'] and readback['ownership_closed'] and
        not readback['active'] and readback['fresh_faults']==[] and readback['boot_unchanged'] and
        readback['binary']==diagnostic['artifacts']['readback'] and
        [x['mode'] for x in readback['checks']]==['small','full'] and
        all(x['normal_owned_exit'] and x['state']=='complete' and x['exit_code']==0 for x in readback['checks']),
        'matching small/full checked readback required')
    guards.require(smoke['passed'] and smoke['complete'] and smoke['ownership_closed'] and
        not smoke['active'] and smoke['fresh_faults']==[] and smoke['boot_unchanged'] and
        smoke['binary']==replay['binary'] and all(smoke['checks'][k] for k in
        ('original_row_output_bits','V_perturbation','guards','immutable')),'implemented qualified replay consumer required')
    lineage=checked.replay.runtime_lineage()
    return diagnostic,dict(diagnostic_build_sha256=gen.common.digest(checked.replay.diagnostic.RESULT),
        readback_sha256=gen.common.digest(checked.replay.READBACK_RESULT),
        replay_build_sha256=gen.common.digest(checked.replay.RESULT),
        replay_smoke_sha256=gen.common.digest(checked.paths('smoke')[1]),
        replay_binary=replay['binary'],runtime_lineage=lineage,review=review())


def environment(setup):
    _,restored,_=resource.modules();env=restored.environment(setup,ROOT,'0')
    env.update(STRATA_CPU_WORK_CENSUS='0',STRATA_IQ2_PAIR_AUDIT='0',STRATA_DECODE_TIMING='0',
        STRATA_SNAPSHOT_VERIFY='1',STRATA_SYCL_CACHE_LOAN_AUDIT='1',STRATA_QSA_LOCALITY_CAPTURE=str(BASE))
    guards.require('LD_PRELOAD' not in env and 'STRATA_QSA_DUMP' not in env,'private native validation;legacy dump forbidden')
    for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',SYCL_CACHE_PERSISTENT='0').items():
        guards.require(env.get(k)==v,'qualified runtime safety setting changed')
    return env


def pins(revision):
    result=gen.benchmark_pins(revision)
    for name in ('bench/sycl/qsa_order_replay.cpp','src/prefill/prefill.cpp','src/prefill/layer_major.inc',
        'src/program/generate.cpp','include/strata/prefill/locality_capture.hpp','data/expert-profile.bin'):
        raw=gen.git('show',revision+':'+name)
        guards.require(raw==(P/name).read_bytes(),'commit exact capture input: '+name)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def current(record,full_models=False):
    full,proof,setup=resource.qualification();diagnostic,check=qualification();assets,restored,_=resource.modules()
    guards.require(record['binary']==diagnostic['artifacts']['engine'] and
        record['build_source_commit']==diagnostic['artifact_source_commits']['engine'] and
        record['selected_binary']==full['binary'] and record['qualification']==proof and
        record['diagnostic_qualification']==check and record['native_validation_probe']==validation.preload_qualification(),
        'capture native/diagnostic qualification changed')
    guards.require(record['source_fingerprint']==assets.fingerprint(pins(record['source_commit'])) and
        record['environment']==environment(setup) and record['config']==restored.recipe.config(setup,ROOT),
        'capture source/config/environment changed')
    for p,h in record['generated_inputs'].items():guards.require(gen.common.digest(p)==h,'generated input changed')
    guards.require(record['machine_identity']==assets.machine_identity() and
        assets.model_pins(record['config'],record['model_assets'],full=full_models)==record['model_assets'],
        'capture model/platform changed')


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh single-use capture only')
    full,proof,setup=resource.qualification();diagnostic,check=qualification();assets,restored,_=resource.modules()
    revision=gen.git('rev-parse','HEAD').decode().strip();source=pins(revision);config=restored.recipe.config(setup,ROOT)
    for k,v in (('--expert-cache','1256'),('--prefill','8192'),('--max-context','131072'),('--kv','int8')):
        guards.require(config['args'][config['args'].index(k)+1]==v,'selected configuration changed: '+k)
    models=assets.model_pins(config,setup['model_assets'],full=True)
    ROOT.mkdir(mode=0o700)
    for n in ('cache','tmp'):(ROOT/n).mkdir()
    shutil.copyfile(P/'data/expert-profile.bin',ROOT/'expert-profile.bin')
    guards.require(gen.common.digest(ROOT/'expert-profile.bin')==setup['expert_profile_sha256'],'profile changed')
    probe=validation.preload_qualification();binary=diagnostic['artifacts']['engine'];transaction=secrets.token_hex(16)
    write(ROOT/'config.json',config);write(ROOT/'plan.json',dict(binary=binary,native_validation_probe=probe))
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    record=dict(stage='prepared',mode='actual-QSA-tail-and-prefix-count-capture',source_commit=revision,
        build_source_commit=diagnostic['artifact_source_commits']['engine'],binary=binary,selected_binary=full['binary'],
        qualification=proof,diagnostic_qualification=check,native_validation_probe=probe,
        source_count=len(source),source_fingerprint=assets.fingerprint(source),config=config,environment=environment(setup),
        model_assets=models,machine_identity=assets.machine_identity(),runtime_asset_delta=setup['runtime_asset_delta'],
        generated_inputs={str(ROOT/n):gen.common.digest(ROOT/n) for n in ('config.json','plan.json','expert-profile.bin')},
        transaction_id=transaction,
        command=['/bin/sh',str(P/'bench.sh'),'--config',str(ROOT/'config.json'),'--port',str(port),
            '--prefill-only','--prefill-tokens','98304','--think','--runs','1','--profile-request-gates',
            '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(ROOT/'bench')],
        prospective_gates='Fresh0700 transaction and exclusive claim before child. One owned native,short warmup then '
            'one fresh98304..131048-input thinking request/1..24output. Same31runtime/15model/selected13assets. '
            'Private same-PID validation,immutable native environment and warmup journal verified before long request; '
            'both capture paths MUST still be absent then. Require actual BF16/mode10/private48-layer processing,1256-slot '
            'cache/map and dense48 restoration,exact official/journal/native closure. Exactly12 paired512x2051 IDs '
            'and512x4 step records,tail extent/causal sorted unique IDs checked by the committed replay host parser. '
            'Every expected layer/chunk prefix submission is counted with bounded geometry. Full model hashes after normal exit. '
            'Only all checks plus no fresh fault permit exclusive completion binding session/request/both file hashes. '
            'First failure stops,no retry,no speed/adoption claim.',
        retention='Raw selected.ids/steps <=64MiB exclusively for imminent same-generation actual-mask GPU replay '
            'validation and three clean order comparisons. Commit useful samples then retire raw files,successful traces '
            'and this additional diagnostic generation at its decision. Keep compact conserved prefix counts,not full logs/arrays.',
        budgets=dict(wall=1800,cpu=54000,rss_cap=96<<30,text_cap=32<<20,file_cap=128<<20,total_cap=384<<20),
        complete=False,passed=False,active=False,ownership_closed=True,GPU_executed=False,
        clean_speed_qualified=False,hardware_saturation_qualified=False,model_adoption_qualified=False,actual_mask_replay_qualified=False)
    current(record);write(ROOT/'capture-plan.json',record);write(RESULT,record)


def route_frame(log,cells):
    headers=list(re.finditer(r'^strata serve: prompt .*$',log,re.M));guards.require(len(headers)==2,'two native frames')
    segment=log[headers[0].end():headers[1].start()]
    route=re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 '
        r'residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) '
        r'residual_gpu=(\d+) residual_host=0 residual_staging=0 ',segment,re.M)
    guards.require(len(route)==1 and tuple(map(int,route[0]))==(cells,cells*20480,cells,cells*20480) and
        len(re.findall(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='+
            str(min(cells,81920))+r' mixed_fp16_bytes=0$',segment,re.M))==1,'actual long GPU BF16/mode10 route')
    cycle,pub=bank.admission.loan_cycle(segment);dense=bank.admission.records(segment,'strata prefill dense',3)
    guards.require([(r['phase'],r['residency'],r['activations']) for _,r in dense]==
        [('retired','Full','0'),('bank','Bank','0'),('restored','Full','48')] and
        pub[0][0]<dense[0][0]<dense[1][0]<dense[2][0]<pub[1][0],'dense/cache handoff not restored')
    return dict(actual_GPU_BF16_rows=cells,actual_MoE_mode=10,private_bank=bank.geometry(segment,1),
        restored_dense_layers=48,cache_payload_and_map_match=True,loan_generation=cycle['generation']),segment


def prefix_counts(segment,cells):
    pattern=r'^strata QSA locality prefix: layer=(\d+) first=(\d+) queries=(\d+) live=(\d+) page=(\d+) packed_logical_rows=(\d+) submitted=1$'
    matches=re.findall(pattern,segment,re.M);lines=[s for s in segment.splitlines() if s.startswith('strata QSA locality prefix:')]
    expected={(layer,first,min(8192,cells-first),min(first+8192,cells),4,
        ((min(first+8192,cells)+3)//4)*8) for layer in range(12) for first in range(0,cells,8192)}
    actual=[tuple(map(int,x)) for x in matches]
    guards.require(len(actual)==len(lines)==len(expected) and set(actual)==expected,'prefix submission geometry/count differs')
    per_layer=sum(x[-1] for x in expected if x[0]==0)
    return dict(layers=12,chunks_per_layer=(cells+8191)//8192,submitted_calls=len(actual),
        packed_logical_rows_per_layer=per_layer,packed_logical_rows_total=per_layer*12,
        cells=cells,page_size=4,physical_traffic_measured=False,completion_or_service_time_measured=False)


def bind_tail(files,cells):
    """Bind the parser's internally consistent pair to this actual request."""
    guards.require(98303<=cells<=131047,'actual request cells outside committed scope')
    for suffix,width in (('.ids',2051),('.steps',4)):
        item=files[suffix];record_bytes=16+512*width*4
        guards.require(item['bytes']==12*record_bytes,'exact bounded paired payload extent')
        with Path(item['path']).open('rb') as stream:
            for layer in range(12):
                stream.seek(layer*record_bytes);raw=stream.read(16)
                guards.require(len(raw)==16 and struct.unpack('<4i',raw)==(layer,cells-512,512,width),
                    'captured header does not match actual request tail')
            stream.seek(12*record_bytes);guards.require(not stream.read(1),'capture trailing bytes')
    return dict(layers=12,first=cells-512,queries=512,cap=2051,steps=4,cells=cells)


def official(record,boot,entry):
    path,=(ROOT/'bench').glob('*/results.json');result=json.loads(path.read_text());cohort=result['results']['prefill_only']
    guards.require(result['complete'] and not result['results']['failures'] and result['params']['mode']=='prefill-only' and
        result['params']['think'] is True and result['params']['prefill_tokens']==98304 and cohort['complete'] and
        cohort['eligible_runs']==1,'incomplete official capture request')
    sample,=cohort['requests']
    guards.require(all(sample[k] is True for k in ('ok','eligible','stream_complete','thinking_requested','reasoning_observed')) and
        98304<=sample['actual_prompt_tokens']==sample['actual_read_tokens']<=131048 and sample['actual_reused_tokens']==0 and
        1<=sample['generated_tokens']<=24 and math.isfinite(sample['prompt_ms']) and sample['prompt_ms']>0,'ineligible capture request')
    journal,_=gen.journal_evidence(ROOT);journal_path,=(ROOT/'bench').glob('*/requests.jsonl')
    events=[json.loads(s) for s in journal_path.read_text().splitlines()]
    guards.require(journal['valid'] and journal['complete'] and len(events)==5 and
        [e['kind'] for e in events]==['identity','start','result','start','result'] and
        events[-1]['sample']=={k:v for k,v in sample.items() if k!='run'} and
        events[-1]['request_index']==events[-2]['index'] and
        events[-1]['sample']['request_sha256']==events[-2]['request_sha256'],'journal/request binding')
    native=gen.native_end(ROOT,record,dict(config=record['config']),boot,entry['pid']);log=(ROOT/'engine.stderr').read_text(errors='replace')
    activation=validation.preload_evidence(log,native['native']['pid'])
    guards.require(activation['loader']==record['native_validation_probe']['loader']['path'],'validation provider differs')
    body=log.split(f'strata UR validation: pid={native["native"]["pid"]} phase=before-main',1)[1]
    guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:',body),'runtime error')
    footers=re.findall(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$',log,re.M)
    guards.require(len(footers)==2 and tuple(map(int,footers[-1]))==
        (sample['actual_read_tokens'],0,sample['actual_read_tokens'],sample['generated_tokens']),'native request work differs')
    warm=tuple(map(int,footers[0]));guards.require(1<warm[0]<1024 and warm[1]==0 and warm[0]==warm[2] and 1<=warm[3]<=24,'warmup extent')
    resource.modules()[0].verify_pool(log,5);route,segment=route_frame(log,sample['actual_read_tokens']-1)
    cells=sample['actual_read_tokens']-1
    files={suffix:dict(path=str(BASE)+suffix,bytes=Path(str(BASE)+suffix).stat().st_size,
        sha256=gen.common.digest(str(BASE)+suffix)) for suffix in ('.ids','.steps')}
    tail=bind_tail(files,cells)
    return dict(request=sample,native_identity=native['native'],validation_activation=activation,files=files,
        tail=tail,
        actual_long_request_route=route,prefix=prefix_counts(segment,cells),journal_sha256=gen.common.digest(journal_path))


def run():
    record=gen.committed(RESULT)
    guards.require(record==json.loads((ROOT/'capture-plan.json').read_text()) and record['stage']=='prepared' and
        not any((ROOT/p).exists() for p in ('record.json','launch.json','transaction.claim','capture-complete.json')) and
        not any(Path(str(BASE)+s).exists() for s in ('.ids','.steps')),'fresh committed capture transaction required')
    current(record);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in before.splitlines() if x.startswith('-- cursor: '));owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(record,owned_commands=owner.commands if owner else []))
    def interact(io,leader):
        for stage in ('begin','end'):
            while True:
                line=io.line(1500)
                if line.startswith('STRATA_PROFILE_GATE '):break
            guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==
                dict(stage=stage,phase='prefill-only',run=0),'request gate differs')
            if stage=='begin':
                native=json.loads((ROOT/'engine-start.json').read_text())['native'];actual=supervision.identity(native['pid'])
                guards.require(actual and actual['start_ticks']==native['start_ticks'] and actual['sid']==leader['pid'] and
                    Path('/proc',str(native['pid']),'exe').resolve()==Path(record['binary']['path']).resolve(),'native identity differs')
                env=dict(s.split(b'=',1) for s in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in s)
                guards.require(all(env.get(k.encode())==v.encode() for k,v in record['environment'].items()) and
                    env.get(b'LD_PRELOAD')==record['native_validation_probe']['library']['path'].encode(),'native immutable environment differs')
                path,=(ROOT/'bench').glob('*/requests.jsonl');events=[json.loads(s) for s in path.read_text().splitlines()]
                guards.require(len(events)==4 and events[2]['kind']=='result' and events[2]['sample']['ok'] is True and
                    events[3]['phase']=='prefill-only','warmup before long dispatch failed')
                guards.require(not any(Path(str(BASE)+s).exists() for s in ('.ids','.steps')),'warmup produced capture;refuse stale pair')
                validation.preload_evidence((ROOT/'engine.stderr').read_text(errors='replace'),native['pid'])
                record.update(native_identity=native,effective_environment_verified=True,warmup_capture_absent=True);persist()
            io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();guards.require(time.monotonic()<deadline,'benchmark teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    owner=supervision.Owner(ROOT/'owned',persist);persist();completion=None
    try:
        with (ROOT/'transaction.claim').open('x') as f:f.write(record['transaction_id']+'\n');f.flush();os.fsync(f.fileno())
        entry,_,_=owner.run('bench',record['command'],record['environment'],P,interaction=interact,**record['budgets'])
        record['GPU_executed']=(ROOT/'engine-start.json').exists()
        guards.require(supervision.closed(entry) and entry['exit_code']==0,'capture benchmark did not close normally')
        record.update(official(record,boot,entry));current(record,full_models=True)
        args=[record['diagnostic_qualification']['replay_binary']['path'],'--host-check',str(BASE)]
        host,out,err=owner.run('actual-host-check',args,record['environment'],P,wall=60,cpu=120,rss_cap=2<<30,
            text_cap=1<<20,file_cap=1<<20,total_cap=384<<20)
        guards.require(supervision.closed(host) and host['exit_code']==0 and
            out.read_text()=='capture_parser=PASS records=12 gpu=UNRUN\n' and not err.read_text(),'actual captured pair parser failed')
        for item in record['files'].values():guards.require(gen.common.digest(item['path'])==item['sha256'],'captured bytes changed')
        probe=dict(passed=True,ownership_closed=True);postchecks(probe,boot,cursor);guards.require(probe['passed'],'fault/incomplete observation')
        completion=dict(state='complete',transaction_id=record['transaction_id'],native_source=record['build_source_commit'],
            binary=record['binary'],native_identity=record['native_identity'],boot_id=boot,
            request_sha256=record['request']['request_sha256'],journal_sha256=record['journal_sha256'],
            config_sha256=record['generated_inputs'][str(ROOT/'config.json')],tail=record['tail'],files=record['files'],
            normal_owned_exit=True,actual_host_check_passed=True)
        record.update(passed=True,complete=True,actual_host_check_passed=True)
    except BaseException as error:record['error']=repr(error)
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(e) for e in owner.commands)
        postchecks(record,boot,cursor);record['stage']='closed'
        if record['passed']:
            try:
                guards.require(completion is not None,'checked payload completion absent')
                with (ROOT/'capture-complete.json').open('x') as f:
                    json.dump(completion,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
                record['completion']=dict(path=str(ROOT/'capture-complete.json'),sha256=gen.common.digest(ROOT/'capture-complete.json'))
            except BaseException as error:record.update(error=repr(error),passed=False,complete=False)
        if (ROOT/'engine-start.json').exists() and not record['GPU_executed']:record['GPU_executed']=None
        persist();write(RESULT,record)
    print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare() if sys.argv[1]=='prepare' else run())
