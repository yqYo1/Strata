#!/usr/bin/env python3
"""Bounded actual streamed-QSA16 model admission; Main, repository LICENSE.

Reuse this fork's owned protocol/runtime readers. Full configured capacity is
not a full-context fill; token-tiled thinking prompts are not speed/task evidence.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
import check_qsa16_stream_model as device
import q6_head_model_entry as previous
import q6_dense_model_entry as dense

P,B,RESULT=device.P,device.B,device.RESULT
ROOT,KEY=B/'qsa16-stream-model-entry-v1','qsa16_stream_model_entry_v1'
ATTEMPT='v1'
THIS='tools/sycl/qsa16_stream_model_entry.py'
REFERENCE=B/'q6-dense-head-backing-model-v1/record.json'
require,sha,write,signature=device.require,device.sha,device.write,device.common.signature
owner,guards,validation=previous.owner,previous.guards,previous.validation
INFO_FIELDS=set('context kv kv_resident expert_slots expert_cache_mib expert_slots_primary '
    'expert_cache_primary_mib spec mtp_max lookup vram_free_mib cvec arena_mib pool_workers '
    'pcie_frac spec_min_p conversation_cache_mib conversation_cache_slots '
    'conversation_cache_min_free_mib engine'.split())


def configure(attempt):
    global ROOT,KEY,ATTEMPT
    require(attempt in ('v1','v2'),'declared model attempt only')
    ATTEMPT=attempt;ROOT=B/('qsa16-stream-model-entry-'+attempt)
    KEY='qsa16_stream_model_entry_'+attempt


def native(expected=None):
    n=device.native(observers={device.THIS,THIS})
    prior=validation.committed(RESULT,(device.KEY,))
    require(prior['stage']=='closed' and prior['passed'] and prior['complete'] and
        prior['ownership_closed'] and not prior['active'] and prior['binaries']==n['binaries'] and
        prior['native_source_commit']==n['native_source_commit'] and
        [v['mode'] for v in prior['results']]==list(device.STAGES) and
        all(v['passed'] and v['ownership_closed'] for v in prior['results']) and
        prior['controller_admission']['controls']['passed'] and
        prior['native_observer_admission']['exact_seed_recipe_sha256']==
            n['observer_admission']['exact_seed_recipe_sha256'],
        'closed five-stage same-native/fixture/recipe qualification required')
    admission=prior['controller_admission']['source_commit']
    original=device.common.source.git(P,'show',admission+':'+device.THIS)
    require(hashlib.sha256(original).hexdigest()==
        prior['native_observer_admission']['added_observers'][device.THIS],
        'original qualified observer source must remain recoverable')
    proof=dict(n['observer_admission'],device_key=device.KEY,
        device_evidence_sha256=signature(prior),qualified_device_observer_source=admission,
        qualified_device_observer_sha256=hashlib.sha256(original).hexdigest(),
        changed_controller_scope='Explicit observer-set argument and this new model reader only;all original native,helper,fixture,recipe,SDK and ELF inputs exact.',
        earlier_model_result_inherited=False)
    require(expected is None or proof==expected,'streamed model admission changed')
    n['model_observer_admission']=proof
    return n


def requests(tokenizer):
    old=previous.requests(tokenizer);base=old['long']['ids']
    prefix,suffix=base[:16],base[-16:];body=base[16:-16]
    needed=32769-len(prefix)-len(suffix)
    ids=prefix+(body*((needed+len(body)-1)//len(body)))[:needed]+suffix
    require(len(ids)==32769 and ids[-16:]==base[-16:] and all(0<=v<248320 for v in ids),
        'exact32K thinking diagnostic extent')
    long=dict(ids=ids,prompt_tokens=len(ids),generated=32,thinking=True,
        wire_sha256=hashlib.sha256(previous.model.wire(ids,32)).hexdigest(),
        generator='Token-tile the middle of the maintained merge-requirement thinking prompt;preserve first/last16IDs. Diagnostic only,no benchmark or quality claim.')
    return dict(long=long,short=old['short'])


def context(admission=None):
    ctx=device.load_module('qsa16_stream_actual_model',previous.__file__)
    ctx.ROOT,ctx.KEY=ROOT,KEY
    ctx.build=SimpleNamespace(**{**vars(device.common),'ROOT':device.NATIVE,'KEY':device.NATIVE_KEY})
    ctx.device=SimpleNamespace(KEY=device.KEY)
    ctx.native=lambda:native(admission)
    ctx.controller_pin=lambda:device.common.source.pin(P,
        ['tools/sycl','tools/strata_tokenizer.py','data/expert-profile.bin'])
    ctx.requests=requests
    return ctx


def parse_info(text):
    pairs=[p.split('=',1) for p in text.split()]
    require(text.isascii() and all(len(p)==2 and p[0] and p[1] for p in pairs) and
        len({p[0] for p in pairs})==len(pairs) and {p[0] for p in pairs}==INFO_FIELDS,
        'exact producer INFO schema,without duplicates,missing or extra fields')
    fields=dict(pairs);placement(fields);return fields


def placement(fields):
    expected=dict(context='262144',kv='int8',kv_resident='20480',expert_slots='1256',
        expert_slots_primary='1256',spec='4',pool_workers='5',pcie_frac='0.00',mtp_max='0',
        lookup='0',cvec='0',spec_min_p='0.00',conversation_cache_mib='0')
    integers=INFO_FIELDS-{'kv','pcie_frac','spec_min_p','engine'}
    require(set(fields)==INFO_FIELDS and all(fields.get(k)==v for k,v in expected.items()) and
        all(type(fields[k]) is str and fields[k].isascii() and fields[k].isdecimal() for k in integers) and
        int(fields['arena_mib'])>0 and fields['expert_cache_primary_mib']==fields['expert_cache_mib'] and
        re.fullmatch(r'\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?',fields['engine']),
        'actual streamed placement or producer field format differs')


def route(text):
    plans=re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=(\d+) residual_dtype=(\w+) ',text,re.M)
    require(plans==[('32768','8192','bf16')]*2,'both long requests must take exact layer-major BF16 route')
    require(text.count('QSA tensor IO=BF16, indexer/GDN internals FP32;')==2,
        'both long requests need actual typed-QSA layer-major entry')
    require(len(re.findall(r'^strata trace: layer-major layer \d+ of 48: loading ',text,re.M))==96,
        'complete48layers twice required')
    return dict(actual_mode=1,logical_cells=262144,resident_cells=20480,
        long_prefill_rows=32768,layer_major_entries=2,layers_completed=96,
        typed_QSA_entry=True,legacy_source_label='The diagnostic source=resident-Mode0 text is stale;INFO and exact checked production source establish mode1. No extra kernel dispatch telemetry.',
        decode_routes=dense.routes(text,1),head_handoffs=dense.handoffs(text,0))


def prepare():
    require(not ROOT.exists() and KEY not in device.common.public(),'fresh actual-model plan only')
    n=native();prior=validation.committed(RESULT,(device.KEY,));reference=json.loads(REFERENCE.read_text())
    require(reference['passed'] and reference['complete'] and reference['ownership_closed'],
        'closed original model asset/configuration source required')
    argv=list(reference['argv']);argv[0]=n['binaries']['strata']['path']
    for flag,value in (('--max-context','262144'),('--kv-resident','20480'),('--prefill','8192')):
        require(argv.count(flag)==1,'one explicit argument: '+flag);argv[argv.index(flag)+1]=value
    old=reference['environment']['STRATA_MEASUREMENT_ROOT']
    env={k:v.replace(old,str(ROOT)) for k,v in reference['environment'].items()}
    require('STRATA_KV_STAGE_OWN' not in env,'stage allocation must be an explicit treatment')
    if ATTEMPT=='v2':env['STRATA_KV_STAGE_OWN']='1'
    env.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),
        STRATA_MEASUREMENT_ROOT=str(ROOT),STRATA_SYCL_PREFILL_QSA_STREAM='1',
        STRATA_SYCL_DECODE_Q6_DPAS_HEAD='2',STRATA_SYCL_DECODE_Q6_DPAS_DENSE='1',
        STRATA_PREFILL_HEAD_HOST_BACKING='0',STRATA_TRACE='1',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/'validation.log'))
    env.pop('UR_LOG_TRACING',None)
    require(env['STRATA_SYCL_PREFILL_NATIVE16']=='bf16' and env['STRATA_SYCL_PREFILL_NATIVE16_QSA']=='1' and
        env['STRATA_SYCL_PREFILL_QSA_MATH']=='bf16-prefix-ordered-exp' and
        env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']=='1' and env['EnableDirectSubmission']=='0' and
        env['STRATA_PREFILL_RELEASE_HEADS']=='1' and env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE']=='0',
        'preserve typed route,independent current decode and qualified safety')
    ctx=context(n['model_observer_admission'])
    row=dict(stage='prepared',source_commit=device.common.source.git(P,'rev-parse','HEAD').decode().strip(),
        controller=ctx.controller_pin(),native_source_commit=n['native_source_commit'],
        native_build=dict(binaries=n['binaries']),native_record=dict(path=str(device.NATIVE/'record.json'),sha256=sha(device.NATIVE/'record.json')),
        native_observer_admission=n['model_observer_admission'],device_evidence_sha256=signature(prior),
        models=reference['models'],models_sha256=reference['models_sha256'],
        runtime_manifest=prior['runtime_manifest'],runtime_files=reference['runtime_files'],
        probe=prior['native_validation_probe'],generator_runtime=previous.prompts.generator_runtime(),
        tokenizer=reference['tokenizer'],requests=requests(Path(reference['tokenizer'])),
        environment=env,argv=argv,boot=prior['boot'],driver=prior['driver'],
        capacity=dict(main_QSA_state_reduction_vs_131072_resident_bytes=1164967936,
            full_stage_payload_bytes=262144*1056,logical_host_KV_payload_bytes=12*262144*1056,
            scope='Pinned allocation formula only;32Kmodel must establish actual READY/scratch/residual admission. Shared scratch,largest layer,head retirement and maximum single allocation remain independently enforced by production planner.'),
        budgets=dict(wall=1200,cpu=14400,rss_cap=104<<30,text_cap=32<<20,file_cap=32<<20,total_cap=128<<20),
        plan='Host placement guards,then one fresh actual mode1 process with32769/82/32769thinking prompts and32outputs each. Full262144configured logical capacity,20480resident main cells,INT8KV,8192chunks,1256expert slots. Long prompts exceed resident extent and short request exercises decode placement. Require READY/INFO,immutable environment/providers,complete zero-reuse GEN/MTP output,96layer-major layer visits,typedBF16QSA entry,two complete head retire/restore cycles,quiet validation,normal0,closed owners,unchanged boot/no fresh fault. Hash all15model assets before and after. Stop first failure,no retries or refreshed pins.',
        limits='Actual streamed route/handoff admission only. No numerical/task-quality,full262144fill/tail/RESTORE,clean PP/TG/TTFT,sustained decode,ordinary provider teardown or adoption claim. Subsequent comparisons must use bench.sh with long/agent/strong-and-weak thinking workloads and physical full-capacity checks.',
        results=[],commands=[],passed=False,complete=False,active=False,ownership_closed=True,
        model_executed=False,speed_eligible=False,model_adoption_qualified=False,
        retention='Retain compact actual placement,progress/head/mode extents and original failures. Retire successful full stage/UR logs and generated token/wire copies after evidence verification. No new native generation;keep a347/552b for imminent independent-phase comparisons and review extra candidate after comparison.')
    if ATTEMPT=='v2':
        old_root=B/'qsa16-stream-model-entry-v1';old=json.loads((old_root/'record.json').read_text())
        require(validation.committed(RESULT,('qsa16_stream_model_entry_v1',))==ctx.compact(old) and
            old['stage']=='failed' and not old['passed'] and old['ownership_closed'] and not old['active'] and
            old['failure_analysis']['actual_chunk']==4096 and old['failure_analysis']['native_exit_code']==0 and
            old['failure_analysis']['all_three_requests_completed'] and old['native_build']==row['native_build'],
            'preserve the original closed4096route rejection;no native result inherited')
        row['previous_attempt']=dict(key='qsa16_stream_model_entry_v1',source_commit=old['source_commit'],
            private_record_sha256=sha(old_root/'record.json'),passed=False,native_exit_code=0,actual_chunk=4096)
        row['capacity']['stage_allocation']='Separately owned264MiB excluded from expert-cache scratch loan;actual8192admission still required.'
        row['plan']+=' V2 changes only the pre-exec stage allocation to STRATA_KV_STAGE_OWN=1. Preserve the failed V1,original native source/five ELFs,decoder,logical/resident capacities,1256expert slots,8192actual-route requirement and all other criteria. Correct the INFO reader to validate the full20-field producer schema with duplicate/extra/missing/numeric guards before admission.'
    ctx.assets(row,full=True);ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir()
    ctx.save(row);print('Prepared physical262144-capacity actual mode1 route;commit before host/GPU.',flush=True)


def execute(stage):
    row=json.loads((ROOT/'record.json').read_text());ctx=context(row['native_observer_admission']);ctx.verify(row)
    if stage=='host':
        require(row['stage']=='prepared','one prospective host admission only')
        good=dict(context='262144',kv='int8',kv_resident='20480',expert_slots='1256',
            spec='4',pool_workers='5',pcie_frac='0.00',mtp_max='0',expert_cache_mib='2455',
            expert_slots_primary='1256',expert_cache_primary_mib='2455',lookup='0',vram_free_mib='942',
            cvec='0',arena_mib='47962',spec_min_p='0.00',conversation_cache_mib='0',
            conversation_cache_slots='4',conversation_cache_min_free_mib='2560',engine='0.1.38')
        wire=lambda f:' '.join(k+'='+v for k,v in f.items())
        text=wire(good);parse_info(text);parse_info(wire(dict(reversed(list(good.items())))))
        bad=[]
        for key,value in (('context','131072'),('kv_resident','0'),('kv_resident','262144'),('expert_slots','1536')):
            bad.append(wire(dict(good,**{key:value})))
        bad.extend(['context=131072 '+text,text+' extra=1',wire({k:v for k,v in good.items() if k!='engine'}),
            text.replace('context=262144','context262144'),text.replace('engine=0.1.38','engine='),
            wire(dict(good,arena_mib='nan')),wire(dict(good,vram_free_mib='-1')),
            wire(dict(good,spec_min_p='nan')),wire(dict(good,expert_cache_primary_mib='2454'))])
        for broken in bad:
            try:parse_info(broken)
            except RuntimeError:continue
            raise RuntimeError('INFO guard accepted malformed or wrong placement')
        require(('STRATA_KV_STAGE_OWN' in row['environment'])==(ATTEMPT=='v2') and
            (ATTEMPT!='v2' or row['environment']['STRATA_KV_STAGE_OWN']=='1'),'frozen stage treatment')
        row.update(stage='host_passed',host_controls=dict(positive=2,negative=len(bad),passed=True,GPU_executed=False))
        ctx.save(row);print('Host placement guards pass;commit before actual model.',flush=True);return
    require(stage=='run' and row['stage']=='host_passed' and not row['results'],'one native process;no retry')
    frozen=copy.deepcopy(row);scope=None;result=dict(passed=False,requests=[])
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    def persist():write(ROOT/'execution.json',dict(active=bool(scope and scope.active),commands=scope.commands if scope else []))
    try:
        scope=owner.Owner(ROOT/'owned',persist)
        def interact(io,command):
            startup=[]
            while True:
                line=io.line(480);startup.append(line);require(len(startup)<=32,'startup stdout cap')
                if line.startswith('READY '):break
            require(line=='READY 262144 stop','full configured capacity required')
            info=[v[5:] for v in startup if v.startswith('INFO ')];require(len(info)==1,'one actual INFO')
            result['placement']=parse_info(info[0])
            actual=dict(v.split('=',1) for v in Path('/proc',str(command['pid']),'environ').read_text().split('\0') if '=' in v)
            require(actual==row['environment'],'immutable native environment')
            result.update(previous.model.runtime_evidence(row,command['pid'],
                (scope.output/'model.stderr').read_text(errors='replace')))
            result['memory_at_READY']=previous.memory_snapshot(command['pid'])
            print('Actual mode1 READY:logical262144,resident20480.',flush=True)
            for name in ('long','short','long'):
                r=row['requests'][name];parser=previous.protocol.RequestProtocol(r['prompt_tokens'],r['generated'])
                io.send(previous.model.wire(r['ids'],r['generated']))
                while parser.done is None:
                    line=io.line(480);parser.feed(line)
                    if line.startswith('PP '):print(name+': '+line,flush=True)
                require(int(parser.done[7])>0,'actual MTP offers required')
                result['requests'].append(dict(label=name,prompt_tokens=r['prompt_tokens'],generated=len(parser.tokens),
                    accepted=int(parser.done[6]),offered=int(parser.done[7]),output_ids_sha256=signature(parser.tokens),complete=True))
                row['model_executed']=True;print(name+' complete with decode handoff.',flush=True)
            result['memory_after_requests']=previous.memory_snapshot(command['pid']);io.send(b'QUIT\n');io.exit(30)
        e,out,err=scope.run('model',row['argv'],row['environment'],P,interaction=interact,**row['budgets'])
        require(owner.closed(e) and e['exit_code']==0,'actual model process failed')
        text=err.read_text(errors='replace');result['routes']=route(text)
        log=ROOT/'validation.log';require((not log.exists() or log.stat().st_size==0) and
            not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain',text),
            'runtime/parameter-validation failure')
        ctx.assets(row,full=True);ctx.verify(frozen)
        result.update(host_probe_accepted=True,native_exit_code=0)
    except BaseException as error:result.update(host_probe_accepted=False,error=repr(error))
    finally:
        if scope:validation.finish(result,scope,cursor,row['boot'])
        result['passed']=bool(result.get('passed') and len(result['requests'])==3 and result.get('driver_after')==row['driver'])
        row['results'].append(result);row['commands']+=scope.commands if scope else []
        row.update(stage='closed' if result['passed'] else 'failed',passed=result['passed'],complete=result['passed'],
            active=bool(scope and scope.active),ownership_closed=bool(result.get('ownership_closed')))
        ctx.save(row)
    require(result['passed'],result.get('error','actual streamed model gate failed'))
    print('Actual streamed long/short/long route closes;speed/full-context adoption remain separate.',flush=True)


if __name__=='__main__':
    require(len(sys.argv) in (2,3) and (len(sys.argv)==2 or sys.argv[1]=='v2'),
        '[v2] prepare|host|run')
    configure('v2' if len(sys.argv)==3 else 'v1');action=sys.argv[-1]
    require(action in ('prepare','host','run'),'prepare|host|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        prepare() if action=='prepare' else execute(action)
