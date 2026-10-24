#!/usr/bin/env python3
"""Main-owned fixed-MTP2 route qualification on the exact selected6e40 binary.

Repository LICENSE. No native rebuild, kernel/precision change or adoption.
The short synthetic requests exercise wiring and graph retirement, not speed.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode = True
import mtp_resident_serve as protocol
import check_full_context_source as selected
import generation_engine as gen
import ur_validation as validation

P, B = protocol.P, protocol.B
ROOT = B / 'mtp2-route-v1'
RESULT = protocol.RESULT.with_name('mtp2-route-current.json')
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
SNAPSHOT = P / '.cache/strata-research/20261024/mtp2-source-audit/source-manifest.json'
SNAPSHOT_SHA = '1300c58a593dce8975c0d061022add49a5386eabb47886662ba322789f1fb8ab'
require, digest, write = protocol.require, protocol.digest, protocol.write
fingerprint = protocol.gate.native.fingerprint


def requests():
    return [[i % 4096 for i in range(n)] for n in (33, 257, 16385, 33)]


def compact(row):
    return {k:v for k,v in row.items() if k not in ('environment','model_assets','commands')}


def seal(row):
    return fingerprint({k:row[k] for k in ('environment','model_assets')})


def source_check():
    require(digest(SNAPSHOT) == SNAPSHOT_SHA, 'original source snapshot manifest changed')
    snapshot = json.loads(SNAPSHOT.read_text()); require(snapshot['source_commit'] == BASE, 'wrong source')
    for name, raw in selected.build.blobs(P, BASE, snapshot['files']):
        require(hashlib.sha256(raw).hexdigest() == snapshot['files'][name] ==
            digest(SNAPSHOT.parent/'source'/name), 'original audit source differs: '+name)
    return dict(source_commit=BASE,manifest_sha256=SNAPSHOT_SHA,files=len(snapshot['files']),
        contracts={
            'generate.cpp:5759-5760':'--spec 4 keeps allocated width4;--mtp-max-t 2 sets S_mtp2 and max_drafts1.',
            'generate.cpp:6582-6665':'Fresh first targetT1,then fixedT2 without confidence/suffix;commit accepted prefix before drafting,skip final draft at EOS/output cap.',
            'generate.cpp:122-178':'Short n<=8192 does not detach MTP experts;16384 prompt cells do. Prompt checked owners end before expert restoration.',
            'mtp.cpp:580-649':'Detach requires allocated max_t4,drains MTP stream,destroys all width-keyed round/chain graphs and preserves borrowed verifier window_R. Restore drains prompt consumers before expert upload.',
            'mtp.cpp:1091-1148,1340-1425':'Round graph keyed by actualT;catch-upT rows plus one full proposal. max_drafts1 skips every chain step. Buffers and row7 staging retain max_t4 extents.',
            'mtp_graph_observer.hpp:34-103':'Host observer request4,skip0,limit2 must see T1/T2,one round replay each,zero chain work;not a device-service or speed measurement.'})


def prepare():
    require(not ROOT.exists() and not RESULT.exists(), 'fresh fixed2 plan only;no duplicate/retry')
    original, proof = selected.baseline(); source = source_check(); base = protocol.model.setup()
    protocol.model.model_pins(base['config'],base['models'])
    speed = validation.committed(RESULT.with_name('phase-prefill-speed-current.json'))
    require(speed['selection']['selected_source'] == BASE and not speed['selection']['candidate_adopted'], 'original prefill required')
    runtime = validation.committed(RESULT.with_name('client-resources-current.json'));gen.runtime_assets_unchanged(runtime)
    preload = validation.preload_qualification()
    revision = gen.git('rev-parse','HEAD').decode().strip(); pins = gen.benchmark_pins(revision)
    args = list(base['config']['args']);require('--mtp-max-t' not in args and args[args.index('--spec')+1]=='4','reference width differs')
    args[args.index('--expert-profile')+1] = str(P/'data/expert-profile.bin');args += ['--mtp-max-t','2']
    env = dict(speed['environment_template'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),
        STRATA_MEASUREMENT_ROOT=str(ROOT),LD_PRELOAD=preload['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=preload['loader']['path'],STRATA_TRACE='1',STRATA_SNAPSHOT_VERIFY='1',
        STRATA_SYCL_CACHE_LOAN_AUDIT='1',STRATA_DECODE_TIMING='1',STRATA_SYCL_MTP_OBSERVER='host',
        STRATA_SYCL_MTP_OBSERVER_REQUEST='4',STRATA_SYCL_MTP_OBSERVER_SKIP='0',STRATA_SYCL_MTP_OBSERVER_WINDOWS='2',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/'model/validation.log'))
    env.pop('UR_LOG_TRACING',None)
    require('STRATA_SYCL_PREFILL_SEPARATE_LAYOUT' not in env and env['STRATA_PREFILL_RELEASE_MTP_EXPERTS']=='1', 'original prefill/release required')
    for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE','SYCL_CACHE_PERSISTENT'):
        require(env[name] == original['environment'][name], 'runtime safety changed: '+name)
    vocab = set(json.loads((Path(base['config']['tokenizer'])/'vocab.json').read_text()).values())
    require(vocab == set(range(len(vocab))) and set(range(4096)) <= vocab, 'actual vocabulary differs')
    cases = requests()
    row = dict(stage='prepared',source_commit=revision,source_audit=source,baseline=proof,binary=original['binary'],
        fixture=original['fixtures']['sycl_submission_test'],benchmark_files=len(pins),benchmark_fingerprint=fingerprint(pins),
        model_assets=base['models'],model_reference=base['reference'],environment=env,preload=preload,
        references={str(p):digest(p) for p in (RESULT.with_name('phase-prefill-speed-current.json'),RESULT.with_name('client-resources-current.json'))},
        command=[original['binary']['path'],'--serve',*args],requests=[dict(prompt_tokens=len(ids),max_new=16,
            wire_bytes=len(protocol.wire(ids,16)),wire_sha256=hashlib.sha256(protocol.wire(ids,16)).hexdigest()) for ids in cases],
        vocab_size=len(vocab),eos_ids=[248044,248046],
        driver=selected.guards.driver_guard(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        criteria='Original6e40 only,no rebuild. Commit each phase before launch. First host controls and same-build10-case submission fixture with bounded fullUR trace/preexec validation. Only then one fresh serve process,spec4/mtp-max-t2,original8192 prefill/1256slots/resident131072/caches0/suffix0. Four synthetic GEN16 inputs33/257/16385/33. Each fresh prompt covers n-1 cells in exact8192 chunks,token fallback below64,mode2;first verifierT1,thenT2,position/accepted/output/offer conservation. Third request must detach/restore MTP experts and check long GPU BF16/mode10/cache payload/map/48dense restoration;fourth must exercise2 host-observed MTP round replays(T1,T2),zero chain calls and valid accounting after that handoff. Accept valid EOS1..16 except require at least2 target windows per request and observer4 complete. Normal exit,empty validation warnings,source/assets/runtime/boot/driver unchanged,no fresh fault,all15 full hashes afterward. Stop first failure;no retry or text-identity/speed/task/full-context/adoption claim.',
        budgets=dict(wall=900,cpu=5400,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=128<<20),
        device_budgets=dict(wall=90,cpu=120,rss_cap=2<<30,text_cap=8<<20,file_cap=8<<20,total_cap=64<<20),
        host_budgets=dict(wall=60,cpu=60,rss_cap=4<<30,text_cap=2<<20,file_cap=2<<20,total_cap=8<<20),
        retention='Keep exact source/configuration,individual request work/count/hash and fault/owner outcomes. Retire normal trace/model logs after replay. Source snapshot remains only through this source/route check;original/private13 assets remain for imminent width comparison,not default adoption.',
        outcomes={},active=False,ownership_closed=True,passed=False,complete=False,production_selected=False,speed_qualified=False,model_adoption_qualified=False)
    row['private_inputs_sha256'] = seal(row)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache','device','model','host'): (ROOT/name).mkdir()
    write(ROOT/'record.json',row);write(RESULT,compact(row))
    print('Prepared original-binary fixed2 source/device/model gates;commit before execution.',flush=True)


def admission(row):
    require(RESULT.read_bytes()==gen.git('show','HEAD:'+str(RESULT.relative_to(P))) and
        json.loads(RESULT.read_text())==compact(row) and seal(row)==row['private_inputs_sha256'], 'committed fixed2 plan differs')
    require(source_check()==row['source_audit'] and selected.baseline()[1]==row['baseline'], 'original source/assets changed')
    pins=gen.benchmark_pins(row['source_commit'])
    require(len(pins)==row['benchmark_files'] and fingerprint(pins)==row['benchmark_fingerprint'], 'controller inputs changed')
    require(protocol.model.setup()['reference']==row['model_reference'] and validation.preload_qualification()==row['preload'], 'model/preload reference changed')
    for p,sha in row['references'].items():require(digest(p)==sha,'comparison reference changed: '+p)
    gen.runtime_assets_unchanged(validation.committed(RESULT.with_name('client-resources-current.json')))
    protocol.model.model_pins(protocol.model.setup()['config'],row['model_assets'])
    require(selected.guards.driver_guard()==row['driver'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'platform changed')


def routes(text,samples):
    headers=list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$',text,re.M))
    require(len(headers)==len(samples)==4,'four native request footers required')
    out=[];previous=0
    for i,(header,s) in enumerate(zip(headers,samples)):
        require(tuple(map(int,header.groups()))==tuple(s[k]for k in ('prompt_tokens','reused_tokens','read_tokens','generated_tokens')) and s['reused_tokens']==0,'request footer differs')
        segment=text[previous:header.start()];cells=s['read_tokens']-1
        chunks=[8192]*(cells//8192)+([cells%8192]if cells%8192 else[])
        require(gen.draft_rows(segment)==[dict(path='token'if n<64 else'batched',mode=2,cells=n)for n in chunks],'prompt callback path/cells differ')
        windows=[tuple(map(int,v))for v in re.findall(r'^strata trace: window (\d+) (\d+)$',segment,re.M)]
        require(len(windows)>=2 and windows[0]==(cells,1) and all(t==2 for _,t in windows[1:]) and
            all(1<=b[0]-a[0]<=a[1]for a,b in zip(windows,windows[1:])) and
            s['drafts_offered']==len(windows)-1,'actual fixed2 target route differs')
        accepted=sum(b[0]-a[0]-1 for a,b in zip(windows,windows[1:]))
        require(0<=s['drafts_accepted']-accepted<2 and 0<=len(windows)+s['drafts_accepted']-s['generated_tokens']<2,'accepted/output conservation differs')
        out.append(dict(target_windows=len(windows),target_rows=sum(t for _,t in windows),prompt_chunks=chunks))
        if i==2:
            import generation_compare as comparison
            route=re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=0 residual_staging=0 ',segment,re.M)
            require(len(route)==1 and tuple(map(int,route[0]))==(cells,cells*20480,cells,cells*20480),'long GPU BF16 residual route differs')
            require(re.search(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=16384 mixed_fp16_bytes=0$',segment,re.M),'MoE10 route differs')
            cycle,pub=comparison.bank.admission.loan_cycle(segment)
            dense=comparison.bank.admission.records(segment,'strata prefill dense',3)
            require([(v['phase'],v['residency'],v['activations'])for _,v in dense]==[('retired','Full','0'),('bank','Bank','0'),('restored','Full','48')] and pub[0][0]<dense[0][0]<dense[1][0]<dense[2][0]<pub[1][0],'dense/cache handoff differs')
            phases=re.findall(r'^strata mtp: prompt expert phase=(\S+) ',segment,re.M)
            require(phases==['before-detach','after-detach','after-prompt-before-restore','after-restore'] and len(re.findall(r'^strata mtp: experts restored bytes=707788800,',segment,re.M))==1,'MTP owner detach/restore differs')
            out[-1].update(GPU_BF16_rows=cells,MoE_mode=10,dense_layers_restored=48,cache_payload_and_map_match=True,MTP_experts_restored=True,loan_generation=cycle['generation'])
        previous=header.end()
    return out


def observer(text):
    def fields(raw):
        tokens = raw.split()
        require(all(t.count('=') == 1 for t in tokens), 'malformed observer field')
        values = dict(t.split('=',1) for t in tokens)
        require(len(values) == len(tokens), 'duplicate observer field')
        for key,value in values.items():
            if key.endswith('_s'):
                require(math.isfinite(float(value)) and float(value) >= 0, 'invalid observer clock')
        return values
    rows=re.findall(r'^strata MTP_OBSERVER (.*)$',text,re.M);require(len(rows)==1,'one bounded observer request required')
    values=fields(rows[0])
    expected=dict(v=1,request=4,profile=0,request_complete=1,sample_complete=1,accounting_valid=1,skip=0,limit=2,windows=2,completed=2,failed=0,replays=2,rows=3,T_mask=6,valid_graph_clocks=0,valid_queue_clocks=0,graph_envelope_ns=0,round_clocks=0,round_envelope_ns=0,chain_clocks=0,chain_envelope_ns=0,raw_retained=0,raw_omitted=0,observer_charged=1,scope='draft_entry_through_return',clocks='HOST_DISJOINT_DEVICE_NONADDITIVE',arithmetic_service='UNKNOWN',hardware_saturation='UNKNOWN')
    require(all(values.get(k)==str(v)for k,v in expected.items()) and int(values['seen'])>=2,'fixed2 proposal observer differs')
    phases=re.findall(r'^strata MTP_OBSERVER_PHASE (.*)$',text,re.M);require(len(phases)==8,'missing observer scopes')
    counts={};phase_rows=[]
    for raw in phases:
        v=fields(raw);key=(v['kind'],v['phase'])
        require(v['v']=='1' and v['request']=='4' and key not in counts and v['throws']=='0','observer identity/exception differs')
        counts[key]=int(v['calls'])
        require(counts[key]>0 or float(v['wall_s'])==float(v['thread_cpu_s'])==0, 'observer clock without calls')
        phase_rows.append(v)
    require(counts=={(k,p):(2 if k=='round' and p!='query' else 0)for k in ('round','chain')for p in ('capture','submit','drain','query')},'unexpected chain/capture/replay work')
    for total,other,key in [('wall_s','other_wall_s','wall_s'),('thread_cpu_s','other_cpu_s','thread_cpu_s')]:
        require(float(values[total])>0 and abs(float(values[total])-float(values[other])-
            sum(float(v[key])for v in phase_rows))<=1e-6, 'observer accounting does not close')
    require('strata MTP_GRAPH_CLOCK ' not in text, 'host observer leaked graph clock')
    return dict(request=4,proposal_windows=2,proposal_rows=3,T_mask=6,round_replays=2,chain_calls=0,accounting_valid=True)


def run(kind):
    row=json.loads((ROOT/'record.json').read_text());admission(row)
    require(kind in ('device','model') and not row['active'] and kind not in row['outcomes'] and
        (row['stage']=='prepared'if kind=='device'else row['stage']=='device-checked'and row['outcomes']['device']['passed']),'ordered fresh stage only')
    before=copy.deepcopy(row);outcome=dict(passed=False);owner=None
    raw=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(v.split(': ',1)[1]for v in raw.splitlines()if v.startswith('-- cursor: '))
    def persist():
        row['active']=bool(owner and owner.active is not None);write(ROOT/'record.json',dict(row,commands=owner.commands if owner else []))
    try:
        owner=selected.supervision.Owner(ROOT/kind/'owned',persist)
        if kind=='device':
            for label,name in [('protocol-tests','tools/sycl/test_mtp_resident_serve.py'),('route-tests','tools/sycl/test_mtp2_route.py')]:
                env={k:row['environment'][k]for k in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','PYTHONDONTWRITEBYTECODE')}
                entry,_,_=owner.run(label,['/usr/bin/python3','-B',name],env,P,**row['host_budgets'])
                require(selected.supervision.closed(entry)and entry['exit_code']==0,'host control failed: '+label)
            env=dict(row['environment'],UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',UR_LOG_TRACING='level:info;flush:info;output:stderr',UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/'device/validation.log'))
            entry,out,err=owner.run('submission',[row['fixture']['path'],'--submission'],env,ROOT,**row['device_budgets'])
            require(selected.supervision.closed(entry)and entry['exit_code']==0 and len(re.findall(r'^case ',out.read_text(),re.M))==10 and 'fixture_status=PASS mode=submission'in out.read_text(),'bounded submission check failed')
        else:
            def interact(io,entry):
                actual=dict(v.split(b'=',1)for v in Path('/proc',str(entry['pid']),'environ').read_bytes().split(b'\0')if b'='in v)
                require(all(actual.get(k.encode())==v.encode()for k,v in row['environment'].items()),'effective native environment differs')
                protocol.interact(io,row,outcome,2,request_ids=requests(),max_new=16)
            entry,out,err=owner.run('model',row['command'],row['environment'],ROOT,interaction=interact,**row['budgets'])
            require(selected.supervision.closed(entry)and entry['exit_code']==0,'native model exit/owner failed')
            text=err.read_text();outcome['routes']=routes(text,outcome['requests']);outcome['observer']=observer(text)
        outcome['activation']=validation.preload_evidence(err.read_text(),entry['pid'])
        warnings=ROOT/kind/'validation.log';require(not warnings.exists()or warnings.stat().st_size==0,'parameter validation warnings')
        admission(before)
        if kind=='model':protocol.model.model_pins(protocol.model.setup()['config'],row['model_assets'],full=True);outcome['final_full_model_hashes']=15
        outcome.update(passed=True,host_probe_accepted=True,exit_code=0,stdout_sha256=digest(out),stderr_sha256=digest(err))
    except BaseException as error:outcome['error']=repr(error)
    finally:
        if owner is not None:validation.finish(outcome,owner,cursor,row['boot'])
        else:outcome['ownership_closed']=True
        outcome['passed']=bool(outcome.get('passed')and outcome.get('driver_after')==row['driver'])
        row['outcomes'][kind]=outcome;row['ownership_closed']=outcome['ownership_closed']
        row['stage']=('closed'if kind=='model'else'device-checked')if outcome['passed']else'failed'
        row['passed']=row['complete']=bool(kind=='model'and outcome['passed']);persist();write(RESULT,compact(row))
    print(json.dumps({k:outcome.get(k)for k in ('passed','ownership_closed','exit_code','error')}),flush=True)
    return 0 if outcome['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:]in(['prepare'],['device'],['model']),'prepare|device|model')
    with (B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);selected.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:sys.exit(run(sys.argv[1]))
