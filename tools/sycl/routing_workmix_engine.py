#!/usr/bin/env python3
"""Main-owned framed route collection on the unchanged qualified6e40 engine."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cpu_work_census as census
import generation_engine as gen
import resource_engine as resources
import routing_workmix as routing

assets,restored,comparison = resources.modules()
guards,supervision = resources.guards,resources.supervision
P,B,A = gen.P,gen.B,gen.A
ROOT,RESULT = B/'routing-workmix-v1',A/'routing-workmix-current.json'
NATIVE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
SOURCES = ('src/program/generate.cpp','src/core/verify.cpp','src/core/expert_source.cpp')


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def plan():
    return dict(question='Can exact per-expert verifier-window/NT frequencies identify static cache alternatives that reduce logical CPU work,unlike the rejected aggregate-hit/global-rank selection?',
        source_contract='Exact6e40 serve path only. generate.cpp: drive_pool_multi writes synchronous layer,k,IDs,zero placeholder weights after each dispatch;only this thread writes the FILE. STRATA_TRACE explicitly frames request,prompt-done and each window(position,T). short-read0/prompt-cache0 use batched prefill,which has no Drive routing callback. Verifier::record_window uses G1 with --no-spec-split;run iterates layers0..47 once in order,calling pool(T rows) synchronously,then drains before return. Target loop emits window trace immediately before that call;firstT1,remainingT4 without suffix/min-p/adaptation. MTP has no Drive writer. Request DONE fflush precedes routing fflush and native stderr footer;wait for that footer before gate offsets. Wrapper permits one native exec. This combines explicit external frames and audited source ordering;the route file or matching aggregate totals alone are insufficient.',
        execution='No build or inference selection change. One independently owned official bench.sh thinking regular-code subset,runs1,4 warmups then >=32768 fresh input/>=1024 output,131072 context,8192 prefill,spec4,1256 per-layer slots,5workers,pcie0/adapt0/caches0. Immutable existing diagnostic environment;add only --dump-routing. No passive collector or runtime mutation.',
        admission='Recover9853 native inputs/recipe and exact13 artifacts,18 safety groups,committed controller/parser/tests and generated config/profile. Pin all benchmark sources,three audited native files,qualified runtime and model identities. At begin gate reconcile4 complete warmups: explicit request/prompt/window frames,exact callback byte count,all seven format/CPU/VRAM/job/NT counters and static restored map. Record quiescent routing offset and prefix hash before authorizing long request. Refuse all missing/reordered/truncated/extra frames or bytes. Existing small traced packed-bank success at71d0b6e63 is same-native continuity,not new trace qualification;warmup gate exercises new file writer before long dispatch.',
        decision='Collection qualifies only with identical recovered sources/effective native environment,complete official journal and substantive answer,strong observedMTP>=.60,owned normal native exit,all five request frame/provider/NT reconciliations,unchanged begin prefix/exact final offset,GPU BF16/mode10/private bank/48-layer restoration,post-process15 full model hashes and no fresh GPU fault. No speed gate,gain,hardware-saturation or adoption claim. Any failure/budget stop is incomplete;no retry.',
        budgets='1800s wall,54000CPU-s,96GiB RSS,128MiB total/per-file output. Trace at the existing3611-token reference is17.5MB;128MiB allows longer valid output plus diagnostic text. Stop rather than silently truncate if it grows beyond budget.',
        retention='Keep one individual request,compact frame/census/correctness checks and source identity. Keep only the393236-byte versioned per-expert NT1..4 count matrix outside Git for the immediate static-cache offline ranking decision;review/retire after that decision. Successful raw routing,model logs,JIT and one-use controller remain only through evidence replay,then retire. Logical counts are not CPU service or physical RAM/PCIe bytes.')


def pins(revision):
    values=gen.benchmark_pins(revision)
    for name in (*SOURCES,'data/expert-profile.bin','tests/core/test_routing_workmix.py'):
        raw=gen.git('show',revision+':'+name)
        guards.require((P/name).read_bytes()==raw,'uncommitted routing input: '+name)
        values[name]=hashlib.sha256(raw).hexdigest()
    return values


def config(setup):
    value=restored.recipe.config(setup,ROOT)
    args=value['args']
    for flag,expected in (('--spec','4'),('--short-read','0'),('--suffix-draft','0'),('--adapt-swaps','0'),
            ('--pcie-frac','0'),('--prompt-cache','0'),('--conversation-cache-mib','0'),('--expert-cache','1256')):
        guards.require(args.count(flag)==1 and args[args.index(flag)+1]==expected,'unsupported routing configuration: '+flag)
    guards.require('--expert-cache-per-layer' in args and '--no-spec-split' in args and
        not any(flag in args for flag in ('--dump-routing','--spec-min-p','--mtp-max-t','--layer-split','--peer-device',
            '--expert-cache-remote','--no-split-rows','--no-pool')), 'unsupported routing provider/window policy')
    args+=['--dump-routing',str(ROOT/'routing.bin')]
    return value


def current(record,full_models=False):
    full,proof,setup=resources.qualification()
    guards.require(full['source_commit']==NATIVE and record['binary']==full['binary'] and
        record['qualification']==proof and record['plan']==plan(),'native/qualification/plan changed')
    original={name:hashlib.sha256(gen.git('show',NATIVE+':'+name)).hexdigest() for name in SOURCES}
    guards.require(record['native_frame_source_sha256']==original and
        all(gen.common.digest(P/name)==sha for name,sha in original.items()),'audited callback sources changed')
    actual=pins(record['source_commit'])
    guards.require(len(actual)==record['source_count'] and assets.fingerprint(actual)==record['source_fingerprint'] and
        gen.common.digest(gen.SETTINGS)==record['settings_sha256'],'routing controller inputs changed')
    for path,sha in record['generated_inputs'].items():
        guards.require(gen.common.digest(path)==sha,'generated routing input changed: '+path)
    guards.require(record['process_plan']['config']==config(setup) and
        record['process_plan']['environment']==restored.environment(setup,ROOT,'0'),'routing configuration changed')
    guards.require(assets.machine_identity()==record['machine_identity'] and
        assets.model_pins(setup['config'],record['model_assets'],full=full_models)==record['model_assets'],'models/platform changed')
    gen.runtime_assets_unchanged(record)
    return full,setup


def prepare():
    guards.require(not ROOT.exists(),'single-use route root;never retry')
    prior=gen.committed(A/'cache-placement-current.json')
    guards.require(prior['stage']=='closed' and prior['passed'] and prior['artifact_retirement']['complete'] and
        not prior['decision']['advance_to_new_clean_plan'] and prior['ownership_closed'],'closed rejected placement screen required')
    full,proof,setup=resources.qualification();source=gen.git('rev-parse','HEAD').decode().strip()
    guards.require(full['source_commit']==NATIVE,'exact audited native required')
    inputs=pins(source);models=assets.model_pins(setup['config'],setup['model_assets'],full=True)
    ROOT.mkdir(mode=0o700)
    for part in ('tmp','cache'):(ROOT/part).mkdir()
    (ROOT/'expert-profile.bin').write_bytes((P/'data/expert-profile.bin').read_bytes())
    cfg,env=config(setup),restored.environment(setup,ROOT,'0')
    write(ROOT/'config.json',cfg);write(ROOT/'plan.json',dict(binary=full['binary'],native_validation_probe=None))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    process=dict(root=str(ROOT),config=cfg,environment=env,command=['/bin/sh',str(P/'bench.sh'),'--config',str(ROOT/'config.json'),
        '--port',str(port),'--long-generation-only','--long-generation-workload','regular-code','--think','--runs','1',
        '--profile-request-gates','--workload-id',comparison.WORKLOAD_ID,'--fail-fast-engine-error','--ready-timeout','600','--out',str(ROOT/'bench')])
    record=dict(schema='strata-framed-routing-workmix-v1',mode='engine-compare',stage='prepared',source_commit=source,
        build_source_commit=full['source_commit'],binary=full['binary'],qualification=proof,plan=plan(),process_plan=process,
        native_frame_source_sha256={n:inputs[n] for n in SOURCES},source_count=len(inputs),source_fingerprint=assets.fingerprint(inputs),
        settings_sha256=gen.common.digest(gen.SETTINGS),generated_inputs={str(ROOT/n):gen.common.digest(ROOT/n)
            for n in ('config.json','plan.json','expert-profile.bin')},model_assets=models,machine_identity=assets.machine_identity(),
        runtime_asset_delta=setup['runtime_asset_delta'],native_validation_probe=None,active=False,ownership_closed=True,
        complete=False,passed=False,clean_speed_qualified=False,adoption_qualified=False)
    current(record);write(ROOT/'frozen-plan.json',record);write(RESULT,record)
    print('Prepared framed route collection;commit the exact plan before execution.',flush=True)


def observed(root,record,samples,keep_counts=False):
    text=(root/'engine.stderr').read_text(errors='replace')
    guards.require(not re.search(comparison.FAILURE,text),'native error in routing run')
    guards.require(re.findall(r'^strata generate: expert cache (\d+) slots,',text,re.M)==['1256'] and
        'R4.2g PER-LAYER:' in text and 'pre-filled 1256 of 1256 slots from the profile; slot 0 verified' in text,
        'actual routing cache admission differs')
    assets.verify_pool(text,5);gen.request_evidence(text,samples)
    timing=gen.decode_timing_evidence(text,samples)
    bindings=[dict(prompt_tokens=s['prompt_tokens'],windows=t['windows'],rows=t['work']['target_rows'],
        cpu_jobs=t['work']['cpu_unique_jobs'],cpu_entries=t['work']['cpu_entries'],pool_ms=t['ms_per_window']['CPU_pool_run']*t['windows'])
        for s,t in zip(samples,timing)]
    args=record['process_plan']['config']['args']
    formats=census.manifest_formats(Path(args[args.index('--pack')+1])/'native_experts.txt')
    membership=census.expected_membership(root/'expert-profile.bin')
    work=census.extract(text,bindings,formats,membership)
    audits=gen.iq2_pair_evidence(text,samples,1,restored.recipe.component.base.machine()['physical_cpus'])
    for audit,w in zip(audits,work):
        guards.require(audit['row_ops']==640*w['cpu_entries'] and
            audit['nt1_rows']==640*sum(b['jobs_NT'][0] for b in w['format_bins'] if b['gu']==22),'IQ2/routing census mismatch')
    frames=routing.request_windows(text,samples);spans=[]
    with (root/'routing.bin').open('rb') as stream:
        for schedule,w in zip(frames,work):
            begin=stream.tell();counts,summary=routing.decode(stream,schedule,membership,formats,w)
            spans.append(dict(begin=begin,end=stream.tell(),**summary))
        guards.require(not stream.read(1),'unbound trailing routing bytes')
    return dict(requests=len(samples),frames=spans,final_offset=spans[-1]['end'],
        prefix_sha256=gen.common.digest(root/'routing.bin'),census=work[-1],iq2_dispatch=audits[-1]),counts if keep_counts else None


def run():
    record=gen.committed(RESULT);_,setup=current(record)
    guards.require(record==json.loads((ROOT/'frozen-plan.json').read_text()) and record['stage']=='prepared' and
        not (ROOT/'record.json').exists() and not (ROOT/'launch.json').exists(),'changed/closed/existing route plan')
    row=dict(passed=False,complete=False,active=False,ownership_closed=True);owner=None
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in before.splitlines() if s.startswith('-- cursor: '))
    def persist():
        row['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(row,commands=owner.commands if owner else []))
    def interact(io,leader):
        for stage in ('begin','end'):
            while True:
                line=io.line(1500)
                if line.startswith('STRATA_PROFILE_GATE '):break
            guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==
                dict(stage=stage,phase='long-generation/regular-code',run=0),'routing request gate differs')
            if stage=='begin':
                native=json.loads((ROOT/'engine-start.json').read_text())['native'];identity=supervision.identity(native['pid'])
                guards.require(identity and identity['start_ticks']==native['start_ticks'] and identity['sid']==leader['pid'] and
                    Path('/proc',str(native['pid']),'exe').resolve()==Path(record['binary']['path']).resolve(),'native routing owner/executable differs')
                env=dict(x.split(b'=',1) for x in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in x)
                guards.require(all(env.get(k.encode())==v.encode() for k,v in record['process_plan']['environment'].items()) and
                    b'LD_PRELOAD' not in env,'native routing environment differs')
                path,=(ROOT/'bench').glob('*/requests.jsonl');events=[json.loads(x) for x in path.read_text().splitlines()]
                guards.require(len(events)==10 and events[-1]['kind']=='start','routing warmup journal differs')
                gen.warmup_reconciliation(events[:-1],workloads=())
                samples=[e['sample'] for e in events[:-1] if e['kind']=='result']
                # DONE is flushed before routing/footer; the client gate can
                # arrive first. The footer proves that the FILE fflush ran.
                deadline=time.monotonic()+5
                while True:
                    text=(ROOT/'engine.stderr').read_text(errors='replace')
                    footers=len(re.findall(r'^strata serve: prompt .*$',text,re.M))
                    guards.require(footers<=4,'extra warmup footer before routing gate')
                    if footers==4:break
                    io.poll();guards.require(time.monotonic()<deadline,'routing warmup fflush/footer deadline')
                    time.sleep(.01)
                checked,_=observed(ROOT,record,samples)
                row.update(begin_checkpoint=checked,native_identity=native,effective_environment_verified=True);persist()
            io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();guards.require(time.monotonic()<deadline,'routing teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    owner=supervision.Owner(ROOT/'owned',persist)
    try:
        process=record['process_plan']
        entry,_,_=owner.run('bench',process['command'],process['environment'],P,interaction=interact,
            wall=1800,cpu=54000,rss_cap=96<<30,text_cap=128<<20,file_cap=128<<20,total_cap=128<<20)
        guards.require(supervision.closed(entry) and entry.get('exit_code')==0,'routing benchmark incomplete/failed')
        path,=(ROOT/'bench').glob('*/results.json')
        samples,warm,events=comparison.official(ROOT,json.loads(path.read_text()),('regular-code',))
        sample,=samples
        guards.require(sample['acceptance_rate']>=.60 and sample['request_sha256']==setup['request_sha256'],
            'fixed strong-MTP request required')
        native=gen.native_end(ROOT,record,process,boot,entry['pid'])
        guards.require(native['native']==row['native_identity'] and row['effective_environment_verified'],'native routing identity changed')
        all_samples=[e['sample'] for e in events if e['kind']=='result']
        checked,counts=observed(ROOT,record,all_samples,keep_counts=True)
        begin=row['begin_checkpoint'];offset=begin['final_offset']
        with (ROOT/'routing.bin').open('rb') as stream:
            prefix=hashlib.sha256(stream.read(offset)).hexdigest()
        guards.require(checked['frames'][:4]==begin['frames'] and prefix==begin['prefix_sha256'] and
            checked['frames'][4]['begin']==offset,'quiescent routing prefix/offset changed')
        text=(ROOT/'engine.stderr').read_text(errors='replace')
        routes=comparison.native_routes(text,events,1,('regular-code',))
        # Both paths use the original per-layer layout; check every refill cycle.
        footers=list(re.finditer(r'^strata serve: prompt .*$',text,re.M));previous=0
        for footer in footers:
            comparison.bank.admission.loan_cycle(text[previous:footer.start()]);previous=footer.end()
        current(record,full_models=True)
        count_path=ROOT/'expert-window-counts.bin';count_path.write_bytes(routing.encode_counts(counts))
        row.update(request=sample,warmups=warm,native_exit_code=native['exit_code'],observation=checked,routes=routes,
            per_expert_counts=dict(path=str(count_path),bytes=count_path.stat().st_size,sha256=gen.common.digest(count_path),
                consumer='Immediate offline static-cache ranking by unique verifier-window/NT work;not observed CPU service.',
                review_point='After next offline cache decision;retire if no eligible prospective comparison needs these actual counts.'),
            full_model_hashes_verified_after_process=15,passed=True,complete=True)
    except BaseException as error:
        row['error']=str(error)
        if (ROOT/'engine.stderr').exists():row['error_excerpt']=(ROOT/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row.update(ownership_closed=owner.active is None and all(supervision.closed(e) for e in owner.commands),
            fresh_faults=None,fault_probe_successful=False,boot_unchanged=False)
        try:
            after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            row.update(fresh_faults=[s for s in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)],
                fault_probe_successful=True,boot_unchanged=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            guards.driver_guard()
        except Exception as error:row['fault_probe_error']=str(error)
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row['fault_probe_successful'] and
            row['fresh_faults']==[] and row['boot_unchanged'] and 'fault_probe_error' not in row)
        persist();record.update(stage='closed',complete=row['complete'],passed=row['passed'],active=False,
            ownership_closed=row['ownership_closed'],process=row)
        write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ('passed','complete','ownership_closed','error')}),flush=True)
    return 0 if row['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.action=='prepare' else run()


if __name__=='__main__':sys.exit(main())
