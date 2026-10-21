#!/usr/bin/env python3
"""Same-native packed-bank117K clean comparison; Main owns every process."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

sys.dont_write_bytecode=True
import packed_bank as diagnostic

c, gen, assets, guards, supervision = diagnostic.evidence, diagnostic.gen, diagnostic.assets, diagnostic.guards, diagnostic.supervision
P, B, A = diagnostic.P, diagnostic.B, diagnostic.A
ROOT=B/'packed-expert-bank-clean-v1'
RESULT=A/'prefill-packed-bank-speed-current.json'
DROP={'STRATA_TRACE','UR_ENABLE_LAYERS','STRATA_SNAPSHOT_VERIFY',
    'STRATA_PREFILL_PHASE_AUDIT','STRATA_PREFILL_BOUNDARY_AUDIT',
    'STRATA_PREFILL_LAYER_MAJOR_PROFILE','STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE',
    'STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE','STRATA_PREFILL_CPU_WAIT_PROFILE',
    'STRATA_SYCL_PREFILL_COMMAND_PROFILE','STRATA_SYCL_QUERY_PROFILE',
    'STRATA_SYCL_CACHE_LOAN_AUDIT','STRATA_SYCL_VERIFY_PCIE_EARLY_AUDIT'}


def comparison_plan():
    plan=c.comparison_plan()
    plan.update(arms=['same-native-packed-off','same-native-packed-on'],
        classification='Isolated private expert-bank layout; identical native binary,Q8 MTP,cache/scratch placement and request bodies.',
        prerequisites='Both packed-bank117K diagnostic processes pass with conserved MoE work,fewer requested H2D bytes,closed ownership,full before/after model hashes and no fresh fault. Verify same-build engine,nine fixtures,original inputs and all runtime/model identities.',
        limits='Independent process pairs are repetition units. Four requests share each process; first-long and subsequent-long results remain separate. Short thinking output does not qualify decode,task quality,agent-session or physical262144/session/RESTORE. Requested bytes are not measured physical traffic. Neither screen adopts a model route.')
    return plan


def summarize(processes):
    result=c.summarize(processes)
    guards.require(all(len(r['work'])==4 and r['binary_sha256']==processes[0]['binary_sha256'] and
        r['build_source_commit']==processes[0]['build_source_commit'] for r in processes),
        'four work records and one native build required')
    for before,after in c.PAIRS:
        for left,right in zip(processes[before]['work'],processes[after]['work']):
            guards.require(left['logical_work']==right['logical_work'],'paired MoE work changed')
            a,b=left['expert_bank'],right['expert_bank']
            guards.require(a['uploads']+a['permanent_hits']==b['uploads']+b['permanent_hits'] and
                b['requested_H2D_bytes']<a['requested_H2D_bytes'],'paired transfer mechanism not conserved')
    result['limits']=comparison_plan()['limits']
    return result


def qualification():
    full,proof=diagnostic.qualification();record=gen.committed(diagnostic.RESULT)
    guards.require(record['complete'] and record['passed'] and record['stage']=='closed' and
        not record['active'] and record['ownership_closed'] and len(record['processes'])==2 and
        record['decision']['work_screen_passed'] and record['decision']['logical_lookups_conserved'] and
        record['decision']['additional_conserved_MoE_work'] and
        record['binary']==full['binary'] and record['qualification']==proof,
        'closed same-build paired diagnostic and conserved work required')
    for arm,row in enumerate(record['processes']):
        guards.require(row['arm']==arm and row['passed'] and row['complete'] and row['ownership_closed'] and
            not row['active'] and row['fresh_faults']==[] and row['boot_unchanged'] and
            row['native_exit_code']==0 and row['capacity']['checked_retirements']==96 and
            row['capacity']['exact_payload_and_map_cycles']==2 and
            row['geometry']['actual_private_layers']==(48 if arm else 0),'diagnostic arm is incomplete')
    return full,dict(native=proof,diagnostic_sha256=gen.common.digest(diagnostic.RESULT)),record


def environment(original,plan):
    env={k:v.replace(original['root'],plan['root']) for k,v in original['environment'].items()
        if k not in DROP and not k.startswith('UR_LOG_')}
    env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED=str(plan['arm']),STRATA_MTP_BATCH_F16='0')
    guards.require('LD_PRELOAD' not in env and not any(k in env for k in DROP),'observer environment leaked')
    settings=json.loads(gen.SETTINGS.read_text())['environment']
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[key]==settings[key],'qualified runtime safety changed')
    return env


def completed_work(text):
    (_,bank),=diagnostic.admission.records(text,'strata MoE expert bank complete',1)
    guards.require(set(bank)=={'uploads','requested_H2D_bytes','permanent_hits','permanent_slots','total_slots','final_drain'} and
        bank['final_drain']=='checked','expert bank did not complete')
    bank={k:diagnostic.admission.integer(bank,k) for k in bank if k!='final_drain'}
    guards.require(bank['uploads']>0 and bank['requested_H2D_bytes']>0 and
        bank['permanent_slots']<=383 and bank['total_slots']==384,'final bank extent changed')
    logical={}
    for name in ('GU','Down'):
        (_,row),=diagnostic.admission.records(text,'strata MoE '+name+' fusion',1)
        guards.require(set(row)=={'fused_jobs','fused_rows','separate_jobs','min_rows'} and row['min_rows']=='2334',
            'fusion geometry changed')
        logical[name]={k:diagnostic.admission.integer(row,k) for k in row}
    (_,row),=diagnostic.admission.records(text,'strata MoE accumulator window',1)
    guards.require(set(row)=={'windows','dequants','jobs','source_H2D_bytes','GPU_resident_windows_only','operator_max','maximum_rows'} and
        row['GPU_resident_windows_only']=='1' and row['operator_max']=='8192' and row['maximum_rows']=='81920',
        'actual accumulator route changed')
    logical['accumulator']={k:diagnostic.admission.integer(row,k) for k in row}
    acc=logical['accumulator']
    guards.require(acc['windows']==96 and acc['dequants']==bank['uploads']+bank['permanent_hits'] and
        acc['source_H2D_bytes']==225576960 and all(v['fused_jobs']+v['separate_jobs']==acc['jobs']
            for v in (logical['GU'],logical['Down'])),'completed logical work is inconsistent')
    return dict(expert_bank=bank,logical_work=logical)


def native_work(log,samples,pid,loan,admitted):
    c.native_evidence(log,samples,1,False,loan,pid,None)
    guards.require(not any(s in log for s in ('strata packed expert bank:','strata MoE planned job census:',
        'strata prefill cpu','strata prefill layer profile:','strata prefill QSA profile:',
        'strata native dense host profile:','strata MoE status observer:')),'clean run emitted diagnostic records')
    for prefix in ('strata MoE expert bank complete:','strata MoE GU fusion:',
                   'strata MoE Down fusion:','strata MoE accumulator window:'):
        guards.require(log.count(prefix)==len(samples),'completed work escaped request bounds')
    headers=list(re.finditer(r'^strata serve: prompt \d+ tokens = [^\n]+$',log,re.M))
    work=[completed_work(log[a.end():b.start()]) for a,b in zip(headers,headers[1:])]
    guards.require(work[0]['expert_bank']==admitted['geometry']['requested_work'] and
        samples[0]['request_sha256']==admitted['sample']['request_sha256'],
        'first request differs from same-arm admitted packed dispatch')
    return work


def current(record,full_models=False):
    full,proof,admitted=qualification()
    guards.require(full['binary']==record['binary'] and proof==record['qualification'] and
        record['plan']==comparison_plan(),'qualification or frozen comparison changed')
    pins=gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins)==record['source_count'] and assets.fingerprint(pins)==record['benchmark_fingerprint'],
        'benchmark inputs changed')
    original,ref=diagnostic.seed.original_plan()
    guards.require(ref==record['original_configuration'] and gen.common.digest(gen.SETTINGS)==record['settings_sha256'],
        'original setup changed')
    for plan in record['plans']:
        for name,sha in plan['generated_inputs'].items():guards.require(gen.common.digest(name)==sha,'generated input changed')
        guards.require(assets.fingerprint(environment(original,plan))==plan['environment_fingerprint'],'environment differs')
    guards.require(assets.machine_identity()==record['machine_identity'] and
        assets.model_pins(record['plans'][0]['config'],record['model_assets'],full=full_models)==record['model_assets'],
        'model/platform changed')
    gen.runtime_assets_unchanged(record)
    return original,admitted


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh clean comparison required')
    full,proof,admitted=qualification();original,ref=diagnostic.seed.original_plan()
    source=gen.git('rev-parse','HEAD').decode().strip();pins=gen.benchmark_pins(source)
    models=assets.model_pins(original['config'],admitted['model_assets'],full=True)
    ROOT.mkdir(mode=0o700);plans=[]
    for cohort,arm in enumerate(c.ORDER):
        root=ROOT/str(cohort);root.mkdir(mode=0o700)
        for name in ('cache','tmp'):(root/name).mkdir()
        config=copy.deepcopy(original['config'])
        config['args'][config['args'].index('--expert-profile')+1]=str(root/'expert-profile.bin')
        config.update(log=str(root/'engine.stderr'),env={})
        plan=dict(cohort=cohort,arm=arm,root=str(root),config=config)
        env=environment(original,plan)
        shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
        guards.require(gen.common.digest(root/'expert-profile.bin')==original['profile_sha256'],'profile differs')
        (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (root/'plan.json').write_text(json.dumps(dict(binary=full['binary']),indent=2)+'\n')
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan.update(environment_fingerprint=assets.fingerprint(env),
            generated_inputs={str(root/n):gen.common.digest(root/n) for n in ('config.json','plan.json','expert-profile.bin')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--prefill-only','--prefill-tokens','98304','--think','--runs','4','--workload-id',
                'native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')])
        plans.append(plan)
    record=dict(mode='engine-compare',stage='prepared',source_commit=source,build_source_commit=full['source_commit'],
        binary=full['binary'],native_validation_probe=None,qualification=proof,original_configuration=ref,
        plan=comparison_plan(),plans=plans,processes=[],runtime_asset_delta=original['runtime_asset_delta'],
        settings_sha256=gen.common.digest(gen.SETTINGS),model_assets=models,machine_identity=assets.machine_identity(),
        source_count=len(pins),benchmark_fingerprint=assets.fingerprint(pins),
        model_assets_fingerprint=assets.fingerprint(models),complete=False,passed=False,active=False,ownership_closed=True,
        adoption_qualified=False,sustained_decode_qualified=False,
        model_hash_policy='Full model digests at preparation and after final process; inode,size,mtime,ctime and inventory guards before/after every process.',
        retention='Keep individual first/subsequent samples,paired work/route/ownership/fault checks and decisions. Retire normal output/JIT after closure; no full API trace or tensors.',
        budgets=dict(wall=2400,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20))
    current(record);(ROOT/'plan.json').write_text(json.dumps(record,indent=2)+'\n')
    RESULT.write_text(json.dumps(record,indent=2)+'\n');print('Prepared six-process clean cohort; commit exact plan before run.')


def run():
    record=gen.committed(RESULT);original,admitted=current(record)
    frozen=json.loads((ROOT/'plan.json').read_text());mutable={'processes','stage','complete','passed','active','ownership_closed','decision'}
    guards.require({k:v for k,v in record.items() if k not in mutable}=={k:v for k,v in frozen.items() if k not in mutable} and
        not record['complete'] and not record['active'] and record['ownership_closed'] and
        all(r['passed'] for r in record['processes']),'closed or changed clean comparison')
    cohort=len(record['processes']);guards.require(cohort<6,'comparison closed')
    plan=record['plans'][cohort];root=Path(plan['root'])
    guards.require(not (root/'record.json').exists() and not (root/'launch.json').exists(),'existing owner; never retry')
    boot=c.read_boot();before=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in before.splitlines() if s.startswith('-- cursor: '))
    row=dict(cohort=cohort,arm=plan['arm'],passed=False,complete=False);owner=None
    def persist():
        row['active']=bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(row,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner=supervision.Owner(root/'owned',persist)
    try:
        entry,_,_=owner.run('bench',plan['command'],environment(original,plan),P,**record['budgets'])
        guards.require(supervision.closed(entry) and entry.get('exit_code')==0,'owned official benchmark failed')
        path,=(root/'bench').glob('*/results.json');samples,journal=c.official(root,json.loads(path.read_text()),4)
        row.update(requests=samples,journal=journal)
        native=diagnostic.native_end(root,record,plan,boot,entry['pid'])
        log=(root/'engine.stderr').read_text(errors='replace');assets.verify_pool(log,5)
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:',log),'native runtime failure')
        work=native_work(log,samples,native['native']['pid'],admitted['processes'][0]['capacity']['long_loan'],
            admitted['processes'][plan['arm']])
        current(record,full_models=cohort==5)
        row.update(passed=True,complete=True,native_identity=native['native'],native_exit_code=native['exit_code'],work=work,
            build_source_commit=record['build_source_commit'],binary_sha256=record['binary']['sha256'],
            machine_identity=record['machine_identity'],model_assets_fingerprint=record['model_assets_fingerprint'],
            benchmark_fingerprint=record['benchmark_fingerprint'],observer_records_absent=True,actual_route_checked=True)
    except BaseException as error:
        row['error']=str(error)
        if (root/'engine.stderr').exists():row['error_excerpt']=(root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row.update(ownership_closed=owner.active is None and all(supervision.closed(e) for e in owner.commands),
            fresh_faults=None,fault_probe_successful=False,boot_unchanged=False)
        try:
            after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            row.update(fresh_faults=[s for s in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)],fault_probe_successful=True)
        except Exception as error:row['fault_probe_error']=str(error)
        try:row.update(boot_unchanged=boot==c.read_boot(),driver_after=guards.driver_guard())
        except Exception as error:row['driver_error']=str(error)
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row['fault_probe_successful'] and
            row['fresh_faults']==[] and row['boot_unchanged'] and 'driver_error' not in row)
        persist();record['processes'].append(row)
        record.update(stage='collecting' if row['passed'] and cohort<5 else 'closed',
            ownership_closed=row['ownership_closed'],active=row['active'],
            complete=bool(cohort==5 and row['passed']),passed=bool(cohort==5 and row['passed']))
        if record['passed']:
            try:record['decision']=summarize(record['processes'])
            except Exception as error:record.update(passed=False,error='comparison eligibility: '+str(error))
        RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:row.get(k) for k in ('cohort','arm','passed','complete','ownership_closed','error')}))
    return 0 if row['passed'] and 'error' not in record else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.action=='prepare' else run()


if __name__=='__main__':sys.exit(main())
