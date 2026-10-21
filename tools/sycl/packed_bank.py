#!/usr/bin/env python3
"""Main-owned117K packed-bank work/admission check; not clean speed adoption."""
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

sys.dont_write_bytecode = True
import prefill_status as seed
import cache_long as admission
import cache_long_compare as evidence

q, gen, assets, guards, supervision = admission.q, admission.gen, admission.assets, admission.guards, admission.supervision
P, B, A = admission.P, admission.B, admission.A
ROOT = B/'packed-expert-bank-model-v1'
RESULT = A/'prefill-packed-bank-current.json'
BANK = A/'moe-expert-bank-current.json'
ORIGINAL_BUILD = 'b62a63744'


def qualification():
    full, count = q.current_build()
    bank = gen.committed(BANK)['packed_gpu']
    old_raw = gen.git('show', ORIGINAL_BUILD+':'+str(q.BUILD.relative_to(P)))
    old = json.loads(old_raw)
    guards.require(bank['passed'] and bank['complete'] and bank['ownership_closed'] and
        not bank['fresh_faults'] and bank['boot_unchanged'] and bank['mode']=='packed-validate' and
        len(bank['checks'])==2 and bank['build_source_commit']==old['source_commit'] and
        bank['fixture']==old['fixtures']['sycl_expert_bank_test'] and
        bank['provider_fixture']==old['fixtures']['file_expert_source_test'], 'passed original packed component required')
    guards.require(all(full[k]==old[k] for k in ('binary','fixtures','compiler','configure_argv','recipe_reference')) and
        [{k:s[k] for k in ('root','revision','count','aggregate_sha256')} for s in full['dependencies']]==old['inputs'][1:] and
        not gen.git('diff','--name-only',old['source_commit'],full['source_commit'],'--',
                    'src','include','cmake','CMakeLists.txt','third_party','bench/sycl'),
        'packed component native/recipe/fixture identity changed')
    return full, dict(original_inputs_recovered=count, source_build_sha256=gen.common.digest(q.BUILD_RECORD),
        shared_component_sha256=gen.common.digest(q.COMPONENT), packed_component_sha256=gen.common.digest(BANK),
        original_packed_build=dict(commit=ORIGINAL_BUILD,path=str(q.BUILD.relative_to(P)),sha256=hashlib.sha256(old_raw).hexdigest()),
        native_origin_source_commit=old['source_commit'], controller_only_native_reuse=True)


def geometry(log, arm):
    rows = admission.records(log,'strata packed expert bank',48 if arm else 0)
    census = {}
    for expected, (_, row) in enumerate(rows):
        guards.require(set(row)=={'layer','slots','stride','used_bytes','backing_bytes','full_fit'}, 'unexpected geometry fields')
        layer,slots,stride,used,backing,full = [admission.integer(row,k) for k in
            ('layer','slots','stride','used_bytes','backing_bytes','full_fit')]
        guards.require(layer==expected and stride>0 and stride%256==0 and backing==1022361600 and
            slots==min(512,backing//stride) and 0<slots<=512 and used==slots*stride<=backing and
            full==int(slots==512), 'actual private geometry differs from checked extent')
        census[str(stride)]=census.get(str(stride),0)+1
    complete, = admission.records(log,'strata MoE expert bank complete',1)
    row=complete[1]
    guards.require(not rows or rows[-1][0]<complete[0], 'layout appeared after final drain')
    guards.require(set(row)=={'uploads','requested_H2D_bytes','permanent_hits','permanent_slots','total_slots','final_drain'} and
        row['final_drain']=='checked', 'missing completed expert uploads')
    counters={k:admission.integer(row,k) for k in row if k!='final_drain'}
    guards.require(counters['uploads']>0 and counters['requested_H2D_bytes']>0 and
        counters['permanent_slots']<=counters['total_slots']<=512, 'invalid expert upload totals')
    if arm:
        guards.require(counters['total_slots']==int(rows[-1][1]['slots']), 'final private layout differs')
    else:
        guards.require(counters['total_slots']==384 and counters['permanent_slots']<=383, 'baseline bank changed')
    return dict(actual_private_layers=len(rows),stride_layers=census,requested_work=counters,
                physical_PCIe_traffic_measured=False)


def environment(original, plan):
    env={k:v.replace(original['root'],plan['root']) for k,v in original['environment'].items()}
    env.update(plan['environment_overrides'])
    return env


def current(record, full_models=False):
    full, proof = qualification()
    guards.require(full['binary']==record['binary'] and proof==record['qualification'], 'qualification changed')
    pins=gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins)==record['source_count'] and assets.fingerprint(pins)==record['benchmark_fingerprint'], 'benchmark inputs changed')
    original, ref=seed.original_plan()
    guards.require(ref==record['original_configuration'] and gen.common.digest(gen.SETTINGS)==record['settings_sha256'], 'original setup changed')
    for plan in record['plans']:
        for path,sha in plan['generated_inputs'].items(): guards.require(gen.common.digest(Path(path))==sha,'generated input changed: '+path)
        guards.require(assets.fingerprint(environment(original,plan))==plan['environment_fingerprint'],'environment differs')
    guards.require(assets.machine_identity()==record['machine_identity'] and
        assets.model_pins(record['plans'][0]['config'],record['model_assets'],full=full_models)==record['model_assets'], 'model/platform changed')
    q.runtime_assets_unchanged(record)
    return original


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh packed model plan required')
    full,proof=qualification();original,ref=seed.original_plan()
    source=gen.git('rev-parse','HEAD').decode().strip();pins=gen.benchmark_pins(source)
    probe=evidence.validation.preload_qualification()
    models=assets.model_pins(original['config'],original['model_assets'],full=True)
    ROOT.mkdir(mode=0o700);plans=[]
    for arm in (0,1):
        root=ROOT/str(arm);root.mkdir(mode=0o700)
        for name in ('tmp','cache'): (root/name).mkdir()
        config=copy.deepcopy(original['config'])
        config['args'][config['args'].index('--expert-profile')+1]=str(root/'expert-profile.bin')
        config.update(log=str(root/'engine.stderr'),env={});admission.require_batched_warmup(config)
        plan=dict(arm=arm,root=str(root),config=config,environment_overrides=dict(
            STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED=str(arm),STRATA_MTP_BATCH_F16='0',
            STRATA_SYCL_CACHE_LOAN_AUDIT='1',STRATA_SNAPSHOT_VERIFY='1'))
        env=environment(original,plan)
        settings=json.loads(gen.SETTINGS.read_text())['environment']
        for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
            guards.require(env[key]==settings[key],'qualified runtime safety changed')
        guards.require(env['UR_ENABLE_LAYERS']=='UR_LAYER_PARAMETER_VALIDATION' and env['STRATA_TRACE']=='1', 'diagnostic observers required')
        shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
        guards.require(gen.common.digest(root/'expert-profile.bin')==original['profile_sha256'],'profile differs')
        (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (root/'plan.json').write_text(json.dumps(dict(binary=full['binary'],native_validation_probe=probe),indent=2)+'\n')
        with socket.socket() as sock: sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan.update(environment_fingerprint=assets.fingerprint(env),generated_inputs={str(root/name):gen.common.digest(root/name)
            for name in ('config.json','plan.json','expert-profile.bin')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
            '--prefill-only','--prefill-tokens','98304','--think','--runs','1','--workload-id',
            'native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')])
        plans.append(plan)
    record=dict(mode='cache-capacity-check',diagnostic='packed-expert-bank117K',stage='prepared',source_commit=source,
        build_source_commit=full['source_commit'],binary=full['binary'],qualification=proof,
        original_configuration=ref,native_validation_probe=probe,plans=plans,processes=[],
        runtime_asset_delta=original['runtime_asset_delta'],settings_sha256=gen.common.digest(gen.SETTINGS),
        model_assets=models,machine_identity=assets.machine_identity(),source_count=len(pins),benchmark_fingerprint=assets.fingerprint(pins),
        complete=False,passed=False,active=False,ownership_closed=True,clean_speed_qualified=False,model_adoption_qualified=False,
        gate='Two independent processes,off then on; identical same-build Q8-MTP117489-input/24-output thinking workload. Require one short warmup,96 checked borrowed FFN retirements,two windows per48 layers,exact decode-cache/dense restoration,all-GPU BF16 rows,48 checked private layouts only when on,complete upload totals,native exit0 and no fresh fault. Stop after any failure; no retry. Work screen requires fewer requested H2D bytes with equal logical expert lookups; diagnostics cannot qualify speed.',
        limits='Short thinking output checks restoration only; no sustained decode,task quality,agent-session,physical262144/session/RESTORE or clean-speed qualification. Requested bytes are not wire traffic or saturation.',
        retention='Keep per-arm samples,compact route/work/fault/provenance and decision. Retire successful logs/JIT after closure; no tensors or full API history.',
        budgets=dict(wall=1800,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20))
    current(record,full_models=True)
    (ROOT/'plan.json').write_text(json.dumps(record,indent=2)+'\n');RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared off/on117K diagnostic; commit the exact plan before run.')


def run():
    record=gen.committed(RESULT);original=current(record,full_models=True)
    frozen=json.loads((ROOT/'plan.json').read_text())
    mutable={'processes','stage','complete','passed','active','ownership_closed','decision'}
    guards.require({k:v for k,v in record.items() if k not in mutable}=={k:v for k,v in frozen.items() if k not in mutable} and
        not record['complete'] and record['ownership_closed'] and not record['active'] and
        all(p['passed'] for p in record['processes']), 'closed or changed comparison plan')
    cohort=len(record['processes']);guards.require(cohort<2,'comparison closed')
    plan=record['plans'][cohort];root=Path(plan['root'])
    guards.require(not (root/'record.json').exists() and not (root/'launch.json').exists(),'inspect existing model owner; never restart')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
    cursor=next(s.split(': ',1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
    row=dict(arm=cohort,passed=False,complete=False)
    owner=None
    def persist():
        row['active']=bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(row,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner=supervision.Owner(root/'owned',persist)
    try:
        entry,_,_=owner.run('bench',plan['command'],environment(original,plan),P,**record['budgets'])
        guards.require(supervision.closed(entry) and entry.get('exit_code')==0,'owned official benchmark failed')
        path,=(root/'bench').glob('*/results.json')
        sample,journal_proof=admission.official_evidence(root,json.loads(path.read_text()))
        native=gen.native_end(root);log=(root/'engine.stderr').read_text(errors='replace');assets.verify_pool(log,5)
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:',log),'native runtime failure')
        headers=list(re.finditer(r'^strata serve: prompt \d+ tokens = [^\n]+$',log,re.M))
        guards.require(len(headers)==2,'warmup/long request binding missing')
        row.update(sample=sample,journal=journal_proof,native=native['native'],native_exit_code=native['exit_code'],
            validation_activation=evidence.activation(log,native['native']['pid'],record['native_validation_probe']),
            capacity=admission.mixed_evidence(log,sample),geometry=geometry(log[headers[0].end():headers[1].start()],cohort))
        current(record,full_models=True);row.update(passed=True,complete=True)
    except BaseException as error:
        row['error']=str(error)
        if (root/'engine.stderr').exists():row['error_excerpt']=(root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
        row['fresh_faults']=[s for s in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)]
        row['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try:row['driver_after']=guards.driver_guard()
        except Exception as error:row.update(driver_error=str(error),passed=False)
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row['boot_unchanged'] and not row['fresh_faults'])
        persist();record['processes'].append(row)
        record.update(ownership_closed=row['ownership_closed'],active=row['active'],stage='partial' if row['passed'] and cohort==0 else 'closed',
            complete=bool(row['complete'] and (cohort==1 or not row['passed'])),passed=bool(row['passed'] and cohort==1))
        if record['passed']:
            before,after=[r['geometry']['requested_work'] for r in record['processes']]
            same=before['uploads']+before['permanent_hits']==after['uploads']+after['permanent_hits']
            record['decision']=dict(logical_lookups_conserved=same,requested_H2D_saved_bytes=before['requested_H2D_bytes']-after['requested_H2D_bytes'],
                work_screen_passed=same and after['requested_H2D_bytes']<before['requested_H2D_bytes'],speed_qualified=False)
        RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:row.get(k) for k in ('arm','passed','complete','ownership_closed','error')}))
    return 0 if row['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.action=='prepare' else run()


if __name__=='__main__':sys.exit(main())
