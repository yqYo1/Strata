#!/usr/bin/env python3
"""Replay a closed failed diagnostic; never launch or relabel an engine run."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import model_aggregate as model

P, B, RESULT, ROOT = model.P, model.B, model.RESULT, model.ROOT
KEY = 'native_model_aggregate_v2_reanalysis'
FAILURE_REV = '292bbc441'
guards, gen, assets = model.guards, model.gen, model.assets


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def original(container):
    raw = gen.git('show', FAILURE_REV+':'+str(RESULT.relative_to(P)))
    record = json.loads(raw)[model.KEY]
    guards.require(container[model.KEY] == record and record['stage'] == 'closed'
        and not record['active'] and not record['passed'] and not record['complete']
        and record['ownership_closed'] and record['session_removed']
        and record['fresh_faults'] == [] and record['boot_unchanged']
        and record['error'] == 'ValueError(\'invalid device table row: "zeCommandListAppendBarrier"\')',
        'preserve the exact closed original parser failure')
    full, proof, setup = model.resource.qualification()
    guards.require(record['binary'] == full['binary'] and record['qualification'] == proof
        and model.profile_assets(container) == record['profile_assets']
        and model.environment(setup) == record['environment']
        and model.resource.modules()[1].recipe.config(setup, ROOT) == record['config'],
        'original native/qualification/configuration changed')
    pins = gen.benchmark_pins(record['source_commit'], match_worktree=False)
    guards.require(len(pins) == record['source_count']
        and assets.fingerprint(pins) == record['source_fingerprint'], 'original inputs unrecoverable')
    differences = [name for name, sha in pins.items() if gen.common.digest(P/name) != sha]
    guards.require(differences == ['tools/sycl/profile_aggregate.py'],
        'only the separately committed parser correction may differ')
    for path, sha in record['generated_inputs'].items():
        guards.require(gen.common.digest(path) == sha, 'original generated input changed')
    gen.runtime_assets_unchanged(record)
    return record, setup, dict(commit=gen.git('rev-parse',FAILURE_REV).decode().strip(),
        path=str(RESULT.relative_to(P)), sha256=hashlib.sha256(raw).hexdigest())


def raw_pins():
    # The closed evidence readers consume these normal logs/receipts. Disposable
    # JIT/cache/temp/trace directories are not inputs to the offline analysis.
    return {str(p): gen.common.digest(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p.relative_to(ROOT).parts[0] not in ('cache','tmp','trace')
        and p.name not in ('reanalysis-plan.json','reanalysis.json')}


def prepare(container):
    guards.require(KEY not in container and not (ROOT/'reanalysis-plan.json').exists(), 'fresh analysis plan only')
    _, _, reference = original(container)
    revision = gen.git('rev-parse','HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    record = dict(stage='prepared', source_commit=revision, source_count=len(pins),
        source_fingerprint=assets.fingerprint(pins), original_failure=reference, raw_inputs=raw_pins(),
        question='Recover complete service evidence from the normally exited v2 diagnostic; original run remains failed.',
        criteria='No inference execution. Recover original controller/native/recipe/artifact identities; allow only the committed parser difference. Verify frozen closed raw inputs, original exit/ownership/session/fault evidence, official thinking32K+ request,route/work/dispatch/clocks,validation and loader order,full table conservation. Barrier minimum may be0 while its total/average and all compute intervals stay positive. Recheck all15 full model hashes afterward; distinguish this later check from original run completion.',
        limits='Whole-process summed service includes load,warmups,prefill,decode; no phase critical path,physical traffic,saturation,observer-overhead,clean speed oradoption claim.',
        retention='Compact request samples,route/work checks,total/calls andtop families; full table outside Git only through source attribution/replay,then retire normal logs.',
        passed=False,complete=False,original_run_passed=False,model_adoption_qualified=False)
    write(ROOT/'reanalysis-plan.json', record);container[KEY]=record;write(RESULT,container)
    print('Prepared closed-data replay; commit exact inputs and plan before analysis.',flush=True)


def table_families(table):
    families = {}
    for row in table['rows']:
        name = re.sub(r'\[SIMD(?:_ANY|\d+) \{\d+; \d+; \d+\} \{\d+; \d+; \d+\}\](?="?$)', '', row['name'])
        if name.startswith('"zeCommandListAppendMemory'):
            name = re.sub(r'\[\d+\](?="?$)', '', name)
        group = families.setdefault(name,dict(name=name,calls=0,total_ns=0,shape_rows=0))
        for key in ('calls','total_ns'):group[key]+=row[key]
        group['shape_rows']+=1
    rows=sorted(families.values(),key=lambda r:r['total_ns'],reverse=True)
    guards.require(sum(r['total_ns'] for r in rows)==table['total_device_ns'], 'family aggregation lost time')
    return rows


def analyze(container):
    record=container[KEY]
    guards.require(record['stage']=='prepared' and record==json.loads((ROOT/'reanalysis-plan.json').read_text()), 'frozen analysis required')
    pins=gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins)==record['source_count'] and assets.fingerprint(pins)==record['source_fingerprint']
        and raw_pins()==record['raw_inputs'], 'analysis source/raw inputs changed')
    run,setup,reference=original(container)
    guards.require(reference==record['original_failure'], 'original failure archive changed')
    owned=json.loads((ROOT/'record.json').read_text())['owned_commands']
    entry,=[r for r in owned if r['label']=='bench']
    boot=json.loads((ROOT/'engine-start.json').read_text())['boot_id']
    evidence=model.closed_evidence(run,setup,entry,boot)
    table=evidence.pop('table');families=table_families(table)
    models=assets.model_pins(run['config'],run['model_assets'],full=True)
    guards.require(models==run['model_assets'] and raw_pins()==record['raw_inputs'], 'model assets/raw input changed during replay')
    full=dict(evidence,table=table,families=families)
    write(ROOT/'reanalysis.json',full)
    record.update(stage='closed',passed=True,complete=True,evidence=evidence,
        posthoc_full_model_hashes_verified=len(models),original_failure_preserved=True,
        table_summary=dict(shape_rows=len(table['rows']),family_count=len(families),
            calls=sum(r['calls'] for r in table['rows']),total_device_ns=table['total_device_ns'],
            zero_minimum_rows=[r for r in table['rows'] if r['min_ns']==0],
            top_families=[dict(name_prefix=r['name'][:240],name_sha256=hashlib.sha256(r['name'].encode()).hexdigest(),
                **{k:r[k] for k in ('calls','total_ns','shape_rows')}) for r in families[:12]]),
        full_analysis=dict(path=str(ROOT/'reanalysis.json'),sha256=gen.common.digest(ROOT/'reanalysis.json')))
    container[KEY]=record;write(RESULT,container)
    print(json.dumps(dict(passed=True,original_run_passed=False,shape_rows=len(table['rows']),
        family_count=len(families),total_device_ns=table['total_device_ns'],full_model_hashes=len(models))),flush=True)


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        container=gen.committed(RESULT)
        return prepare(container) if action=='prepare' else analyze(container)


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','analyze'), 'use prepare|analyze')
    main(sys.argv[1])
