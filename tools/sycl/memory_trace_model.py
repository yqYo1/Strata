#!/usr/bin/env python3
"""Main-only bounded NEO allocation logging qualification; never clean speed.

Main-authored Strata diagnostic, repository LICENSE. Reuses the selected
submission fixture and the independently qualified UR memory observer.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import memory_trace_device as device
host=device.host
model=host.model
P,B,RESULT=host.P,host.B,host.RESULT
PARENT_ROOT=B/'ur-memory-client-v1'
PARENT_KEY='ur_memory_client_v1'
ROOT=B/'neo-memory-fixture-v1'
KEY='neo_memory_fixture_v1'
REVIEW_KEY='neo_logging_source_review'
INPUTS=sorted(set([*device.INPUTS,'tools/sycl/memory_trace_model.py',
    'bench/sycl/client_resources.py','tests/core/test_client_resources.py']))
require,digest,write=host.require,host.digest,host.write
LOGGING_KEYS={'LogAllocationType','LogAllocationMemoryPool','LogAllocationStdout',
    'PrintBOCreateDestroyResult','PrintBOBindingResult','PrintBOPrefetchingResult'}


def pins(revision):
    result={}
    for name,raw in model.gate.native.build.blobs(P,revision,INPUTS):
        require(raw==(P/name).read_bytes(),'commit exact logging input: '+name)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def compact(r):
    return {k:v for k,v in r.items() if k not in ('environment','dependencies','commands')}


def parent(container):
    r=json.loads((PARENT_ROOT/'record.json').read_text())
    require(compact(r)==container[PARENT_KEY] and r['complete'] and r['passed'] and
        not r['active'] and r['ownership_closed'] and r['fresh_faults']==[] and
        r['full_model_hashes_matched']==15,'completed model/client observer required')
    for name,h in r['input_sha256'].items():
        raw=model.gate.native.build.git(P,'show',r['source_commit']+':'+name)
        require(hashlib.sha256(raw).hexdigest()==h,'original client source unrecoverable: '+name)
        if name!='tools/sycl/memory_trace_model.py':require(digest(P/name)==h,'native/observer source changed: '+name)
    require(all(digest(p)==h for p,h in r['dependencies'].items()) and
        digest(r['collector']['path'])==r['collector']['sha256'] and
        model.model.fingerprint(r['environment'])==r['environment_sha256'],
        'qualified collector/dependencies/environment changed')
    return r


def runtime(r):
    native=model.gate.engine();preload=model.gate.validation.preload_qualification()
    require(native['binary']==r['binary'] and native['source_commit']==r['native_source_commit'] and
        r['environment']['LD_PRELOAD']==preload['library']['path'] and
        r['environment']['STRATA_UR_PROBE_EXPECTED_LOADER']==preload['loader']['path'],
        'full native SDK/runtime/validation preload differs')
    return dict(binary=native['binary'],source_commit=native['source_commit']),preload


def prepare(container):
    require(KEY not in container and not ROOT.exists(),'fresh logging fixture only')
    review=container['owner_trace_obligations'][REVIEW_KEY]
    require(review['native_review_passed'] and review['logging_only'] and
        review['flag_map'] and set(review['flag_map'])<=LOGGING_KEYS and
        all(v=='1' for v in review['flag_map'].values()),'reviewed logging-only flags required')
    p=parent(container);native,preload=runtime(p)
    selected,proof=model.gate.native.source.baseline();fixture=selected['fixtures']['sycl_submission_test']
    require(digest(fixture['path'])==fixture['sha256'],'selected fixture changed')
    info=container['owner_trace_obligations']['post_collection_driver_identification']
    # Exact installed DSO identity is a new pre-exec pin, not retrospective
    # qualification of its source mapping or prior runtime teardown.
    driver=review['installed_driver']
    require(driver=={k:info[k] for k in ('path','sha256')},'source review driver identity differs')
    require(digest(driver['path'])==driver['sha256'],'observed installed driver changed')
    binary=Path(driver['path']).read_bytes()
    require(all(k.encode()+b'\0' in binary for k in review['flag_map']),'logging key absent in installed driver')
    revision=model.gate.native.build.git(P,'rev-parse','HEAD').decode().strip();source=pins(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    environment=dict(p['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),**review['flag_map'])
    changed={k for k in set(environment)|set(p['environment']) if environment.get(k)!=p['environment'].get(k)}
    require(changed<=set(review['flag_map'])|{'TMPDIR','XDG_CACHE_HOME'},'unplanned runtime environment change')
    r=dict(stage='prepared',source_commit=revision,input_sha256=source,
        parent_qualification_sha256=model.model.fingerprint(container[PARENT_KEY]),
        source_review_sha256=model.model.fingerprint(review),source_review=review,
        selected_original=proof,runtime_native=native,preload=preload,
        fixture=fixture,fixture_source_commit=selected['source_commit'],
        collector=p['collector'],dependencies=p['dependencies'],
        driver_library=driver,environment=environment,environment_sha256=model.model.fingerprint(environment),
        command=[fixture['path'],'--submission'],cwd=str(ROOT),
        driver_before=model.gate.native.guards.driver_guard(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        active=False,passed=False,complete=False,ownership_closed=True,GPU_executed=False,
        speed_eligible=False,physical_owner_attribution_qualified=False,ordinary_runtime_teardown_qualified=False,
        model_adoption_qualified=False,decode_performance_qualified=False,
        budgets=dict(wall_s=120,cpu_s=120,rss_bytes=4<<30,text_bytes=16<<20,file_bytes=16<<20,total_bytes=64<<20),
        criteria='One selected original10-check submission fixture,no model or engine rebuild. Preserve the complete immutable qualified safety/preload/UR observer environment;add only reviewed logging keys and private temporary/cache paths before exec. Pin original source/recipe,full SDK,runtime,collector239 dependencies,driver DSO and fixture. Require the original10 numerical/copy/event/profiling/lifetime checks,complete UR memory ledger,each predeclared minimum NEO log category,normal direct native exit0,closed ownership,unchanged inputs/boot/driver and no fresh fault. Capture stdout/stderr in private cwd under finite budgets. Stop first failure;no retry. NEO memory pool names describe creation decisions,not necessarily current KMD placement;UMD logs alone do not establish migration,physical residency,PCIe traffic or speed. Exit summary and ordinary teardown are not required or qualified.',
        retention='Main consumes small output to select one finite same-native model capture. Keep compact sample counts,representative output,identities and original outcome;retire raw logs/JIT after that plan or refusal. No extra engine generation.')
    write(ROOT/'record.json',r);container[KEY]=compact(r);write(RESULT,container)
    print('Prepared bounded selected-fixture NEO logging gate;commit before run.',flush=True)


def admission(r,container):
    require(RESULT.read_bytes()==model.gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        compact(r)==container[KEY] and pins(r['source_commit'])==r['input_sha256'],
        'commit exact logging capability plan')
    p=parent(container);native,preload=runtime(p)
    require(native==r['runtime_native'] and preload==r['preload'] and
        model.model.fingerprint(container[PARENT_KEY])==r['parent_qualification_sha256'] and
        model.model.fingerprint(container['owner_trace_obligations'][REVIEW_KEY])==r['source_review_sha256'],
        'native/preload/parent/source review qualification changed')
    require(model.model.fingerprint(r['environment'])==r['environment_sha256'] and
        all(digest(p)==h for p,h in r['dependencies'].items()),'logging environment/dependencies changed')
    for asset in (r['fixture'],r['collector'],r['driver_library']):
        require(digest(asset['path'])==asset['sha256'],'logging artifact changed')
    require(model.gate.native.source.baseline()[1]==r['selected_original'] and
        model.gate.native.guards.driver_guard()==r['driver_before'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot'],
        'selected baseline/driver/boot changed')


def logging_evidence(stdout,stderr,contract):
    require(contract['required_patterns'],'nonempty predeclared NEO output contract required')
    result={}
    required=contract['required_patterns'];optional=contract.get('optional_patterns',{})
    require(not set(required)&set(optional),'overlapping required/optional log categories')
    for category,pattern in {**required,**optional}.items():
        require(0<len(pattern)<=512 and '\n' not in pattern,'bounded log decoder pattern required')
        rows=[]
        for sink,text in (('stdout',stdout),('stderr',stderr)):
            for line in text.splitlines():
                if re.search(pattern,line):
                    require(len(line)<=4096,'NEO record exceeds bounded line size')
                    rows.append((sink,line))
        require(rows or category not in required,'required NEO output missing: '+category)
        result[category]=dict(records=len(rows),required=category in required,
            sinks=sorted({sink for sink,_ in rows}),representative=rows[0][1] if rows else None)
    return dict(categories=result,scope='Observed logging capability only;not complete BO identity/lifetime,UMD-to-BO pointer join,KMD migration or ordinary teardown.')


def run(container):
    r=json.loads((ROOT/'record.json').read_text());admission(r,container)
    require(r['stage']=='prepared','closed logging gate cannot be retried')
    original=json.loads(json.dumps(r));owner=None
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
    def persist():
        r['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(r,commands=owner.commands if owner else []))
    owner=model.gate.native.supervision.Owner(ROOT/'owned',persist)
    r.update(stage='running',GPU_executed=True);persist()
    try:
        b=r['budgets'];entry,out,err=owner.run('device',r['command'],r['environment'],Path(r['cwd']),
            wall=b['wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],
            file_cap=b['file_bytes'],total_cap=b['total_bytes'])
        r['exit_code']=entry['exit_code']
        require(model.gate.native.supervision.closed(entry) and entry['exit_code']==0,'normal owned direct native exit required')
        stdout,stderr=out.read_text(errors='replace'),err.read_text(errors='replace')
        r['device_evidence']=device.evidence(stderr,stdout,entry['pid'])
        r['logging_evidence']=logging_evidence(stdout,stderr,r['source_review']['output_contract'])
        admission(original,container);r['host_probe_accepted']=True
    except BaseException as error:r['error']=repr(error)
    finally:
        model.gate.validation.finish(r,owner,cursor,r['boot'])
        r['passed']=bool(r['passed'] and r.get('driver_after')==r['driver_before'])
        r['complete']=r['passed'];r['stage']='closed' if r['passed'] else 'failed';persist()
        container[KEY]=compact(r);write(RESULT,container)
    print(json.dumps({k:r.get(k) for k in ('passed','complete','ownership_closed','exit_code','logging_evidence','error','fresh_faults')}),flush=True)
    return 0 if r['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['run']),'prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model.gate.native.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare':prepare(container)
        else:sys.exit(run(container))
