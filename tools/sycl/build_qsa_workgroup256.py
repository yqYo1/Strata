#!/usr/bin/env python3
"""Fresh component-only WG128/WG256 build;Main-authored,repository LICENSE.

All project objects are fresh. Recover the qualified compiler/SDK/flags and
compile identical baseline/runtime sources;no old archive or object is linked.
This does not rebuild,select or qualify an inference engine.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode=True
import coherent_original_reuse as original
import measurement_owner as owner

P,B,RESULT=original.P,original.B,original.RESULT
ROOT,KEY=B/'qsa-prefill-workgroup256-build-v1','qsa_prefill_workgroup256_build_v1'
THIS='tools/sycl/build_qsa_workgroup256.py'
BASE_SOURCE='14508acbf1b70eb994fdf37f97fb6d0f5cff7d43'
BASE_ROOT=B/'coherent-prefill-native-v1'
SOURCES=['src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp',
    'src/kernels/sycl/prompt_attention.cpp','src/kernels/sycl/qsa_bf16_prefix.cpp',
    'src/kernels/sycl/qsa_prefill_workgroup256.cpp']
NEW=['include/strata/kernels/qsa_prefill_workgroup256.hpp',SOURCES[-1],
     'tests/sycl/qsa_prefill_workgroup256.cpp','tests/sycl/qsa_prefill_workgroup256_bench.cpp',
     'tests/sycl/qsa_prefill_workgroup256_admission.hpp']


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def blob(revision,name):
    return subprocess.check_output(['git','-C',str(P),'show',revision+':'+name],timeout=30)


def compact(row):
    return {k:v for k,v in row.items() if k not in ('sources','environment','SDK')}


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row,indent=2)+'\n')
    data=json.loads(RESULT.read_text());data[KEY]=compact(row)
    RESULT.write_text(json.dumps(data,indent=2)+'\n')


def prepare():
    data=json.loads(RESULT.read_text());require(KEY not in data and not ROOT.exists(),'fresh component build only')
    retirement=data['phase_kv_split_prefill_v1']['B1_build_retirement']
    require(retirement['stage']=='deleted_verified' and retirement['retained_ELFs']==26
        and not Path(retirement['root']).exists(),'retire ended B1before a bounded third generation')
    prior=data['qsa_prefill_workgroup64_build_v3']['retirement']
    require(prior['stage']=='deleted_verified' and prior['retained_engine_ELFs']==26
        and not any(Path(p).exists() for p in prior['paths']),
        'rejected WG64component retired before new coherent geometry')
    base,proof=original.verify_original()
    require(type(proof) is dict and proof['original_native_source']==BASE_SOURCE,
        'typed original native proof required')
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',source_commit,'--','include'],text=True).splitlines()
    names=sorted(set(names+SOURCES+NEW+[THIS,'tests/sycl/qsa_bf16_prefix.cpp',
        'tools/sycl/measurement_owner.py','tools/sycl/coherent_original_reuse.py',
        'tools/sycl/coherent_native_reuse.py','AGENTS.md','docs/ARTIFACT_RETENTION.md']))
    pins={name:hashlib.sha256(blob(source_commit,name)).hexdigest() for name in names}
    require(all(sha(P/n)==h for n,h in pins.items()),'commit exact component source before prepare')
    for name in SOURCES[:-1]:
        require((P/name).read_bytes()==blob(BASE_SOURCE,name),'keep baseline/runtime native bytes: '+name)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache','objects'):(ROOT/name).mkdir()
    database=json.loads((BASE_ROOT/'build/compile_commands.json').read_text())
    def command(source,target,output,remove=(),extra=()):
        candidates=[x for x in database if '/'+target+'.dir/' in x['command']]
        if source in SOURCES[:-1]:candidates=[x for x in candidates if x['file']==str(P/source)]
        require(len(candidates)==1,'one original compile command: '+source)
        argv=shlex.split(candidates[0]['command'])
        argv=[x for x in argv if x not in remove]
        old=argv.index('-o');argv[old+1]=str(output)
        argv[argv.index('-c')+1]=str(P/source)
        return argv+list(extra)+['-MD','-MF',str(output)+'.d']
    commands=[];objects=[]
    for i,name in enumerate(SOURCES):
        output=ROOT/'objects'/('native'+str(i)+'.o');objects.append(str(output))
        if name.startswith('src/sycl/'):target='strata_sycl_runtime'
        else:target='strata_kernels'
        # The new implementation uses exactly the old prefix's compile flags.
        if name==SOURCES[-1]:
            selected=[x for x in database if x['file']==str(P/SOURCES[-2]) and '/strata_kernels.dir/' in x['command']]
            require(len(selected)==1,'one baseline prefix command')
            argv=shlex.split(selected[0]['command']);argv[argv.index('-o')+1]=str(output)
            argv[argv.index('-c')+1]=str(P/name);argv+=['-MD','-MF',str(output)+'.d']
        else:argv=command(name,target,output)
        commands.append(dict(label='native'+str(i),argv=argv,object=str(output)))
    tests=[]
    for label,extra in [('oracle4',['-DSTRATA_QSA_TEST_PAGE_SIZE=4','-DSTRATA_QSA_REAL_HEAD_STATE=1']),
                        ('oracle16',['-DSTRATA_QSA_TEST_PAGE_SIZE=16'])]:
        output=ROOT/'objects'/(label+'.o')
        argv=command(NEW[2],'sycl_qsa_bf16_prefix_cached_subgroup_test',output,
            remove=('-DSTRATA_QSA_COMPARE_SUBGROUP=1','-Dqsa_bf16_prefix_batch=qsa_bf16_prefix_cached_subgroup_batch'),extra=extra)
        commands.append(dict(label=label,argv=argv,object=str(output)));tests.append((label,str(output)))
    output=ROOT/'objects/bench.o'
    argv=command(NEW[3],'sycl_qsa_bf16_prefix_cached_subgroup_bench',output,
        remove=('-DSTRATA_QSA_BENCH_CACHED_SUBGROUP=1',))
    commands.append(dict(label='bench',argv=argv,object=str(output)));tests.append(('bench',str(output)))
    raw=subprocess.check_output(['/usr/bin/ninja','-C',str(BASE_ROOT/'build'),'-t','commands',
        'sycl_qsa_bf16_prefix_cached_subgroup_bench'],text=True,timeout=30).splitlines()[-1]
    link=shlex.split(raw);require(link[:2]==[':','&&'] and link[-2:]==['&&',':'],'original simple link recipe')
    link=link[2:-2];old_object=next(x for x in link if x.endswith('.o'))
    require(len([x for x in link if x.endswith('.o')])==1,'one old fixture object')
    link=[x for x in link if not x.endswith('.a') and x!=old_object]
    for label,obj in tests:
        argv=list(link);argv[argv.index('-o')+1]=str(ROOT/label);argv.extend([*objects,obj])
        commands.append(dict(label='link-'+label,argv=argv,output=str(ROOT/label)))
    env={k:v.replace(str(BASE_ROOT),str(ROOT)) for k,v in base['recipe']['environment'].items()}
    env.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),PYTHONDONTWRITEBYTECODE='1')
    actual=json.loads(Path(base['actual_inputs']['path']).read_text())
    sdk={f:h for f,h in actual['compiled'].items() if not Path(f).is_relative_to(P)}
    sdk.update(actual['links']);sdk.update(base['toolchain'])
    require(all(sha(f)==h for f,h in sdk.items()),'qualified external compiler/SDK bytes changed')
    row=dict(stage='prepared',source_commit=source_commit,sources=pins,SDK=sdk,environment=env,
        input_sha256=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest(),
        SDK_sha256=hashlib.sha256(json.dumps(sdk,sort_keys=True).encode()).hexdigest(),
        environment_sha256=hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest(),
        baseline_native_source=BASE_SOURCE,baseline_recovery=proof,commands=commands,results=[],
        host_compile_only=True,GPU_executed=False,engine_built=False,production_selected=False,
        complete=False,passed=False,active=False,ownership_closed=True,
        budgets=dict(wall=300,cpu=1200,rss_cap=8<<30,text_cap=8<<20,file_cap=256<<20,total_cap=768<<20),
        plan='Fresh nine-object/three-fixture component generation after B1retirement. No project archive/object reuse. '
            'Original five baseline/runtime source bytes and compiler/SDK/flags stay exact;new kernel and fixtures '
            'are committed. Validate actual-MD dependency closure and all owned headers before accepting each compile. '
            'Link only fresh objects through the pinned SYCL driver;capture actual external link files. Stop first '
            'source/dependency/compiler/owner/budget failure;no retry or hash refresh. This builds no inference engine '
            'and advances no GPU/model/speed gate.',
        retention='Bounded third component generation,three ELFs,only through WG256numerical and matched timing decision;'
            'fast14508/12ELFs and namingA/14ELFs remain unchanged. Retire failed/rejected component after original source/recipe recovery.')
    save(row);print('Prepared fresh WG256component build;commit recipe before compile.',flush=True)


def current(row):
    require(json.loads(blob('HEAD',str(RESULT.relative_to(P))))[KEY]==compact(row)
        and RESULT.read_bytes()==blob('HEAD',str(RESULT.relative_to(P))),'commit exact current build plan')
    require(all(sha(P/n)==h for n,h in row['sources'].items())
        and all(sha(n)==h for n,h in row['SDK'].items()),'frozen build source/SDK changed')
    require(hashlib.sha256(json.dumps(row['environment'],sort_keys=True).encode()).hexdigest()==row['environment_sha256'],
        'frozen compiler environment changed')
    original.verify_original(expected=row['baseline_recovery'])


def dependencies(path,row):
    text=Path(path).read_text().replace('\\\n',' ')
    require(text.count(':')==1,'one complete compiler dependency rule')
    names=shlex.split(text.split(':',1)[1]);require(names,'nonempty actual compiler inputs')
    actual={}
    for name in names:
        p=Path(name).resolve();h=sha(p)
        wanted=row['sources'].get(str(p.relative_to(P))) if p.is_relative_to(P) else row['SDK'].get(str(p))
        require(wanted==h,'uncommitted/changed/uncovered actual compiler input: '+str(p))
        actual[str(p)]=h
    return actual


def compile_all():
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']=='prepared' and not row['results'],'one fresh build attempt')
    job=owner.Owner(ROOT/'owned');row['stage']='compiling';closure={};links={}
    try:
        for item in row['commands']:
            entry,out,err=job.run(item['label'],item['argv'],row['environment'],P,**row['budgets'])
            good=owner.closed(entry) and entry['exit_code']==0
            row['results'].append(dict(label=item['label'],passed=bool(good),exit_code=entry['exit_code'],
                ownership_closed=bool(owner.closed(entry))))
            require(good,'component build failed: '+item['label']+' '+err.read_text()[-4000:])
            if 'object' in item:closure.update(dependencies(item['object']+'.d',row))
            else:
                for line in out.read_text().splitlines():
                    if line.startswith('/') and Path(line).is_file():
                        p=Path(line).resolve();h=sha(p)
                        require(row['SDK'].get(str(p))==h or p.is_relative_to(ROOT/'objects'),
                            'unqualified actual external link input: '+str(p))
                        links[str(p)]=h
            print('Closed WG256build '+item['label'],flush=True)
        require(len(row['results'])==12 and len(closure)>100 and links,'complete fresh dependency/link evidence required')
        census=dict(compiler_inputs=closure,link_inputs=links)
        (ROOT/'actual-inputs.json').write_text(json.dumps(census,indent=2)+'\n')
        row.update(stage='built',passed=True,complete=True,
            actual_inputs=dict(path=str(ROOT/'actual-inputs.json'),sha256=sha(ROOT/'actual-inputs.json'),
                compiler_files=len(closure),external_and_object_link_files=len(links),uncovered=0),
            binaries={n:dict(path=str(ROOT/n),sha256=sha(ROOT/n),bytes=(ROOT/n).stat().st_size)
                for n in ('oracle4','oracle16','bench')})
    except BaseException as error:row.update(stage='failed',error=repr(error))
    finally:
        row['ownership_closed']=job.active is None and all(owner.closed(x) for x in job.commands)
        row['passed']=bool(row['passed'] and row['ownership_closed']);save(row)
    require(row['passed'],row.get('error','component build failed'))


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['compile']),'prepare|compile')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (prepare if sys.argv[1]=='prepare' else compile_all)()
