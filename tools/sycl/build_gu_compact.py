#!/usr/bin/env python3
"""Build a current-API compact-GU fixture from exact qualified native libraries.

Main-authored; repository LICENSE applies. No engine rebuild or GPU execution.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
import check_moe_census_model as device
import measurement_owner as owner

P,B,RESULT=device.P,device.B,device.RESULT
ROOT,KEY=B/'gu-compact-component-v1','gu_compact_component_v1'
THIS,CPP='tools/sycl/build_gu_compact.py','bench/sycl/gu_compact.cpp'
NATIVE=device.build.ROOT
require,sha=device.require,device.sha


def native(extra_observers=frozenset()):
    return device.native(observers={device.THIS,'tools/sycl/moe_host_census_model.py',
        THIS,CPP,*extra_observers})


def compact(row):
    return {k:v for k,v in row.items() if k not in ('execution','source_files')}


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row,indent=2)+'\n')
    public=json.loads(RESULT.read_text());public[KEY]=compact(row)
    RESULT.write_text(json.dumps(public,indent=2)+'\n')


def current(row,extra_observers=frozenset()):
    require(device.original.validation.committed(RESULT,(KEY,))==compact(row),'commit exact component plan')
    n=native(extra_observers)
    require(n['native_source_commit']==row['native_source_commit']
        and n['binaries']['strata']==row['engine'],'qualified engine changed')
    for name,h in row['source_files'].items():
        raw=device.original.common.source.git(P,'show',row['source_commit']+':'+name)
        require(hashlib.sha256(raw).hexdigest()==h==sha(P/name),'component source changed')
    require(sha(row['compile_seed']['path'])==row['compile_seed']['sha256'],'compiler recipe changed')
    require(all(sha(p)==h for p,h in row['static_libraries'].items()),'same-build library changed')
    return n


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh component only')
    n=native();model=device.original.validation.committed(RESULT,('moe_host_census_model_v1',))
    require(model['stage']=='closed' and model['passed'] and model['complete']
        and model['ownership_closed'] and not model['active']
        and model['native_build']['binaries']==n['binaries'],'closed exact model handoff required')
    build=NATIVE/'build';seed=build/'compile_commands.json';entries=json.loads(seed.read_text())
    entry,=[e for e in entries if 'sycl_verifier_retirement_test.dir' in e['command']]
    compile_argv=shlex.split(entry['command'])
    compile_argv=[x for x in compile_argv if not x.startswith('-DSTRATA_VERSION=')]
    compile_argv[compile_argv.index(entry['file'])]=str(P/CPP)
    compile_argv[compile_argv.index('-o')+1]=str(ROOT/'fixture.o')
    compile_argv+=['-fno-fast-math','-ffp-contract=off','-MD','-MF',str(ROOT/'fixture.d')]
    text=subprocess.check_output(['ninja','-C',str(build),'-t','commands',
        'sycl_verifier_retirement_test'],text=True).splitlines()[-1]
    require(text.startswith(': && ') and text.endswith(' && :'),'known exact link rule required')
    link=shlex.split(text[5:-5]);old=entry['output']
    require(link.count(old)==1,'one exact fixture object required')
    link[link.index(old)]=str(ROOT/'fixture.o');link[link.index('-o')+1]=str(ROOT/'fixture')
    libraries={str((build/x).resolve()):sha((build/x).resolve()) for x in link if x.endswith('.a')}
    require(len(libraries)==7,'seven same-build static libraries required')
    revision=device.original.common.source.git(P,'rev-parse','HEAD').decode().strip()
    sources={x:sha(P/x) for x in (THIS,CPP)}
    require(all(hashlib.sha256(device.original.common.source.git(P,'show',revision+':'+x)).hexdigest()==h
        for x,h in sources.items()),'commit exact component source first')
    env=dict(n['recipe']['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir()
    row=dict(stage='prepared',source_commit=revision,source_files=sources,
        native_source_commit=n['native_source_commit'],engine=n['binaries']['strata'],
        native_admission=n['observer_admission'],compile_seed=dict(path=str(seed),sha256=sha(seed)),
        static_libraries=libraries,recipe=dict(compile=compile_argv,link=link,cwd=str(build),environment=env),
        plan='Compile only this fixture and link the seven exact97fda7ba same-build static libraries;do not overwrite/reconfigure the native engine. CPU source/native/recipe/SDK checks precede any build. Commit build evidence before fresh bounded traced smoke,numerics and separately planned clean component timing.',
        prospective_checks='Smoke compact M3 finite/edge;numeric compact/wide M1,3,17,33,64,127,129,640,1280 plus edge17. Exact BF16capture,independent small integer-dot oracle,<=1FP16ULP,guards,immutableinputs. No profiling queue or GPU timing markers. Later3processes x7rowshapes x4matched wall-time pairs;150mscalibration,100mseligibilityfloor,max8192repeats. Select a shape only from consistent per-process gains;no blanket minimum gain. Resident reused operands omit dequant,gather,PCIe and production dispatcher;no model advancement solely from these times.',
        retention='One current-API component fixture using the existing native generation;no new engine generation. Retire object/build output after useful compiled-input/safety/speed evidence is committed and the next production decision closes. Five existing engine/proof holds remain separate and are under source review.',
        budget=dict(wall=300,cpu=600,rss_cap=8<<30,text_cap=32<<20,file_cap=128<<20,total_cap=512<<20),
        active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,commands=[])
    save(row);print('Prepared current-library compact-GU fixture;commit before building.',flush=True)


def build():
    row=json.loads((ROOT/'record.json').read_text());current(row);frozen=copy.deepcopy(row)
    require(row['stage']=='prepared' and not row['active'] and row['ownership_closed'],'one prospective build')
    scope=None
    def persist():
        (ROOT/'execution.json').write_text(json.dumps(dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []),indent=2)+'\n')
    try:
        scope=owner.Owner(ROOT/'owned-build',persist)
        for label in ('compile','link'):
            e,out,err=scope.run(label,row['recipe'][label],row['recipe']['environment'],
                Path(row['recipe']['cwd']),**row['budget'])
            require(owner.closed(e) and e['exit_code']==0,'component '+label+' failed')
            row['commands'].append(dict(label=label,argv=row['recipe'][label],exit_code=e['exit_code'],
                ownership_closed=True,stdout_sha256=sha(out),stderr_sha256=sha(err)))
        deps=(ROOT/'fixture.d').read_text().replace('\\\n',' ').split(':',1)[1]
        headers={str((Path(row['recipe']['cwd'])/p).resolve()):sha((Path(row['recipe']['cwd'])/p).resolve())
            for p in shlex.split(deps)}
        (ROOT/'actual-headers.json').write_text(json.dumps(headers,indent=2)+'\n')
        row['actual_headers']=dict(path=str(ROOT/'actual-headers.json'),sha256=sha(ROOT/'actual-headers.json'),files=len(headers))
        row['binary']=dict(path=str(ROOT/'fixture'),sha256=sha(ROOT/'fixture'),bytes=(ROOT/'fixture').stat().st_size)
        current(frozen);row.update(stage='built',passed=True,complete=True)
    except BaseException as error:
        row.update(stage='failed',error=repr(error));raise
    finally:
        row['active']=bool(scope and scope.active)
        row['ownership_closed']=bool(scope and not scope.active and all(owner.closed(x) for x in scope.commands))
        save(row)
    print('Built compact-GU fixture only;GPU and speed unqualified.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build'),'prepare|build')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device.original.guards.no_model_owners([])
        prepare() if sys.argv[1]=='prepare' else build()
