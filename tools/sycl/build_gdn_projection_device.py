#!/usr/bin/env python3
"""Build the guarded GDN fixture with the actual AOT object; no GPU launch. MIT."""
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import sys

sys.dont_write_bytecode = True
import check_gdn_projection_fp32 as image

base, RESULT = image.base, image.RESULT
ROOT = image.ROOT / 'device-fixture-v1'
KEY = 'official_gdn_fp32_projection_fixture_build_v1'
FIXTURE = 'bench/sycl/gdn_projection_fp32_device.cpp'
require, digest, write = base.require, base.digest, base.write


def save(row):
    write(ROOT / 'record.json', row)
    data=json.loads(RESULT.read_text());data[KEY]=base.compact(row);write(RESULT,data)


def prepare():
    data=base.committed(RESULT); prior=json.loads(image.RECORD.read_text())
    require(KEY not in data and not ROOT.exists() and data[image.KEY]==base.compact(prior) and
        prior['passed'] and prior['ownership_closed'] and not prior['active'] and
        prior['codegen_member']['manual_FP32_store_review_passed'],'closed original FP32 image required')
    original=json.loads((image.ROOT/'record.json').read_text())
    db=Path(original['original_compile_database']['path'])
    require(digest(db)==original['original_compile_database']['sha256'],'original compile database changed')
    entries=json.loads(db.read_text()); selected=image.build.retained.ROOT
    revision=base.basis.build.git(base.P,'rev-parse','HEAD').decode().strip()
    sha=hashlib.sha256(base.basis.build.git(base.P,'show',revision+':'+FIXTURE)).hexdigest()
    require(digest(base.P/FIXTURE)==sha,'commit exact guarded fixture')
    commands=[]
    for oldfile,newfile,label in [('tests/sycl/vllm_native16.cpp',str(base.P/FIXTURE),'fixture-object'),
                                 ('src/sycl/runtime.cpp',str(image.ROOT/'source/src/sycl/runtime.cpp'),'runtime-object')]:
        entry=[e for e in entries if e['file'].endswith('/'+oldfile)];require(len(entry)==1,'one production compile recipe required')
        entry=entry[0];argv=entry.get('arguments') or shlex.split(entry['command'])
        argv=[a.replace(str(selected/'source'),str(image.ROOT/'source')) for a in argv]
        argv[argv.index('-c')+1]=newfile;argv[argv.index('-o')+1]=str(ROOT/'build'/(label+'.o'))
        argv+=['-MD','-MF',str(ROOT/(label+'.d'))];commands.append((label,argv))
    mkl=Path('/opt/intel/oneapi/mkl/2026.1/lib')
    link=[base.COMPILER,'-O3','-fsycl','-fsycl-id-queries-range=size_t','-fsycl-allow-device-image-dependencies',
          '-Wl,-t',str(ROOT/'build/fixture-object.o'),str(ROOT/'build/runtime-object.o'),
          str(image.ROOT/'image.o'),str(image.ROOT/'host.o'),'-Wl,-rpath='+str(mkl)]
    link += [str(mkl/n) for n in ('libmkl_sycl_blas.so','libmkl_intel_ilp64.so','libmkl_sequential.so','libmkl_core.so')]
    link += ['-lm','-ldl','-lpthread','-o',str(ROOT/'gdn-fp32-check')];commands.append(('compile',link))
    ROOT.mkdir(mode=0o700)
    for part in ('tmp','cache','build'):(ROOT/part).mkdir()
    row=dict(stage='prepared',source_commit=revision,controller_inputs=base.controls(revision,controller=Path(__file__).resolve()),
        native_sources={FIXTURE:sha},native_implementation_commit=prior['implementation_commit'],
        original_image_record=dict(path=str(image.RECORD),sha256=digest(image.RECORD)),
        commands=commands,commands_executed=[],
        environment=dict(original['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
        budgets=dict(wall=300,cpu=900,rss_cap=5<<30,text_cap=16<<20,file_cap=256<<20,total_cap=1<<30),
        plan=dict(scope='Compile and link only;no execution of the new fixture.',
            runtime='One unchanged555-source Runtime translation unit;one allocation registry,no second runtime library.',
            acceptance='Production flags,unchanged AOT objects,pinned original dependencies,covered actual compiler/link inputs,all owned commands exit0.',
            device_gate='No launch until corrected host D-copy proof passes and Main commits a separate bounded,traced guard/numerical/retirement plan.'),
        active=False,ownership_closed=True,passed=False,GPU_executed=False,device_launch_permitted=False)
    save(row);print('Prepared exact-source guarded fixture CPU build;commit before build.',flush=True)


def run():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY]==base.compact(row) and row['stage']=='prepared' and
        not (ROOT/'owned-build').exists(),'fresh committed fixture compile required')
    require(base.controls(row['source_commit'],controller=Path(__file__).resolve())==row['controller_inputs'],
        'fixture build controls changed')
    ref=row['original_image_record'];require(digest(ref['path'])==ref['sha256'],'original image proof changed')
    prior=json.loads(Path(ref['path']).read_text());original=image.admission(prior)
    allowed=image.build.admit(original)
    full,_=image.build.source.sources()
    for group in full['dependencies']:
        for n,sha in group['source_sha256'].items():
            p=str((Path(group['root'])/n).resolve());require(digest(p)==sha,'dependency changed');allowed[p]=sha
    for n,sha in row['native_sources'].items():
        require(digest(base.P/n)==sha,'fixture source changed');allowed[str(base.P/n)]=sha
    for name in ('image.o','host.o'):
        item=prior['outputs'][name];require(digest(item['path'])==item['sha256'],'AOT object changed');allowed[item['path']]=item['sha256']
    owner=None
    def persist():
        write(ROOT/'record.json',dict(row,commands_executed=owner.commands if owner else [],active=bool(owner and owner.active)))
    def monitor(io,entry):
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(base.supervision.artifact_bytes(ROOT)<=row['budgets']['total_cap'],'fixture build artifact budget')
            if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        row['stage']='building';owner=base.supervision.Owner(ROOT/'owned-build',persist)
        for label,argv in row['commands']:
            entry,out,err=owner.run(label,argv,row['environment'],ROOT,interaction=monitor,**row['budgets'])
            require(base.supervision.closed(entry) and entry['exit_code']==0,
                label+' failed: '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-4000:])
        found=set()
        for label in ('fixture-object','runtime-object'):
            for line in (ROOT/(label+'.d')).read_text().replace('\\\n','').splitlines():
                if not line.strip():continue
                left,colon,right=line.partition(':');require(colon and left.strip(),'invalid fixture depfile')
                found.update(str((ROOT/p).resolve()) for p in shlex.split(right))
        require(all(allowed.get(p)==digest(p) for p in found),'uncovered actual fixture compiler input')
        write(ROOT/'compiler-inputs.json',{p:allowed[p] for p in sorted(found)})
        row['actual_compiler_inputs']=dict(path=str(ROOT/'compiler-inputs.json'),sha256=digest(ROOT/'compiler-inputs.json'),files=len(found),uncovered=0)
        row['actual_link_inputs']=base.traced_link_inputs(allowed,ROOT)
        row.update(stage='built-not-executed',passed=True,binary=dict(path=str(ROOT/'gdn-fp32-check'),sha256=digest(ROOT/'gdn-fp32-check'),bytes=(ROOT/'gdn-fp32-check').stat().st_size))
    except BaseException as error:
        row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row.update(active=False,ownership_closed=bool(owner and all(base.supervision.closed(v) for v in owner.commands)),commands_executed=owner.commands if owner else [])
        require(owner is None or owner.active is None,'fixture build owner unresolved');save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','GPU_executed','error')}),flush=True)
    require(row['passed'],'guarded fixture build failed')


if __name__=='__main__':
    with (base.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if sys.argv[1:]==['prepare']:prepare()
        elif sys.argv[1:]==['build']:run()
        else:raise RuntimeError('build_gdn_projection_device.py prepare|build')
