#!/usr/bin/env python3
"""CPU-only lossless fixture build from qualified native objects;repository LICENSE.

Numerical launch/interval policy belongs to a separately committed controller.
This module never submits a GPU kernel or selects an inference route.
"""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
sys.dont_write_bytecode=True
import qsa_head_native as n
q,r,P,B=n.q,n.r,n.P,n.B
ROOT=n.ROOT/'numerics-v2'
KEY='numerical_fixture_v2'
CPP='bench/sycl/qsa_head_trace.cpp'
RESULT=n.RESULT
require,digest,write=n.require,n.digest,n.write


def sources(revision):
    return q.sources(revision,Path(__file__).resolve(),[CPP,n.FIXTURE,n.ORACLE,
        'tools/sycl/qsa_sum_interval.py','tools/test_qsa_sum_interval.py'])


def native():
    record=r.validation.committed(RESULT);scope=record[n.KEY]
    row=json.loads((n.ROOT/'record.json').read_text())
    require(n.compact(row)==scope['build'] and row['passed'] and row['ownership_closed'] and not row['active'],
            'closed exact native build first')
    props=scope['properties4']['outcome']
    require(props['passed'] and props['complete'] and props['ownership_closed'] and not props['fresh_faults'],
            'four compiled native properties first')
    allowed,rt=n.admitted(row);return record,row,allowed,rt


def compact(row):
    return {k:v for k,v in row.items() if k not in ['source_sha256','commands_executed']}


def admitted(row):
    record,built,_,rt=native()
    require(sources(row['source_commit'])==row['source_sha256'],'lossless fixture/controller sources changed')
    require(digest(row['inventory']['path'])==row['inventory']['sha256'],'frozen input inventory changed')
    allowed=json.loads(Path(row['inventory']['path']).read_text())
    for name,h in allowed.items():require(digest(name)==h,'frozen compiler/object/header input changed:'+name)
    for binary in row.get('binaries',{}).values():require(digest(binary['path'])==binary['sha256'],'lossless fixture ELF changed')
    return record,built,allowed,rt


def prepare():
    record,built,allowed,rt=native()
    require(KEY not in record[n.KEY] and not ROOT.exists(),'fresh lossless fixture only')
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip();pins=sources(rev)
    ROOT.mkdir(mode=0o700)
    for name in ['tmp','cache']:(ROOT/name).mkdir()
    compile_base=next(argv for label,argv in built['commands'] if label=='original')
    flags=compile_base[1:compile_base.index('-MD')];commands=[]
    for page in [4,16]:
        obj=ROOT/('trace'+str(page)+'.o')
        commands.append(['compile'+str(page),[built['compiler']['path'],*flags,
            '-DSTRATA_QSA_TEST_PAGE_SIZE='+str(page),'-MD','-MF',str(ROOT/('trace'+str(page)+'.d')),
            '-c',str(P/CPP),'-o',str(obj)]])
        link=list(next(argv for label,argv in built['commands'] if label=='link'+str(page)))
        link[link.index(str(n.ROOT/('oracle'+str(page)+'.o')))]=str(obj)
        link[link.index('-o')+1]=str(ROOT/('trace'+str(page)))
        commands.append(['link'+str(page),link])
        for name in link:
            if name.endswith('.o') and name!=str(obj):allowed[name]=digest(name)
    allowed.update({str(P/name):h for name,h in pins.items()});write(ROOT/'inventory.json',allowed)
    row=dict(stage='prepared',source_commit=rev,source_sha256=pins,native_source_commit=built['source_commit'],
        commands=commands,compiler=built['compiler'],native_geometry=built['geometry_proof'],
        runtime_sha256=digest(r.RESULT),boot=rt['boot'],
        inventory=dict(path=str(ROOT/'inventory.json'),sha256=digest(ROOT/'inventory.json'),files=len(allowed)),
        environment=dict(built['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
        budgets=dict(wall=600,cpu=1800,rss_cap=8<<30,text_cap=32<<20,file_cap=256<<20,total_cap=2<<30),
        passed=False,active=False,ownership_closed=True,GPU_executed=False,
        criterion='CPUbuild only from exact original/candidate native objects and frozen full closure. Page4/16lossless p/BF16p/state/A/precast capture fixture must build/link with zero uncovered inputs. No numerical launch until independent arithmetic contract,interval parser and finite source-bound test plan are committed.',
        retention='Same imminent normalizer comparison generation,not an extra independent engine. Retire this fixture/raw output with the candidate after decision and exact source/recipe recovery;selected6e40+12fixtures unchanged.')
    admitted(row);write(ROOT/'record.json',row);record[n.KEY][KEY]=compact(row);write(RESULT,record)
    print('Lossless CPU fixture plan prepared;commit before build.',flush=True)


def build():
    record=r.validation.committed(RESULT);row=json.loads((ROOT/'record.json').read_text())
    require(row['stage']=='prepared' and compact(row)==record[n.KEY][KEY],'fresh committed fixture plan')
    _,_,allowed,_=admitted(row);owner=None
    def persist():write(ROOT/'record.json',dict(row,active=bool(owner and owner.active is not None),commands_executed=owner.commands if owner else []))
    owner=r.supervision.Owner(ROOT/'build-owned',persist)
    try:
        links={}
        for label,argv in row['commands']:
            print('CPUfixture:'+label,flush=True)
            entry,out,err=owner.run(label,argv,row['environment'],ROOT,**row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code']==0,'fixture compile/link failed:'+err.read_text()[-3000:])
            if label.startswith('link'):links[label]=q.linking.traced_inputs(out.read_text(),err.read_text(),allowed,ROOT,ROOT)
        observed={}
        for dep in ROOT.glob('*.d'):
            for name in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
                p=Path(name).resolve();h=digest(p);require(allowed.get(str(p))==h,'unfrozen actual fixture compiler input:'+str(p));observed[str(p)]=h
        require(str(P/CPP) in observed and str(P/n.ORACLE) in observed and len(observed)>600,'incomplete actual fixture source closure')
        write(ROOT/'compiler-inputs.json',observed);write(ROOT/'link-inputs.json',links)
        row.update(stage='built',passed=True,binaries={str(page):dict(path=str(ROOT/('trace'+str(page))),sha256=digest(ROOT/('trace'+str(page))),bytes=(ROOT/('trace'+str(page))).stat().st_size) for page in [4,16]},
            actual_compiler_inputs=dict(path=str(ROOT/'compiler-inputs.json'),sha256=digest(ROOT/'compiler-inputs.json'),files=len(observed),uncovered=0),
            actual_link_inputs=dict(path=str(ROOT/'link-inputs.json'),sha256=digest(ROOT/'link-inputs.json'),uncovered=0));admitted(row)
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row.update(active=False,ownership_closed=owner.active is None and all(r.supervision.closed(c) for c in owner.commands),commands_executed=owner.commands)
        row['passed']=bool(row['passed'] and row['ownership_closed']);write(ROOT/'record.json',row);record[n.KEY][KEY]=compact(row);write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ['stage','passed','ownership_closed','error']}),flush=True)
    require(row['passed'],'lossless CPU fixture build failed')


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'prepare':prepare,'build':build}[sys.argv[1]]()
