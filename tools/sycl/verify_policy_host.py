#!/usr/bin/env python3
"""Current phase-policy host/syntax qualification; repository LICENSE.

No engine build, GPU enumeration or model execution. Main owns the finite job.
The later engine still requires a coherent full build and model qualification.
"""
import copy
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import fixture_build as common
import decode_window_bench as baseline

P, B, RESULT = common.P, common.B, common.RESULT
ROOT, KEY = B/'decode-kv-policy-host-v2', 'decode_kv_policy_host_v2'
EXPECTED_STDOUT = ('verify-window-policy: passed selections=30 centering=18 native-half=26 '
    'dpas-head=1 dpas-dense=4 phase-roundtrip=1 invalid-keys=4 retirement=1 route-budget=64')
require, sha = common.require, common.sha


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('commands', 'actual_inputs')}
    out['project'] = {k:v for k,v in row['project'].items() if k != 'source_sha256'}
    out['dependencies'] = [{k:v for k,v in d.items() if k != 'source_sha256'} for d in row['dependencies']]
    if 'actual_inputs' in row:
        out['actual_inputs'] = {k:dict(files=len(v),sha256=common.signature(v)) for k,v in row['actual_inputs'].items()}
    return out


def save(row):
    common.write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text());data[KEY]=compact(row);common.write(RESULT,data)


def check(row,committed=True):
    if committed: require(common.validation.committed(RESULT,(KEY,))==compact(row),'commit exact host plan/result')
    common.build.check_inputs(row)
    require(all(sha(p)==h for p,h in row['generated_reference'].items()),'read-only generated reference changed')
    require(row['expected_stdout']==EXPECTED_STDOUT,'frozen fixture marker differs')
    for key in ('object','binary'):
        if key in row:
            item=row[key]
            require(Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256'],
                'before-launch host artifact changed: '+key)


def allowed(row):
    values=json.loads(Path(row['external_inputs']['path']).read_text()) | row['generated_reference']
    for pin in (row['project'],*row['dependencies']):
        values.update({str((Path(pin['root'])/n).resolve()):h for n,h in pin['source_sha256'].items()})
    values.update({str(Path(p).resolve()):h for p,h in row['toolchain'].items()})
    return values


def dependencies(path,row):
    raw=path.read_text().replace('\\\n','')
    names=shlex.split(raw.split(':',1)[1]);require(names,'actual compiler dependencies required')
    permits=allowed(row);found={}
    for name in names:
        p=Path(name);p=p if p.is_absolute() else P/p
        n=str(p.resolve());h=sha(p)
        require(permits.get(n)==h,'unadmitted host compiler input: '+n);found[n]=h
    return found


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh host-policy attempt only')
    native,proof=baseline.retained_native()
    prior=common.validation.committed(RESULT,('decode_kv_policy_host_v1',))
    require(prior['stage']=='failed' and not prior['passed'] and prior['ownership_closed'] and not prior['active']
        and len(prior['outcomes'])==4 and all(c['exit_code']==0 and c['ownership_closed'] for c in prior['outcomes'])
        and prior['marker_failure']['actual_stdout'].strip()==EXPECTED_STDOUT,
        'preserve original four-command success with failed marker qualification')
    compiler='/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    flags=[compiler,'--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13','-std=c++20',
        '-O2','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-I'+str(P/'include')]
    database=json.loads((baseline.reuse.ROOT/'build/compile_commands.json').read_text())
    found=[c for c in database if c['file']==str(P/'src/core/verify.cpp')]
    require(len(found)==1,'one qualified verifier recipe required')
    args=shlex.split(found[0]['command']);syntax=[];skip=False
    for arg in args:
        if skip:skip=False;continue
        if arg in ('-o','-MF','-MT','-MQ'):skip=True;continue
        if arg in ('-c','-MD','-MMD','-MP'):continue
        syntax.append(arg)
    require(not skip and syntax[0]==compiler and '-DSTRATA_ENABLE_SYCL=1' in syntax,
        'qualified actual SYCL verifier conditions required')
    env={k:v for k,v in native['recipe']['environment'].items() if k in
        ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH')}
    env.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),PYTHONDONTWRITEBYTECODE='1')
    recipe=dict(environment=env,commands=[
        ('compile',[*flags,'-MD','-MF',str(ROOT/'policy.d'),'-c',str(P/'tests/core/verify_window_policy_test.cpp'),'-o',str(ROOT/'policy.o')]),
        ('syntax',[*syntax,'-fsyntax-only','-MD','-MF',str(ROOT/'verify.d')]),
        ('link',[compiler,'--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',str(ROOT/'policy.o'),'-Wl,-t','-o',str(ROOT/'policy')]),
        ('run',[str(ROOT/'policy')])])
    row=dict(stage='prepared',project=common.source.pin(P,common.PREFIXES),dependencies=copy.deepcopy(native['dependencies']),
        toolchain=native['toolchain']|{str(Path(sys.executable).resolve()):sha(sys.executable)},external_inputs=native['external_inputs'],
        fallbacks=native['binaries'],generated_reference=native['generated_inputs'],recipe=recipe,
        reference_source=proof['original_native_source'],reference_recipe_sha256=proof['original_recipe_sha256'],
        budget=dict(wall=180,cpu=240,rss_cap=4<<30,text_cap=2<<20,file_cap=32<<20,total_cap=64<<20),
        commands=[],outcomes=[],passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,model_executed=False,
        expected_stdout=EXPECTED_STDOUT,
        previous_failure=dict(key='decode_kv_policy_host_v1',evidence_commit='2a1630dd24b2b266e6f475ca4f7278eee12eac10',
            qualification='failed;all4CPUcommands0but wrong controller marker;no retroactive pass'),
        acceptance='Exact committed source/controller/recipe/compiler/SDK/header/link inputs;actual policy fixture0andPASS;actual SYCL verifier syntax0;all owned processes normally reaped with empty sessions. No native engine build,ODR/model/physical-context or performance qualification.',
        consumer='Current explicit decode KVhook;retire this temporary CPUELF after result/source recovery. Full engine rebuild remains mandatory before runtime tests.')
    check(row,False);ROOT.mkdir(mode=0o700)
    for n in ('tmp','cache'):(ROOT/n).mkdir(mode=0o700)
    save(row);print('Prepared current policy/actual-verifier host checks;commit before compilation.',flush=True)


def run():
    row=json.loads((ROOT/'record.json').read_text());check(row)
    require(row['stage']=='prepared' and not (ROOT/'attempt.json').exists(),'one frozen host attempt')
    common.write(ROOT/'attempt.json',dict(source_commit=row['project']['revision']))
    owner=common.ownership.Owner(ROOT/'owned');inputs=dict(compiled={},links={})
    try:
        for label,args in row['recipe']['commands']:
            check(row,False)
            entry,out,err=owner.run(label,args,row['recipe']['environment'],P,**row['budget'])
            require(common.ownership.closed(entry) and entry['exit_code']==0,'host command failed: '+label)
            row['outcomes'].append(dict(label=label,exit_code=0,ownership_closed=True))
            if label in ('compile','syntax'):
                inputs['compiled'].update(dependencies(ROOT/('policy.d' if label=='compile' else 'verify.d'),row))
            if label=='compile':
                row['object']=dict(path=str(ROOT/'policy.o'),sha256=sha(ROOT/'policy.o'),bytes=(ROOT/'policy.o').stat().st_size)
            if label=='link':
                permits=allowed(row)|{str(ROOT/'policy.o'):sha(ROOT/'policy.o')}
                for line in out.read_text().splitlines():
                    p=Path(line.strip())
                    if p.is_absolute() and p.is_file():
                        n=str(p.resolve());h=sha(p);require(permits.get(n)==h,'unadmitted host link input: '+n)
                        inputs['links'][n]=h
                require(len(inputs['links'])>=8,'complete host link census required')
                row['binary']=dict(path=str(ROOT/'policy'),sha256=sha(ROOT/'policy'),bytes=(ROOT/'policy').stat().st_size)
            if label=='run':
                require(out.read_text().strip()==row['expected_stdout']
                    and sha(ROOT/'policy')==row['binary']['sha256'],'all actual policy/cache controls required')
        check(row,False);row.update(stage='closed',passed=True,complete=True,actual_inputs=inputs)
    except BaseException as error:
        row.update(stage='failed',passed=False,error=repr(error),actual_inputs=inputs);raise
    finally:
        row.update(commands=owner.commands,active=bool(owner.active),
            ownership_closed=owner.active is None and all(common.ownership.closed(c) for c in owner.commands))
        save(row)
    print('Current policy and actual SYCL verifier syntax pass;no GPU/model executed.',flush=True)


if __name__=='__main__':
    require(sys.executable==baseline.PYTHON and len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),
        'qualified interpreter: prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);common.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:run()
