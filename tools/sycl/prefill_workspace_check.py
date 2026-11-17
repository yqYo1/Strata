#!/usr/bin/env python3
"""Commit-bound, CPU-only long-prefill workspace check. Repository LICENSE.

No SYCL loader/model/GPU process. Keep the fixture only through its immediate
capacity check, then retire it before the coherent GPU candidate is built.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_owner as owner

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B/'prefill-window-plan-v1'
RESULT = P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'prefill_window_workspace_v1'
THIS = 'tools/sycl/prefill_workspace_check.py'
INPUTS = (THIS, 'tools/sycl/measurement_owner.py', 'tests/prefill_moe_workspace.cpp',
          'include/strata/prefill/moe_accum_plan.hpp', 'AGENTS.md',
          'docs/ARTIFACT_RETENTION.md', 'docs/BENCHMARKS.md')
require = owner.require


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git','-C',str(P),*args],timeout=20)


def compact(row):
    value={k:v for k,v in row.items() if k not in ('sources','external_inputs')}
    for key in ('sources','external_inputs'):
        value[key+'_summary']=dict(files=len(row[key]),sha256=hashlib.sha256(
            json.dumps(row[key],sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return value


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row,indent=2)+'\n')
    value=json.loads(RESULT.read_text());value[KEY]=compact(row)
    RESULT.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def current(row):
    committed=json.loads(git('show','HEAD:'+str(RESULT.relative_to(P))))[KEY]
    require(committed==compact(row),'commit exact preceding CPU stage first')
    for name,h in row['sources'].items():
        require(hashlib.sha256(git('show',row['source_commit']+':'+name)).hexdigest()==h
            and sha(P/name)==h,'CPU fixture/controller input changed:'+name)
    require(all(sha(p)==h for p,h in row['external_inputs'].items()),'compiler/header/link input changed')
    require(owner.artifact_bytes(ROOT)<=row['whole_bytes'],'CPU capture envelope exceeded')


def prepare():
    require(not ROOT.exists(),'fresh CPU plan only; inspect existing process handle')
    revision=git('rev-parse','HEAD').decode().strip()
    pins={name:sha(P/name) for name in INPUTS}
    require(all(git('show',revision+':'+n)==(P/n).read_bytes() for n in INPUTS),'commit exact CPU sources first')
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir()
    compiler=Path('/usr/bin/c++').resolve()
    env=dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),LANG='C.UTF-8',LC_ALL='C.UTF-8',
             PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'))
    command=[str(compiler),'-std=c++20','-O2','-Wall','-Wextra','-Werror',
        '-I'+str(P/'include'),str(P/'tests/prefill_moe_workspace.cpp'),'-Wl,--trace',
        '-o',str(ROOT/'prefill_moe_workspace')]
    deps=subprocess.check_output([*command[:-3],'-M'],env=env,cwd=P,text=True,timeout=30)
    paths={Path(v).resolve() for v in shlex.split(deps.replace('\\\n',' '))[1:]}
    paths.update((compiler,Path(sys.executable).resolve()))
    for program in ('cc1plus','collect2','as','ld'):
        name=subprocess.check_output([str(compiler),'-print-prog-name='+program],env=env,text=True,timeout=10).strip()
        paths.add(Path(shutil.which(name,path=env['PATH']) or name).resolve())
    libraries=('Scrt1.o','crti.o','crtbeginS.o','crtendS.o','crtn.o','libstdc++.so',
        'libm.so','libm.so.6','libmvec.so.1','libgcc_s.so','libgcc_s.so.1','libgcc.a',
        'libc.so','libc.so.6','libc_nonshared.a','ld-linux-x86-64.so.2','liblto_plugin.so')
    for library in libraries:
        name=subprocess.check_output([str(compiler),'-print-file-name='+library],env=env,text=True,timeout=10).strip()
        require(name!=library and Path(name).is_file(),'unresolved planned library:'+library)
        paths.add(Path(name).resolve())
    external={str(path):sha(path) for path in sorted(paths) if not path.is_relative_to(P)}
    require(all(str(path.relative_to(P)) in pins for path in paths if path.is_relative_to(P)),
            'unplanned project include')
    prior=json.loads(RESULT.read_text())[KEY]
    require(prior['stage']=='source_proposal' and not prior['GPU_executed'],'keep original proposal unexecuted')
    row=dict(prior,stage='prepared',source_commit=revision,sources=pins,external_inputs=external,
        compiler=dict(path=str(compiler),sha256=sha(compiler),version=subprocess.check_output(
            [str(compiler),'--version'],env=env,text=True,timeout=10).splitlines()[0]),
        command=command,environment=env,whole_bytes=32<<20,results=[],active=False,
        complete=False,passed=False,ownership_closed=True,
        host_budget=dict(wall=120,cpu=90,rss_cap=1<<30,text_cap=1<<20,file_cap=8<<20,total_cap=16<<20),
        host_plan='Build separately;pin exact executable and actual linker inputs;commit before running. '
                  'Require49152row cases,old-layout invariants,two relocated planes,disjoint guards/chunk maps,negative refusals,normal0/reaped/empty owner. No GPU/math/speed/model admission follows.')
    save(row)
    print('Prepared CPU source/recipe/dependency closure;commit before build.',flush=True)


def execute(mode):
    row=json.loads((ROOT/'record.json').read_text());current(row)
    require(row['stage']==('prepared' if mode=='build' else 'built'),'ordered fresh CPU stage only')
    root=ROOT/mode;root.mkdir()
    scope=None
    def persist():
        (root/'execution.json').write_text(json.dumps(dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []),indent=2)+'\n')
    scope=owner.Owner(root/'owned',persist)
    row.update(active=True,ownership_closed=False);save(row)
    result=dict(mode=mode,passed=False,GPU_executed=False)
    try:
        argv=row['command'] if mode=='build' else [row['binary']['path']]
        if mode=='build':require(not Path(argv[-1]).exists(),'no CPU executable overwrite')
        else:require(sha(argv[0])==row['binary']['sha256'],'CPU fixture executable changed')
        entry,out,err=scope.run(mode,argv,row['environment'],P,**row['host_budget'])
        require(owner.closed(entry) and entry['exit_code']==0,'normal CPU closure required')
        if mode=='build':
            actual=[]
            for line in out.read_text().splitlines():
                path=Path(line.strip())
                temporary_object=path.is_relative_to(ROOT/'tmp') and path.suffix=='.o'
                require(path.is_absolute() and (path.is_file() or temporary_object),
                        'unrecognized linker trace row')
                real=str(path.resolve())
                # GCC's one temporary object is compiled from the pinned single
                # source in this owned invocation,then removed by the driver.
                actual.append(real)
            for path in actual:
                require(path in row['external_inputs'] or Path(path).is_relative_to(ROOT/'tmp'),
                        'linker consumed an unplanned input:'+path)
            row['actual_link_inputs']=actual
            binary=Path(row['command'][-1]);row['binary']=dict(path=str(binary),sha256=sha(binary),bytes=binary.stat().st_size)
            row['stage']='built'
        else:
            expected=('prefill_moe_workspace CPU_PASS rows_checked=49152 moved_planes=2 '
                'extra_131071=729799424 historical_margin=22492160 GPU=UNRUN runtime_admission=UNRUN '
                'model_math=UNRUN model_performance=UNRUN physical262144_model=UNRUN\n')
            require(out.read_text()==expected and not err.read_text(),'complete CPU capacity checks required')
            result.update(rows_checked=49152,relocated_planes=2);row['stage']='cpu_closed'
        result.update(passed=True,exit_code=0,ownership_closed=True)
    except BaseException as error:
        row['stage']='failed';result.update(error=repr(error))
    finally:
        result['ownership_closed']=scope.active is None and all(owner.closed(c) for c in scope.commands)
        row['results'].append(result);row.update(active=bool(scope.active),ownership_closed=result['ownership_closed'],
            passed=row['stage']=='cpu_closed',complete=row['stage']=='cpu_closed')
        save(row)
    print(json.dumps(result),flush=True);require(result['passed'],result.get('error','CPU check failed'))


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','build','check'),'prepare|build|check required')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare() if sys.argv[1]=='prepare' else execute(sys.argv[1])
