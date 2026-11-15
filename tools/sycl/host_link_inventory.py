"""Freeze this official compiler's host linker inputs before a NEW build.

Repository LICENSE. The previous scope remains failed; this is prospective
inventory from the compiler's queries under the exact build environment.
"""
import hashlib
import json
import sys
from pathlib import Path
import subprocess

NAMES=('crtbeginS.o','crtendS.o','libgcc.a','libgcc_s.so','Scrt1.o','crti.o','crtn.o',
       'libc.so','libc_nonshared.a','libdl.a','libm.so','libmvec.so.1','libpthread.a')


def inventory(compiler,environment):
    files={};queries=[];owners=set()
    for option in ['-print-prog-name=ld',*['-print-file-name='+name for name in NAMES]]:
        argv=[compiler,'-fsycl',option]
        reported=subprocess.check_output(argv,env=environment,text=True,timeout=15).strip()
        path=Path(reported)
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError('compiler did not resolve host link input: '+option+':'+reported)
        path=path.resolve()
        if not path.is_relative_to('/usr'):
            raise RuntimeError('unexpected non-system host linker input: '+str(path))
        files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        queries.append(dict(argv=argv,reported=reported,resolved=str(path)))
    # Read the package file lists once for the whole resolved set. Preserve the
    # exact per-file ownership requirement and returned inventory, not14 scans.
    ownership=subprocess.check_output(['dpkg-query','-S',*files],
                                     env=environment,text=True,timeout=45)
    covered=set()
    for line in ownership.splitlines():
        package,separator,name=line.partition(': /')
        if separator and '/'+name in files:
            covered.add('/'+name);owners.add(package)
    if covered!=set(files):
        raise RuntimeError('unowned host linker inputs: '+repr(sorted(set(files)-covered)))
    versions=subprocess.check_output(['dpkg-query','-W','-f=${binary:Package} ${Version}\n',*sorted(owners)],
                                    env=environment,text=True,timeout=15).splitlines()
    return dict(files=files,compiler_queries=queries,packages=versions,
                scope='Exact host linker,CRT,archive/linker-script and math-library bytes from official driver queries. No GPU runtime,safety key or old failure status is changed.')


def verification_main():
    """Main-owned CPU-only equality check before this metadata control is reused."""
    import fcntl
    import subprocess
    import measurement_owner as supervision
    import measurement_guards as guards
    self_path=Path(__file__).resolve();project=self_path.parents[2]
    native_path=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/cache-map-model-v1/record.json')
    root=native_path.parents[1]/'host-link-inventory-batch-v1'
    result=project/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
    key='host_link_inventory_batch_v1'
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    native=json.loads(native_path.read_text())
    if sys.argv[1:] == ['--emit']:
        print(json.dumps(inventory(native['compiler']['path'],native['environment']),sort_keys=True))
        return
    guards.require(sys.argv[1:] in (['prepare'],['check']), 'use prepare|check')
    with (root.parent/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        committed=json.loads(subprocess.check_output(['git','show','HEAD:'+str(result.relative_to(project))],cwd=project))
        guards.require(committed==json.loads(result.read_text()),'commit exact verification plan first')
        if sys.argv[1]=='prepare':
            guards.require(not root.exists() and key not in committed,'one fresh inventory comparison')
            revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
            sources={str(path.relative_to(project)):digest(path) for path in
                     (self_path,Path(supervision.__file__),Path(guards.__file__))}
            for name,sha in sources.items():
                guards.require(hashlib.sha256(subprocess.check_output(['git','show',revision+':'+name],cwd=project)).hexdigest()==sha,'commit checker sources')
            root.mkdir(mode=0o700)
            row=dict(stage='prepared',source_commit=revision,sources=sources,
                selected_record=dict(path=str(native_path),sha256=digest(native_path)),
                interpreter=dict(path=sys.executable,sha256=digest(sys.executable)),
                expected_sha256=hashlib.sha256(json.dumps(native['host_link_inputs'],sort_keys=True).encode()).hexdigest(),
                command=[sys.executable,str(self_path),'--emit'],
                budgets=dict(wall=180,cpu=120,rss_cap=512<<20,text_cap=1<<20,file_cap=1<<20,total_cap=2<<20),
                plan='Query every original compiler/linker path;one batched dpkg ownership lookup must cover every resolved file;compare the entire result with the unchanged selected native inventory,including bytes,driver queries and package versions. No engine/GPU execution.',
                passed=False,complete=False,ownership_closed=True)
        else:
            row=json.loads((root/'record.json').read_text())
            guards.require(committed[key]==row and row['stage']=='prepared','one committed check')
            guards.require(digest(native_path)==row['selected_record']['sha256'] and
                all(digest(project/name)==sha for name,sha in row['sources'].items()) and
                digest(row['interpreter']['path'])==row['interpreter']['sha256'] and
                digest(native['compiler']['path'])==native['compiler']['sha256'],'frozen inputs changed')
            owner=supervision.Owner(root/'owned')
            try:
                entry,out,err=owner.run('equal-inventory',row['command'],native['environment'],root,**row['budgets'])
                guards.require(supervision.closed(entry) and entry['exit_code']==0 and not err.stat().st_size,
                               'inventory command failed: '+err.read_text()[-2000:])
                actual=json.loads(out.read_text())
                guards.require(actual==native['host_link_inputs'],'batched inventory differs from selected reference')
                guards.require(digest(native_path)==row['selected_record']['sha256'] and
                    all(digest(project/name)==sha for name,sha in row['sources'].items()) and
                    all(digest(path)==sha for path,sha in actual['files'].items()) and
                    digest(native['compiler']['path'])==native['compiler']['sha256'] and
                    digest(row['interpreter']['path'])==row['interpreter']['sha256'], 'post-check inputs changed')
                row.update(stage='closed',passed=True,complete=True,native_exit=0,
                    actual_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest(),
                    files=len(actual['files']),queries=len(actual['compiler_queries']),packages=len(actual['packages']))
            except BaseException as error:
                row.update(stage='failed',error=repr(error));raise
            finally:
                row['ownership_closed']=owner.active is None and all(supervision.closed(x) for x in owner.commands)
                write(root/'record.json',row);data=json.loads(result.read_text());data[key]=row;write(result,data)
        write(root/'record.json',row);data=json.loads(result.read_text());data[key]=row;write(result,data)
        print('Host link inventory',row['stage'],flush=True)


if __name__=='__main__':
    verification_main()
