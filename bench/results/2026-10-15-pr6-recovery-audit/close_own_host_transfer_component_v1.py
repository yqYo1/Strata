#!/usr/bin/env python3
"""Retire raw closed-success transfer logs after verifying compact evidence."""
import fcntl
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import require,sha,load_owner


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=A/'own-host-transfer-component-v1.json';r=json.loads(source.read_text())
        root=B/'main-own-host-transfer-component-root-v1'
        target=A/'closed-own-host-transfer-retention-v1.json'
        compact=A/'closed-own-host-transfer-compact-v1.json'
        require(not target.exists() and not compact.exists(),'fresh closure')
        require(r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'],'normal healthy component')
        require(sha(root/'record.json')==sha(source),'durable receipt identical')
        for command in r['commands']:
            require(module.closed(command) and command['exit_code']==0,'normal owned closure')
            for old in command['owners']:
                now=module.identity(old['pid']);require(now is None or now['start_ticks']!=old['start_ticks'],'owned identity present')
        evidence=dict(receipt_sha256=sha(source),all_individual_samples_retained_in_receipt=True,
            build_stdout=(root/'owned/build-transfer-probe.stdout').read_text(),successful_smoke_API_returns={},raw_files=[])
        for label,v in r['cases'].items():
            out=root/('owned/'+label+'.stdout');err=root/('owned/'+label+'.stderr')
            require(out.read_text()==v['stdout'] and sha(err)==v['stderr_sha256'] and v['correct'],'exact original copy/check evidence retained')
            if v['trace']:evidence['successful_smoke_API_returns'][label]=err.read_text().count('UR_RESULT_SUCCESS')
        deletions=[]
        for path in root.rglob('*'):
            if not path.is_file():continue
            require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),'private raw artifact')
            item=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),status='planned',reason='Closed successful component; source/commands/env/allindividualsamples/fulloracleoutcomes/setup/owner/faults retained in receipt; API repetitions and duplicate receipt retired.')
            deletions.append(item);evidence['raw_files'].append(dict(path=str(path.relative_to(root)),bytes=item['bytes'],sha256=item['sha256']))
        compact.write_text(json.dumps(evidence,indent=2)+'\n')
        manifest=dict(status='planned',owner='Main',recipe_sha256=sha(Path(__file__)),receipt_sha256=sha(source),compact_sha256=sha(compact),original_result_status_unchanged=True,physical_free_space_claim=False,deletions=deletions)
        target.write_text(json.dumps(manifest,indent=2)+'\n')
        for item in deletions:
            path=Path(item['path']);require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'raw identity before unlink');path.unlink();item['status']='removed'
        for path in sorted((p for p in root.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):path.rmdir()
        root.rmdir()
        manifest.update(status='complete',files=len(deletions),removed_logical_bytes=sum(v['bytes'] for v in deletions))
        target.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ('status','files','removed_logical_bytes')}))


if __name__=='__main__':main()
