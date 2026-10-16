#!/usr/bin/env python3
"""Close reviewed storage diagnostics without changing original outcomes."""
import fcntl
import json
import re
import subprocess
import tempfile
from pathlib import Path
from check_gemm_orientation import P,B,D,sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dest=D/'closed-artifact-retention-v3.json';assert not dest.exists()
        names=['main-primary-layer-major-dispatch-build-root-v1',
            'main-primary-layer-major-storage-build-root-v1',
            'main-primary-layer-major-storage-small-root-v1',
            'main-primary-layer-major-long-root-v1',
            'main-primary-current-diagnostics-build-root-v1']
        originals={};plans=[];excerpts=[]
        for name in names:
            root=B/name;record=root/'record.json';r=json.loads(record.read_text())
            assert not r['active']
            assert all(c['direct_child_reaped'] and c['session_empty'] for c in r['commands'])
            originals[str(record)]=sha(record)
            for f in sorted((root/'owned').glob('*')):
                if not f.is_file() or f.suffix not in ['.stdout','.stderr']:continue
                assert f.resolve()==f and f.stat().st_size<=64 << 20
                lines=[s for s in f.read_text(errors='replace').splitlines()
                       if re.search(r'(?i)warning|error|\bFAIL\b|cannot|aborting',s)]
                assert sum(len(s.encode()) for s in lines)<=256 << 10
                if lines:excerpts.append(dict(path=str(f),lines=lines))
                plans.append((f,'Closed text: outcomes, commands, CPU/GPU proof, model comparisons and diagnostics retained in compact receipts.'))
        small=json.loads((D/'layer-major-storage-small-v1.json').read_text())
        assert small['passed'] and small['complete'] and not small['fresh_faults']
        old=json.loads((D/'layer-major-small-v1.json').read_text())
        for arm in small['arms']:
            if arm['label']=='fp16-auto':continue
            reference=old['arms'][0] if arm['label'].startswith('fp32') else small['arms'][2]
            for candidate,control in zip(arm['fixtures'],reference['fixtures']):
                assert candidate['sha256']==control['sha256'] and sha(Path(control['path']))==control['sha256']
                plans.append((Path(candidate['path']),'Closed bitwise duplicate; original FP32 control and one FP16 GPU-arm fixture are canonical for the next compiled-source regression.'))
        long=json.loads((D/'layer-major-long-v1.json').read_text())
        natural=json.loads((D/'natural-layout-long-pair-v2.json').read_text())
        assert long['passed'] and long['complete']
        for candidate,control in zip(long['arms'][0]['fixtures'],natural['arms'][1]['fixtures']):
            assert candidate['sha256']==control['sha256'] and sha(Path(control['path']))==control['sha256']
            plans.append((Path(candidate['path']),'Closed FP32 long comparison; natural0 remains sole long reference for selected-head/model regression.'))

        proposal=D/'layer-major-fp16-storage-v1'
        base=json.loads((proposal/'source-base-sha256.json').read_text())
        artifacts=json.loads((proposal/'artifact-sha256.json').read_text())
        assert sha(proposal/'source.diff')==artifacts['source.diff']
        # The patch plus a reachable base reproduces every original generated
        # review copy, including Main's later intentionally different fixes.
        with tempfile.TemporaryDirectory(prefix='storage-proposal-replay-',dir=B) as tmp:
            tmp=Path(tmp)
            for name,pin in base.items():
                data=subprocess.check_output(['git','show','005c47b6:'+name],cwd=P,timeout=30)
                target=tmp/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                assert sha(target)==pin
            replay=subprocess.run(['git','apply',str(proposal/'source.diff')],cwd=tmp,
                capture_output=True,timeout=30,check=True)
            for name in base:
                copy=proposal/'files'/name;pin=artifacts['files/'+name]
                assert sha(tmp/name)==pin and sha(copy)==pin
                plans.append((copy,'Integrated source committed at ba78d296; original candidate reproduces byte-for-byte from reachable005c47b6 plus retained source.diff. Full generated review copy no longer has a separate consumer.'))
        evidence=[D/n for n in ['layer-major-dispatch-build-v1.json','layer-major-storage-build-v1.json',
            'layer-major-storage-small-v1.json','layer-major-storage-small-summary-v1.json',
            'layer-major-long-v1.json','current-diagnostics-build-v1.json',
            'natural-layout-long-pair-v2.json','layer-major-small-v1.json',
            'layer-major-fp16-storage-v1/source.diff','layer-major-fp16-storage-v1/artifact-sha256.json']]
        manifest=dict(complete=False,recipe_sha256=sha(Path(__file__)),original_receipts=originals,
            retained_evidence={str(f):sha(f) for f in evidence},diagnostic_excerpts=excerpts,
            original_results_unchanged=True,proposal_replay=dict(base_commit='005c47b6',
                patch_sha256=sha(proposal/'source.diff'),files=len(base),all_hashes_equal=True,exit_code=replay.returncode),
            current_tensor_consumers=dict(FP32_small='Original control: regression after diagnostic/MTP lease source integration',
                FP16_small='fp16-auto: same integration regression; host/mixed duplicates retired',
                long='natural0: selected-position natural-history comparison; candidate duplicate retired'),
            removed=[],removed_logical_bytes=0,removed_allocated_bytes=0,physical_free_space_delta_measured=False)
        assert len({str(f) for f,_ in plans})==len(plans)
        pinned=[dict(path=str(f),bytes=f.stat().st_size,allocated_bytes=f.stat().st_blocks*512,
                     sha256=sha(f),reason=reason) for f,reason in plans]
        assert all(str(Path(v['path']).resolve()).startswith(str(B)+'/') or
                   str(Path(v['path']).resolve()).startswith(str(proposal/'files')+'/') for v in pinned)
        def persist():dest.write_text(json.dumps(manifest,indent=2)+'\n')
        persist()
        for item in pinned:
            f=Path(item['path']);assert sha(f)==item['sha256'] and f.stat().st_size==item['bytes']
            f.unlink();manifest['removed'].append(item)
            manifest['removed_logical_bytes']+=item['bytes'];manifest['removed_allocated_bytes']+=item['allocated_bytes'];persist()
        assert all(sha(Path(f))==pin for f,pin in originals.items())
        assert all(sha(Path(f))==pin for f,pin in manifest['retained_evidence'].items())
        manifest['complete']=True;persist()
        print(json.dumps(dict(complete=True,files=len(pinned),logical_bytes=manifest['removed_logical_bytes'],
            allocated_bytes=manifest['removed_allocated_bytes'],original_results_unchanged=True)))


if __name__=='__main__':main()
