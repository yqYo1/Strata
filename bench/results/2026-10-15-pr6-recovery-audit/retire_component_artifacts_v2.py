#!/usr/bin/env python3
"""Retire only enumerated closed-job logs/timelines and superseded tensors."""
import fcntl
import json
import re
from pathlib import Path
from check_gemm_orientation import P, B, D, sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        dest=D/'closed-artifact-retention-v2.json'
        assert not dest.exists()
        names=[
            'main-primary-gemm-orientation-build-root-v1','main-primary-gemm-orientation-build-root-v2',
            'main-primary-gemm-orientation-validation-root-v1','main-primary-gemm-orientation-microbench-root-v1',
            'main-primary-gemm-orientation-validation-root-v2','main-primary-gemm-orientation-microbench-root-v2',
            'main-primary-pr6-gemm-unitrace-root-v1','main-primary-unitrace-runtime-root-v1',
            'main-primary-unitrace-clock-source-root-v1','main-primary-natural-layout-long-pair-root-v2',
            'main-primary-layer-major-build-root-v1','main-primary-layer-major-small-root-v1']
        originals={}
        plans=[]
        excerpts=[]
        for name in names:
            root=B/name
            record=root/'record.json'
            r=json.loads(record.read_text())
            assert not r['active']
            assert all(c['direct_child_reaped'] and c['session_empty'] for c in r['commands'])
            originals[str(record)]=sha(record)
            for path in sorted((root/'owned').glob('*')):
                if path.is_file() and path.suffix in ['.stdout','.stderr']:
                    assert path.resolve()==path and path.stat().st_size<=64 << 20
                    found=[line for line in path.read_text(errors='replace').splitlines()
                           if re.search(r'(?i)warning|error|\bFAIL\b|cannot|not applicable',line)]
                    assert sum(len(x.encode()) for x in found)<=256 << 10
                    if found: excerpts.append(dict(path=str(path),lines=found))
                    plans.append((path,'Closed compiler/runtime/model text; individual samples, CPU/numeric/fault results, commands and warnings are preserved in structured receipts.'))
        evidence=[D/n for n in ['gemm-orientation-build-v1.json','gemm-orientation-build-v2.json',
            'gemm-orientation-validation-v1.json','gemm-orientation-microbench-v1.json',
            'gemm-orientation-validation-v2.json','gemm-orientation-microbench-v2.json',
            'unitrace-gemm-evidence-v1.json','unitrace-clock-source-v1.json',
            'natural-layout-long-pair-v2.json','natural-layout-long-pair-summary-v2.json',
            'layer-major-build-v1.json','layer-major-build-v2.json','layer-major-small-v1.json']]
        evidence.append(P/'bench/results/2026-10-14-primary-vtune/unitrace-runtime-evidence.json')
        assert all(p.is_file() for p in evidence)
        clock=json.loads((D/'unitrace-clock-source-v1.json').read_text())
        assert clock['passed'] and clock['clock_domain_mismatch_reproduced']
        for path in [B/'main-primary-pr6-gemm-unitrace-root-v1/record.json', B/'main-primary-unitrace-runtime-root-v1/record.json']:
            for t in json.loads(path.read_text())['trace_files']:
                raw=Path(t['path']);assert sha(raw)==t['sha256'] and raw.stat().st_size==t['bytes']
                plans.append((raw,'Complete internal event sequence verified; compact evidence retained. CPU reproduction resolves the RAW/BOOT source-clock question. Original external UTC failure stays false; no historical timestamp correction.'))
        natural=json.loads((D/'natural-layout-long-pair-v2.json').read_text())
        assert natural['passed'] and natural['logits_bitwise_equal'] and natural['sampled_residuals_bitwise_equal']
        for left,right in zip(natural['arms'][0]['fixtures'],natural['arms'][1]['fixtures']):
            assert left['sha256']==right['sha256'] and sha(Path(right['path']))==right['sha256']
            plans.append((Path(left['path']),'Bitwise duplicate old-control tensor; sole current long comparator consumes natural0 arm, retained with matching hash.'))
        small=json.loads((D/'layer-major-small-v1.json').read_text())
        assert small['passed'] and small['complete'] and not small['active']
        for arm in small['arms'][1:]:
            assert arm['logits_bitwise_equal'] and arm['residuals_bitwise_equal']
            for candidate,control in zip(arm['fixtures'],small['arms'][0]['fixtures']):
                assert candidate['sha256']==control['sha256'] and sha(Path(control['path']))==control['sha256']
                plans.append((Path(candidate['path']),'Closed same-input comparison; bitwise equivalent control small fixtures remain for future dispatcher/regression consumer.'))
        assert len({str(p) for p,_ in plans})==len(plans)
        manifest=dict(complete=False,recipe_sha256=sha(Path(__file__)),original_receipts=originals,
            retained_evidence={str(p):sha(p) for p in evidence},diagnostic_excerpts=excerpts,
            current_tensor_consumers=dict(long='natural0 arm: first logits+541sampled residuals for upcoming layer-major comparison',
                small='control arm: first logits+8sampled residuals for dispatcher regression; candidateGPU/host duplicates retired'),
            deletion_status_preserves_original_outcomes=True,external_UTC_failure_unchanged=True,
            removed=[],removed_logical_bytes=0,removed_allocated_bytes=0,physical_free_space_delta_measured=False)

        def persist():
            dest.write_text(json.dumps(manifest,indent=2)+'\n')

        # Validate the whole plan before the first unlink. Never touch an active
        # fixture or original result record; only the enumerated files qualify.
        pinned=[dict(path=str(p),bytes=p.stat().st_size,allocated_bytes=p.stat().st_blocks*512,
                     sha256=sha(p),reason=reason) for p,reason in plans]
        assert all(str(Path(v['path']).resolve()).startswith(str(B)+'/') for v in pinned)
        assert all(sha(Path(p))==h for p,h in originals.items())
        persist()
        for item in pinned:
            path=Path(item['path'])
            assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes']
            path.unlink()
            manifest['removed'].append(item)
            manifest['removed_logical_bytes']+=item['bytes']
            manifest['removed_allocated_bytes']+=item['allocated_bytes']
            persist()
        assert all(sha(Path(p))==h for p,h in originals.items())
        assert all(sha(Path(p))==h for p,h in manifest['retained_evidence'].items())
        manifest['complete']=True;persist()
        print(json.dumps(dict(complete=True,files=len(pinned),logical_bytes=manifest['removed_logical_bytes'],
            allocated_bytes=manifest['removed_allocated_bytes'],original_results_unchanged=True)))


if __name__=='__main__':
    main()
