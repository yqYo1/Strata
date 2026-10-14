#!/usr/bin/env python3
"""Offline same-dtype divergence analysis; leaves original failed gate unchanged."""
import array
import fcntl
import json
import math
import struct
from check_gemm_orientation import B, D, sha
from check_layer_major_storage import floats, logits_difference
from pathlib import Path


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        ref_path=D/'selected-prefill-long-v1.json'
        cand_path=D/'mtp-expert-release-long-v1.json'
        ref=json.loads(ref_path.read_text());cand=json.loads(cand_path.read_text())
        assert ref['passed'] and ref['complete'] and not ref['active']
        assert not cand['passed'] and not cand['complete'] and not cand['active']
        assert not cand['fresh_faults'] and cand['boot_unchanged'] and not cand['driver_dump_exists']
        assert all(c['observation_complete'] and c['direct_child_reaped'] and c['session_empty'] and c['exit_code']==0 for c in cand['commands'])
        reference=next(a for a in ref['arms'] if a['dtype']=='fp32')
        candidate=cand['arms'][0];assert candidate['dtype']=='fp32'
        def file_for(arm,ending):
            files=[x for x in arm['fixtures'] if x['path'].endswith(ending)]
            assert len(files)==1
            x=files[0];p=Path(x['path']);assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes']
            return p
        rp=file_for(reference,'/selected-logits.bin');cp=file_for(candidate,'/selected-logits.bin')
        rr=rp.read_bytes();cr=cp.read_bytes();v,n=struct.unpack_from('ii',rr)
        assert rr[:8]==cr[:8] and len(rr)==len(cr)==8+v*n*4
        result=dict(original_gate_still_failed=True,reference_receipt_sha256=sha(ref_path),
                    candidate_receipt_sha256=sha(cand_path),recipe_sha256=sha(Path(__file__)),
                    comparator_sha256=sha(D/'check_layer_major_storage.py'),positions=[],
                    input_tokens=ref['actual_input_tokens'],dtype='fp32',performance_qualified=False,
                    scope='Offline comparison of the first failed long same-dtype gate; no GPU execution or tolerance/adoption change.')
        for i,p in enumerate(ref['positions']):
            rbytes=rr[8+i*v*4:8+(i+1)*v*4];cbytes=cr[8+i*v*4:8+(i+1)*v*4]
            r=array.array('f');r.frombytes(rbytes);c=array.array('f');c.frombytes(cbytes)
            comparison=logits_difference(r,c)
            comparison.update(position=p,bitwise_equal=rbytes==cbytes,
                              finite=all(math.isfinite(x) for x in c),
                              unequal_floats=sum(x!=y for x,y in zip(r,c)))
            result['positions'].append(comparison)
            print(p,comparison,flush=True)
        result['first_logits']=logits_difference(floats(file_for(reference,'/first-logits.bin')),
                                                 floats(file_for(candidate,'/first-logits.bin')))
        (D/'mtp-expert-release-long-divergence-v1.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
