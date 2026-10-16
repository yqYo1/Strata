#!/usr/bin/env python3
"""Exact offline comparison of closed layer-boundary captures; no GPU work."""
import array
import fcntl
import json
import math
import sys
from pathlib import Path
from check_gemm_orientation import B,D,sha

def main():
    assert sys.byteorder=='little'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inputs={name:D/('boundary-audit-'+name+'-v1.json') for name in ['long-fixed','long-auto']}
        records={name:json.loads(path.read_text()) for name,path in inputs.items()}
        for record in records.values():
            assert record['complete'] and record['localization_complete'] and not record['active']
            assert not record['fresh_faults'] and not record['driver_dump_exists'] and record['boot_unchanged']
            assert all(c['direct_child_reaped'] and c['session_empty'] and c['observation_complete'] for c in record['commands'])
        fixed=records['long-fixed'];auto=records['long-auto']
        assert fixed['passed'] and fixed['binary_sha256']==auto['binary_sha256']
        assert fixed['source_pins']==auto['source_pins'] and fixed['tokens_sha256']==auto['tokens_sha256']
        a=fixed['arms'][0];b=auto['arms'][0]
        assert a['dtype']==b['dtype']=='fp32' and a['audit_on'] and b['audit_on']
        assert a['boundary']['copy_equal'] and b['boundary']['copy_equal']
        assert a['boundary']['output_order']==b['boundary']['output_order']
        for arm in [a,b]:
            info=arm['boundary'];assert sha(Path(info['raw_path']))==info['raw_sha256']
        row_bytes=10240*4
        layers=[dict(layer=l,equal_rows=0,different_rows=[],max_abs=0.) for l in range(48)]
        first=None
        with Path(a['boundary']['raw_path']).open('rb') as af,Path(b['boundary']['raw_path']).open('rb') as bf:
            for layer,row in a['boundary']['output_order']:
                x=af.read(row_bytes);y=bf.read(row_bytes);assert len(x)==len(y)==row_bytes
                out=layers[layer]
                if x==y:
                    out['equal_rows']+=1;continue
                xa=array.array('f');xa.frombytes(x);ya=array.array('f');ya.frombytes(y)
                assert all(math.isfinite(v) for v in xa) and all(math.isfinite(v) for v in ya)
                maximum=max(abs(float(u)-float(v)) for u,v in zip(xa,ya))
                out['different_rows'].append(row);out['max_abs']=max(out['max_abs'],maximum)
                if first is None:
                    byte=next(i for i,(u,v) in enumerate(zip(x,y)) if u!=v)
                    first=dict(layer=layer,relative_row=row,first_different_byte=byte,
                        float_index=byte//4,fixed_value=xa[byte//4],auto_value=ya[byte//4],max_abs=maximum)
            assert af.read(1)==bf.read(1)==b''
        old=json.loads((D/'mtp-expert-release-long-v1.json').read_text())
        assert old['arms'][0]['dtype']=='fp32' and not old['passed']
        result=dict(complete=True,GPU_executed=False,recipe_sha256=sha(Path(__file__)),
            receipts={name:sha(path) for name,path in inputs.items()},
            original_auto_fail_receipt_sha256=sha(D/'mtp-expert-release-long-v1.json'),
            binary_sha256=fixed['binary_sha256'],input_sha256=fixed['tokens_sha256'],
            fixed_gpu_rows=a['residual_plan']['gpu_rows'],auto_gpu_rows=b['residual_plan']['gpu_rows'],
            same_run_source_store_input_equal=True,all_selected_layer_rows_equal=first is None,
            first_selected_difference=first,layers=layers,
            auto_selected_equals_original_non_audited_fail=(b['selected_logits_sha256']==old['arms'][0]['selected_logits_sha256']),
            auto_selected_equals_old_control=b['same_dtype_selected_logits_equal'],
            raw_bytes_each=a['boundary']['raw_bytes'],
            scope='Selected19rows across48 layer outputs, with same-run exact copies. Not first-all-token localization; added synchronization can mask races. Original FAIL unchanged.',
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False)
        target=D/'boundary-audit-long-pair-v1.json';assert not target.exists()
        target.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(complete=True,first_selected_difference=first,
            auto_selected_equals_original_non_audited_fail=result['auto_selected_equals_original_non_audited_fail'])),flush=True)

if __name__=='__main__':main()
