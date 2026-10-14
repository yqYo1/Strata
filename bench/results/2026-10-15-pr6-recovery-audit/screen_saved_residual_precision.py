#!/usr/bin/env python3
"""CPU post-hoc representation screen of saved final residuals, not model quality."""
import fcntl
import json
from pathlib import Path
import numpy as np
from check_gemm_orientation import P, B, D, sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        ref=D/'natural-layout-long-pair-v2.json'
        r=json.loads(ref.read_text())
        gate=json.loads((D/'layer-major-long-v1.json').read_text())
        assert r['passed'] and r['complete'] and not r['active']
        assert gate['passed'] and gate['complete'] and not gate['active']
        fixture=r['arms'][1]['fixtures'][1]
        path=Path(fixture['path'])
        assert path.stat().st_size==fixture['bytes'] and path.stat().st_size<=32 << 20
        assert sha(path)==fixture['sha256']
        fmt=np.dtype([('pos','<i8'),('r','<f4',(10240,))],align=False)
        assert fmt.itemsize==40968
        frames=np.frombuffer(path.read_bytes(),dtype=fmt)
        assert len(frames)==541 and frames['pos'].tolist()==r['arms'][1]['residual_positions']
        values=frames['r'].copy()
        assert np.isfinite(values).all()
        bits=values.view(np.uint32)
        bf_bits=(bits+np.uint32(0x7fff)+((bits>>16)&np.uint32(1)))&np.uint32(0xffff0000)
        formats={'fp16':values.astype(np.float16).astype(np.float32),'bf16':bf_bits.view(np.float32)}
        source_norm=float(np.linalg.norm(values.astype(np.float64)))
        summaries={};errors={}
        for name,roundtrip in formats.items():
            delta=roundtrip.astype(np.float64)-values
            abs_error=np.abs(delta)
            errors[name]=abs_error
            summaries[name]=dict(finite=bool(np.isfinite(roundtrip).all()),
                nonfinite_after_cast=int(np.count_nonzero(~np.isfinite(roundtrip))),
                underflow_to_zero=int(np.count_nonzero((values!=0)&(roundtrip==0))),
                exactly_equal_scalars=int(np.count_nonzero(values==roundtrip)),
                relative_L2_error=float(np.linalg.norm(delta))/source_norm,
                max_absolute_error=float(np.max(abs_error)),
                p50_p99_p999_absolute_error=np.quantile(abs_error,[.5,.99,.999]).tolist())
        result=dict(passed=True,GPU_executed=False,recipe_sha256=sha(Path(__file__)),
            reference_receipt_sha256=sha(ref),fixture=fixture,numpy_version=np.__version__,
            positions=len(frames),scalars=int(values.size),
            minimum=float(values.min()),maximum=float(values.max()),maximum_absolute=float(np.abs(values).max()),
            float32_zeros=int(np.count_nonzero(values==0)),representations=summaries,
            per_position_columns=['position','source_max_abs','fp16_max_abs_error','bf16_max_abs_error'],
            per_position=[[int(p),float(np.max(np.abs(values[i]))),float(errors['fp16'][i].max()),
                           float(errors['bf16'][i].max())] for i,p in enumerate(frames['pos'])],
            actual_device_conversion_qualified=False,model_precision_qualified=False,quality_qualified=False,
            physical_262144_qualified=False,adopted=False,
            scope='Final-layer residual samples at541positions only. NumPy IEEE FP16 and bitwise RNE BF16 post-hoc round-trip. No48-layer accumulated rounding, GPU cast, retrieval, NLL/task quality or speed measurement; cannot adopt from these errors alone.')
        dest=D/'saved-residual-precision-screen-v1.json';assert not dest.exists()
        dest.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,positions=541,maximum_absolute=result['maximum_absolute'],
            representations=summaries,quality_qualified=False)),flush=True)


if __name__=='__main__':
    main()
