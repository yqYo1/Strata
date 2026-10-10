"""Root-run bounded CPU receipt checker; reads an already closed qualifier log."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('case',choices=('valid32','valid64','valid262','collector-bound','off',
                                'unsupported','missing-product','unrouted-copy','unknown-branch'))
    p.add_argument('stderr',type=Path)
    args=p.parse_args()
    with args.stderr.open('rb') as f: raw=f.read((32<<20)+1)
    assert len(raw)<=32<<20
    rows=[json.loads(v) for v in raw.decode().splitlines()]
    if args.case=='off':
        assert rows==[]
        print(json.dumps(dict(passed=True,case=args.case)));return
    receipts=[r for r in rows if r['kind']=='prefill_route_census_receipt']
    layers=[r for r in rows if r['kind']=='prefill_route_layer']
    assert len(receipts)==1 and len(rows)==len(layers)+1
    receipt=receipts[0]
    if args.case.startswith('valid') or args.case=='collector-bound':
        n={'valid32':32767,'valid64':65535,'valid262':262143,'collector-bound':262144}[args.case]
        chunks=(n+8191)//8192
        assert receipt['valid'] is True and receipt['requested_tokens']==n
        assert receipt['chunks']==chunks and receipt['layers']==chunks*48
        assert receipt['capacity_join'] is False and receipt['device_commands_added']==0
        assert len(layers)==chunks*48
        for ci in range(chunks):
            subset=layers[ci*48:(ci+1)*48]
            T=min(8192,n-ci*8192)
            assert [r['layer'] for r in subset]==list(range(48))
            for r in subset:
                assert r['chunk_p0']==ci*8192 and r['T']==T and r['K']==10
                assert r['stream_all'] is False and r['peer_calls']==r['peer_rows']==0
                assert r['calls']==512 and r['rows']==10*T
                assert r['generic_GU_calls']==r['generic_Down_calls']==512
                assert r['transfer_calls']==r['routed_transfer_calls']==384
                assert r['transfer_bytes']==384*1971200
                assert sum(v['calls'] for v in r['bins'])==512
                assert sum(v['routed_rows'] for v in r['bins'])==10*T
                assert sum(v['calls'] for v in r['bins'] if v['source']!=1)==384
        for total,key in [('calls','calls'),('routed_rows','rows'),('transfer_calls','transfer_calls'),
                          ('transfer_bytes','transfer_bytes'),('routed_transfer_calls','routed_transfer_calls')]:
            assert receipt[total]==sum(r[key] for r in layers)
        assert receipt['routed_rows']==n*10*48
    else:
        assert receipt['valid'] is False and layers==[]
        assert all(receipt[k]==0 for k in ('calls','routed_rows','transfer_calls','transfer_bytes',
                                          'routed_transfer_calls','output_bytes'))
        assert receipt['capacity_join'] is False
        expected={'unsupported':'unsupported_multi_gpu_or_pipeline',
                  'missing-product':'call_pair_or_transfer_reconciliation',
                  'unrouted-copy':'transfer_extent_or_unrouted_copy',
                  'unknown-branch':'layer_geometry_or_unknown_branch'}
        assert receipt['reason']==expected[args.case]
    print(json.dumps(dict(passed=True,case=args.case,gpu_executed=False)))

if __name__=='__main__': main()
