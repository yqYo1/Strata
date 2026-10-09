"""Frozen r6 entry-count prefix analysis; CPU-only, no policy/config writes.

Fit request 1 A-read0 only. B and repeated requests are not independent holdout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import types

READER_SHA = 'b65dcabbc75614ef06209782c9ebe9b9540f4e9f9b66d6056c47b58153e54f0f'
DP_SHA = '6025cbef7718d3f26f086f7de03cd65370b6edf4dce511c434357bc6c79f0220'
RECEIPT_SHA = 'f606f5a1f2d00dba240213a7feeb3cc84c9da4f1487075d1e97c336a558e4c89'
HELPER_ROOT = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEFAULT = [2]*47+[34]
U64 = (1 << 64)-1


def require(ok, message):
    if not ok: raise ValueError(message)


def pinned_source(path, expected):
    """Bootstrap source read: bounded owned no-follow file, stable open identity."""
    require(type(expected) is str and len(expected)==64 and all(c in '0123456789abcdef' for c in expected), 'exact SHA256 required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_nlink==1 and 0<=before.st_size<=131072,'source file admission')
        data=bytearray()
        while len(data)<before.st_size:
            chunk=os.read(fd,min(65536,before.st_size-len(data)))
            require(chunk,'source short read');data.extend(chunk)
        after=os.fstat(fd)
        fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_uid','st_nlink')
        require(not os.read(fd,1) and all(getattr(before,k)==getattr(after,k) for k in fields),'source changed during read')
        require(hashlib.sha256(data).hexdigest()==expected,'source SHA256 mismatch')
        return bytes(data)
    finally: os.close(fd)


def load_pinned(name,path,sha):
    source=pinned_source(path,sha)
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module


def cleanup_gate(record):
    cleanup=record.get('cleanup')
    require(type(cleanup) is dict,'explicit cleanup object required')
    for key in ('forced','inferior_survived','gdb_survived'):
        require(key in cleanup and cleanup[key] is False,'explicit false cleanup: '+key)


def prefix_curves(ranked,pairs,layers=48,experts=512,cap=128):
    """Keep immutable within-layer profile order, including unseen zero counts.

    Generic bounded geometry serves literal CPU fixtures. Production caller pins
    48x512 and checks the complete STRP inverse using the existing parser first.
    """
    for value,upper in ((layers,48),(experts,512),(cap,128)):
        require(type(value) is int and 1<=value<=upper,'curve geometry')
    require(type(ranked) in (list,tuple) and len(ranked)==layers*experts,'complete rank geometry')
    order=[[] for _ in range(layers)];seen=set()
    for pair in ranked:
        require(type(pair) in (tuple,list) and len(pair)==2,'profile pair shape')
        l,e=pair
        require(type(l) is int and type(e) is int and 0<=l<layers and 0<=e<experts,'profile pair bounds')
        require((l,e) not in seen,'duplicate profile pair');seen.add((l,e));order[l].append(e)
    require(type(pairs) in (list,tuple) and len(pairs)<=layers*experts,'count geometry')
    counts={}
    for p in pairs:
        require(type(p) is dict,'count record')
        l,e,n=p.get('layer'),p.get('expert'),p.get('entries')
        require(type(l) is int and type(e) is int and 0<=l<layers and 0<=e<experts,'count pair bounds')
        require(type(n) is int and 0<=n<=U64,'uint64 entry count')
        require((l,e) not in counts,'duplicate count pair');counts[l,e]=n
    curves=[]
    for l,ids in enumerate(order):
        curve=[0]
        for e in ids[:cap]:
            n=counts.get((l,e),0)
            require(n<=U64-curve[-1],'uint64 prefix overflow');curve.append(curve[-1]+n)
        curves.append(curve)
    return curves


def curves_hash(curves):
    return hashlib.sha256((json.dumps(curves,separators=(',',':'))+'\n').encode('ascii')).hexdigest()


def analyze(receipt,receipt_sha,controller,helper_root,dp_path):
    require(Path(helper_root)==HELPER_ROOT,'exact helper root required')
    require(receipt_sha==RECEIPT_SHA,'exact closed r6 receipt required')
    reader=load_pinned('frozen_route_replay',HELPER_ROOT/'replay_cache_route_pairs_v1.py',READER_SHA)
    dp=load_pinned('frozen_prefix_dp',dp_path,DP_SHA)
    record=json.loads(reader.verified_file(receipt,RECEIPT_SHA,16*1024**2))
    cleanup_gate(record)
    replay=reader.replay(receipt,RECEIPT_SHA,controller)
    parser=reader.module('frozen_route_parser',HELPER_ROOT/'cache_route_pairs_parser_v1.py',reader.PARSER_SHA)
    profile=record['profile']
    profile_bytes=reader.verified_file(profile['path'],reader.PROFILE_SHA,196632)
    residents,fnv=parser.static_profile(profile_bytes)
    ranked=list(struct.iter_unpack('<HH',profile_bytes[24:24+24576*4]))
    raw=reader.verified_file(record['requests'][0]['cache_route_pairs']['replay_input']['stderr_path'],record['engine_log_sha256'])
    fit=None;candidate=None;results=[];fit_hash=None
    names=('A-read0','B-read1','A-read2','B-read3')
    for index,request in enumerate(record['requests']):
        require(request['name']==names[index],'frozen request ordering')
        ref=request['cache_route_pairs']['replay_input']
        interval=reader.verified_interval(raw,ref['begin'],ref['end'],ref['sha256'])
        done=[line for line in request['protocol'] if line.startswith('DONE ')]
        require(len(done)==1,'one DONE')
        parsed=parser.parse_request(interval,index+1,done[0],residents,fnv)
        curves=prefix_curves(ranked,parsed['pairs'])
        default_score=dp.score_prefix_quota(curves,DEFAULT)
        require(default_score==parsed['end']['hits']==replay['requests'][index]['default_resident_coverage']['covered_entries'],'default entry coverage reconciliation')
        if index==0:
            fit=dp.select_prefix_quotas(curves,128,DEFAULT)
            candidate=fit['quotas'];fit_hash=curves_hash(curves)
        results.append(dict(ordinal=index+1,name=request['name'],candidate_entries=dp.score_prefix_quota(curves,candidate),default_entries=default_score,total_entries=parsed['end']['entries'],curves_sha256=curves_hash(curves),raw_reference=ref,canonical_pair_shapes_sha256=replay['requests'][index]['canonical_pair_shapes_sha256']))
    return dict(schema='static-prefix-entry-trace-v1',objective='routed entries N',fit_request=dict(ordinal=1,name='A-read0'),uniform_MAXBLOB_slots=128,default_quotas=DEFAULT,fit=fit,fit_curves_sha256=fit_hash,requests=results,sources=dict(adapter_sha256=None,reader_sha256=READER_SHA,dp_sha256=DP_SHA,controller_sha256=reader.CONTROLLER_SHA,parser_sha256=reader.PARSER_SHA,shape_sha256=reader.SHAPE_SHA),inputs=dict(receipt_sha256=RECEIPT_SHA,profile_sha256=reader.PROFILE_SHA,metadata_sha256=reader.METADATA_SHA,raw_stderr_sha256=record['engine_log_sha256']),adopted=False,performance_eligible=False,full_lifecycle_passed=False,limitations=['Captured default-route entry counts only; changed residency may change future routes.','B is a different diagnostic fixture, not certified independent holdout; repetitions are not independent samples.','No LRU order, traffic, service cost, timing, packed-byte allocation or live-policy qualification.','Receipt/source hashes bind existing evidence; no reopening head/state/model payloads or full-payload validation.'])


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--self-sha256',required=True)
    cli.add_argument('--helper-root',type=Path,required=True)
    cli.add_argument('--receipt',type=Path,required=True)
    cli.add_argument('--receipt-sha256',required=True)
    cli.add_argument('--controller',type=Path,required=True)
    args=cli.parse_args()
    pinned_source(__file__,args.self_sha256)
    result=analyze(args.receipt,args.receipt_sha256,args.controller,args.helper_root,Path(__file__).with_name('static_cache_quota.py'))
    result['sources']['adapter_sha256']=args.self_sha256
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
