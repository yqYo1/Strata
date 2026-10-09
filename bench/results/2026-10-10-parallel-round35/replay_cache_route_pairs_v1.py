"""CPU-only closed-v6 static coverage replay. Never import controller top-level.

Counts reproduce captured evaluated routes; no ordering/LRU/cost/timing oracle.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

CONTROLLER_SHA='9304f84dfce23b099c02d0090ea3b8f9b158100d157b0175ab3e4cfda1a9f423'
PARSER_SHA='2ff44099f7385054dc5663f04a8e226d364d4d349a84b2493bc2bb1d6f3585cd'
SHAPE_SHA='0355f450b1c0b3b4dc59ff28db09c5be48ea914e40808f94daa822b38b5dc2ff'
PROFILE_SHA='8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf'
METADATA_SHA='d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d'
MAX_BYTES=64*1024**2


def require(ok,message):
    if not ok: raise ValueError(message)


def pin(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value),'exact SHA256 required')
    return value


def identity(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_uid,info.st_nlink)


def bounded_file(path,limit=MAX_BYTES):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_nlink==1,
                'reader requires owned regular single-link file')
        require(0<=before.st_size<=limit,'file byte budget')
        chunks=[];left=before.st_size
        while left:
            chunk=os.read(fd,min(left,65536));require(chunk,'short read')
            chunks.append(chunk);left-=len(chunk)
        require(not os.read(fd,1),'file grew during read')
        require(identity(before)==identity(os.fstat(fd)),'file changed during read')
        return b''.join(chunks)
    finally: os.close(fd)


def verified_file(path,sha,limit=MAX_BYTES):
    data=bounded_file(path,limit)
    require(hashlib.sha256(data).hexdigest()==pin(sha),'file SHA256 mismatch')
    return data


def module(name,path,sha):
    source=verified_file(path,sha,131072)
    result=type(sys)(name)
    exec(compile(source,str(path),'exec'),result.__dict__)
    return result


def histogram_parser(controller_path):
    source=verified_file(controller_path,CONTROLLER_SHA,262144)
    tree=ast.parse(source)
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id.startswith('HISTOGRAM_'):
            selected.append(node)
        if isinstance(node,ast.FunctionDef) and node.name=='parse_histogram': selected.append(node)
    require(sum(isinstance(n,ast.FunctionDef) for n in selected)==1,'histogram parser extraction')
    scope={'re':re}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(controller_path),'exec'),scope)
    return scope['parse_histogram']


def closed_gate(record):
    require(record.get('active') is False,'receipt active/unknown')
    for key in ('completed','healthy','math_gate_passed','pair_gate_passed','histogram_gate_passed',
                'pair_shape_gate_passed','protocol_boundary_gate_passed','text_budget_gate_passed'):
        require(record.get(key) is True,'closed success gate: '+key)
    require(record.get('exit_code')==0 and record.get('exit_signal') is None and not record.get('error') and
            not record.get('new_fault_messages') and not any(record.get('cleanup',{}).values()),'exit/fault/cleanup gate')
    require(record.get('controller_sha256')==CONTROLLER_SHA and record.get('pair_parser_sha256')==PARSER_SHA and
            record.get('shape_helper_sha256')==SHAPE_SHA,'source identity gate')
    require(record.get('C_original_math_gate_passed') is False and record.get('D_original_healthy') is False and
            record.get('D_original_full_math_passed') is False and record.get('prior_r5_original_math_passed') is False and
            record.get('prior_r5_original_healthy') is False,'preserve prior failures')
    require(record.get('performance_eligible') is False and record.get('full_lifecycle_passed') is False and record.get('adopted') is False,'diagnostic qualification boundary')
    require(len(record.get('requests',[]))==4,'four request gate')


def canonical_hash(pair_shapes):
    return hashlib.sha256((json.dumps(pair_shapes,separators=(',',':'))+'\n').encode('ascii')).hexdigest()


def owned_frames(raw,intervals):
    require(len(raw)<=MAX_BYTES and raw.endswith(b'\n'),'raw stderr byte/newline bound')
    previous=0
    for begin,end in intervals:
        require(type(begin) is int and type(end) is int and 0<=previous<=begin<end<=len(raw),'interval bounds/order')
        previous=end
    offset=0;markers=0
    for line in raw.splitlines(keepends=True):
        end=offset+len(line)
        require(len(line)<=65536 and line.endswith(b'\n'),'raw stderr line bound')
        if b'CACHE_ROUTE_PAIRS_V1' in line:
            require(line.startswith(b'CACHE_ROUTE_PAIRS_V1 '),'misplaced frame prefix')
            require(sum(begin<=offset<end<=stop for begin,stop in intervals)==1,'orphan/extra frame outside request')
            markers+=1
        offset=end
    return markers


def verified_interval(raw, begin, end, expected_sha):
    require(type(begin) is int and type(end) is int and 0<=begin<end<=len(raw)<=MAX_BYTES,'interval bounds/order')
    data=raw[begin:end]
    require(hashlib.sha256(data).hexdigest()==pin(expected_sha),'interval SHA256 mismatch')
    return data


def coverage(parsed,residents,formats):
    covered_entries=0;covered_callbacks=0;cells={}
    for p in parsed['pairs']:
        if (p['layer'],p['expert']) not in residents: continue
        n,j=p['entries'],p['callback_pairs'];covered_entries+=n;covered_callbacks+=j
        # Shape helper has already admitted J<=N<=2J for all pairs.
        for phase,typ,rows in (('GU',formats[p['layer']][0],640),('Down',formats[p['layer']][1],2560)):
            for nt,jobs in ((1,2*j-n),(2,n-j)):
                if jobs:
                    values=cells.setdefault((phase,typ,nt),[0,0,0])
                    for i,value in enumerate((jobs,jobs*nt,jobs*rows)): values[i]+=value
    return {'covered_entries':covered_entries,'covered_callback_pairs':covered_callbacks,
            'cells':[dict(phase=k[0],type=k[1],nt=k[2],experts=v[0],tokens=v[1],output_rows=v[2]) for k,v in sorted(cells.items())]}


def replay(receipt_path,receipt_sha,controller_path):
    require(sys.flags.optimize==0,'assertions must remain enabled')
    record=json.loads(verified_file(receipt_path,receipt_sha,16*1024**2))
    closed_gate(record)
    base=Path(__file__).parent
    parser=module('pinned_route_parser',base/'cache_route_pairs_parser_v1.py',PARSER_SHA)
    shape=module('pinned_route_shapes',base/'cache_route_shapes_v1.py',SHAPE_SHA)
    parse_hist=histogram_parser(controller_path)
    profile=record['profile'];require(profile['sha256']==PROFILE_SHA,'profile identity')
    residents,fnv=parser.static_profile(verified_file(profile['path'],PROFILE_SHA,196632))
    require(len(residents)==128 and fnv==profile['expected_host_res_fnv1a64'],'profile residency identity')
    args=record['argv'];metadata_path=Path(args[args.index('--pack')+1])/'native_experts.txt'
    require(record['pack_metadata_sha256']['native_experts.txt']==METADATA_SHA,'metadata identity')
    formats=shape.formats_from_metadata(verified_file(metadata_path,METADATA_SHA,131072),METADATA_SHA)
    requests=record['requests'];raw_path=Path(requests[0]['cache_route_pairs']['replay_input']['stderr_path'])
    require(raw_path.resolve()==(Path(receipt_path).parent/'debugger/inferior.stderr').resolve(),'raw stderr path binding')
    raw=verified_file(raw_path,record['engine_log_sha256'])
    require(len(raw)==record['engine_log_bytes'],'raw stderr size')
    intervals=[(q['stderr_begin_offset'],q['stderr_end_offset']) for q in requests]
    markers=owned_frames(raw,intervals);expected_markers=0;frame_bytes=0;results=[]
    for index,(request,(begin,end)) in enumerate(zip(requests,intervals)):
        require(request['math_gate_passed'] is True,'request math gate')
        ref=request['cache_route_pairs']['replay_input']
        require(Path(ref['stderr_path']).resolve()==raw_path.resolve() and (ref['begin'],ref['end'])==(begin,end),'interval path/range mismatch')
        require(ref['parser_sha256']==PARSER_SHA and ref['shape_helper_sha256']==SHAPE_SHA and
                ref['resident_profile_sha256']==PROFILE_SHA,'interval provenance')
        interval=verified_interval(raw,begin,end,ref['sha256'])
        done=[line for line in request['protocol'] if line.startswith('DONE ')]
        require(len(done)==1,'one DONE per request')
        parsed=parser.parse_request(interval,index+1,done[0],residents,fnv)
        hist_lines=[line[:-1].decode('ascii') for line in interval.splitlines(keepends=True) if line.startswith(b'strata native dispatch exposure: ')]
        require(len(hist_lines)==1,'one raw native histogram per request')
        histogram=parse_hist(hist_lines[0]);joined=shape.reconcile_shapes(parsed,residents,formats,histogram)
        sha=canonical_hash(joined.pop('pair_shapes'))
        require(sha==request['cache_route_shape_reconciliation']['canonical_pair_shapes_sha256'],'canonical pairshape SHA256 mismatch')
        expected_markers+=len(parsed['pairs'])+3;frame_bytes+=parsed['frame_bytes']
        require(frame_bytes<=16*1024**2,'job frame byte budget')
        results.append(dict(ordinal=index+1,name=request['name'],raw_reference=ref,
                            pair_count=len(parsed['pairs']),canonical_pair_shapes_sha256=sha,
                            observed_CPU_cells=joined['cells'],default_resident_coverage=coverage(parsed,residents,formats)))
    require(markers==expected_markers,'extra frame count')
    return dict(passed=True,receipt_sha256=receipt_sha,controller_sha256=CONTROLLER_SHA,parser_sha256=PARSER_SHA,
                shape_sha256=SHAPE_SHA,profile_sha256=PROFILE_SHA,metadata_sha256=METADATA_SHA,
                raw_stderr_sha256=record['engine_log_sha256'],requests=results,performance_eligible=False,
                scope='fixed captured static default resident coverage counts only; no order/LRU/traffic/service cost/policy benefit')


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--receipt',type=Path,required=True)
    cli.add_argument('--receipt-sha256',required=True) # Future closed v6 digest: deliberately unknown now.
    cli.add_argument('--controller',type=Path,required=True)
    cli.add_argument('--self-sha256',required=True)
    args=cli.parse_args();verified_file(__file__,args.self_sha256,131072)
    print(json.dumps(replay(args.receipt,pin(args.receipt_sha256),args.controller),indent=2))


if __name__=='__main__': main()
