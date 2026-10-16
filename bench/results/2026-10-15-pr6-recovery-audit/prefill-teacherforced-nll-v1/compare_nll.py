#!/usr/bin/env python3
"""SOURCE ONLY. Strict paired initial-batched-prefill NLL consumer; no quality threshold."""
import argparse, hashlib, json, math, pathlib, struct, sys

def read(path):
    p=pathlib.Path(path)
    # Exact hard bounded reads; reject even an oversized metadata or raw fixture.
    raw=p.read_bytes() if p.stat().st_size <= 4*1024*1024 else None
    mp=pathlib.Path(str(p)+'.meta')
    if mp.stat().st_size>4096 or raw is None: raise ValueError('budget exceeded')
    lines=mp.read_text().splitlines()
    if not lines or lines[0]!='strata-prefill-nll-v1': raise ValueError('metadata magic')
    meta={}
    for line in lines[1:]:
        k,v=line.split('=',1)
        if k in meta: raise ValueError('duplicate metadata')
        meta[k]=v
    if meta.get('complete')!='true' or meta.get('scope')!='initial-full-batched-prefill': raise ValueError('incomplete/wrong scope')
    magic,n,vocab,written=struct.unpack_from('=4i',raw)
    if magic!=0x314c4c4e or not 2<=n<=262144 or vocab<=0 or written!=n-1: raise ValueError('header')
    if len(raw)!=16+4*n+8*written: raise ValueError('dimensions/size')
    if any(int(meta[k])!=written for k in ('expected_rows','written_rows','covered_rows')): raise ValueError('coverage')
    if int(meta['raw_bytes'])!=len(raw) or int(meta['vocab'])!=vocab or int(meta['input_tokens'])!=n: raise ValueError('metadata dimensions')
    token_bytes=raw[16:16+4*n]; tokens=struct.unpack('='+str(n)+'i',token_bytes)
    if any(t<0 or t>=vocab for t in tokens): raise ValueError('invalid input ID')
    values=[]
    for i,(gold,nll) in enumerate(struct.iter_unpack('=if',raw[16+4*n:])):
        if gold!=tokens[i+1] or not math.isfinite(nll) or nll<0: raise ValueError('gold/NLL record')
        values.append(nll)
    return {'raw_sha256':hashlib.sha256(raw).hexdigest(),'token_sha256_native_i32':hashlib.sha256(token_bytes).hexdigest(),'tokens':token_bytes,'nll':values,'meta':meta}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('reference'); parser.add_argument('candidate'); args=parser.parse_args()
    a,b=read(args.reference),read(args.candidate)
    if a['tokens']!=b['tokens']: raise ValueError('input identity mismatch')
    for k in ('vocab','head','head_ggml_type','scope'):
        if a['meta'][k]!=b['meta'][k]: raise ValueError('incompatible '+k)
    av,bv=a['nll'],b['nll']; count=len(av)
    delta=[y-x for x,y in zip(av,bv)]; mean=math.fsum(delta)/count
    print(json.dumps({'status':'observation-only','quality_qualified':False,'physical_262144_qualified':False,
      'restore_qualified':False,'predictions':count,'input_tokens':count+1,
      'input_token_sha256_native_i32':a['token_sha256_native_i32'],
      'reference_raw_sha256':a['raw_sha256'],'candidate_raw_sha256':b['raw_sha256'],
      'mean_nll_reference':math.fsum(av)/count,'mean_nll_candidate':math.fsum(bv)/count,
      'mean_nll_delta':mean,'min_nll_delta':min(delta),'max_nll_delta':max(delta),
      'relative_ppl_change':math.expm1(mean) if mean<700 else None,
      'required_external_pins':'recipe/source/binary/model/input/env/chunk/residual dtype and placement/MTP/driver/fault and retirement receipt'},indent=2,allow_nan=False))
if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError,struct.error) as e: sys.exit(str(e))
