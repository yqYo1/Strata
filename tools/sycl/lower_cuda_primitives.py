#!/usr/bin/env python3
"""Lower complete pinned native state/attention primitives using real streams."""
import argparse
import hashlib
import json
from pathlib import Path
from lower_cuda_products import lower

SOURCES=['native_gr_norm','native_gr_postops','native_moe','native_gdn',
         'native_gdn_preprocess','native_qsa','native_router','native_rope',
         'native_ple_postops','native_flash_attn','native_qsa_indexer','native_qsa_score']

def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or path.read_text()!=text:path.write_text(text)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ggml',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'docs/sycl-audit/source-inventory.json').read_text())
    hashes={x['path']:x['upstream_sha256'] for x in manifest['files']}
    def checked(path):
        data=(root/path).read_bytes();assert hashlib.sha256(data).hexdigest()==hashes[path],path
        return data.decode()
    records={}
    for name in SOURCES:
        path='src/kernels/cuda/'+name+'.cu';original=checked(path);text=original
        changes=[]
        if name=='native_router':
            # Seven complete CUDA warps exit before this barrier. Only warp
            # zero remains; its registers need no cross-warp communication.
            # SYCL requires a workgroup barrier to be reached by every member.
            assert text.count('__syncthreads();')==1
            text=text.replace('__syncthreads();','__syncwarp();')
            changes.append('Exited CUDA warps: sole surviving router warp uses native subgroup barrier.')
        if name in ['native_gr_norm','native_ple_postops']:
            # Runtime-width and constant-width normalization disagreed by one
            # ULP between original single/batch PLE paths. Materialize the same
            # correctly rounded F32 mean at both original division sites.
            divisor='n_cols' if name=='native_gr_norm' else 'N'
            expr='const float mean = partial / '+divisor+';'
            assert text.count(expr)==1
            text=text.replace(expr,'const float mean = __fdiv_rn(partial, float('+divisor+'));')
            changes.append('RMS mean division uses explicit F32 RN with original operands; single/batch PLE bits are checked.')
        if name in ['native_rope','native_qsa_indexer']:
            # Standard GPU powf(base,1) moved two ULPs on B570 and the
            # ensuing angle caused an independently measured rotation error.
            # Bind the same F32 operands to Intel's high-accuracy device pow.
            expr='powf(theta_scale, float(pair))'
            assert text.count(expr)==(1 if name=='native_rope' else 2)
            text=text.replace(expr,'sycl::ext::intel::math::ha::pow(theta_scale, float(pair))')
            changes.append('Analytic F32 rotation power uses native IMF high-accuracy pow with original operands.')
        if name=='native_qsa_score':
            # Retain the original portable non-PTX score algorithm already
            # present in this source. Do not invent a CUDA architecture on Intel.
            assert text.count('#if !defined(__HIPCC__)')==1
            assert text.count('#if defined(__HIPCC__)')==1
            text=text.replace('#if !defined(__HIPCC__)','#if !defined(__HIPCC__) && !defined(STRATA_USE_SYCL)')
            text=text.replace('#if defined(__HIPCC__)','#if defined(__HIPCC__) || defined(STRATA_USE_SYCL)')
            changes.append('SYCL selects original portable F32 score branch; PTX tensor path remains unbound.')
        text,record=lower(text);out=args.output/('original_'+name+'.cpp');write(out,text)
        records[path]={'original_sha256':hashes[path],'generated_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),**record,'native_semantic_lowering':changes}
    # Expose the original device-only read helpers without pretending to be a
    # CUDA/HIP compiler. Their bodies are retained; __ldg is a native typed load.
    path='include/strata/kernels/mrope.hpp';text=checked(path)
    text=text.replace('#if defined(__CUDACC__) || defined(__HIPCC__)','#if defined(__CUDACC__) || defined(__HIPCC__) || defined(STRATA_USE_SYCL)')
    text=text.replace('__device__ __forceinline__','inline')
    text=text.replace('namespace strata::kernels {','namespace strata::kernels {\nusing sycl_upstream::cuda_kernel::__ldg;')
    text='#include "strata/sycl_upstream/cuda_kernel.hpp"\n'+text
    out=args.output/'native_include/strata/kernels/mrope.hpp';write(out,text)
    records[path]={'original_sha256':hashes[path],'generated_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    write(args.output/'original-primitive-lowering.json',json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
