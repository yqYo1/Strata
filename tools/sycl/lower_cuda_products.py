#!/usr/bin/env python3
"""Lower pinned original product translation units; retain arithmetic and dispatch.

This is deliberately limited to the four named original sources. Source hashes,
every kernel, shared array and launch are checked; unfamiliar syntax fails closed.
Generated files stay in the component build directory.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

SOURCES = ['native_mmvq', 'iq_kernels', 'native_bf16', 'bf16_gemv']


def split_args(text):
    depth = 0
    start = 0
    out = []
    for i, c in enumerate(text):
        if c in '(<[':
            depth += 1
        elif c in ')>]':
            depth -= 1
        elif c == ',' and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
    out.append(text[start:].strip())
    return out


def closing(text, at):
    depth = 1
    for i in range(at + 1, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('unclosed launch arguments')


def lower(source):
    text = source
    # Q8_1 integer rounding depends on correctly rounded division. On B570 a
    # plain SYCL divide changed a 63.5 tie despite -foffload-fp32-prec-div.
    # Keep the original operands/order with explicit native rounding operations.
    quant_divisions = text.count('const float d = amax / 127.0f;')
    assert quant_divisions == int('roundf(xi / d)' in text)
    text = text.replace('const float d = amax / 127.0f;', 'const float d = __fdiv_rn(amax, 127.0f);')
    text = text.replace('roundf(xi / d)', 'roundf(__fdiv_rn(xi, d))')
    text = re.sub(r'__launch_bounds__\([^\n]*?\)\s*', '', text)
    kernels = {}
    # Kernel signatures do not contain function-pointer parameters. Bodies end
    # at the original column-zero closing brace, including all nested scopes.
    pattern = re.compile(r'__global__ void (\w+)\((.*?)\)\s*\{(.*?)\n\}', re.S)
    def kernel(match):
        name, args, body = match.groups()
        prefix = text[:match.start()].rstrip()
        template = re.search(r'(template\s*<([^\n]+)>)(?:\s*)$', prefix)
        template_text = template.group(1) if template else ''
        template_names = [a.split()[-1] for a in split_args(template.group(2))] if template else []
        declarations = re.findall(r'__shared__ float (\w+)((?:\[[^\]]+\])+);', body)
        dynamic = re.findall(r'extern __shared__ float (\w+)\[\];', body)
        static = [(n,d) for n,d in declarations if d != '[]']
        assert len(dynamic) <= 1 and len(static) <= 1, name
        sizes = []
        for n,d in static:
            body = body.replace('__shared__ float '+n+d+';', 'auto& '+n+' = *reinterpret_cast<float (*)'+d+'>(strata_shared);')
            sizes.append('sizeof(float)' + ''.join('*('+s+')' for s in re.findall(r'\[([^\]]+)\]',d)))
        for n in dynamic:
            body = body.replace('extern __shared__ float '+n+'[];', 'float* '+n+' = strata_shared;')
        assert '__shared__' not in body, name
        # ROWS in single-column kernels is a body-local compile-time value.
        constants = re.findall(r'constexpr int (\w+)\s*=\s*([^;]+);', body)
        used = ' '.join(sizes)
        constants = [f'constexpr int {n} = {v};' for n,v in constants if re.search(r'\b'+n+r'\b',used)]
        size_func = (template_text+'\n' if template_text else '')+'inline constexpr size_t '+name+'_shared_bytes() { '+ ' '.join(constants)+' return '+(' + '.join(sizes) if sizes else '0')+'; }\n'
        # The original template declaration preceding this match applies to the
        # size helper; repeat it for the transformed kernel body below.
        if template_text:
            size_func = size_func[len(template_text)+1:]
        kernels[name] = {'template_names':template_names, 'shared':bool(static or dynamic), 'warp32': name not in ['bf16_gemv_naive_kernel','bf16_gemv_split_kernel','swiglu_entries_kernel']}
        return size_func+(template_text+'\n' if template_text else '')+'inline void '+name+'(float* strata_shared, '+args+') {'+body+'\n}'
    text = pattern.sub(kernel, text)
    assert kernels, 'no original kernels'
    launch = re.compile(r'\b(\w+)(<[^<>;\n]*>)?\s*<<<(.*?)>>>\s*\(',re.S)
    launches = []
    while (m := launch.search(text)):
        name, templates, config = m.groups()
        assert name in kernels, name
        configs = split_args(config)
        assert 2 <= len(configs) <= 4, config
        grid, block = configs[:2]
        shared = configs[2] if len(configs)>2 else '0'
        stream = configs[3] if len(configs)>3 else 'nullptr'
        end = closing(text, m.end()-1)
        args = text[m.end():end]
        warp = 'true' if kernels[name]['warp32'] else 'false'
        uses_shared = 'true' if kernels[name]['shared'] else 'false'
        suffix = templates or ''
        replacement = f'sycl_upstream::cuda_kernel::launch<{warp}, {uses_shared}>({grid}, {block}, ({shared}) + {name}_shared_bytes{suffix}(), {stream}, [=](float* strata_shared) {{ {name}{suffix}(strata_shared, {args}); }})'
        text = text[:m.start()]+replacement+text[end+1:]
        launches.append(name)
    assert '<<<' not in text and '__global__' not in text
    assert set(kernels) <= set(launches), sorted(set(kernels)-set(launches))
    for original,replacement in [('threadIdx','thread_index()'),('blockIdx','block_index()'),('blockDim','block_dimensions()'),('gridDim','grid_dimensions()')]:
        text = re.sub(r'\b'+original+r'\b',replacement,text)
    text = re.sub(r'__device__\s+__align__\((\d+)\)\s+',r'alignas(\1) inline constexpr ',text)
    text = text.replace('__device__ ', '').replace('__forceinline__','inline')
    text = text.replace('#include "strata/kernels/dp4a.hpp"','#define STRATA_DP4A(a,b,c) __dp4a((a),(b),(c))')
    text = text.replace('#include <cuda_fp16.h>','')
    text = text.replace('#define GGML_COMMON_DECL_CUDA','').replace('#define GGML_COMMON_IMPL_CUDA','').replace('#include "ggml-common.h"','')
    needle = 'namespace strata::kernels {\nnamespace {'
    assert text.count(needle)==1
    text = text.replace(needle,'namespace strata::kernels {\nnamespace {\nusing namespace sycl_upstream::cuda_kernel;')
    text = '#include "strata/sycl_upstream/cuda_kernel.hpp"\n'+text+'\n#undef STRATA_DP4A\n'
    return text, {'kernels':sorted(kernels), 'launch_sites':len(launches),
                  'q8_1_explicit_rn_divisions':2*quant_divisions}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ggml',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'docs/sycl-audit/source-inventory.json').read_text())
    hashes={x['path']:x['upstream_sha256'] for x in manifest['files']}
    external={x['path']:x['sha256'] for x in manifest['external_sources']}
    def checked(base,path,wanted):
        data=(base/path).read_bytes()
        assert hashlib.sha256(data).hexdigest()==wanted[path], 'Pinned source mismatch: '+path
        return data.decode()
    checked(args.ggml,'ggml/src/ggml-common.h',external)
    args.output.mkdir(parents=True,exist_ok=True)
    records={}
    for name in SOURCES:
        path='src/kernels/cuda/'+name+'.cu'
        original=checked(root,path,hashes)
        translated,record=lower(original)
        out=args.output/('original_'+name+'.cpp')
        if not out.exists() or out.read_text()!=translated:
            out.write_text(translated)
        records[path]={'original_sha256':hashes[path],'generated_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),**record}
    (args.output/'original-product-lowering.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    main()
