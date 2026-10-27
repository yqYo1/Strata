#!/usr/bin/env python3
"""Closed-data selected6e40 QSA pack/attention conservation; no GPU execution.

Independently authored under the repository LICENSE. Names include the complete
source specialization and lambda path, not a broad prefix substring. This does
not provide a per-enqueue identity join, critical path or physical traffic.
"""
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P/'tools/sycl'))
import generation_engine as gen
import profile_aggregate
import runtime_probe as runtime

SOURCE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
RESULT = gen.A/'fused-moe-service-current.json'
SOURCE_FILES = ('src/kernels/sycl/qsa_bf16_prefix.cpp','include/strata/kernels/qsa_bf16_prefix.hpp',
    'src/prefill/prefill.cpp','src/prefill/layer_major.inc','src/prefill/sycl/qsa_native16.cpp')
BASE = ('strata::kernels::prefix_batch_impl<33, false, false, true, true, true, true, false>'
    '(unsigned short const*, strata::kernels::QsaAttnPools const&, int const*, int const*, long, '
    'strata::kernels::QsaShapes const&, unsigned short*, long, strata::kernels::QsaPromptBf16Bounds const&, '
    'long, strata::kernels::QsaBf16PackedScratch const&, void*, strata::kernels::QsaPrefixStateProbe)')
SUFFIXES = dict(pack='::{lambda(sycl::_V1::nd_item<1>)#1}',
    attention='::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::{lambda(sycl::_V1::nd_item<1>)#1}')
require = runtime.require


def geometry(rows):
    if type(rows) is not int or not 8192 < rows <= 262144:
        raise ValueError('selected long-prefill row extent required')
    attention, pack = Counter(), Counter()
    for first in range(0,rows,8192):
        t = min(8192,rows-first)
        attention[2*t] += 12  # both KV heads are in this one launch's grid
        pack[2*((first+t+3)//4)*4] += 12
    return dict(attention=dict(attention),pack=dict(pack))


def reconcile(summary, rows):
    expected = geometry(rows)
    found = {k:[] for k in expected}
    for row in summary['rows']:
        name = row['name'].strip('"')
        if 'prefix_batch_impl<' not in name: continue
        family = None
        for candidate,suffix in SUFFIXES.items():
            base = BASE+suffix
            if name.startswith(base):
                family = candidate
                simd,local = (16,128) if candidate == 'attention' else (32,64)
                match = re.fullmatch(re.escape(base)+r'\[SIMD'+str(simd)+r' \{([1-9][0-9]*); 1; 1\} \{'+str(local)+r'; 1; 1\}\]',name)
                if not match: raise ValueError('unexpected QSA launch geometry')
                group = int(match[1])
                if group not in expected[candidate] or row['calls'] != expected[candidate][group]:
                    raise ValueError('QSA source/grid call counts differ')
                found[candidate].append(dict(groups=group,calls=row['calls'],service_ns=row['total_ns']))
                break
        if family is None: raise ValueError('unqualified QSA prefix specialization or lambda')
    result = {}
    for family,shapes in found.items():
        if len(shapes) != len(expected[family]) or {r['groups'] for r in shapes} != set(expected[family]):
            raise ValueError('missing or duplicate QSA shapes')
        result[family] = dict(calls=sum(r['calls'] for r in shapes),service_ns=sum(r['service_ns'] for r in shapes),
            geometries=len(shapes),source_grid_counts_match=True)
        if family == 'attention':
            result[family]['query_KV_head_workgroups'] = sum(r['groups']*r['calls'] for r in shapes)
            result[family]['query_rows_across_12_layers'] = rows*12
            result[family]['full_tail_shapes'] = sorted(shapes,key=lambda r:r['groups'])
        else:
            result[family]['BF16_KV_output_bytes'] = sum(r['groups']*r['calls'] for r in shapes)*256*2*2
            result[family]['output_byte_meaning'] = 'Source BF16 K+V writes,including padded rows;not physical VRAM traffic.'
    return result


def source_counter_check(text,rows):
    expected_batches = (rows+8191)//8192
    parsed = re.findall(r'^strata lm profile: layer=(-?\d+) wall_ms=[^\n]*? samples=([0-9,]+) ',text,re.M)
    require(len(parsed) == 49 and [int(p[0]) for p in parsed] == [*range(48),-1], 'one complete coarse layer census required')
    samples = [list(map(int,p[1].split(','))) for p in parsed]
    require(all(len(s) == 6 for s in samples), 'coarse sample schema differs')
    for layer,s in enumerate(samples[:48]):
        qsa = layer%4 == 3
        require(s[1] == (expected_batches if qsa else 0) and
            s[2] == (0 if qsa else expected_batches), 'QSA/GDN coarse source multiplicity differs')
    require(samples[-1] == [sum(s[i] for s in samples[:48]) for i in range(6)] and
        samples[-1][1] == 12*expected_batches, 'coarse aggregate does not conserve')
    require(text.count('source=resident-Mode0 score_pitch=33 operand_layout=row-major softmax=parallel-exp-ascending-sum') == 1 and
        text.count('strata prefill: QSA cached_query=1 grf_policy=default source=qualified-component') == 1 and
        text.count('strata prefill: QSA k_staging=typed4 parallel_rows=1 source=qualified-component') == 1 and
        text.count('strata prefill: QSA v_staging=typed4 layout=row-major source=qualified-component') == 1,
        'selected QSA route markers differ')
    return dict(layer_census=True,QSA_calls=12*expected_batches,GDN_calls=36*expected_batches)


def analyse():
    import measure_fused_moe_service as measurement
    row = gen.committed(RESULT)
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and row['ownership_closed'] and
        not row['active'] and row['overhead']['passed'] and len(row['checks']) == 7 and
        'selected_qsa_service' not in row, 'closed complete three-pair cost gate required')
    measurement.admission(row)
    revision = gen.git('rev-parse','HEAD').decode().strip()
    self_name = str(Path(__file__).resolve().relative_to(P))
    require(gen.git('show',revision+':'+self_name) == Path(__file__).read_bytes(),'commit exact QSA analysis first')
    source = {name:hashlib.sha256(data).hexdigest() for name,data in gen.blobs(P,SOURCE,list(SOURCE_FILES))}
    callers = gen.git('grep','-l','qsa_bf16_prefix_ordered_exp_batch(',SOURCE,'--','src','include').decode().splitlines()
    require({line.split(':',1)[1] for line in callers} == set(SOURCE_FILES[:3]),'selected prefix caller inventory differs')
    checks = {c['mode']:c for c in row['checks']}
    observations = []
    for mode,plan in row['plans'].items():
        if plan['native_profile'] is None: continue
        c = checks[mode]
        require(c['passed'] and c['complete'] and c['ownership_closed'] and c['fresh_faults'] == [] and c['session_removed'],
            'closed native profile required')
        path = Path(plan['root'])/'engine.stderr'
        require(runtime.digest(path) == c['evidence_sha256']['engine.stderr'],'closed native output changed')
        text = path.read_text();rows = c['request']['actual_read_tokens']-1
        counter = source_counter_check(text,rows)
        service = reconcile(profile_aggregate.device_summary(text),rows)
        require(all(r['calls'] == counter['QSA_calls'] for r in service.values()), 'pack/attention multiplicity differs')
        observations.append(dict(mode=mode,actual_prefill_rows=rows,source_counters=counter,**service,
            native_output_sha256=c['evidence_sha256']['engine.stderr']))
    row['selected_qsa_service'] = dict(source_commit=revision,native_source=SOURCE,source_sha256=source,
        source_facts=['Prefill n>chunk selects layer-major;short74-token warmup stays on original path.',
            'Selected ordered-exp typed branch calls once per QSA chunk,passing live_cells=p0+T.',
            'Each call enqueues one logical-prefix pack and one attention kernel;two KV heads are inside each grid.',
            'T8192 and final tail across12 QSA layers determine every expected launch geometry.'],
        original_cost_plan='d7fd4944d',analysis_passed=True,GPU_executed=False,observations=observations,
        phase_ownership='Source route/count/geometry conservation;no independent per-enqueue phase identity join.',
        limits='Selected family service only. No complete-model/critical-path,physical traffic,hardware saturation,clean speed,decode,quality or physical262144-model qualification.',
        next_action='Use existing actual-list geometry and matrix-mapping reference to preflight issued row-slot savings before a new kernel build. Do not repeat closed GRF/partial-cache/cell64/SLM/sum candidates unchanged.')
    RESULT.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row['selected_qsa_service']['observations']),flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] == ['analyse'],'analyse')
    with (gen.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runtime.guards.no_model_owners([])
        analyse()
