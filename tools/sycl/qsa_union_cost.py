#!/usr/bin/env python3
"""Bounded CPU census of an existing, initialized selected-ID fixture.

Repository LICENSE applies. This costs a hypothetical exact query union; it
does not build a kernel, run a GPU, measure bandwidth or select an engine.
"""
from array import array
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_owner as supervision
import qsa_selection_capture as capture

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'qsa-query-union-cost-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'qsa_query_union_cost'
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
ATTENTION = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
SOURCES = ['tools/sycl/qsa_union_cost.py', 'tools/sycl/qsa_selection_capture.py',
           'tools/sycl/measurement_owner.py']
require, digest = supervision.require, supervision.sha


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=30)


def committed(path):
    raw = git('show', 'HEAD:' + str(path.relative_to(P)))
    require(raw == path.read_bytes(), 'commit result before proceeding')
    return json.loads(raw)


def inputs(revision):
    pins = {}
    for name in SOURCES:
        raw = git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit unchanged input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def admit(plan):
    require(inputs(plan['source_commit']) == plan['source_sha256'], 'source changed')
    require(sys.version == plan['python_version'] and
            str(Path(sys.executable).resolve()) == plan['python']['path'] and
            digest(plan['python']['path']) == plan['python']['sha256'], 'Python changed')
    f = plan['fixture']
    require(Path(f['path']).stat().st_size == f['bytes'] and digest(f['path']) == f['sha256'],
            'captured ID bytes changed')
    require(f['origin_native_source'] == BASE and f['header'] == [0,32768,6491,2051],
            'different fixture domain requires a new plan')
    original = git('show', BASE + ':' + ATTENTION)
    require(hashlib.sha256(original).hexdigest() == plan['matrix_source']['sha256'] and
            all(anchor.encode() in original for anchor in plan['matrix_source']['anchors']),
            'original selected matrix geometry changed')


def prepare():
    require(not ROOT.exists(), 'fresh census only; inspect existing owner before resuming')
    record = committed(RESULT)
    require(KEY not in record and record['passed'] and record['complete'] and
            record['ownership_closed'] and not record['active'] and
            record['native_source'] == BASE, 'closed selected-source service evidence required')
    f = record['qsa_k_staging_preflight']['fixture']
    revision = git('rev-parse', 'HEAD').decode().strip()
    python = str(Path(sys.executable).resolve())
    original = git('show', BASE + ':' + ATTENTION)
    plan = dict(stage='prepared', source_commit=revision, source_sha256=inputs(revision),
        source_result_commit=revision, fixture=f, python_version=sys.version,
        python=dict(path=python, sha256=digest(python)), groups=[2,4],
        matrix_source=dict(commit=BASE,path=ATTENTION,sha256=hashlib.sha256(original).hexdigest(),
            anchors=['constexpr int Dim = 256, Heads = 12, Cells = 32, Workgroup = 128;',
                     'mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16>']),
        prospective_geometry=dict(query_heads_per_KV_head=12, KV_heads=2,matrix_rows=8,
                                  cell_tile=32, nominal_KV_bytes_per_cell=2048),
        budgets=dict(wall=120,cpu=100,rss_cap=512<<20,text_cap=64<<10,
                     file_cap=1<<20,total_cap=2<<20),
        environment=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',
                         PYTHONDONTWRITEBYTECODE='1'),
        question='Exact T=2/4 logical union reuse versus dense masked/padded matrix work and metadata cost in the retained layer0 tail;not whole-model coverage or speed.',
        limits='Logical IDs only. Preserve per-query incidence and independent softmax. Physical aliases/cache traffic,actual GPU preparation cost,occupancy,other layers/full chunks and a kernel implementation remain unmeasured. Padded8-row/32-cell counts use the original available matrix geometry in a prospective construction;not actual issued instructions.',
        retention='Keep source/input identity and conserved aggregate counts/distributions only. No per-query lists,unions,masks or duplicate fixture. Retire successful owned output after verification. Review original fixture at this decision boundary.')
    admit(plan)
    ROOT.mkdir(mode=0o700)
    write(ROOT/'plan.json',plan)
    record[KEY] = dict(plan,passed=False,complete=False,active=False,ownership_closed=True,
                       GPU_executed=False,model_selected=False)
    write(RESULT,record)
    print('Prepared exact-ID union census;commit before CPU-only execution.',flush=True)


def describe(values):
    ordered=sorted(values)
    return dict(count=len(ordered),minimum=ordered[0],median=statistics.median(ordered),
                p95=ordered[math.ceil(.95*len(ordered))-1],maximum=ordered[-1])


def read_fixture(plan):
    f=plan['fixture']; h=f['header']; layer,pos,rows,cap=h
    inspected=capture.inspect(f['path'],[h],selected=h)
    require(inspected['sha256']==f['sha256'] and inspected['selected']['unused_cap_entries']==0,
            'only validated initialized IDs may be read')
    before=Path(f['path']).stat()
    with Path(f['path']).open('rb') as stream:
        require(capture.identity(os.fstat(stream.fileno()))==capture.identity(before) and
                capture.HEADER.unpack(stream.read(capture.HEADER.size))==tuple(h),'fixture replaced')
        ids=array('i');ids.fromfile(stream,rows*cap)
        require(not stream.read(1) and capture.identity(os.fstat(stream.fileno()))==capture.identity(before),
                'fixture changed/trailing data')
    require(capture.identity(Path(f['path']).stat())==capture.identity(before),'fixture changed')
    return ids,h,inspected['selected']['valid_entries']


def analyze(ids,h,g,groups,valid_entries):
    layer,pos,rows,cap=h
    require(groups and len(ids)==rows*cap and rows>=max(groups) and cap>0 and
            valid_entries==rows*cap and 0<min(groups)<=max(groups)<=8,
            'invalid census geometry')
    M=g['matrix_rows']; N=g['cell_tile']; heads=g['query_heads_per_KV_head']
    require(M>0 and N>0 and heads>0,'invalid matrix cost geometry')
    pad=lambda x,t:(x+t-1)//t*t
    result=[]
    for T in groups:
        sums=collections.Counter(); hist=collections.Counter(); full=[]; tail=None
        for first in range(0,rows,T):
            size=min(T,rows-first); masks={}
            for q in range(size):
                width=min(pos+first+q+1,cap)
                for cell in ids[(first+q)*cap:(first+q)*cap+width]:
                    masks[cell]=masks.get(cell,0)|(1<<q)
            k=sum(min(pos+first+q+1,cap) for q in range(size)); u=len(masks)
            counts=collections.Counter(mask.bit_count() for mask in masks.values())
            require(sum(a*b for a,b in counts.items())==k and sum(counts.values())==u and
                    max(min(pos+first+q+1,cap) for q in range(size))<=u<=k,
                    'union/incidence conservation failed')
            old_cells=sum(pad(heads,M)*pad(min(pos+first+q+1,cap),N) for q in range(size))
            new_cells=pad(size*heads,M)*pad(u,N)
            values=dict(queries=size,selected=k,union=u,dense_incidences=size*u,
                        old_padded_matrix_cells_per_KV_head=old_cells,
                        union_padded_matrix_cells_per_KV_head=new_cells,
                        ID_plus_u8_mask_bytes=5*u)
            sums.update(values);hist.update(counts)
            metrics=dict(union=u,logical_KV_read_reduction=1-u/k,
                         dense_useful_fraction=k/(size*u),
                         padded_matrix_work_ratio=new_cells/old_cells,
                         ID_plus_u8_mask_bytes=5*u)
            if size==T:full.append(metrics)
            else:tail={'first_query':first,**values,**metrics}
        require(sums['queries']==rows and sums['selected']==valid_entries and
                sum(a*b for a,b in hist.items())==sums['selected'],'whole-capture conservation failed')
        result.append(dict(group_queries=T,full_groups=len(full),tail=tail,totals=dict(sums),
            incidence_multiplicity_histogram=dict(sorted(hist.items())),
            full_group_distributions={key:describe([v[key] for v in full]) for key in full[0]},
            whole_logical_KV_read_reduction=1-sums['union']/sums['selected'],
            whole_dense_useful_fraction=sums['selected']/sums['dense_incidences'],
            whole_padded_matrix_work_ratio=sums['union_padded_matrix_cells_per_KV_head']/sums['old_padded_matrix_cells_per_KV_head'],
            original_nominal_KV_read_bytes=sums['selected']*g['nominal_KV_bytes_per_cell'],
            ideal_union_nominal_KV_read_bytes=sums['union']*g['nominal_KV_bytes_per_cell']))
    return dict(header=h,valid_entries=valid_entries,initialized=True,
                conservation_passed=True,groups=result)


def worker(plan):
    admit(plan)
    ids,h,valid_entries=read_fixture(plan)
    result=analyze(ids,h,plan['prospective_geometry'],plan['groups'],valid_entries)
    admit(plan)
    print(json.dumps(result))


def run():
    record=committed(RESULT);plan=json.loads((ROOT/'plan.json').read_text())
    require(record[KEY]==dict(plan,passed=False,complete=False,active=False,ownership_closed=True,
                             GPU_executed=False,model_selected=False) and not (ROOT/'owned').exists(),
            'fresh committed plan required')
    admit(plan)
    outcome=dict(passed=False,complete=False,active=False,ownership_closed=False,GPU_executed=False)
    owner=None
    def persist():
        write(ROOT/'record.json',dict(outcome,active=bool(owner and owner.active),
                                      commands=owner.commands if owner else []))
    owner=supervision.Owner(ROOT/'owned',persist)
    try:
        entry,out,err=owner.run('union-census',[plan['python']['path'],str(Path(__file__).resolve()),'_worker'],
                              plan['environment'],ROOT,**plan['budgets'])
        require(supervision.closed(entry) and entry['exit_code']==0,'CPU census failed: '+err.read_text()[-1500:])
        data=json.loads(out.read_text());require(data['conservation_passed'],'unqualified census')
        admit(plan)
        outcome.update(passed=True,complete=True,analysis=data)
    except BaseException as error:
        outcome['error']=repr(error);raise
    finally:
        outcome['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        outcome['passed']=bool(outcome['passed'] and outcome['ownership_closed'] and 'error' not in outcome)
        persist();record[KEY].update(stage='closed',outcome=outcome,passed=outcome['passed'],
                                     complete=outcome['complete'],ownership_closed=outcome['ownership_closed'])
        write(RESULT,record)
    print('CPU-only union cost census closed;no kernel or model qualification.',flush=True)


if __name__=='__main__':
    if sys.argv[1]=='_worker':
        worker(json.loads((ROOT/'plan.json').read_text()))
    else:
        with (B/'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            {'prepare':prepare,'run':run}[sys.argv[1]]()
