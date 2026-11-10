#!/usr/bin/env python3
"""Own full-engine build for the independently selected long-prefill sum route.

Uses the qualified current full-build recipe; does not execute a GPU/model.
Component evidence is a prerequisite, never inherited model qualification.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import check_qsa16_stream as original
import qsa_cached_subgroup as component

P, B, RESULT = original.P, original.B, original.RESULT
ROOT, KEY = B/'qsa-cached-subgroup-model-native-v2', 'qsa_cached_subgroup_model_native_v2'
TARGETS = ['strata', 'sycl_q6_head_owner_test', 'sycl_submission_test',
    'sycl_verifier_retirement_test', 'sycl_verify_window_policy_test',
    'sycl_qsa_bf16_prefix_cached_subgroup_test',
    'sycl_qsa_bf16_prefix_cached_subgroup_bench', 'sycl_native16_qsa_test']
require, sha = original.require, original.sha

def component_reference():
    path=component.ROOT/'record.json';row=json.loads(path.read_text())
    public=original.validation.committed(RESULT,(component.KEY,))
    require(public==component.compact(row) and row['stage']=='closed' and
        row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'] and
        row['performance']['component_advance_eligible'], 'closed eligible paired component required')
    require(list(row['outcomes'])==['smoke','component','perf0','perf1','perf2'] and
        all(v['passed'] and v['complete'] and v['ownership_closed'] and v['fresh_faults']==[]
            for v in row['outcomes'].values()), 'all original component gates required')
    original_source=row['native_source_commit']
    before={n:hashlib.sha256(raw).hexdigest() for n,raw in
        original.common.source.blobs(P,row['source_commit'],row['input_sha256'])}
    require(before==row['input_sha256'],'component original inputs unrecoverable')
    permitted='include/strata/kernels/qsa_bf16_prefix.hpp'
    changed=[]
    for family,values in row['actual_inputs'].items():
        for name,h in values.items():
            if sha(name)==h: continue
            path0=Path(name)
            require(family=='compiled' and path0==P/permitted,
                'undeclared compiled component input change: '+name)
            raw=original.common.source.git(P,'show',original_source+':'+permitted)
            require(hashlib.sha256(raw).hexdigest()==h and
                raw.replace(b'// No model or decode dispatch selects this export.',
                    b'// Only the explicitly selected long-prefill route may use this export;\n'
                    b'// decode has its independent implementation and selection.')==path0.read_bytes(),
                'only the declared integration comment may change')
            changed.append(name)
    require(changed==[str(P/permitted)], 'expected one comment-only compiled-input delta')
    for artifact in row['binaries'].values():
        require(sha(artifact['path'])==artifact['sha256'] and
            Path(artifact['path']).stat().st_size==artifact['bytes'],'original component fixture changed')
    require(row['historical_crosswalk']['template_and_admission_body_identical'] and
        row['historical_crosswalk']['previous_outcome']=='failed_speed_gate_preserved',
        'preserve old stricter-screen result and same-mechanism crosswalk')
    return dict(path=str(path),sha256=sha(path),key=component.KEY,
        evidence_commit='5dcdec7e0a0048ab619c6ab519887383699ec91b',
        native_source_commit=original_source,binaries=row['binaries'],
        complete_original_sources_recovered=True,actual_compiler_inputs=len(row['actual_inputs']['compiled']),
        actual_link_inputs=len(row['actual_inputs']['links']),
        compiled_code_unchanged_except_declared_comment=changed,
        same_algorithm_as_historical_candidate=True,
        model_speed_quality_decode_RESTORE_inherited=False)

def context():
    spec=importlib.util.spec_from_file_location('qsa_cached_model_build_context',original.__file__)
    ctx=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctx)
    ctx.ROOT,ctx.KEY,ctx.TARGETS,ctx.MODEL_BUILD=ROOT,KEY,TARGETS,True
    ctx.SEED=B/'qsa16-bulk-mirror-model-native-v1/record.json'
    ctx.SEED_KEY='qsa16_bulk_mirror_model_native_v1'
    ctx.component_reference=component_reference
    return ctx

def prepare():
    ctx=context();ctx.prepare()
    row=json.loads((ROOT/'record.json').read_text())
    decode=json.loads((B/'q6-dense-head-backing-native-v3/record.json').read_text())
    require(decode['passed'] and decode['complete'] and decode['ownership_closed'],
        'preserve independent sustained-decode generation')
    row['preserved_binaries'].append(decode['binaries']['strata'])
    row['previous_preparation']=dict(source_commit='deeddb764ed83d346481721c7b12fe9fdb232a4b',
        evidence_commit=original.common.source.git(P,'rev-parse','HEAD').decode().strip(),
        key='qsa_cached_subgroup_model_native_v1',stage='preparation_failed',
        reason='Wrong top-level binary lookup;actual reference is binaries.strata. No configure/build/test/GPU process started. Original failed preparation remains in Git.')
    row['plan']=('Commit every stage. Reuse exact56cad9 compiler/SDK/configure recipe and '
        'complete source/generated/link census;build the engine,four owner/submission/retirement/'
        'phase fixtures,the fresh cached-subgroup numerical/timing fixtures and streamed producer. '
        'The new startup-only bf16-prefix-cached-subgroup selection is confined to long-prefill '
        'dispatch,scratch admission and truthful diagnostics. Existing ordered route and independent '
        'decode stay selectable. No GPU/model qualification inherited. After compile,commit fresh '
        'small traced native/numerical/producer checks before bounded long/short/long actual-model '
        'handoff and a separately predeclared same-binary comparison. Preserve parameter validation,'
        'all safety settings,thinking,real token counts,zero reuse and phase/full262144/RESTORE/task gates.')
    row['retention']=('One temporary fourth full engine for the imminent combined-prefill/model '
        'qualification without overwriting current56cad9,fast117K552b or independent-decodea347. '
        'Eight same-build executables are one generation. This is not a larger default quota. '
        'Review at the first closed native/model/paired32K64K/decode-preservation boundary;select '
        'candidate or56cad9 and retire the superseded build after exact source/recipe recovery. '
        'The separate two component ELFs are retained only until this successor same-build '
        'qualification resolves their proof consumer. Legacy configured-deployment holds stay separate.')
    ctx.current(row,committed=False);ctx.save(row)
    print('Prepared8target full candidate;commit before configure. No model/adoption claim.',flush=True)

if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','configure','compile'),
        'use prepare|configure|compile')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);original.guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:context().build(sys.argv[1])
