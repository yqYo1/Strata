#!/usr/bin/env python3
"""Prove reuse of unchanged native fixtures after generator-only changes;Main."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import check_full_context as gate

P,B,RESULT=gate.P,gate.B,gate.RESULT
KEY='prefill_layout_components_v4'
ORDER=('layout','restore-smoke','component','resident','attention-smoke','attention-component','restore-component')
require=gate.native.guards.require


def qualify(container):
    require(KEY not in container,'fresh component continuity proof only')
    require(RESULT.read_bytes()==gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))),
            'commit exact predecessor results first')
    revision=gate.native.build.git(P,'rev-parse','HEAD').decode().strip()
    path=str(Path(__file__).resolve().relative_to(P))
    require(gate.native.build.git(P,'show',revision+':'+path)==Path(__file__).read_bytes(),
            'commit exact continuity controller first')
    engine=gate.engine();previous=container['native_prefill_layout_v2_build']
    old=container['prefill_layout_components_v2']
    require(old['passed'] and old['complete'] and old['ownership_closed'] and not old['active'] and
        old['native_source_commit']==previous['source_commit']==engine['predecessor']['source_commit'] and
        old['binary']==previous['binary']==engine['predecessor']['binary'] and
        set(old['outcomes'])==set(ORDER) and set(old['reused_unchanged_gates'])=={'host','layout-smoke'},
        'closed original component cohort required')
    delta=set(gate.native.build.git(P,'diff','--name-only',previous['source_commit'],engine['source_commit'],
        '--',*gate.native.PREFIXES).decode().splitlines())
    require(delta=={'src/program/generate.cpp','include/strata/program/diagnostic_kv_ranges.hpp',
        'tests/core/full_context_diagnostic_test.cpp','tools/sycl/build_full_context.py',
        'tools/sycl/check_layout_source.py'},'changed native fixture/library input;do not reuse')
    dependencies=[{k:d[k] for k in ('root','revision','count','aggregate_sha256')}
                  for d in engine['dependencies']]
    require(all(engine[k]==previous[k] for k in ('compiler','sdk','tools','configure_argv','generated_expected')) and
        dependencies==previous['sources'][1:] and engine['fixtures']==previous['fixtures'],
        'compiler/recipe/dependency/generated or actual fixture bytes changed')
    ninja=(gate.native.ROOT/'build/build.ninja').read_text()
    require('CMakeFiles/strata.dir/src/program/generate.cpp.o' in
        ninja[ninja.index('build strata:'):].split('\n\n',1)[0], 'generator must remain executable-only')
    preload=gate.validation.preload_qualification()
    require(preload==old['preload'] and old['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        and old['driver_before']==gate.native.guards.driver_guard(),'runtime/boot/driver changed')
    refs={mode:('prefill_layout_components_v2',mode) for mode in ORDER}
    refs.update({mode:(r['result_key'],r['outcome']) for mode,r in old['reused_unchanged_gates'].items()})
    verified=[]
    for mode,(key,name) in refs.items():
        origin=container[key];outcome=origin['outcomes'][name];case=origin['cases'][name]
        require(outcome['passed'] and outcome['complete'] and outcome['ownership_closed'] and
            outcome['fresh_faults']==[] and case['fixture'] in engine['fixtures'].values() and
            gate.native.build.digest(case['fixture']['path'])==case['fixture']['sha256'],
            'original gate/actual fixture mismatch: '+mode)
        suffix='v2' if key=='prefill_layout_components_v2' else 'v1'
        root=B/('prefill-layout-component-'+suffix)/name
        env=dict(engine['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
            CUDA_MODULE_LOADING='EAGER',UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
            LD_PRELOAD=preload['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=preload['loader']['path'])
        for variable in ('UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):
            env[variable]='level:warning;flush:warning;output:stderr'
        if name in ('layout-smoke','attention-smoke','restore-smoke'):
            env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:stderr')
        else:env.pop('UR_LOG_TRACING',None)
        if name.startswith('layout'):
            env.update(STRATA_SYCL_PREFILL_SCRATCH_ALIAS='1',STRATA_PREFILL_MMQ='0',
                STRATA_PREFILL_MOE_COMPACT_FP16='1',STRATA_PREFILL_RING='8',
                STRATA_GR_UNFUSED='0',STRATA_PREFILL_BF16X2='0',STRATA_PREFILL_LAYER_MAJOR='1')
        require(hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest()==case['environment_sha256'],
            'original effective runtime differs: '+mode)
        verified.append(dict(result_key=key,outcome=name,fixture_sha256=case['fixture']['sha256']))
    record=dict(kind='unchanged-fixture-qualification-proof',stage='closed_proof',source_commit=revision,
        controller_sha256=gate.native.build.digest(__file__),native_source_commit=engine['source_commit'],
        binary=engine['binary'],previous_native_source_commit=previous['source_commit'],
        changed_native_inputs=sorted(delta),fixtures_verified=6,individual_gate_references=verified,
        compiler_recipe_dependencies_generated_matched=True,effective_runtime_matched=True,
        passed=True,complete=True,active=False,ownership_closed=True,GPU_executed=False,
        native_model_qualification_inherited=False,model_adoption_qualified=False,
        physical262144_model_qualified=False,decode_performance_qualified=False,
        ordinary_runtime_teardown_qualified=False,
        scope='Only exact six fixture bytes and their9 already-executed component gates are reused. Changed generator and diagnostic model behavior require new actual model gates. Original cohort statuses remain unchanged;no new GPU execution or speed result.')
    container[KEY]=record;gate.write(RESULT,container)
    print(json.dumps(dict(passed=True,fixtures_verified=6,individual_gates_referenced=len(verified),GPU_executed=False)),flush=True)


if __name__=='__main__':
    require(sys.argv[1:]==['qualify'],'qualify')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gate.native.guards.no_model_owners([])
        qualify(json.loads(RESULT.read_text()))
