"""SOURCE ONLY CPU runner for exported production gates; Main owns execution.

Synthetic dictionaries below exercise rejection logic, never completed receipts.
Owned subprocesses are tiny CPU QUIT protocol children, not Strata/model/GPU work.
"""
import argparse
import copy
import json
from pathlib import Path

import admission as a
import fixtures
import protocol
import run_root


def rejected(function, *args):
    try:
        function(*args)
    except (ValueError, RuntimeError, KeyError):
        return
    raise AssertionError('invalid production admission accepted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a.require(args.output.is_absolute() and not args.output.exists(), 'fresh absolute CPU fixture root')
    # The real reviewed42 identity is used as fixture input; no runtime state is consulted.
    expected = json.loads((run_root.HERE/'source-review.expected.json').read_text())['source_pins']
    assert len(expected) == 42
    q = dict(status='passed', main_reviewed=True, base_revision=a.BASE_REVISION,
        source_review_sha256=a.SOURCE_REVIEW_SHA256, production_source_pins=expected,
        owner_sha256=a.OWNER_SHA256, build_receipts={k:'/synthetic/'+k for k in ('free','contrib-llvm','free-asan','contrib-llvm-asan')},
        controller_source_pins=dict(a.SHARED, **{k:'synthetic' for k in a.CODE_NAMES-set(a.SHARED)}),
        full262144_qualified=False, broad_quality_qualified=False, restore_qualified=False,
        performance_eligible=False, receipt_sha256={})
    a.require_qualification(q, expected)
    for key, value in [('status','pending'), ('main_reviewed',False), ('owner_sha256','wrong'),
                       ('base_revision','wrong'), ('full262144_qualified',True), ('performance_eligible',True)]:
        bad=copy.deepcopy(q); bad[key]=value
        rejected(a.require_qualification,bad,expected)
    bad=copy.deepcopy(q); del bad['build_receipts']['free-asan']; rejected(a.require_qualification,bad,expected)
    bad=copy.deepcopy(q); bad['controller_source_pins']['protocol.py']='wrong'; rejected(a.require_qualification,bad,expected)
    bad=copy.deepcopy(q); bad['production_source_pins']['src/prefill/prefill.cpp']='wrong'; rejected(a.require_qualification,bad,expected)
    rejected(a.safe_record_path, '/excluded/prefill-gemm-only-contract-root-v1/record.json')

    q.update(build_receipt_sha256={k:'synthetic' for k in q['build_receipts']}, lint_receipt_sha256='synthetic')
    bridge=dict(status='passed',main_reviewed=True,base_revision=a.BASE_REVISION,
        source_review_sha256=a.SOURCE_REVIEW_SHA256,source_tree=str(run_root.W),clean=True,
        published_revision='a'*40,source_pins=expected,built_source_pins=expected,
        build_receipt_sha256=q['build_receipt_sha256'],controller_source_pins=q['controller_source_pins'],
        lint_receipt_sha256='synthetic')
    a.require_publication(bridge,q,expected,run_root.W)
    for key,value in [('status','pending'),('clean',False),('published_revision',a.BASE_REVISION),('built_source_pins',{})]:
        bad=copy.deepcopy(bridge);bad[key]=value;rejected(a.require_publication,bad,q,expected,run_root.W)
    a.require_published_state(expected,expected,'a'*40,'a'*40,'','', 'binary', 'binary')
    for index,value in [(0,dict(expected, **{'src/prefill/prefill.cpp':'wrong'})),(2,'b'*40),(4,' M src/prefill/prefill.cpp\n'),(5,'diff'),(6,'wrong')]:
        values=[expected,expected,'a'*40,'a'*40,'','', 'binary', 'binary'];values[index]=value
        rejected(a.require_published_state,*values)

    base=dict(STRATA_PREFILL_NATIVE_DEQUANT_BATCH='0',STRATA_PREFILL_FUSED='0',STRATA_PRELOAD_KERNELS='0',
              STRATA_TRACE='1',ZE_ENABLE_VALIDATION_LAYER='1',ZE_ENABLE_PARAMETER_VALIDATION='1')
    off=a.compact_environment(base,'off');on=a.compact_environment(base,'on')
    a.require_exact_variant_env(off,on)
    for key in ('STRATA_PREFILL_FUSED','STRATA_PREFILL_NATIVE_DEQUANT_BATCH','STRATA_PRELOAD_KERNELS'):
        rejected(a.compact_environment,dict(base, **{key:'1'}),'on')
    rejected(a.compact_environment,dict(base,STRATA_DBG_NAN='0'),'on')
    rejected(a.compact_environment,dict(base,UR_LOG_TRACING='full'),'off')
    rejected(a.require_exact_variant_env,off,dict(on,STRATA_TRACE='0'))

    for index,(n,variant) in enumerate(a.SEQUENCE):
        assert a.admitted_variant(n,16,variant,Path('/synthetic/ref') if variant=='on' else None)==index
        wanted=Path('/synthetic/first') if index==0 else a.root_path(run_root.B,*a.SEQUENCE[index-1])/'record.json'
        assert a.predecessor_path(run_root.B,n,variant,'/synthetic/first')==wanted
        a.require_sequence_prefix(a.SEQUENCE[:index],n,variant)
        rejected(a.require_sequence_prefix,a.SEQUENCE[:index]+((n,variant),),n,variant)
        if index:rejected(a.require_sequence_prefix,a.SEQUENCE[:index-1],n,variant)
    for values in [(True,16,'off',None),(65,16,'off',None),(64,8,'off',None),(64,16,'on',None),(64,16,'off',Path('/ref'))]:
        rejected(a.admitted_variant,*values)

    command=dict(label='model',observation_complete=True,direct_child_reaped=True,session_empty=True,
                 session_anchored=True,cleanup=[],errors=[],survivors=[],exit_code=0)
    previous=dict(passed=True,complete=True,active=False,boot_id='fixture-boot',boot_unchanged=True,
        kernel_gate_passed=True,reset_executed=False,retry_count=0,commands=[command],model_normal_close=True,
        model_close=dict(quit_requested=True,quit_send_outcome='sent',quit_exit_outcome='exit observed',exit_code=0,normal=True,forced=False))
    a.require_normal_model(previous,'fixture-boot')
    for key,value in [('active',True),('passed',False),('boot_id','other'),('retry_count',1),('reset_executed',True)]:
        bad=copy.deepcopy(previous);bad[key]=value;rejected(a.require_normal_model,bad,'fixture-boot')
    for key,value in [('session_empty',False),('session_anchored',False),('cleanup',[{'signal':'SIGTERM'}]),('exit_code',-15)]:
        bad=copy.deepcopy(previous);bad['commands'][0][key]=value;rejected(a.require_normal_model,bad,'fixture-boot')
    bad=copy.deepcopy(previous);bad['model_close']['quit_send_outcome']='pending';rejected(a.require_normal_model,bad,'fixture-boot')

    mapping=dict(path='/fixture/compiler',canonical_path='/fixture/compiler',sha256='hash',stat={'inode':1},elf_build_id='id',original_runtime=False)
    snapshot=dict(complete=True,selected_rows_stable=True,files={'/fixture/compiler':mapping})
    a.require_additional_mappings(snapshot,{'/fixture/compiler':mapping})
    bad=copy.deepcopy(snapshot);bad['files']['/fixture/compiler']['elf_build_id']='wrong'
    rejected(a.require_additional_mappings,bad,{'/fixture/compiler':mapping})
    rejected(a.require_additional_mappings,snapshot,{})
    for n in (64,2049,32768):
        lines=fixtures.protocol_case(n)
        value=dict(request_protocol=lines,validation=protocol.validate(lines,n=n))
        a.compare_strict_outputs(value,value)
        bad=copy.deepcopy(value);bad['request_protocol'][bad['request_protocol'].index('T 123')]='T 124'
        rejected(a.compare_strict_outputs,value,bad)
        bad=copy.deepcopy(value);bad['request_protocol'][bad['request_protocol'].index('LP -0.1 123:-0.1 124:-1 125:-2 126:-3 127:-4')]='LP -0.2 123:-0.2 124:-1 125:-2 126:-3 127:-4'
        rejected(a.compare_strict_outputs,value,bad)
        early=fixtures.protocol_case(n,1,'stop')
        rejected(a.compare_strict_outputs,value,dict(request_protocol=early,validation=protocol.validate(early,n=n)))

    # Actual exported QUIT helpers against anchored CPU children. No AST extraction or mocked owner closure.
    args.output.mkdir(mode=0o700,parents=True)
    owner=run_root.m.Owner(args.output/'commands')
    cases=[]
    for phase in ('normal','early-eos','ready-failure','done-failure'):
        record=dict(gen_send_attempted=phase!='ready-failure',ready_seen=True,request_done_seen=phase!='ready-failure')
        if phase.endswith('failure'):record['primary_model_failure']='synthetic primary gate failure'
        def interaction(io,entry):
            assert io.line(5)=='READY 262144'
            if phase!='ready-failure':assert io.line(5)=='DONE stop'
            if phase.endswith('failure'):
                assert a.close_after_gate_failure(io,record,io.poll,lambda:None)
                assert record['primary_model_failure']=='synthetic primary gate failure'
            else:a.normal_quit(io,record)
        child="import sys\nprint('READY 262144',flush=True)\n"+("print('DONE stop',flush=True)\n" if phase!='ready-failure' else '')+"sys.exit(0 if sys.stdin.readline()=='QUIT\\n' else 7)\n"
        # Literal newline in the child comparison: no backslash-n byte string accepted as a QUIT substitute.
        child=child.replace("'QUIT\\n'",repr('QUIT\n'))
        entry,_,_=owner.run(phase,['/usr/bin/python3','-u','-c',child],
            dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1'),
            run_root.HERE,interaction=interaction,wall=20,cpu=40,text_cap=65536,total_cap=2<<20,rss_cap=128<<20)
        assert a.strong_closed_command(entry) and entry['exit_code']==0
        assert record['model_close']['quit_send_outcome']=='sent' and record['model_close']['quit_exit_outcome']=='exit observed'
        cases.append(dict(phase=phase,record=record,command=entry))
    print(json.dumps(dict(passed=True,gpu_executed=False,model_executed=False,
        scope='CPU exported admission/rejection/provenance/sequence/environment/output and actual CPU-child QUIT fixtures only',
        owner_sha256=run_root.m.sha(run_root.OWNER),source_pins={name:run_root.m.sha(run_root.HERE/name) for name in sorted(a.CODE_NAMES)},
        owned_quit_cases=cases)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
