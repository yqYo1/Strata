"""SOURCE ONLY CPU runner for quiet/phase exported production gates; Main owns execution.

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


def prerequisite_cases(lint_path, review):
    """Read existing immutable closed structures; never manufacture a completed prerequisite."""
    consumed = {}
    def actual(path):
        a.safe_record_path(path)
        a.require(path.stat().st_size <= 8 << 20, 'fixture retained receipt byte budget')
        value = json.loads(path.read_text())
        consumed[str(path)] = run_root.m.sha(path)
        return value
    expected = review['source_pins']
    compiler_path = run_root.B/'xe-llvm-v711-source-root-v5/record.json'
    compiler = actual(compiler_path)
    compiler_sha = consumed[str(compiler_path)]
    a.require_compiler_prerequisite(compiler)
    for key, value in [('active', True), ('owner_sha256', 'wrong'), ('queue_created', True)]:
        bad = copy.deepcopy(compiler); bad[key] = value; rejected(a.require_compiler_prerequisite, bad)
    host_path = run_root.B/'xe-compact-moe-host-qualification-root-v2/record.json'
    host = actual(host_path)
    host_sha = consumed[str(host_path)]
    a.require_host_prerequisite(host, expected, run_root.W, compiler_sha)
    for key, value in [('main_review_sha256', 'wrong'), ('phase', 'free'), ('source_stable', False)]:
        bad = copy.deepcopy(host); bad[key] = value
        rejected(a.require_host_prerequisite, bad, expected, run_root.W, compiler_sha)
    build_hashes = {}
    for mode in a.BUILD_MODES:
        path = run_root.B/f'xe-compact-moe-{mode}-qualification-root-v2/record.json'
        value = actual(path)
        build_hashes[mode] = consumed[str(path)]
        cache = Path(value['build_recipe']['CMakeCache.txt']['path']).read_text()
        arguments = [mode, expected, run_root.W, host_sha, compiler_sha, value['binary'], cache]
        a.require_build_prerequisite(value, *arguments)
        for key, wrong in [('phase', 'wrong'), ('main_review_sha256', 'wrong'), ('active', True),
                           ('host_fixture_receipt_sha256', 'wrong'), ('compiler_receipt_sha256', 'wrong'),
                           ('source_stable', False), ('cpu_test_output', 'tests unavailable')]:
            bad = copy.deepcopy(value); bad[key] = wrong; rejected(a.require_build_prerequisite, bad, *arguments)
        bad = copy.deepcopy(value); bad['commands'][0]['session_anchored'] = False
        rejected(a.require_build_prerequisite, bad, *arguments)
        bad = copy.deepcopy(value); bad['source_pins']['src/prefill/prefill.cpp'] = 'wrong'
        rejected(a.require_build_prerequisite, bad, *arguments)
        if not mode.endswith('-asan'):
            # These are actual exit2 commands from the closed release receipts, not invented CLI results.
            bad = copy.deepcopy(value)
            next(c for c in bad['commands'] if c['exit_code'] == 2)['argv'][0] = '/wrong/binary'
            rejected(a.require_build_prerequisite, bad, *arguments)
        else:
            bad = copy.deepcopy(value); bad['commands'][0]['exit_code'] = 2
            rejected(a.require_build_prerequisite, bad, *arguments)
    lint = actual(lint_path)
    original = None
    original_sha = None
    if lint.get('reused_commands'):
        original_path = Path(lint['original_failed_comparison']['path'])
        original = actual(original_path)
        original_sha = consumed[str(original_path)]
    arguments = [expected, review['changed_source_pins'], build_hashes, original, original_sha]
    a.require_lint_prerequisite(lint, *arguments)
    for key, value in [('main_review_sha256', 'wrong'), ('active', True), ('passed', False)]:
        bad = copy.deepcopy(lint); bad[key] = value; rejected(a.require_lint_prerequisite, bad, *arguments)
    bad = copy.deepcopy(lint)
    del bad['main_review_sha256']; bad['source_review_sha256'] = a.SOURCE_REVIEW_SHA256
    rejected(a.require_lint_prerequisite, bad, *arguments)
    bad = copy.deepcopy(lint); del bad['comparisons']['contrib-llvm']
    rejected(a.require_lint_prerequisite, bad, *arguments)
    bad = copy.deepcopy(lint); bad['comparisons']['free']['new_findings'] = ['negative mutation']
    rejected(a.require_lint_prerequisite, bad, *arguments)
    bad = copy.deepcopy(lint)
    bad['comparisons']['free']['candidate_diagnostics'] += copy.deepcopy(bad['comparisons']['free']['baseline_diagnostics'])
    rejected(a.require_lint_prerequisite, bad, *arguments)
    bad = copy.deepcopy(lint); bad['comparisons']['free']['candidate_exit_code'] = 2
    rejected(a.require_lint_prerequisite, bad, *arguments)
    if original is not None:
        bad = copy.deepcopy(lint); bad['reused_commands'][0]['argv'].append('--unexpected')
        rejected(a.require_lint_prerequisite, bad, *arguments)
        bad_original = copy.deepcopy(original); bad_original['passed'] = True
        rejected(a.require_lint_prerequisite, lint, *arguments[:-2], bad_original, original_sha)
        bad = copy.deepcopy(lint); bad['original_failed_comparison']['original_status'] = 'PASSED'
        rejected(a.require_lint_prerequisite, bad, *arguments)
    # Retained read-only bytes must remain unchanged during these fixture reads.
    assert all(run_root.m.sha(Path(path)) == digest for path, digest in consumed.items())
    return consumed


def main():
    import window_protocol as wp
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lint-receipt', type=Path, required=True)
    parser.add_argument('--matrix-manifest', type=Path, required=True,
                        help='Main pins all six only after final original6144ON closes; no live reads')
    parser.add_argument('--three-case-manifest',type=Path,required=True,
                        help='Main pins all three ONLY after actual closed PASS; no live artifacts')
    args = parser.parse_args()
    a.require(args.output.is_absolute() and not args.output.exists(), 'fresh absolute CPU fixture root')
    review = json.loads((run_root.HERE/'source-review.expected.json').read_text())
    expected = review['source_pins']
    consumed = prerequisite_cases(args.lint_receipt, review)
    def actual(path):
        a.safe_record_path(path)
        value = json.loads(path.read_text())
        consumed[str(path)] = run_root.m.sha(path)
        return value
    original_path = run_root.B/'xe-compact-moe-main-admission-converter-root-v1/qualification.resolved.json'
    original = actual(original_path)
    original_sha = consumed[str(original_path)]
    a.require_qualification(original, expected, a.ORIGINAL_CODE_NAMES)
    original_cpu=actual(Path(original['controller_cpu_receipt']))
    original_review=actual(Path(original['controller_review']))
    a.require_original_controller_proofs(original_cpu,original_review,original)
    bad=copy.deepcopy(original_cpu);bad['source_pins']={}
    rejected(a.require_original_controller_proofs,bad,original_review,original)
    bridge = actual(Path(original['publication_bridge']))
    a.require_publication(bridge, original, expected, run_root.W)
    assert bridge['published_revision'] == a.PUBLISHED_REVISION
    pending = json.loads((run_root.HERE/'qualification.pending.json').read_text())
    rejected(a.require_chunk_qualification, pending, expected, original)
    pending_bridge = json.loads((run_root.HERE/'publication.bridge.pending.json').read_text())
    rejected(a.require_publication, pending_bridge, original, expected, run_root.W)
    bad = copy.deepcopy(bridge); bad['built_source_pins'] = {}
    rejected(a.require_publication, bad, original, expected, run_root.W)
    a.require_published_state(expected, expected, a.PUBLISHED_REVISION, a.PUBLISHED_REVISION, '', '',
                              original['free_release_binary']['sha256'], original['free_release_binary']['sha256'])
    rejected(a.require_published_state, expected, expected, 'wrong', a.PUBLISHED_REVISION, '', '', 'binary', 'binary')

    manifest = actual(args.matrix_manifest)
    pins = manifest['six_case_receipts']
    a.require(set(pins) == {f'n{n}-{v}' for n,v in a.MATRIX}, 'Main supplies exact closed six matrix')
    matrix = {}
    for n, variant in a.MATRIX:
        key = f'n{n}-{variant}'
        path = a.matrix_path(run_root.B, n, variant)
        a.require(pins[key]['path'] == str(path) and run_root.m.sha(path) == pins[key]['sha256'], 'actual Main matrix pin')
        matrix[key] = actual(path)
    boot = matrix['n32768-on']['boot_id']
    matrix_args = [pins, original, original_sha, boot, run_root.B]
    a.require_completed_matrix(matrix, *matrix_args)
    # All positives above are retained actual completed receipt structures, never invented PASS records.
    bad = copy.deepcopy(matrix); del bad['n32768-on']; rejected(a.require_completed_matrix, bad, *matrix_args)
    for key, changed in [('active', True), ('published_revision', 'wrong'), ('requested_chunk',8192),
                         ('qualification_sha256','wrong'), ('controller_sha256','wrong')]:
        bad = copy.deepcopy(matrix); bad['n32768-on'][key] = changed
        rejected(a.require_completed_matrix, bad, *matrix_args)
    bad = copy.deepcopy(matrix); bad['n64-on']['commands'][0]['session_anchored'] = False
    rejected(a.require_completed_matrix, bad, *matrix_args)
    bad_pins = copy.deepcopy(pins); bad_pins['n32768-on']['sha256'] = 'PENDING'
    rejected(a.require_completed_matrix, matrix, bad_pins, original, original_sha, boot, run_root.B)
    bad = copy.deepcopy(matrix); bad['n2049-on']['request_protocol'][bad['n2049-on']['request_protocol'].index(
        next(x for x in bad['n2049-on']['request_protocol'] if x.startswith('T ')))] = 'T 248320'
    rejected(a.require_completed_matrix, bad, *matrix_args)

    chunk_path=run_root.B/'xe-compact-moe-chunk8192-main-admission-root-v1/qualification.resolved.json'
    chunk_q=actual(chunk_path)
    chunk_sha=consumed[str(chunk_path)]
    a.require_chunk_qualification(chunk_q,expected,original)
    chunk_cpu=actual(Path(chunk_q['controller_cpu_receipt']))
    chunk_review=actual(Path(chunk_q['controller_review']))
    a.require_original_controller_proofs(chunk_cpu,chunk_review,chunk_q)
    rejected(a.require_quiet_qualification,pending,expected,chunk_q,original)
    triple_manifest=actual(args.three_case_manifest)
    triple_pins=triple_manifest['three_case_receipts']
    diagnostics={}
    for n,chunk in a.SEQUENCE:
        key=f'n{n}-t{chunk}'
        path=a.root_path(run_root.B,n,chunk)/'record.json'
        a.require(triple_pins[key]['path']==str(path) and run_root.m.sha(path)==triple_pins[key]['sha256'], 'Main closed diagnostic pin')
        diagnostics[key]=actual(path)
    triple_args=[triple_pins,chunk_q,chunk_sha,boot,run_root.B]
    a.require_closed_chunk_diagnostics(diagnostics,*triple_args)
    rejected(a.require_prior_quiet,diagnostics['n65536-t6144'],boot,1,6144,
        run_root.m.sha(run_root.HERE/'run_root.py'),'PENDING',triple_pins['n65536-t6144'])
    bad=copy.deepcopy(diagnostics);del bad['n65536-t8192']
    rejected(a.require_closed_chunk_diagnostics,bad,*triple_args)
    for field,wrong in [('active',True),('passed',False),('requested_chunk',6144),('qualification_sha256','wrong'),
                        ('controller_sha256','wrong'),('source_guard_phases',['admission'])]:
        bad=copy.deepcopy(diagnostics);bad['n65536-t8192'][field]=wrong
        rejected(a.require_closed_chunk_diagnostics,bad,*triple_args)
    bad=copy.deepcopy(diagnostics);bad['n65536-t8192']['model_close']['quit_send_outcome']='failed'
    rejected(a.require_closed_chunk_diagnostics,bad,*triple_args)
    bad=copy.deepcopy(diagnostics);bad['n65536-t8192']['reference_comparison']['strict_continuation_passed']=False
    rejected(a.require_closed_chunk_diagnostics,bad,*triple_args)
    bad_pins=copy.deepcopy(triple_pins);bad_pins['n65536-t6144']['sha256']='PENDING'
    rejected(a.require_closed_chunk_diagnostics,diagnostics,bad_pins,*triple_args[1:])
    # Finite prefix and balanced independent process roots; original three-case namespace remains distinct.
    assert a.QUIET_SEQUENCE==((1,6144),(1,8192),(2,8192),(2,6144),(3,6144),(3,8192))
    for index,(sample,chunk) in enumerate(a.QUIET_SEQUENCE):
        assert a.admitted_quiet_variant(sample,chunk,'on')==index
        a.require_quiet_prefix(a.QUIET_SEQUENCE[:index],sample,chunk)
        rejected(a.require_quiet_prefix,a.QUIET_SEQUENCE[:index]+((sample,chunk),),sample,chunk)
        if index:rejected(a.require_quiet_prefix,a.QUIET_SEQUENCE[:index-1],sample,chunk)
        first=Path(triple_pins['n65536-t8192']['path'])
        wanted=first if index==0 else a.quiet_root_path(run_root.B,*a.QUIET_SEQUENCE[index-1])/'record.json'
        assert a.quiet_predecessor_path(run_root.B,sample,chunk,first)==wanted
        assert a.quiet_root_path(run_root.B,sample,chunk)!=a.root_path(run_root.B,65536,chunk)
    for args_bad in [(0,6144,'on'),(4,8192,'on'),(1,8192,'off'),(True,6144,'on'),(1,6144,'on',1)]:
        rejected(a.admitted_quiet_variant,*args_bad)
    diag=diagnostics['n65536-t6144']
    quiet=a.quiet_environment(diag['environment'])
    a.require_quiet_environment(diag['environment'],quiet)
    assert set(diag['environment'])-set(quiet)==a.QUIET_REMOVED
    assert quiet==a.quiet_environment(diagnostics['n65536-t8192']['environment'])
    for key in a.QUIET_REMOVED:
        bad=dict(quiet);bad[key]=diag['environment'][key]
        rejected(a.require_quiet_environment,diag['environment'],bad)
    rejected(a.require_quiet_environment,diag['environment'],dict(quiet,STRATA_PREFILL_TIMING='0'))
    rejected(a.require_quiet_environment,diag['environment'],dict(quiet,STRATA_PREFILL_RING='16'))
    a.require_quiet_capture('bounded ordinary progress')
    for forbidden in ('strata trace: prompt chunk 0','strata prefill expert H2D: x',
                      'strata prefill route census v1: x','strata prefill timing: x'):
        rejected(a.require_quiet_capture,forbidden)
    # Protocol capture below is parser input only, not an actual GPU/quiet receipt.
    quiet_protocol=wp.validate(diag['request_protocol'],n=65536,chunk=6144)
    assert quiet_protocol['generated_tokens']==16
    assert wp.output_deltas(diag,diag)['strict_continuation_passed']
    # Minimal parser inputs, explicitly not quiet execution/closure receipts.
    left=dict(sample_index=1,pair_order='AB',requested_chunk=6144,
        request_protocol=diag['request_protocol'],validation=diag['validation'])
    right=dict(sample_index=1,pair_order='AB',requested_chunk=8192,
        request_protocol=diagnostics['n65536-t8192']['request_protocol'],validation=diagnostics['n65536-t8192']['validation'])
    pair=a.paired_quiet_timing(left,right)
    assert pair['b_minus_a_prompt_ms']==right['validation']['prompt_ms']-left['validation']['prompt_ms']
    assert a.paired_quiet_timing(right,left)==pair
    bad=copy.deepcopy(right);bad['sample_index']=2
    rejected(a.paired_quiet_timing,left,bad)
    bad=copy.deepcopy(right);bad['validation']['decode_ms']=0
    rejected(a.paired_quiet_timing,left,bad)
    import phase_observer
    phase_capture=('strata prefill timing: 63 tokens, GPU timeline 200 ms, wall 250 ms, host staging 17 ms:'
        ' embed+steps 20 (10.0%) hc read 30 (15.0%) gdn 40 (20.0%) qsa proj 50 (25.0%) gemm gate/up 60 (30.0%)\n'
        'strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) 3 ms, waiting for each chunk 4 ms, after each chunk (the draft layer, progress) 5 ms, PLE 6 ms\n'
        'strata prefill timer overhead: host wall CPU API overhead: mark 1.000 ms (marker calls 10, new event creations 4), fold 2.000 ms (fold calls 3, completed intervals 8); not GPU service or timer calibration\n')
    parsed=phase_observer.validate(phase_capture,batched_positions=63)
    assert parsed['host_fields']['ple_host_ms']==6 and parsed['observed_phases']['hc read']['ms']==30
    assert parsed['critical_path_complete'] is False and parsed['h2d_service'] is False
    assert parsed['instrumentation_total_overhead_measured'] is False and parsed['omitted_phase_labels']
    for before,after in [('wall 250 ms','wall nan ms'),('host staging 17 ms','host staging -1 ms'),
                         ('hc read 30 (15.0%)','unknown 30 (15.0%)'),('PLE 6 ms','PLE infinity ms'),
                         ('63 tokens','62 tokens'),('completed intervals 8','completed intervals 99')]:
        try:phase_observer.validate(phase_capture.replace(before,after),batched_positions=63)
        except ValueError:pass
        else:raise AssertionError('invalid current phase report accepted')
    pending_phase=json.loads((run_root.HERE/'phase.recipe.pending.json').read_text())
    rejected(phase_observer.require_phase_admission,pending_phase,pending,'PENDING')
    class NeverClose:
        def send(self,*args,**kwargs):raise AssertionError('mid-GEN must not claim safe normal QUIT boundary')
        def exit(self,*args,**kwargs):raise AssertionError('mid-GEN must remain unqualified owned-stop path')
    for state in (dict(ready_seen=False,gen_send_attempted=False,request_done_seen=False),
                  dict(ready_seen=True,gen_send_attempted=True,request_done_seen=False)):
        state['primary_model_failure']='synthetic incomplete request'
        assert a.close_after_gate_failure(NeverClose(),state,lambda:None,lambda:None) is False

    for index, (n, chunk) in enumerate(a.SEQUENCE):
        ref = Path('/reference') if chunk == 8192 else None
        assert a.admitted_variant(n, chunk, 'on', ref) == index
        a.require_sequence_prefix(a.SEQUENCE[:index], n, chunk)
        rejected(a.require_sequence_prefix, a.SEQUENCE[:index] + ((n,chunk),), n, chunk)
        if index: rejected(a.require_sequence_prefix, a.SEQUENCE[:index-1], n, chunk)
        wanted = a.matrix_path(run_root.B,32768,'on') if index == 0 else a.root_path(run_root.B,*a.SEQUENCE[index-1])/'record.json'
        assert a.predecessor_path(run_root.B,n,chunk,a.matrix_path(run_root.B,32768,'on')) == wanted
    for values in [(32768,6144,'on',None),(32768,8192,'off',Path('/reference')),
                   (65536,8192,'on',None),(65536,6144,'on',Path('/reference')),(True,8192,'on',Path('/reference'))]:
        rejected(a.admitted_variant,*values)
    anchor = matrix['n32768-on']
    a.require_same_recipe(anchor['argv'], anchor['argv'], 6144, 6144)
    argv8192 = list(anchor['argv']); argv8192[argv8192.index('--prefill')+1] = '8192'
    a.require_same_recipe(anchor['argv'], argv8192, 6144, 8192)
    wrong = list(argv8192); wrong[wrong.index('--max-context')+1] = '65536'
    rejected(a.require_same_recipe, anchor['argv'], wrong, 6144, 8192)
    env = a.compact_environment(anchor['environment'], 'on')
    assert env == anchor['environment']
    rejected(a.compact_environment,dict(env,STRATA_PREFILL_FUSED='1'),'on')

    # Literal expected schedules test coverage/order/full/tail independently of a formula-only check.
    expected_windows = {
        (32768,8192):[(0,8192),(8192,8192),(16384,8192),(24576,8191)],
        (65536,6144):[(0,6144),(6144,6144),(12288,6144),(18432,6144),(24576,6144),
                       (30720,6144),(36864,6144),(43008,6144),(49152,6144),(55296,6144),(61440,4095)],
        (65536,8192):[(0,8192),(8192,8192),(16384,8192),(24576,8192),(32768,8192),
                       (40960,8192),(49152,8192),(57344,8191)],
    }
    for (n,chunk), schedule in expected_windows.items():
        assert wp.windows(n,chunk) == schedule
        lines=['RESUME 0']+[f'PP {o+t} {n} 25000 100' for o,t in schedule]+['REUSED 0']
        lines += ['T 123','LP -0.1 123:-0.1 124:-1 125:-2 126:-3 127:-4']*16
        lines += [f'DONE 16 {n} 50000 2000 length 8 12 0 774 46560 0 0 0.0 {n} 0']
        value=dict(request_protocol=lines,validation=wp.validate(lines,n=n,chunk=chunk))
        assert wp.output_deltas(value,value)['strict_continuation_passed']
        bad=list(lines);bad[1]=f'PP {schedule[0][1]-1} {n} 25000 100'
        try:wp.validate(bad,n=n,chunk=chunk)
        except ValueError:pass
        else:raise AssertionError('missed row accepted')
        changed=copy.deepcopy(value);changed['request_protocol'][changed['request_protocol'].index('T 123')]='T 124'
        assert not wp.output_deltas(value,changed)['strict_continuation_passed']
        changed=copy.deepcopy(value);changed['request_protocol'][changed['request_protocol'].index(
            'LP -0.1 123:-0.1 124:-1 125:-2 126:-3 127:-4')]='LP -0.2 123:-0.2 124:-1 125:-2 126:-3 127:-4'
        delta=wp.output_deltas(value,changed)
        assert delta['lp5_differences'] and not delta['strict_continuation_passed']
        first_output=lines.index('T 123')
        early=lines[:first_output]+lines[first_output:first_output+2]+[f'DONE 1 {n} 50000 2000 stop 8 12 0 774 46560 0 0 0.0 {n} 0']
        early_value=dict(request_protocol=early,validation=wp.validate(early,n=n,chunk=chunk))
        assert not wp.output_deltas(value,early_value)['strict_continuation_passed']
        trace=''.join(f'strata trace: prompt chunk {o} of {n-1}\n' for o,t in schedule)
        assert wp.validate_trace(trace,n=n,chunk=chunk)['chunk_start_positions']==[o for o,t in schedule]
        # New geometry consumers exercise unchanged transfer/census validators with a complete
        # synthetic48-layer capture, explicitly never a completed GPU/measurement receipt.
        import transfer_ledger
        import route_census
        h2d=[]; census=[]
        for offset,size in schedule:
            for layer in range(48):
                head=f'device=0 stage=[0,48) chunk_offset={offset} pos={offset} tokens={size} layer={layer} mode=full_nonresident_stream issuer=inline'
                h2d.append('strata prefill expert H2D: '+head+' format=native_gguf gu_type=10 down_type=10 blob_bytes=16 submitted_calls=509 submitted_bytes=8144')
                census.append('strata prefill route census v1: '+head+
                    ' routing_observed=1 mmq_enabled=0 plan_fallback=1 mmq_direct=0 group_gather=0'+
                    f' routed_rows={size*10} expected_rows={size*10} active_experts=1 active_nonresident=1 zero_nonresident=508'+
                    ' gather_group_accepted_calls=0 gather_group_declined_calls=0 gather_individual_calls=0 native_dequant_gu_calls=1 native_dequant_down_calls=1 completion=checked')
        calls=len(h2d)*509;extent=calls*16
        capture=h2d+census+[f'strata prefill expert H2D total: scope=this_prefill_stage_run device=0 stage=[0,48) pos=0 tokens={n-1} chunks={len(schedule)} routed_calls=0 routed_bytes=0 full_stream_calls={calls} full_stream_bytes={extent} submitted_calls={calls} submitted_bytes={extent} completion=checked; request extents, not PCIe-exclusive bytes or DMA service']
        expectations=dict(expected_device=0,expected_stage=[0,48],expected_pos0=0,expected_n=n-1,
            expected_chunk_limit=chunk,expected_mode='full_nonresident_stream',expected_issuer='inline',
            expected_format='native_gguf',expected_layers={i:dict(blob_bytes=16,gu_type=10,down_type=10) for i in range(48)})
        ledger=transfer_ledger.validate(capture,**expectations)
        assert len(ledger['rows'])==len(schedule)*48
        route=route_census.validate(capture,enabled=True,h2d_validate=transfer_ledger.validate,h2d_expectations=expectations,
            n_experts=512,route_k=10,moe_layers=set(range(48)),nonresident_population={i:509 for i in range(48)})
        assert route['aggregates']['routed_rows']==(n-1)*480
        bad=capture.copy();bad[0]=bad[0].replace(f'tokens={schedule[0][1]} ',f'tokens={schedule[0][1]-1} ')
        try:transfer_ledger.validate(bad,**expectations)
        except ValueError:pass
        else:raise AssertionError('missing transfer row position accepted')
        bad=capture.copy();bad.pop(len(h2d))
        try:route_census.validate(bad,enabled=True,h2d_validate=transfer_ledger.validate,h2d_expectations=expectations,
                n_experts=512,route_k=10,moe_layers=set(range(48)),nonresident_population={i:509 for i in range(48)})
        except ValueError:pass
        else:raise AssertionError('missing census row accepted')
    # New 32768/6144 parser must agree with original qualified protocol on actual retained output.
    assert wp.validate(anchor['request_protocol'],n=32768,chunk=6144)==protocol.validate(anchor['request_protocol'],n=32768)

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
        entry,_,_=owner.run(phase,['/usr/bin/python3','-u','-c',child],
            dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1'),
            run_root.HERE,interaction=interaction,wall=20,cpu=40,text_cap=65536,total_cap=2<<20,rss_cap=128<<20)
        assert a.strong_closed_command(entry) and entry['exit_code']==0
        assert record['model_close']['quit_send_outcome']=='sent' and record['model_close']['quit_exit_outcome']=='exit observed'
        cases.append(dict(phase=phase,record=record,command=entry))
    assert all(run_root.m.sha(Path(path))==digest for path,digest in consumed.items())
    print(json.dumps(dict(passed=True,gpu_executed=False,model_executed=False,
        scope='CPU exported quiet balanced-prefix/env/reference and retained six+three diagnostic prerequisite structures; phase parser and finite normalQUIT fake CPU children only; no GPU/quiet measurement or invented PASS receipts',
        owner_sha256=run_root.m.sha(run_root.OWNER),source_pins={name:run_root.m.sha(run_root.HERE/name) for name in sorted(a.CODE_NAMES)},
        owned_quit_cases=cases,prerequisite_fixture_receipt_pins=consumed)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
