"""SOURCE ONLY parser for 39e adjacent compute-queue profiling-tag spans.

Not a launch controller or a quiet timing consumer. Main must separately admit
the pending phase recipe after CPU qualification and diagnostic prerequisites.
"""
import math
import re
from timer_overhead import validate as validate_overhead

PHASES = ('embed+steps','hc read','gdn','qsa proj','qsa indexer','qsa select','qsa attn',
    'router+shared','host grouping','gather','wait copy','dequant','gemm gate/up','gemm down',
    'combine','ple','kv stage','gdn conv+gates','gdn recurrence','gdn out proj')


def require(value,message):
    if not value: raise ValueError(message)


def finite(value, *, positive=False):
    result=float(value)
    require(math.isfinite(result) and (result>0 if positive else result>=0), 'finite nonnegative timing field')
    return result


def validate(text, *, batched_positions):
    require(type(batched_positions) is int and batched_positions in (63,32767), 'pending N64/N32768 phase scope')
    lines=text.splitlines()
    phase_lines=[x for x in lines if x.startswith('strata prefill timing: ') and not x.startswith('strata prefill timing: host:')]
    host_lines=[x for x in lines if x.startswith('strata prefill timing: host:')]
    require(len(phase_lines)==len(host_lines)==1, 'one complete phase and host report')
    head=re.fullmatch(r'strata prefill timing: (\d+) tokens, GPU timeline (\d+) ms, wall (\d+) ms, host staging (\d+) ms:(.*)',phase_lines[0])
    require(head is not None and int(head[1])==batched_positions, 'current n-1 Prefill invocation report')
    timeline,wall,staging=finite(head[2],positive=True),finite(head[3],positive=True),finite(head[4])
    rest=head[5]; observed={};last=-1
    pattern=re.compile(r' ('+'|'.join(re.escape(x) for x in sorted(PHASES,key=len,reverse=True))+r') (\d+) \((\d+\.\d)%\)')
    at=0
    for match in pattern.finditer(rest):
        require(match.start()==at, 'no unknown or malformed phase field')
        label=match[1];index=PHASES.index(label)
        require(index>last and label not in observed, 'source order and unique phase labels')
        duration,percent=finite(match[2]),finite(match[3])
        require(percent<=100 and duration<=timeline+1, 'bounded phase duration/percentage')
        observed[label]=dict(ms=duration,percent=percent)
        at=match.end();last=index
    require(at==len(rest) and observed, 'complete current emitted phase fields')
    require(abs(sum(x['ms'] for x in observed.values())-timeline)<=len(PHASES), 'rounded phase/timeline consistency')
    host=re.fullmatch(r'strata prefill timing: host: chunk setup \(PLE rows, the expert stream plan\) (\d+) ms, waiting for each chunk (\d+) ms, after each chunk \(the draft layer, progress\) (\d+) ms, PLE (\d+) ms',host_lines[0])
    require(host is not None, 'all four current host fields')
    overhead=validate_overhead(text,timing_enabled=True)
    return dict(capture_valid=True,batched_positions=batched_positions,compute_adjacent_marker_timeline_ms=timeline,
        prefill_invocation_wall_ms=wall,host_source_staging_ms=staging,observed_phases=observed,
        omitted_phase_labels=[x for x in PHASES if x not in observed],
        omitted_phase_semantics='source suppresses nonpositive internal buckets; no expected-marker/phase-coverage proof',
        host_fields=dict(zip(('chunk_setup_ms','chunk_wait_ms','post_chunk_callback_ms','ple_host_ms'),[finite(host[i]) for i in range(1,5)])),
        reported_timer_cpu_api_overhead=overhead,instrumentation_total_overhead_measured=False,
        event_query_success_implied_by_completed_report=True,hardware_timestamp_support_independently_qualified=False,
        exclusive_kernel_service=False,h2d_service=False,critical_path_complete=False,performance_eligible=False,
        scope='ordered adjacent command_end profiling-tag differences on compute queue; queue idle/copy dependency/host submission gaps can be included; fields overlap and must not be added as service times')


def require_phase_admission(recipe, qualification, qualification_sha256):
    require(recipe.get('status')=='passed' and recipe.get('main_reviewed') is True and recipe.get('cpu_qualified') is True,
            'separate actual Main/CPU phase admission; pending recipe cannot launch')
    require(qualification.get('status')=='passed' and qualification.get('main_reviewed') is True and
            re.fullmatch('[0-9a-f]{64}',qualification_sha256) is not None and
            recipe.get('qualification_sha256')==qualification_sha256, 'exact Main-resolved phase prerequisite reference')
    require(recipe.get('sequence')==[[64,8192],[32768,8192]] and recipe.get('timing_environment_delta')=={'STRATA_PREFILL_TIMING':'1'},
            'N64 sanity then32K8192; only coarse timer added to qualified diagnostic env')
    require(recipe.get('all_three_diagnostics_closed_pass') is True and recipe.get('quiet_arm') is False and
            recipe.get('total_overhead_measured') is False, 'separate coarse diagnostic; overhead unmeasured')
    require(recipe.get('source_revision')==qualification.get('published_revision')=='39e395a103af36323262cdc0dbaea27cba81f3e4' and
            recipe.get('source_pins')==qualification.get('production_source_pins') and
            isinstance(recipe.get('source_pins'),dict) and len(recipe['source_pins'])==42 and
            recipe.get('binary')==qualification.get('free_release_binary') and
            recipe.get('required_controller_cpu_receipt')==qualification.get('controller_cpu_receipt'),
            'actual unchanged qualified source/binary/CPU proof reference')
    require(all(recipe.get(k) is False for k in ('performance_eligible','broad_quality_qualified','full262144_qualified','restore_qualified')),
            'coarse observer cannot qualify broader claims')
