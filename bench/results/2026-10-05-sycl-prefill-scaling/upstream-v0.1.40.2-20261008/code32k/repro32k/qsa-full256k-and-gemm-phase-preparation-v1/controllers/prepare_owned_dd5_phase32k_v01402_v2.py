"""Harden diagnostic phase-report validation without launching a model."""
from pathlib import Path
import ast, datetime, hashlib, json, math

base = Path(__file__).parent
parent = base/'run_owned_dd5_phase32k_v01402_v1.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(parent) == '49f191e74cd85e81a6cd7711d6e5a71f6d60d147a097472430c9d304f062b877'
s = parent.read_text()
s = s.replace('owned-dd5-phase-v01402-code32k-', 'owned-dd5-phase-v2-v01402-code32k-')
start = s.index("            profile_checks={'reported_complete_prefill_prefix'")
end = s.index("            result['phase_reports']=phase_reports", start)
s = s[:start]+'''            profile_checks=validate_phase_reports(phase_reports,phase_tokens,profiling_errors,len(tokens)-1,result['diagnostic_wall_seconds'])
'''+s[end:]
helper = '''def validate_phase_reports(reports,prefixes,errors,expected_prefix,wall_seconds):
    expected_phases={'embed+steps','hc read','gdn','qsa proj','qsa indexer','qsa select',
                     'qsa attn','router+shared','host grouping','gather','wait copy','dequant',
                     'gemm gate/up','gemm down','combine','ple','kv stage','gdn conv+gates',
                     'gdn recurrence','gdn out proj','ple read wait','hc write+norm',
                     'layer weight preload','residual copy'}
    host_fields=['host_setup_ms','host_chunk_wait_ms','host_after_chunk_ms','ple_gather_wall_ms','ple_host_wait_ms']
    def finite(v): return type(v) in (int,float) and math.isfinite(v) and v>=0
    checks={'reported_complete_prefill_prefix':expected_prefix in prefixes,
            'phase_reports_present':bool(reports),'no_profiling_errors':not errors,
            'complete_finite_nonnegative_consistent_timeline':True,'timeline_not_exceed_request_wall':True}
    for report in reports:
        if not isinstance(report,dict):
            checks['complete_finite_nonnegative_consistent_timeline']=False
            checks['timeline_not_exceed_request_wall']=False
            continue
        values=report.get('phase_ms');timeline=report.get('gpu_timeline_ms')
        valid=isinstance(values,dict) and set(values)==expected_phases and all(finite(v) for v in values.values())
        valid=valid and finite(timeline) and timeline>0 and all(finite(report.get(k)) for k in host_fields)
        if valid: valid=abs(sum(values.values())-timeline)<0.001
        checks['complete_finite_nonnegative_consistent_timeline'] &= valid
        checks['timeline_not_exceed_request_wall'] &= finite(timeline) and timeline<=wall_seconds*1000+1000
    return checks

'''
marker="base = Path(__file__).parent\n"
assert s.count(marker)==1
s=s.replace(marker,helper+marker)
s=s.replace('Flushed diagnostic UR/ZE/ZEL/Strata logs retained; no profiler, additional phase waits, cache retirement or prefetch.',
            'Flushed diagnostic UR/ZE/ZEL/Strata logs retained; existing compute-phase marker barriers and two waits per chunk enabled. Copy timing disabled; no interposer, cache retirement or prefetch. Diagnostic intervals include host/copy gaps.')
s=s.replace("'reference_binary_sha256']='79a4b363d33f66f41d910be6274e609b4eb73f62afb0dc49bca72bf2a538d223'",
            "'reference_binary_sha256']=qualified['binary_sha256']")
s=s.replace("if not record['healthy'] or not record.get('math_gate_passed'):raise SystemExit(1)",
            "if not record['healthy'] or not record.get('math_gate_passed') or not record.get('profiling_gate_passed'):raise SystemExit(1)")
tree=ast.parse(s)
function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='validate_phase_reports')
namespace={'math':math}
exec(compile(ast.Module(body=[function],type_ignores=[]),str(parent),'exec'),namespace)
validate=namespace['validate_phase_reports']
names=['embed+steps','hc read','gdn','qsa proj','qsa indexer','qsa select','qsa attn','router+shared','host grouping','gather','wait copy','dequant','gemm gate/up','gemm down','combine','ple','kv stage','gdn conv+gates','gdn recurrence','gdn out proj','ple read wait','hc write+norm','layer weight preload','residual copy']
good={'phase_ms':{k:1.0 for k in names},'gpu_timeline_ms':24.0,**{k:0.0 for k in ['host_setup_ms','host_chunk_wait_ms','host_after_chunk_ms','ple_gather_wall_ms','ple_host_wait_ms']}}
cases=[('valid',[good],[32767],[],True),('missing',[],[32767],[],False),
       ('nonobject',[[1]],[32767],[],False),('badphase',[dict(good,phase_ms=[1])],[32767],[],False),
       ('missingphase',[dict(good,phase_ms={'gdn':24.0})],[32767],[],False),
       ('nan',[dict(good,gpu_timeline_ms=float('nan'))],[32767],[],False),
       ('negative',[dict(good,host_setup_ms=-1)],[32767],[],False),
       ('inconsistent',[dict(good,gpu_timeline_ms=25)],[32767],[],False),
       ('too_long',[dict(good,phase_ms={k:1000.0 for k in names},gpu_timeline_ms=24000.0)],[32767],[],False),
       ('wrongprefix',[good],[8191],[],False),('api_error',[good],[32767],['UR_RESULT_ERROR_INVALID_OPERATION'],False)]
results=[]
for name,reports,prefixes,errors,expected in cases:
    checks=validate(reports,prefixes,errors,32767,1.0)
    assert all(checks.values())==expected,(name,checks)
    results.append({'name':name,'expected_accept':expected,'checks':checks,'passed':True})
assert s.index("assert not preceding['active']") < s.index("out = base/f'owned-dd5-phase-v2")
assert "env['STRATA_PREFILL_TIMING']='1'" in s and 'STRATA_PREFILL_TRANSFER_TIMING\']=\'1' not in s
target=base/'run_owned_dd5_phase32k_v01402_v2.py'
assert not target.exists();target.write_text(s)
record={'active':False,'prepared':True,'gpu_launched':False,'engine_built':False,'AST_passed':True,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'preparer_sha256':sha(__file__),'parent_controller_sha256':sha(parent),
        'controller':str(target),'controller_sha256':sha(target),'CPU_validation_cases':results,
        'minimum_performance_input_tokens':32768,
        'scope':'Prepared only; malformed, incomplete, error and invalid phase timelines rejected without exceptions. Same DD5 binary, no GPU invocation. Initial logged numerical/instrumentation run still required.'}
p=base/'prepare-owned-dd5-phase32k-v01402-v2.json';assert not p.exists();p.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'prepared':True,'controller_sha256':sha(target),'CPU_validation_cases':len(results),'gpu_launched':False}))
