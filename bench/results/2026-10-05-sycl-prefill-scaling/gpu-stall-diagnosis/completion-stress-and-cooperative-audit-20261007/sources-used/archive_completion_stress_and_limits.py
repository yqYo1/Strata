"""Freeze completed stress/resource tests, excluding executables and whole outputs."""
from pathlib import Path
import datetime,hashlib,json,sys
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exec(compile((base/'archive_lazy_full_and_dequant.py').read_text().split('\nfull = json.loads',1)[0],str(base/'archive_lazy_full_and_dequant.py'),'exec'))
directory=parent/'completion-stress-and-cooperative-audit-20261007'
assert not directory.exists();directory.mkdir()

def save_nonbinary_folder(source,prefix):
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name in ['probe','query'] or path.suffix in ['.bin','.o','.a','.d'] or '__pycache__' in path.parts:continue
        target=Path(prefix)/path.relative_to(source)
        if path.stat().st_size<=300000:copy(path,directory,target)
        else:
            dest=directory/target.with_name(path.name+'.tail');dest.parent.mkdir(parents=True,exist_ok=True)
            with path.open('rb') as stream:stream.seek(max(0,path.stat().st_size-65536));dest.write_bytes(stream.read())
            meta=dict(private_path=str(path),private_bytes=path.stat().st_size,private_sha256=digest(path),saved_tail=str(dest.relative_to(directory)))
            dest.with_name(dest.name+'.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

runs=[]
for name in ['dequant-completion-stress-usm','dequant-completion-stress-retire','dequant-gemm-completion-stress-usm','dequant-gemm-completion-stress-retire']:
    r=json.loads((base/name/'record.json').read_text())
    assert r['passed'] and not r['active'] and r['exit_code']==0 and not r['new_fault_messages']
    assert not any(r['cleanup'].values()) and r['events'][-1]['full_checks']==40 and r['events'][-1]['iterations']==20000
    for k in ['inferior','debugger']:
        owner=r[k];now=process_identity(owner['pid']);assert not now or now['start_ticks']!=owner['start_ticks']
    assert digest(base/('dequant_gemm_completion_stress.cpp' if 'gemm' in name else 'dequant_completion_stress.cpp'))==r['source_sha256']
    for head in r['reference_heads']+r.get('pipeline_heads',[]):
        p=base/name/head['file'];assert p.stat().st_size==head['bytes'] and digest(p)==head['sha256']
    runs.append(dict(case=name,passed=True,binary_sha256=r['binary_sha256'],mode=r['mode'],
                     final=r['events'][-1],diagnostic_seconds=r['elapsed_seconds'],
                     reference_heads=r['reference_heads'],pipeline_heads=r.get('pipeline_heads',[]),
                     source_regions=r['input_regions'],cleanup=r['cleanup']))
    save_nonbinary_folder(base/name,name)
assert runs[0]['reference_heads']==runs[1]['reference_heads']==runs[2]['reference_heads']==runs[3]['reference_heads']
assert runs[2]['pipeline_heads']==runs[3]['pipeline_heads']
for name in ['dequant-completion-stress-build','dequant-gemm-completion-stress-build','dequant-gemm-completion-stress-build-v2','prefill-cooperative-query']:
    save_nonbinary_folder(base/name,name)
query=json.loads((base/'prefill-cooperative-query/query-record.json').read_text())
assert query['passed'] and not query['active'] and not query['new_fault_messages'] and query['driver_kernel_submissions_observed']==0
assert query['results'][-1]==dict(stage='PASS',queried=25,exceeds=22,kernels_submitted=0)
for s in query['steps']:
    assert s['exit_code']==0 and not s['timed_out'] and not s['still_alive'] and not Path('/proc',str(s['pid'])).exists()
for name in ['dequant_completion_stress.cpp','dequant_gemm_completion_stress.cpp',
             'build_dequant_completion_stress.py','prepare_dequant_gemm_stress.py',
             'build_dequant_gemm_completion_stress.py','build_dequant_gemm_completion_stress_v2.py',
             'run_dequant_completion_stress.py','run_dequant_gemm_completion_stress.py',
             'prefill_cooperative_query.cpp','build_prefill_cooperative_query.py',
             'run_prefill_cooperative_query.py','archive_completion_stress_and_limits.py']:
    copy(base/name,directory,Path('sources-used')/name)
copy(base/'observed-cooperative-launches-upstream-recheck.json',directory,Path('observed-cooperative-launches.json'))
copy(base/'observed-cooperative-source-targets.json',directory,Path('observed-cooperative-source-targets.json'))
mapping=json.loads((base/'observed-cooperative-source-targets.json').read_text())
assert len(mapping['rows'])==45 and all(len(r['source_candidates'])==1 for r in mapping['rows'])
source_paths={r['source_candidates'][0]['path'] for r in mapping['rows']}
assert len(source_paths)==8
for r in mapping['rows']:
    candidate=r['source_candidates'][0]
    assert digest(root/candidate['path'])==candidate['sha256']
assessment=dict(scope='Bounded real-weight dequant and GEMM/SwiGLU completion stress and observed cooperative resource audit; not model parity, speed or full capacity',
                recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=runs,
                cross_mode_all_dequant_heads_equal=True,cross_mode_all_pipeline_heads_equal=True,
                cooperative_query=query['results'],kernel_launches_by_resource_query=0,
                observed_cooperative_source_mapping=dict(combinations=45,source_files=8,complete_call_graph_proof=False),
                producer_consumer_queue='same in-order queue',
                input_scope='One real layer17/expert0 blob replicated across512 aligned slots in1363148800-byte device arena; IQ3_S GU andIQ4_NL down; eight synthetic activation rows for GEMM cases',
                VMM_scope='One64MiB mapping consumed and synchronized, fully unmapped, physical memory deleted and virtual range freed before stress; descriptor exists and GEMM consumers run before retirement in pipeline case; no decode graph or other model state',
                limit_scope='25 observed original kernel/local-size combinations; zero dynamic local bytes. Exceeding this optimistic bound is a violation; within it does not prove a launch with additional dynamic local bytes is valid.',
                inference='High regular dequant count, large source arena and one isolated VMM retirement do not reproduce the model stall. Real GEMM/SwiGLU consumers at safe20-group SwiGLU also complete. Multiple other model cooperative launches exceed actual limits; correcting only the two dequant flags leaves these invalid settings.',
                source_rewrite_adopted=False,full_model_hang_cause_proven=False,all262144_cell_suite_passed=False,
                manual_reset_rebind_reboot_service_or_package_changes=False)
(directory/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
(directory/'README.md').write_text('''# Completion stress and cooperative audit on 2026-10-07

Four owned diagnostic processes exit normally on Arc B570 10 GiB / Ryzen
5600X / 128 GiB RAM, kernel 7.0.0-38-generic, NEO 26.31.39395.14 and
oneAPI 2026.1.1. Each performs 20,000 GU/down pairs through the actual updated
IQ wrappers with regular launches, full Level Zero/UR logs and validation.
Each checks all guarded FP16 output bytes 40 times. The same real
layer17/expert0 IQ3_S/IQ4_NL weights are replicated across 512 aligned slots
in a 1,363,148,800-byte device arena. This is a real-weight component stress
test, not all experts or a complete model request.

The two extended cases also use unchanged production Gemm::f16 and
swiglu_interleaved wrappers, eight synthetic FP16 activation rows, and
the original producer/consumer wait boundaries. All GU FP32, SwiGLU FP16
and down FP32 outputs are finite, guarded and byte-identical to each
process's initial reference. Their reference outputs also match each other
between modes. The actual SwiGLU cooperative limit is 144 groups; this
fixture uses 20. No arithmetic or production launch geometry was rewritten.

One mode per fixture consumes a 64 MiB virtual-memory mapping, waits,
unmaps the full range, releases physical memory and frees the virtual range
before the stress. The extended fixture also performs real GEMM consumers
before retiring the mapping. This follows the ownership/order requirements
of the [SYCL virtual-memory extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_virtual_mem.asciidoc).
It excludes recorded decode graphs, other model state, routing and staging
threads. A single mapping does not cover the model's whole restoration path.

Diagnostic process durations are 9.100/9.269 seconds for the basic cases and
29.352/29.199 seconds for the extended cases. These include logs, validation,
GDB and initialization; none is accepted as clean throughput. All owners
are gone and no new xe fault is recorded. No reset, rebind, reboot, service
or package change was used. These results fail to reproduce the model stall;
they do not establish general hang prevention or model output parity.

The separate resource query submits no kernels. Native handles and complete
group/local shapes are recovered from the failed unchanged-control trace,
then 25 original prefill/IQ kernel/local-size combinations are queried using
unchanged production objects. Twenty-two observed cooperative launches
exceed the measured maximum. Examples are GDN output normalization
49,152 groups versus 288, broadcast 40,960 versus 144, RMS rows 24,576 versus 144,
and the original large SwiGLU 1,988 versus 144. Removing only the two IQ
dequant flags leaves these invalid settings. See the complete
[query receipt](prefill-cooperative-query/query-record.json) and
[observed launch metadata](observed-cooperative-launches.json).

The query uses zero additional dynamic local bytes, so its limit is an
optimistic bound for any launch with extra dynamic local storage. Counts
above it violate the [Level Zero cooperative launch contract](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#cooperative-kernels);
counts below it do not validate every other launch requirement. Resource
limits alone do not identify which operation caused the completion stall.
All 45 observed kernel/local-size combinations map uniquely to eight source
files; [source hashes and class names](observed-cooperative-source-targets.json)
record that mapping. This is not a complete call-graph or synchronization
proof. The observed gather_rows kernel in verify_kernels.dp.cpp is a row
copier, but other kernels in that file include persistent/global coordination.
Review properties per function; do not remove them from the entire file.
Production source and binary remain unchanged.
Full-model and all 262144-cell gates and PP1000/TG70 remain incomplete.

The first extended CPU link failed because the original archive preceded
the explicit regular IQ object and introduced duplicate definitions. Its
failed receipt is retained. The corrected v2 link places the explicit object
before archives and is the only extended fixture used on GPU. Controllers,
sources, input-region hashes, bounded tails and full private-log hashes are
included. Executables, whole output arrays and large logs stay private.
The manifest covers every archived file except itself.
''')
print(json.dumps(manifest(directory),indent=2))
