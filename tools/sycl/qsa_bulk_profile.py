#!/usr/bin/env python3
"""Observe the faster bulk-mirror path with a matched clean/profile/clean bracket.

Main-authored from the owned bench.sh and collector controls; repository LICENSE.
This reuses byte-identical collector fixtures, not old model coverage or speed.
"""
import ast
import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import qsa_bulk_controls as admission

P, B, RESULT = admission.P, admission.B, admission.RESULT
THIS = 'tools/sycl/qsa_bulk_profile.py'
ROOT, KEY = B / 'qsa16-bulk-fast-profile-v1', 'qsa16_bulk_fast_profile_v1'
original, require, sha = admission.original, admission.require, admission.sha


def collector_proof(native):
    ctx = original.collector_context()
    row = json.loads((original.COLLECTOR_ROOT / 'record.json').read_text())
    require(ctx.base.committed(RESULT)[original.COLLECTOR_KEY] == ctx.compact(row)
        and row['stage'] == 'closed' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active'] and len(row['cases']) == 1,
        'original closed exact collector qualification required')
    case, = row['cases']
    trace = case['trace']
    fixture = native['binaries']['sycl_submission_test']
    require(all(row['binary'][k] == fixture[k] for k in ('bytes', 'sha256'))
        and row['native_validation_probe'] == native['native_validation_probe']
        and row['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        and row['driver'] == admission.model.original.guards.driver_guard()
        and case['passed'] and case['native_exit_code'] == 0 and case['ownership_closed']
        and case['fresh_faults'] == [] and trace['collection_active_from_exec']
        and trace['native_host_flow_joins_unique']
        and trace['class_inventory']['observed'] == dict(direct=1, producer=37, consumer=24),
        'same fixture bytes,probe/runtime/boot,exact classes and joins required')
    before = row['sources']
    recovered = {p:__import__('hashlib').sha256(raw).hexdigest() for p,raw in
        admission.common.source.blobs(P,row['source_commit'],before)}
    require(recovered == before, 'original collector source must remain recoverable')
    for p,h in before.items():
        if p.startswith('tools/sycl/') and p not in admission.UPDATED_CONTROLLERS:
            require(sha(P/p) == h, 'original collector control changed: ' + p)
    raw = admission.common.source.git(P,'show',row['source_commit']+':'+original.THIS)
    def functions(text):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body
                if isinstance(n,ast.FunctionDef)}
    old, now = functions(raw), functions((P/original.THIS).read_bytes())
    critical = ('compatibility_inventory','collector_controls','collector_context','collector_prepare')
    require(all(old[k] == now[k] for k in critical),
        'qualified collector/class/environment functions changed')
    assets = ctx.profile.collector_assets(json.loads(RESULT.read_text()))
    require(assets == row['assets'] and all(sha(v['path']) == v['sha256']
        and Path(v['path']).stat().st_size == v['bytes'] for v in assets.values()),
        'exact qualified collector/launcher bytes required')
    return dict(key=original.COLLECTOR_KEY,source_commit=row['source_commit'],
        original_control_files_recovered=len(recovered),
        current_engine=native['binaries']['strata'],
        byte_identical_qualified_fixture=fixture,
        original_fixture_source_commit=row['binary'].get('origin_source_commit'),
        assets=assets,critical_collector_functions_unchanged=True,
        reused_fixture_qualification=True,original_reference_ELF_required=False,
        limits='Same executed submission fixture bytes,collector/launcher,control code,probe/runtime/boot;'
            'no current-model coverage,observer overhead,speed or physical saturation inherited.')


def observer_history():
    value = admission.validation.committed(RESULT,('qsa16_stream_fast_profile_v1',))
    require(value['stage'] == 'failed' and not value['passed'] and not value['complete']
        and not value['active'] and value['post_failure_reconciliation']['named_session_removed'],
        'preserve original256MiB SIGXFSZ failure and completed physical cleanup')
    return dict(key='qsa16_stream_fast_profile_v1',original_passed=False,
        original_failure_evidence='057f7103feb6e413eca04d536a3e0f1ea117c780',
        limits='Historical failure only;raw retired after recorded closure. Fresh bulk cohort uses its own'
            'finite1GiB/file,1.5GiB/job budget and exact named-session cleanup.')


def context():
    ctx = admission.legacy.load_module('qsa_bulk_current_profile', original.__file__)
    ctx.configure_model_profile()
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.basis = lambda: admission.basis({THIS})
    ctx.check_sample = admission.check_sample_exact
    ctx.profile_control_proof, ctx.previous_profile_failure = collector_proof, observer_history
    base_environment, base_route = ctx.environment, ctx.request_route
    def environment(row,plan):
        env = base_environment(row,plan)
        env[admission.FLAG] = '1'
        return env
    def route(text,plan):
        value = base_route(text,plan)
        value.update(bulk_mirror=1,
            selector_evidence='Actual typed-QSA entry plus immutable pre-exec bulk flag checked at both live gates;exact recovered production dispatch.',
            actual_bulk_command_count_measured=False)
        return value
    ctx.environment, ctx.request_route = environment, route
    return ctx


def prepare():
    ctx = context()
    ctx.profile_prepare()
    row = json.loads((ROOT/'record.json').read_text())
    row['plan']['question'] = 'After repeated clean bulk gains,which GPU services,copy requests and disjoint CPU regions remain in the actual32Kfast route?'
    row['plan']['settings'] = row['plan']['settings'].replace('Same b9aa/1e598 five-ELF native generation',
        'Same611a0fd6/56cad9 five-ELF native generation') + ' Bulk mirror1 is fixed in all three native processes;exact24output policy is prospective.'
    row['plan']['retention'] = ('Reduce the current bulk model trace immediately after native exit. Finite1GiB/file,1.5GiB/job,2GiBwhole,2400s/104GiBRSS;'
        'keep useful CPU partitions,services/union/overlap,request/API bytes,clocks/joins,all6samples and observer ratios.'
        'Retire successful traces/logs/JIT after bracket closure and committed reduction. Exact named collector cleanup remains mandatory.'
        'Current five ELFs remain required;old comparator ELFs are provenance-only in this prospective reader. Source-recovery retirement is separate.')
    row['plan']['limits'] += ' Old32Kappend service is not post-bulk arithmetic cost. No new quantizer is selected without this attribution.'
    row['exact_output_host_controls'] = admission.host_controls()
    ctx.save(row)
    print('Prepared current bulk32Kclean/profile/clean bracket;commit before host/native.',flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare','host','0','1','2'),
        'qsa_bulk_profile.py prepare|host|0|1|2')
    require(sys.executable == admission.model.PYTHON,'use pinned frontend interpreter')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        admission.model.original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'host': original.comparison.host(context())
        else: context().profile_run(int(sys.argv[1]))
