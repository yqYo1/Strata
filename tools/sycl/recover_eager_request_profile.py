#!/usr/bin/env python3
"""Analyze the already-captured 64K trace under a separate output budget.

Main-authored, repository LICENSE. The original model campaign remains failed.
The owned collector trace and analysis input are verified against the original
process before this host-only analysis starts.
"""
import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import eager_request_profile as profile

P, B, RESULT = profile.P, profile.B, profile.RESULT
ROOT = B / 'eager-request-analysis-v2'
ARCHIVE = '8173b1aab2940ec394236fe2330b2c29a4971509'
FAILED_ANALYSIS = '7c14323071207aeda1de23930951a08f26072026'
KEY = 'post_run_analysis_v2'
THIS = 'tools/sycl/recover_eager_request_profile.py'
owner_api = profile.comparison.build.supervision
require, sha = profile.require, profile.sha


def committed_source(revision, name):
    raw = profile.probe.pti.blob(P, revision, name)
    require(raw == (P / name).read_bytes(), 'analysis source differs: ' + name)
    return profile.probe.pti.pti.sha(raw)


def input_bytes():
    path = str(RESULT.relative_to(P))
    archived = json.loads(profile.probe.pti.blob(P, ARCHIVE, path))[profile.KEY]
    failure = json.loads(profile.probe.pti.blob(P, FAILED_ANALYSIS, path))[profile.KEY]['post_run_analysis']
    live = profile.workspace.device.build.ur_validation.committed(RESULT, (profile.KEY,))
    original = dict(live)
    require(original.pop('post_run_analysis') == failure and failure['stage'] == 'failed'
        and failure['ownership_closed'] and failure['exit_code'] == 1
        and 'NoneType' in failure['error'], 'preserve original failed standalone analysis')
    original.pop(KEY, None)
    require(archived == original and archived['stage'] == 'failed' and not archived['passed']
        and not archived['complete'] and not archived['active'] and archived['ownership_closed']
        and len(archived['processes']) == 2, 'preserve exact original failed cohort')
    prior = archived['processes'][1]
    require(prior['native_exit_code'] == 0 and prior['ownership_closed'] and prior['boot_unchanged']
        and prior['fresh_faults'] == [] and not prior['passed']
        and prior['analysis']['exit_code'] == -15 and not prior['analysis']['ownership_closed'],
        'closed native process and preserved analysis failure required')
    case = profile.ROOT / '1'
    execution = json.loads((case / 'execution.json').read_text())
    require(not execution['active'] and all(execution.get(k) == v for k, v in prior.items())
        and all(owner_api.closed(c) for c in execution['commands']), 'original native ownership differs')
    trace = Path(prior['analysis']['command'][3])
    receipt = case / 'analysis-input.json'
    require(trace == case / 'trace' / f"strata.{prior['native_identity']['pid']}.json"
        and receipt == Path(prior['analysis']['command'][4])
        and sha(receipt) == prior['analysis']['input_sha256']
        and 0 < trace.stat().st_size <= profile.reader.MAX_BYTES, 'original analysis inputs differ')
    data = json.loads(receipt.read_text())
    require(data['trace_sha256'] == sha(trace) and len(data['requests']) == 3
        and all(q['request_sha256'] == s['request_sha256'] for q, s in zip(data['requests'], prior['samples'])),
        'original trace digest or request identities differ')
    return live, archived, trace, receipt, data


def save(row, evidence):
    values = json.loads(RESULT.read_text())
    values[profile.KEY] = dict(row, **{KEY: evidence})
    profile.core.write(RESULT, values)


def prepare():
    row, archived, trace, receipt, data = input_bytes()
    require(not ROOT.exists() and KEY not in row, 'fresh analysis only')
    source = profile.workspace.device.build.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = {name: committed_source(source, name)
        for name in (THIS, profile.THIS, profile.READER, 'tools/sycl/request_profile_reader.py',
            'tools/test_request_resource_summary.py', 'tools/test_request_profile_reader.py',
            'tools/sycl/measurement_owner.py', *profile.capacity.POLICIES)}
    require(all(committed_source(archived['source_commit'], name) == pins[name]
        for name in (profile.THIS, 'tools/sycl/request_profile_reader.py',
            'tools/sycl/measurement_owner.py')), 'analysis grammar and original owner must be unchanged')
    prior_reader = profile.probe.pti.blob(P, archived['source_commit'], profile.READER)
    old = b"zip(receipt['requests'], original['requests'])"
    new = b"zip(receipt['requests'], check['requests'])"
    require(prior_reader.count(old) == 1 and prior_reader.replace(old, new) == (P / profile.READER).read_bytes(),
        'only the optional comparison fallthrough may change')
    previous_bytes = owner_api.artifact_bytes(receipt.parent)
    require(previous_bytes > profile.ANALYSIS_BUDGET['total_cap'],
        'cannot verify that the original analysis output-root budget counted the capture')
    plan = dict(stage='prepared', source_commit=source, pins=pins, original_evidence_commit=ARCHIVE,
        failed_analysis_commit=FAILED_ANALYSIS,
        source_trace=dict(path=str(trace), bytes=trace.stat().st_size, sha256=data['trace_sha256']),
        source_receipt=dict(path=str(receipt), sha256=sha(receipt)),
        original_analysis_artifact_scope=dict(counted_parent=str(receipt.parent),
            current_parent_bytes=previous_bytes, trace_alone_bytes=trace.stat().st_size,
            total_cap_bytes=profile.ANALYSIS_BUDGET['total_cap'], confirmed_trace_alone_exceeds_cap=True),
        command=[profile.PYTHON, '-B', str(P / profile.READER), str(trace), str(receipt), str(ROOT / 'resources.json')],
        output_root=str(ROOT), budgets=profile.ANALYSIS_BUDGET,
        scope='Offline interpretation of one prior closed native trace. Original failed 4-process campaign and '
              'its observer-overhead decision remain unqualified. No native GPU work or new timing sample.')
    ROOT.mkdir(mode=0o700)
    profile.core.write(ROOT / 'plan.json', plan)
    save(row, plan)
    print('Pinned independent host-only analysis; commit its plan before execution.', flush=True)


def run():
    row, _, trace, receipt, data = input_bytes()
    plan = row[KEY]
    require(plan['stage'] == 'prepared' and json.loads((ROOT / 'plan.json').read_text()) == plan
        and not (ROOT / 'execution.json').exists()
        and all(committed_source(plan['source_commit'], n) == h for n, h in plan['pins'].items())
        and plan['source_trace']['sha256'] == data['trace_sha256']
        and plan['source_receipt']['sha256'] == sha(receipt)
        and not trace.is_relative_to(ROOT) and not receipt.is_relative_to(ROOT)
        and plan['budgets'] == profile.ANALYSIS_BUDGET, 'frozen host-only plan differs')
    owner = owner_api.Owner(ROOT / 'owned',
        lambda: profile.core.write(ROOT / 'execution.json', owner.commands))
    entry, _, err = owner.run('summarize', plan['command'], dict(PATH='/usr/bin:/bin', LANG='C'), P,
        **plan['budgets'])
    evidence = dict(plan, stage='failed', exit_code=entry['exit_code'],
        ownership_closed=owner_api.closed(entry), GPU_executed=False)
    try:
        require(evidence['ownership_closed'] and entry['exit_code'] == 0,
            'offline analysis failed: ' + str(entry.get('error')) + ' ' + err.read_text()[-1000:])
        resources = json.loads((ROOT / 'resources.json').read_text())
        require(resources['trace_sha256'] == data['trace_sha256']
            and resources['totals_match_request_reader'] and len(resources['requests']) == 3,
            'original request identity or trace conservation differs')
        evidence.update(stage='closed', result=resources, output_sha256=sha(ROOT / 'resources.json'))
    except BaseException as error:
        evidence['error'] = repr(error)
        raise
    finally:
        save(row, evidence)
    print('Three prior request windows summarized. Original V1 outcome remains failed.', flush=True)


if __name__ == '__main__':
    require(sys.executable == profile.PYTHON and sys.argv[1:] in (['prepare'], ['run']),
        'pinned Python: prepare | run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        profile.comparison.build.guards.no_model_owners([])
        (prepare if sys.argv[1] == 'prepare' else run)()
