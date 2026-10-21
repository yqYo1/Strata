#!/usr/bin/env python3
"""Shared source and request-evidence gates for official long-generation diagnostics.

Closed comparison/native-runtime launchers are preserved in original Git history.
The maintained resource_engine.py owns current passive model execution.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import bench_engine as common
import measurement_guards as guards
from build_engine import blobs

P, B, A = common.P, common.B, common.A
RESULT = A / 'decode-long-generation-current.json'
SETTINGS = common.SETTINGS
WORKLOAD_ID = 'native16_decode_long_20261019'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P)


def committed(path):
    data = path.read_bytes()
    guards.require(data == git('show', 'HEAD:' + str(path.relative_to(P))),
                   'commit the exact input first: ' + str(path))
    return json.loads(data)


def benchmark_pins(revision, match_worktree=True):
    names = git('ls-tree', '-r', '--name-only', revision, '--', 'serve', 'tools', 'bench.sh').decode().splitlines()
    names = [n for n in names if n == 'bench.sh' or
             (n.endswith('.py') and not n.startswith('tools/hip/'))]
    pins = {}
    for name, data in blobs(P, revision, names):
        if match_worktree:
            guards.require((P / name).read_bytes() == data, 'uncommitted benchmark input: ' + name)
        pins[name] = hashlib.sha256(data).hexdigest()
    return pins


def native_end(root, record, plan, boot, leader):
    start=json.loads((root/'engine-start.json').read_text())
    end=json.loads((root/'engine-exit.json').read_text())
    guards.require(type(end.get('exit_code')) is int and end==dict(start,exit_code=0) and start['boot_id']==boot and
        start['binary']==record['binary'] and start['args']==['--serve',*plan['config']['args']] and
        start.get('native_validation_probe')==record['native_validation_probe'] and
        not {'resource_collector','native_profile'}&set(start), 'native identity/closure differs')
    for key in ('native','wrapper'):
        identity=start[key]
        guards.require(set(identity)=={'pid','start_ticks','sid'} and
            all(type(v) is int and v>0 for v in identity.values()) and identity['sid']==leader,
            'native ownership identity differs')
    guards.require(start['native']['pid']!=start['wrapper']['pid'], 'native/wrapper identity aliases')
    return end


def draft_rows(text):
    guards.require(not re.search(r'^strata (?:draft kv|serve: SNAPSHOT_VERIFY)', text, re.M),
                   'unexpected timer or snapshot-restore observer')
    pattern = r'strata serve: DRAFT_PREFILL path=(batched|token) mode=(-?\d+) cells=(\d+)'
    parsed = []
    for line in text.splitlines():
        if not line.startswith('strata serve: DRAFT_PREFILL'): continue
        match = re.fullmatch(pattern, line)
        guards.require(match is not None, 'malformed draft-KV marker')
        parsed.append(match.groups())
    return [dict(path=path, mode=int(mode), cells=int(cells)) for path, mode, cells in parsed]


def request_evidence(text, samples):
    """Bind draft work to every native request, including all four warmups."""
    headers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$', text, re.M))
    guards.require(len(headers) == len(samples), 'missing/extra native request footer')
    previous = 0; evidence = []
    for header, sample in zip(headers, samples):
        guards.require(tuple(map(int, header.groups())) == (sample['prompt_tokens'], sample['reused_tokens'],
            sample['read_tokens'], sample['generated_tokens']) and sample['reused_tokens'] == 0,
            'native/journal request binding differs')
        rows = draft_rows(text[previous:header.start()]); cells = sample['read_tokens'] - 1
        expected = [8192] * (cells // 8192) + ([cells % 8192] if cells % 8192 else [])
        guards.require(rows and all(r['path'] == 'batched' and r['mode'] == 2 for r in rows)
            and [r['cells'] for r in rows] == expected, 'draft-KV marker route/chunks do not conserve request')
        evidence.append(dict(read_tokens=sample['read_tokens'], draft_cells=sum(r['cells'] for r in rows),
            batch_count=len(rows), path='batched', mode=2))
        previous = header.end()
    guards.require(not draft_rows(text[previous:]), 'unbound trailing draft batch')
    return evidence


def sample_eligible(row):
    """Independently reject mislabeled short, incomplete or unclocked samples."""
    def integer(key, minimum):
        value = row.get(key)
        return type(value) is int and value >= minimum

    offered, accepted = row.get('drafts_offered'), row.get('drafts_accepted')
    return bool(row.get('eligible') is True and row.get('speed_eligible') is True and
        row.get('ok') is True and row.get('stream_complete') is True and
        row.get('finish_reason') in ('stop', 'length') and row.get('reused_tokens') == 0 and
        integer('prompt_tokens', 32768) and integer('read_tokens', 32768) and
        row['prompt_tokens'] == row['read_tokens'] and integer('usage_generated_tokens', 1024) and
        row['usage_generated_tokens'] <= 16384 and
        row.get('generated_tokens') == row['usage_generated_tokens'] and
        row.get('generated_tokens_source') == 'usage' and integer('output_chars', 512) and
        integer('reasoning_chars', 1) and row.get('completion_includes_thinking') is True and
        type(offered) is int and offered >= 256 and type(accepted) is int and 0 <= accepted <= offered and
        isinstance(row.get('acceptance_rate'), (int, float)) and
        math.isclose(row['acceptance_rate'], accepted / offered, rel_tol=0, abs_tol=1e-12) and
        row.get('decode_tps_source') == 'engine' and all(
            isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool) and
            math.isfinite(row[key]) and row[key] > 0 for key in ('decode_ms', 'decode_tps', 'total_s')))


def collect_samples(generation):
    # Preserve every completed/failed client row before assessing eligibility.
    rows = []
    for workload, case in generation.get('workloads', {}).items():
        for request in case.get('requests', []):
            row = dict(request, workload=workload)
            row['controller_eligible'] = sample_eligible(row)
            rows.append(row)
    return sorted(rows, key=lambda r: r.get('order', -1))


def require_complete_generation(generation, rows, workloads=('regular-code', 'varied-prose')):
    guards.require(generation.get('workload_version') == 4 and generation.get('thinking') is True and
        generation.get('reasoning_budget_tokens') == 0 and generation.get('max_tokens') == 16384,
        'unexpected long-generation workload or thinking/output budget')
    guards.require(len(rows) == len(workloads) and
        [(r.get('workload'), r.get('run'), r.get('order')) for r in rows] ==
        [(name, 0, i) for i, name in enumerate(workloads)] and
        all(r['controller_eligible'] for r in rows), 'ineligible or incomplete long-generation pilot')
    if len(workloads) == 1:
        guards.require(generation.get('diagnostic_subset') is True and
            generation.get('selected_workloads') == list(workloads), 'single-workload diagnostic selection differs')


def journal_evidence(root):
    sys.path.insert(0, str(P / 'tools'))
    from strata_bench import verify_request_journal
    paths = list((root / 'bench').glob('*/requests.jsonl'))
    guards.require(len(paths) == 1, 'missing or ambiguous official request journal')
    journal = verify_request_journal(paths[0].parent)
    rows = []
    for line in paths[0].read_text().splitlines()[:journal['complete_prefix_rows']]:
        row = json.loads(line)
        if row['kind'] == 'result' and row['phase'].startswith('long-generation/'):
            rows.append(dict(row['sample'], workload=row['phase'].split('/', 1)[1],
                run=row['run'], order=len(rows)))
    return journal, rows


def warmup_reconciliation(entries, workloads=('regular-code',)):
    """Require the actual official warmups, not just a successful final request."""
    expected = [('warm-up/chat', None, 24, True), ('calibration', 200, 1, False),
                ('calibration', 800, 1, False), ('warm-up/prompt', None, 8, True)]
    expected += [('long-generation/' + name, 0, 16384, True) for name in workloads]
    events = [row for row in entries if row['kind'] != 'identity']
    guards.require(len(events) == 2 * len(expected), 'missing or extra official request events')
    hashes = []
    for i, (phase, run, maximum, thinking) in enumerate(expected):
        start, result = events[2*i:2*i+2]
        guards.require(start['kind'] == 'start' and result['kind'] == 'result' and
            all(row.get('phase') == phase and row.get('run') == run and row.get('turn') is None
                for row in (start, result)) and start.get('max_tokens') == maximum and
            start.get('thinking') is thinking and result.get('request_index') == start['index'] and
            result.get('sample', {}).get('ok') is True and
            result['sample'].get('request_sha256') == start.get('request_sha256') and
            re.fullmatch('[0-9a-f]{64}', start.get('request_sha256', '')) is not None,
            'official warmup/request identity, order or success differs')
        hashes.append(start['request_sha256'])
    return dict(successful_warmups=4, successful_long_requests=len(workloads),
                phase_order=[row[0] for row in expected], request_sha256=hashes)


def warmup_evidence(root, expected_journal):
    journal, _ = journal_evidence(root)
    guards.require(journal['valid'] and journal['complete'] and journal == expected_journal,
        'warmup journal is invalid, incomplete or differs from the closed receipt')
    path, = (root / 'bench').glob('*/requests.jsonl')
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    return dict(warmup_reconciliation(entries), journal_sha256=common.digest(path),
                journal_state_sha256=common.digest(path.with_name('journal-state.json')))


def runtime_assets_unchanged(record):
    delta = record.get('runtime_asset_delta', {})
    guards.require(not delta or record['mode'] in ('cpu-check', 'cpu-overhead', 'engine-compare', 'client-resources', 'pool-capacity', 'arena-copy-check', 'arena-generation-check', 'arena-generation-screen', 'early-pcie-check', 'early-generation-check', 'early-generation-screen', 'cache-owned-check', 'cache-capacity-clean'), 'proposed runtime is diagnostic-only')
    settings = json.loads(SETTINGS.read_text())
    guards.require(set(delta) <= set(settings['runtime_elfs']), 'unknown proposed runtime asset')
    for name, item in settings['runtime_elfs'].items():
        proposed = delta.get(name)
        if proposed:
            guards.require(proposed['qualified_sha256'] == item['sha256'], 'original runtime baseline changed')
        expected = proposed['candidate_sha256'] if proposed else item['sha256']
        guards.require(common.digest(name) == expected, 'runtime asset changed: ' + name)
