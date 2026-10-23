#!/usr/bin/env python3
"""Closed-data QSA service attribution; repository LICENSE, no device execution."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
A = P / 'bench/results/2026-10-15-pr6-recovery-audit'
RESULT = A / 'qsa-prefix-service-current.json'
PROFILE = A / 'cpu-profiler-current.json'
SOURCE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
FAMILY = 'strata::kernels::prefix_batch_impl<33, false, false, true, true, true, true, false>'
INPUTS = ['tools/sycl/analyze_qsa_prefix_service.py', 'tools/sycl/measurement_guards.py',
          'tests/core/test_qsa_prefix_service.py', 'src/kernels/sycl/qsa_bf16_prefix.cpp',
          'src/prefill/prefill.cpp', 'src/prefill/layer_major.inc']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=30)


def committed(path):
    raw = path.read_bytes()
    guards.require(raw == git('show', 'HEAD:' + str(path.relative_to(P))), 'commit exact input first')
    return json.loads(raw)


def extract(rows, cells, chunk=8192, layers=12):
    guards.require(type(cells) is int and 32768 <= cells <= 262144 and chunk == 8192 and layers == 12,
                   'qualified model shape required')
    expected = {}
    logical_rows = 0
    for first in range(0, cells, chunk):
        queries = min(chunk, cells - first)
        active_rows = ((first + queries + 3) // 4) * 8
        logical_rows += active_rows * layers
        for key in [('pack', active_rows), ('attention', 2 * queries)]:
            expected[key] = expected.get(key, 0) + layers
    found = {}
    for row in rows:
        name = row['name']
        if not name.startswith('"' + FAMILY + '('):
            continue
        shape = re.search(r'\[SIMD(\d+) \{(\d+); 1; 1\} \{(\d+); 1; 1\}\]"$', name)
        guards.require(shape is not None, 'unknown QSA launch geometry')
        simd, groups, local = map(int, shape.groups())
        if '::{lambda(sycl::_V1::handler&)#1}::operator()(sycl::_V1::handler&) const::' in name:
            phase = 'attention'
            guards.require((simd, local) == (16, 128), 'attention source geometry mismatch')
        else:
            phase = 'pack'
            guards.require(name.split('[SIMD', 1)[0].endswith('::{lambda(sycl::_V1::nd_item<1>)#1}') and
                           (simd, local) == (32, 64), 'pack source identity mismatch')
        key = (phase, groups)
        guards.require(key in expected and key not in found and row['calls'] == expected[key],
                       'missing/extra/duplicate QSA work')
        calls, total, average, minimum, maximum = (row[k] for k in
            ['calls', 'total_ns', 'average_ns', 'min_ns', 'max_ns'])
        guards.require(all(type(x) is int for x in [calls, total, average, minimum, maximum]) and
                       calls > 0 and 0 < minimum <= average <= maximum and
                       minimum * calls <= total <= maximum * calls and abs(average - total // calls) <= 1,
                       'invalid positive kernel service')
        found[key] = dict(phase=phase, workgroups=groups, local_size=local, SIMD=simd,
                          name_sha256=hashlib.sha256(name.encode()).hexdigest(),
                          **{k: row[k] for k in ['calls', 'total_ns', 'average_ns', 'min_ns', 'max_ns']})
    guards.require(found.keys() == expected.keys(), 'complete QSA phase inventory required')
    totals = {phase: sum(r['total_ns'] for r in found.values() if r['phase'] == phase)
              for phase in ['pack', 'attention']}
    return dict(rows=sorted(found.values(), key=lambda r: (r['phase'], r['workgroups'])),
                totals_ns=totals, calls_per_phase=sum(expected[k] for k in expected if k[0] == 'pack'),
                packed_logical_rows=logical_rows,
                pack_share_of_attributed_device_service=totals['pack'] / sum(totals.values()))


def evidence():
    profile = committed(PROFILE)
    origin = profile['native_model_aggregate_v2']
    later = profile['native_model_aggregate_v2_reanalysis']
    original = json.loads(git('show', '292bbc441:' + str(PROFILE.relative_to(P))))['native_model_aggregate_v2']
    guards.require(origin == original and not origin['passed'] and not origin['complete'] and
                   origin['ownership_closed'] and not origin['active'] and origin['session_removed'] and
                   origin['fresh_faults'] == [] and origin['boot_unchanged'], 'original failure changed')
    guards.require(later['passed'] and later['complete'] and not later['original_run_passed'] and
                   later['original_failure_preserved'] and later['posthoc_full_model_hashes_verified'] == 15 and
                   later['evidence']['native_exit_code'] == 0, 'qualified closed-data replay required')
    selected = committed(A / 'native-gdn-engine-build.json')
    guards.require(origin['binary'] == selected['binary'] and selected['source_commit'] == SOURCE,
                   'selected native identity differs')
    for name in INPUTS[3:]:
        guards.require((P / name).read_bytes() == git('show', SOURCE + ':' + name), 'native source changed')
    ref = later['full_analysis']
    guards.require(sha(ref['path']) == ref['sha256'], 'closed aggregate bytes changed')
    return later, ref


def write(record):
    RESULT.write_text(json.dumps(record, indent=2) + '\n')


def prepare():
    guards.require(not RESULT.exists(), 'single-use fresh analysis only')
    later, ref = evidence()
    revision = git('rev-parse', 'HEAD').decode().strip()
    pins = {}
    for name in INPUTS:
        raw = git('show', revision + ':' + name)
        guards.require(raw == (P / name).read_bytes(), 'commit analysis source first')
        pins[name] = hashlib.sha256(raw).hexdigest()
    write(dict(stage='prepared', passed=False, complete=False, GPU_executed=False,
               source_commit=revision, source_sha256=pins, native_source_commit=SOURCE,
               profile_sha256=sha(PROFILE), aggregate=ref,
               original_failure=later['original_failure'],
               criteria='No native execution. Preserve original parser failure. Verify exact closed replay and '
                        'all8474rows/5672366calls/211017537985ns. Match every selected prefix-template launch '
                        'to source-derived full-prefix pack or attention geometry,count every33213cell chunk '
                        'across12layers,positive service statistics,one long official request and4warmups. '
                        'No missing/extra/duplicate matched work. Keep source-correlated device service '
                        'separate from phase-recorded critical path,CPU time,wire bytes or saturation.',
               limits='Whole-process profiler data; attribution by exact source/shape/count conservation. '
                      'No per-request event timeline. No causal end-to-end speed bound from sums of service. '
                      'The32K+request does not measure117K or262144 behavior. No inference/adoption change.',
               retention='Keep seven phase/shape service samples and provenance/limits;retire the consumed '
                         'full aggregate/normal logs after reader/source checks. No new build or raw trace.'))


def analyze():
    record = committed(RESULT)
    guards.require(record['stage'] == 'prepared' and sha(PROFILE) == record['profile_sha256'], 'frozen plan changed')
    for name, expected in record['source_sha256'].items():
        guards.require(sha(P / name) == expected and hashlib.sha256(git('show', record['source_commit'] + ':' + name)).hexdigest() == expected,
                       'analysis inputs changed')
    _, ref = evidence()
    guards.require(ref == record['aggregate'], 'aggregate lineage changed')
    data = json.loads(Path(ref['path']).read_text())
    rows = data['table']['rows']
    guards.require(len(rows) == 8474 and sum(r['calls'] for r in rows) == 5672366 and
                   sum(r['total_ns'] for r in rows) == data['table']['total_device_ns'] == 211017537985,
                   'full table conservation failed')
    request, = data['requests']
    guards.require(request['ok'] and request['read_tokens'] == request['prompt_tokens'] == 33214 and
                   request['reused_tokens'] == 0 and request['generated_tokens'] == 3611 and
                   request['stream_complete'] and request['completion_includes_thinking'] and
                   data['warmups']['successful_warmups'] == 4 and data['warmups']['successful_long_requests'] == 1,
                   'single actual long-prefill binding required')
    found = extract(rows, request['read_tokens'] - 1)
    guards.require(sha(ref['path']) == ref['sha256'], 'aggregate changed during analysis')
    record.update(stage='closed', passed=True, complete=True, original_run_passed=False,
                  original_failure_preserved=True, attribution=found,
                  request={k: request[k] for k in ['read_tokens', 'generated_tokens', 'prompt_ms', 'request_sha256']},
                  hardware_saturation_qualified=False, clean_model_speed_qualified=False,
                  model_adoption_qualified=False)
    write(record)
    print(json.dumps(found), flush=True)


if __name__ == '__main__':
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'analyze'), 'prepare|analyze')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else analyze()
