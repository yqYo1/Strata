#!/usr/bin/env python3
"""Own the split-FP32 MoE planner controls; repository LICENSE applies.

CPU-only ASan/UBSan checks. No selected engine, decode route or GPU execution.
The same prospective input/link and ownership helpers serve native experiments.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
import matrix_shapes as q
r, P, B = q.r, q.P, q.B
ROOT = B / 'moe-split-accumulator-v1' / 'host'
KEY = 'moe_split_accumulator'
RESULT = q.RESULT
TESTS = ['tests/moe_accum_plan.cpp', 'tests/moe_accum_routes.cpp']
HEADERS = ['include/strata/prefill/moe_accum_plan.hpp',
           'include/strata/prefill/moe_window_routes.hpp']
require, digest, write = q.require, q.digest, q.write


def sources(revision):
    return q.sources(revision, Path(__file__).resolve(), [*TESTS, *HEADERS])


def compact(row):
    return {k: v for k, v in row.items() if k not in ['source_sha256', 'commands_executed']}


def admitted(row):
    require(sources(row['source_commit']) == row['source_sha256'], 'frozen host source changed')
    require(q.closure.sdk_inputs() == row['sdk'], 'qualified compiler dependency assets changed')
    require(digest(row['input_inventory']['path']) == row['input_inventory']['sha256'], 'inventory changed')
    allowed = json.loads(Path(row['input_inventory']['path']).read_text())
    for path, sha in allowed.items():
        require(digest(path) == sha, 'prospective input changed:' + path)
    for binary in row.get('binaries', {}).values():
        require(digest(binary['path']) == binary['sha256'], 'host test binary changed')
    return allowed


def prepare():
    record = r.validation.committed(RESULT)
    retired = record['qsa_required_workgroup']['retirement']
    require(retired['verified_absent'] and all(not Path(v).exists() for v in retired['roots']),
            'retire completed comparison generation first')
    require(not ROOT.exists() and KEY not in record, 'fresh split comparison only')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
    pins = sources(revision)
    rt = r.validation.committed(r.RESULT)
    selected = r.current(rt)
    sdk = q.closure.sdk_inputs()
    allowed = q.prospective_inputs(selected['compiler'], rt, pins)
    ROOT.mkdir(parents=True, mode=0o700)
    (ROOT / 'tmp').mkdir()
    (ROOT / 'cache').mkdir()
    write(ROOT / 'input-inventory.json', allowed)
    flags = ['-std=c++20', '-O1', '-g0', '-DNDEBUG', '-fno-omit-frame-pointer',
             '-fsanitize=address,undefined', '-fno-fast-math', '-ffp-contract=off',
             '-I' + str(P / 'include'), '-save-temps=obj', '-v', '-Wl,-t']
    commands = []
    for name in TESTS:
        stem = Path(name).stem
        commands.append([stem, [selected['compiler']['path'], *flags, '-MD', '-MF',
            str(ROOT / (stem + '.d')), str(P / name), '-o', str(ROOT / stem)]])
    row = dict(stage='prepared', source_commit=revision, source_sha256=pins,
        source_inputs=dict(files=len(pins), aggregate_sha256=hashlib.sha256(
            json.dumps(pins, sort_keys=True, separators=(',', ':')).encode()).hexdigest()),
        sdk=sdk, compiler=selected['compiler'], selected_binary=selected['binary'],
        input_inventory=dict(path=str(ROOT / 'input-inventory.json'),
            sha256=digest(ROOT / 'input-inventory.json'), files=len(allowed)),
        commands=commands, environment=dict(selected['environment'], TMPDIR=str(ROOT / 'tmp'),
            XDG_CACHE_HOME=str(ROOT / 'cache'), ASAN_OPTIONS='detect_leaks=1:abort_on_error=1',
            UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1'),
        budgets=dict(wall=240, cpu=600, rss_cap=8 << 30, text_cap=16 << 20,
            file_cap=128 << 20, total_cap=512 << 20),
        active=False, passed=False, complete=False, ownership_closed=True, GPU_executed=False,
        criteria='Two fresh CPU-only ASan/UBSan fixtures. Preserve old planner and route admission checks. '
            'Six split capacity cases81921..117488 with complete logical mixed/FP32 bank coverage, '
            'one-byte-short/invalid/full262144 refusal and guarded531988736B tail arithmetic. '
            'Twelve split route cases2..117488 with ordinary/hot routing,complete route mass, '
            'unique source/inverse mappings and <=8192-row jobs,including15jobs/expert. '
            'Exact committed source/compiler/actual compiler and linker inputs,closed owners, '
            'zero sanitizer diagnostics. No actual GPU allocation,arithmetic,speed or fullcontext qualification.',
        retention='Host controls for the imminent split-FP32 candidate;keep selected6e40+12fixtures. '
            'Extract outcomes/source identity and retire disposable test binaries after source recovery.')
    admitted(row)
    write(ROOT / 'record.json', row)
    record[KEY] = dict(status='host_controls_prepared', host=compact(row),
        mechanism='Process one117488-row expert-major window rather than81920+35568;preserve FP32 '
            'expert summation and16-bit tensor boundaries. Split accumulator makes capacity possible. '
            'Fewer expert dequants/submissions are a hypothesis;smaller bank may add PCIe traffic. '
            'Aggregate MoE minus GU/Down is NOT measured dequanttime or physical-resource saturation.',
        inference_adopted=False, decode_changed=False,
        next_gate='Pass host controls,then source-reviewed separate ordinary/fused split consumers, '
            'bounded traced device numerics/bounds/actualallocation before model timing. '
            'Preserve exact selected-source production baseline and independent decode/handoff/fullcontext gates.')
    write(RESULT, record)
    print('Split planner host controls prepared;commit before compile/run.', flush=True)


def run():
    record = r.validation.committed(RESULT)
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and compact(row) == record[KEY]['host'], 'frozen host plan required')
    allowed = admitted(row)
    owner = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, commands_executed=owner.commands if owner else []))
    owner = r.supervision.Owner(ROOT / 'owned', persist)
    try:
        links, observed, binaries, outcomes = {}, {}, {}, {}
        for label, command in row['commands']:
            entry, out, err = owner.run('build-' + label, command, row['environment'], ROOT, **row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code'] == 0,
                    'host build failed:' + err.read_text()[-2500:])
            links[label] = q.linking.traced_inputs(out.read_text(), err.read_text(), allowed, ROOT, ROOT)
            for dep in shlex.split((ROOT / (label + '.d')).read_text().replace('\\\n', ' ').split(':', 1)[1]):
                path = str(Path(dep).resolve())
                require(allowed.get(path) == digest(path), 'unfrozen compiler dependency:' + path)
                observed[path] = digest(path)
            binary = ROOT / label
            binaries[label] = dict(path=str(binary), sha256=digest(binary), bytes=binary.stat().st_size)
        require(all(str(P / v) in observed for v in [*TESTS, *HEADERS]), 'incomplete actual source closure')
        row['binaries'] = binaries
        admitted(row)
        for label in binaries:
            entry, out, err = owner.run('test-' + label, [binaries[label]['path']], row['environment'], ROOT,
                                        **row['budgets'])
            text = out.read_text()
            require(r.supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
                    'host control/sanitizer failed:' + err.read_text()[-2500:])
            require(label + ' PASS GPU=UNRUN' in text, 'missing host final marker')
            if label == 'moe_accum_plan':
                require('split_cases=6 split_negative_checks=28 PASS' in text, 'split capacity coverage')
            else:
                require(text.count('split=1 inverse=PASS') == 12 and text.count('split=0 inverse=PASS') == 18,
                        'split/legacy route coverage')
            outcomes[label] = dict(passed=True, sanitizer_diagnostics=0,
                                  result_lines=[v for v in text.splitlines() if v.startswith(('split_cases=', 'moe_accum_'))])
        write(ROOT / 'compiler-inputs.json', observed)
        write(ROOT / 'link-inputs.json', links)
        row.update(stage='closed', passed=True, complete=True, outcomes=outcomes,
            actual_compiler_inputs=dict(path=str(ROOT / 'compiler-inputs.json'), files=len(observed),
                sha256=digest(ROOT / 'compiler-inputs.json'), uncovered=0),
            actual_link_inputs=dict(path=str(ROOT / 'link-inputs.json'), sha256=digest(ROOT / 'link-inputs.json'),
                frozen_counts={name: len(v['frozen_inputs']) for name, v in links.items()}, uncovered=0))
        admitted(row)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row.update(active=False, ownership_closed=owner.active is None and
            all(r.supervision.closed(c) for c in owner.commands))
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        persist()
        record[KEY].update(status='host_controls_passed' if row['passed'] else 'host_controls_failed', host=compact(row))
        write(RESULT, record)
    print(json.dumps({k: row.get(k) for k in ['stage', 'passed', 'ownership_closed', 'error', 'outcomes']}), flush=True)
    require(row['passed'], 'split host controls failed')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'run': run}[sys.argv[1]]()
