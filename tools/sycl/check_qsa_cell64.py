#!/usr/bin/env python3
"""Main-owned cell64 property admission; repository LICENSE applies.

No pack/attention work is authorized by this first query-only stage.
Own archived208c controller APIs are references; no XeStrata application source.
"""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_qsa_cell64 as build
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from measure_down_scatter import postchecks

P, B, ROOT, RESULT = build.P, build.B, build.ROOT, build.RESULT
KEY = 'qsa_cell64_admission_contract_20261027'


def inputs(revision):
    pending = [Path(__file__).resolve()]; seen = set(); pins = {}
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        name = str(path.relative_to(P))
        raw = build.gen.git('show', revision + ':' + name)
        guards.require(raw == path.read_bytes(), 'commit exact controller input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            modules = [n.name for n in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    return pins


def kernel_ids(record):
    names = []
    for arm, stride in (('original', 33), ('control', 33), ('candidate', 65)):
        asset = record['device_IR'][arm]
        guards.require(build.digest(asset['path']) == asset['sha256'], 'device IR changed')
        text = Path(asset['path']).read_text()
        headers = re.findall(r'^define [^\n]*prefix_batch_implILi' + str(stride) +
            r'ELb0ELb0ELb1ELb1ELb1ELb1ELb0EEE[^\n]*', text, re.M)
        selected = [h for h in headers if '!intel_reqd_sub_group_size ' in h]
        guards.require(len(selected) == 1, 'one exact ordered-exp attention kernel per arm')
        header = selected[0]
        metadata = re.search(r'!intel_reqd_sub_group_size !(\d+)', header)[1]
        guards.require(re.search(r'^!' + metadata + r' = !\{i32 16\}$', text, re.M),
                       'compiled SIMD contract differs')
        name = re.search(r'spir_kernel void @([^ (]+)\(', header)[1]
        guards.require(name.startswith('_ZTS'), 'unexpected kernel identity encoding')
        names.append(name)
    guards.require(len(set(names)) == 3, 'compiled arm identities overlap')
    return names


def contract():
    row = validation.committed(build.gen.A / 'research-index.json', (KEY,))
    guards.require(row['status'] == 'reviewed_accept_conditional' and row['worker_closed'] and
        row['source_revision'] == '9b59e9b511b60941c4d5dd06fb3dae91a83bcac0',
        'closed independent admission API audit required')
    review = row['review']
    guards.require(review['status'] == 'completed' and review['model'] == 'gpt-6-luna' and
        review['reasoning_effort'] == 'max', 'native Luna/max curation required')
    for asset, path in ((row['report_identity'], row['report']), (review, review['path'])):
        file = P / path
        guards.require(file.stat().st_size == asset['bytes'] and
            build.digest(file) == asset['sha256'], 'reviewed admission reference changed')
    return dict(source_revision=row['source_revision'], report_sha256=row['report_identity']['sha256'],
                review_sha256=review['sha256'], status=row['status'])


def admission(record, plan):
    guards.require(record['stage'] == 'built' and record['passed'] and record['complete'] and
        record['ownership_closed'] and not record['active'], 'closed source-bound build required')
    build.admission(record, True)
    guards.require(inputs(plan['source_commit']) == plan['controller_inputs'] and
        kernel_ids(record) == plan['kernel_ids'] and contract() == plan['contract'],
        'controller, compiled identity or independent contract changed')
    _, proof = build.qualified_baseline()
    guards.require(proof == plan['runtime_qualification'], 'runtime qualification changed')
    rt = validation.committed(build.qualified_runtime.RESULT)
    guards.require(rt['runtime_files'] == plan['runtime_files'] and
        rt['preload'] == plan['preload'], 'runtime/preload identities changed')
    return rt


def prepare():
    record = build.gen.committed(RESULT)
    guards.require('admission_plan' not in record and 'admission_check' not in record,
                   'fresh admission only; inspect existing ownership before retry')
    folder = ROOT / 'admission'
    guards.require(not folder.exists(), 'admission output already exists')
    revision = build.gen.git('rev-parse', 'HEAD').decode().strip()
    rt = validation.committed(build.qualified_runtime.RESULT)
    _, proof = build.qualified_baseline()
    names = kernel_ids(record)
    loader = [p for p in rt['runtime_files'] if Path(p).name.startswith('libze_loader.so.')]
    guards.require(len(loader) == 1, 'one pinned Level Zero loader required')
    env = dict(build.gen.committed(P / 'tools/sycl/measurement_environment.json')['environment'],
        TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
        LD_PRELOAD=rt['preload']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
    plan = dict(stage='prepared', source_commit=revision, controller_inputs=inputs(revision),
        kernel_ids=names, contract=contract(), runtime_qualification=proof,
        runtime_files=rt['runtime_files'], preload=rt['preload'], environment=env,
        command=[record['binaries']['fixture']['path'], '--admit', *names, loader[0]],
        budgets=dict(wall=120, cpu=300, rss_cap=4 << 30, text_cap=8 << 20,
                     file_cap=32 << 20, total_cap=128 << 20),
        criteria='One property-only child;all3exact kernels pass WG128/SG16/18argument and per-group local budget checks. UR pre-main proof,quiet validation,no kernel enqueue in trace,closed owned process and no fresh faults. No numerical,occupancy,performance or model qualification.',
        retention='Retain compact queried descriptors and outcome;retire successful trace/normal logs after verification. Current generation remains for the bounded independent oracle comparison.')
    admission(record, plan)
    folder.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (folder / name).mkdir()
    build.write(folder / 'plan.json', plan)
    record['admission_plan'] = plan
    build.write(RESULT, record)
    print('Prepared property-only cell64 admission;commit plan before run.', flush=True)


def run():
    record = build.gen.committed(RESULT); plan = record['admission_plan']
    folder = ROOT / 'admission'
    guards.require(plan['stage'] == 'prepared' and 'admission_check' not in record and
        plan == json.loads((folder / 'plan.json').read_text()), 'fresh committed admission plan required')
    rt = admission(record, plan)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor'], text=True)
    cursor = re.findall(r'^-- cursor: (.+)$', cursor_text, re.M)
    guards.require(len(cursor) == 1, 'fresh fault cursor unavailable')
    result = dict(passed=False, complete=False, active=False, ownership_closed=False,
                  device_initialized=None, kernel_submissions=None, model_qualified=False)
    owner = None
    def persist():
        result['active'] = bool(owner and owner.active is not None)
        build.write(folder / 'owned.json', dict(result, commands=owner.commands if owner else []))
    owner = supervision.Owner(folder / 'owned', persist)
    try:
        entry, out, err = owner.run('properties', plan['command'], plan['environment'], P,
                                   **plan['budgets'])
        stdout, stderr = out.read_text(), err.read_text()
        result['device_initialized'] = 'admission_stage=runtime-ready ' in stdout
        guards.require(supervision.closed(entry) and entry['exit_code'] == 0,
                       'property child failed or did not close')
        proof = validation.preload_evidence(stderr, entry['pid'])
        guards.require(proof['loader'] == rt['preload']['loader'], 'UR provider mismatch')
        body = stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main', 1)[1]
        guards.require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:', body),
                       'native runtime error')
        validation_log = folder / 'validation.log'
        guards.require(not validation_log.exists() or validation_log.stat().st_size == 0,
                       'validation warning/error')
        trace = (folder / 'api.trace').read_text()
        guards.require('urKernelGet' in trace and not re.search(r'urEnqueue\w*Kernel', trace),
                       'query trace absent or unexpected kernel submission')
        result['kernel_submissions'] = 0
        rows = []
        for line in stdout.splitlines():
            if not line.startswith('arm='):
                continue
            fields = dict(part.split('=', 1) for part in line.split())
            guards.require(fields.pop('status') == 'PASS', 'compiled resource check failed')
            rows.append({key: int(value) for key, value in fields.items()})
        guards.require([row['arm'] for row in rows] == [0, 1, 2] and
            stdout.count('admission_status=PASS kernel_submissions=0 ') == 1,
            'property coverage incomplete')
        result.update(passed=True, complete=True, device_initialized=True,
                      properties=rows, validation=proof)
        admission(record, plan)
    except BaseException as error:
        result.update(passed=False, complete=False, error=repr(error))
    finally:
        result['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        postchecks(result, boot, cursor[0]);persist()
        plan['stage'] = 'closed' if result['passed'] else 'failed'
        record['admission_check'] = result
        build.write(RESULT, record)
    print(json.dumps(result), flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    guards.require(sys.argv[1:] in (['prepare'], ['run']), 'prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([]);guards.driver_guard()
        raise SystemExit(prepare() if sys.argv[1] == 'prepare' else run())
