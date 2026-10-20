#!/usr/bin/env python3
"""Main-owned bounded production-source status-copy discriminator; no model run.

Commit source, prepare/commit the generated recipe, build/commit binary identity,
then run the four fixed cells once. No retries or adaptive backlog enlargement.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import query_profile as q
import ur_validation as validation

gen, guards, supervision = q.gen, q.guards, q.supervision
P, B, A = q.P, q.B, q.A
ROOT = B / 'status-copy-component-v1'
RESULT = A / 'status-copy-current.json'
FIXTURE = 'bench/sycl/status_copy.cpp'
CELLS = [('ordinary', 'idle'), ('usm', 'pending'), ('ordinary', 'pending'), ('usm', 'idle')]


def digest(path):
    return supervision.sha(path)


def inputs(source):
    pins = gen.benchmark_pins(source)
    raw = gen.git('show', source + ':' + FIXTURE)
    guards.require(raw == (P / FIXTURE).read_bytes(), 'commit exact fixture input')
    pins[FIXTURE] = hashlib.sha256(raw).hexdigest()
    return pins


def runtime_check():
    settings = gen.committed(gen.SETTINGS)
    guards.require(set(q.DELTA) <= set(settings['runtime_elfs']), 'unknown qualified runtime delta')
    for path, item in settings['runtime_elfs'].items():
        expected = item['sha256']
        if path in q.DELTA:
            original, expected = q.DELTA[path]
            guards.require(original == item['sha256'], 'original baseline differs')
        guards.require(digest(path) == expected, 'runtime changed: ' + path)
    return settings


def admission(record):
    full, count = q.current_build()
    guards.require(full['source_commit'] == record['native_source_commit'] and
        digest(q.BUILD_RECORD) == record['native_build_record_sha256'] and
        count == record['native_inputs_recovered'], 'native source/build changed')
    pins = inputs(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and
        q.assets.fingerprint(pins) == record['source_fingerprint'] and
        digest(full['compiler']['path']) == record['compiler']['sha256'], 'fixture/controller/compiler changed')
    runtime_check()
    guards.require(digest(gen.SETTINGS) == record['settings_sha256'] and
        validation.preload_qualification() == record['validation_probe'], 'runtime/helper proof changed')


def prepare():
    guards.require(not RESULT.exists() and not ROOT.exists(), 'inspect prior job; no retry')
    full, count = q.current_build()
    settings = runtime_check()
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = inputs(source)
    compiler = full['compiler']
    guards.require(digest(compiler['path']) == compiler['sha256'], 'compiler differs')
    env = dict(full['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    native = dict(settings['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
        CUDA_MODULE_LOADING='EAGER', STRATA_SYCL_COMMAND_PROFILE='0')
    probe = validation.preload_qualification()
    native['LD_PRELOAD'] = probe['library']['path']
    executable = ROOT / 'status_copy'
    command = [compiler['path'], '-O3', '-DNDEBUG', '-std=c++20', '-fsycl',
        '-fsycl-id-queries-range=size_t', '-DSTRATA_ENABLE_SYCL=1',
        '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat'),
        str(P / FIXTURE), *[str(P / ('src/sycl/' + name + '.cpp'))
            for name in ('runtime', 'compat', 'command_profile')], '-o', str(executable)]
    record = dict(schema='strata-status-copy-v1', stage='prepared', source_commit=source,
        source_count=len(pins), source_fingerprint=q.assets.fingerprint(pins),
        native_source_commit=full['source_commit'], native_inputs_recovered=count,
        native_build_record_sha256=digest(q.BUILD_RECORD), compiler=compiler,
        settings_sha256=digest(gen.SETTINGS), compile_argv=command, compile_environment=env,
        native_environment=native, validation_probe=probe, cells=CELLS,
        gates='Four separate immutable-environment processes in fixed order;12 samples each. 4-byte production compat D2H, checked drain and sticky postcheck. Pending arm submits exactly8x128MiB fills and queries the last event immediately before copy. A completed pre-query makes that sample ineligible; no retry/enlargement. Require exact phase counts, clocks/conservation, payload/guards, native exit0, owned closure, unchanged inputs and no fresh fault. Pending comparison requires all12 pending samples in each pending cell.',
        budgets=dict(compile_wall_s=240, native_cell_wall_s=45, text_bytes_per_command=16 << 20,
            total_raw_bytes=64 << 20, device_bytes=134217868,
            pending_logical_fill_bytes_per_cell=12 * 8 * (128 << 20)),
        limits='Component source attribution only, with the same production runtime/compat source as3b4998. No model/engine rebuilt and no third engine generation. Pending observation can race completion before enqueue. Fine phase timings include clock overhead, so no microscopic speed/adoption claim. Requested fill bytes are not physical DRAM traffic.',
        retained_generations='396 fastest fallback and3b4998 status engine remain unchanged. This finite production-source fixture is retired after its compact decision; no extra engine build. Review candidate-engine necessity at next native-plan freeze; default integration remains blocked.',
        retention='Keep48 useful phase samples and eligibility/owner/fault/source result. Retire successful full traces, normal compiler output and this fixture build at decision closure after exact source/recipe recovery.',
        complete=False, passed=False, active=False, GPU_executed=False)
    RESULT.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(prepared=True, source=source, source_count=len(pins), GPU_executed=False)))


def execute(build):
    record = gen.committed(RESULT)
    guards.require(record['stage'] == ('prepared' if build else 'built'), 'closed/mismatched stage; no retry')
    admission(record)
    if build:
        guards.require(not ROOT.exists(), 'compile root already exists; inspect original handle')
        ROOT.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    else:
        guards.require(digest(record['binary']['path']) == record['binary']['sha256'], 'fixture executable differs')
        guards.require(not (ROOT / 'run.json').exists(), 'run already attempted; no retry')
    owner = None
    output = ROOT / ('build.json' if build else 'run.json')

    def persist():
        record['active'] = bool(owner and owner.active is not None)
        output.write_text(json.dumps(dict(record, owned_commands=owner.commands if owner else []), indent=2) + '\n')

    owner = supervision.Owner(ROOT / ('owned-build' if build else 'owned-run'), persist)
    record.update(complete=False, passed=False)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursor = None
    persist()
    try:
        if build:
            entry, out, err = owner.run('compile', record['compile_argv'], record['compile_environment'], P,
                wall=240, cpu=240, rss_cap=4 << 30, text_cap=16 << 20, file_cap=128 << 20, total_cap=1 << 30)
            guards.require(supervision.closed(entry) and entry.get('exit_code') == 0,
                'compile failed: ' + err.read_text()[-4096:])
            executable = ROOT / 'status_copy'
            record['binary'] = dict(path=str(executable), bytes=executable.stat().st_size, sha256=digest(executable))
            record['stage'] = 'built'
        else:
            record['GPU_executed'] = True
            guards.driver_guard()
            journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
            cursor = next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
            record['cells_results'] = []
            for index, cell in enumerate(record['cells']):
                runtime_check()
                entry, out, err = owner.run('cell' + str(index), [record['binary']['path'], *cell],
                    record['native_environment'], P, wall=45, cpu=45, rss_cap=2 << 30,
                    text_cap=16 << 20, file_cap=16 << 20, total_cap=64 << 20)
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0, 'cell failed: ' + err.read_text()[-4096:])
                rows = [json.loads(line) for line in out.read_text().splitlines()]
                guards.require(len(rows) == 14 and rows[0] == dict(type='begin',schema=1,
                    host=cell[0],queue=cell[1],samples=12,copy_bytes=4,
                    backlog_fills=8 if cell[1]=='pending' else 0,fill_bytes=128 << 20) and rows[-1] ==
                    dict(type='end', passed=True, samples=12, owners_closed=True), 'missing/ambiguous protocol')
                samples = rows[1:-1]
                guards.require([x['rep'] for x in samples] == list(range(12)) and
                    all(x['type']=='sample' and x['payload_and_guards_passed'] for x in samples), 'sample loss/verification failure')
                for sample in samples:
                    guards.require(sample['eligible'] == (cell[1]=='idle' or sample['pending_observed']) and
                        all(math.isfinite(sample[k]) and sample[k]>=0 for k in
                            ('query_to_copy_wall_s','scope_cpu_s','scope_wall_s')), 'invalid eligibility/clock')
                    for name in ('enqueue','observer','drain','precheck','queue_context','postcheck'):
                        row=sample[name]
                        guards.require(row['calls']==(2 if name in ('precheck','queue_context') else 1) and
                            all(math.isfinite(row[k]) and row[k]>=0 for k in
                                ('cpu_s','wall_s','wait_cpu_s','wait_wall_s')) and
                            row['wait_cpu_s']<=row['cpu_s'] and row['wait_wall_s']<=row['wall_s'], 'invalid phase')
                log = err.read_text()
                proof = validation.preload_evidence(log, entry['pid'])
                guards.require('UR_LAYER_PARAMETER_VALIDATION' in record['native_environment']['UR_ENABLE_LAYERS'] and
                    'strata UR validation failure:' not in log, 'validation activation/error')
                qualified = all(x['eligible'] for x in samples)
                summary = {name:dict(cpu_s=statistics.median(x[name]['cpu_s'] for x in samples),
                    wall_s=statistics.median(x[name]['wall_s'] for x in samples))
                    for name in ('enqueue','observer','drain','precheck','queue_context','postcheck')}
                record['cells_results'].append(dict(cell=cell, samples=samples, medians=summary,
                    all_samples_eligible=qualified, pending_observations=sum(x['pending_observed'] for x in samples),
                    native_pid=entry['pid'], validation_activation=proof, native_exit=entry['exit_code'], ownership_closed=True))
                persist()
            record['pending_comparison_qualified'] = all(x['all_samples_eligible'] for x in record['cells_results'])
            record['stage'] = 'closed'
        admission(record)
        record.update(complete=True, passed=True)
    except BaseException as error:
        record['error'] = str(error)
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        record['passed'] = bool(record['passed'] and record['ownership_closed'])
        if not build and cursor is not None:
            record.update(fresh_faults=None, boot_unchanged=False)
            try:
                text=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                record['fresh_faults']=[x for x in text.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
                record['driver_after']=guards.driver_guard()
                record['boot_unchanged']=boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            except Exception as error: record['final_probe_error']=str(error)
            record['passed']=bool(record['passed'] and record['fresh_faults']==[] and
                record['boot_unchanged'] and 'final_probe_error' not in record)
        persist(); RESULT.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k:record.get(k) for k in ('stage','complete','passed','ownership_closed','error')}))
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','build','run'))
    action=parser.parse_args().action
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        return prepare() if action=='prepare' else execute(action=='build')


if __name__=='__main__': sys.exit(main())
