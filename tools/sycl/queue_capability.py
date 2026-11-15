#!/usr/bin/env python3
"""Bounded metadata-only queue inventory; Main-authored, repository LICENSE.

Commit source -> prepare -> commit recipe -> build -> commit ELF -> run.
No context/queue creation, inference, runtime changes, retries or GPU recovery.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as owner_api

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'queue-capability-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json'
FUSED = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'prefill_queue_capability_v1'
CC = Path('/usr/bin/gcc').resolve()
LOADER = Path('/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0')
NEO = Path('/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758')
SOURCE = P / 'bench/sycl/queue_capability.c'
INPUTS = ('bench/sycl/queue_capability.c', 'tools/sycl/queue_capability.py',
          'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py')
LIMITS = dict(wall=60, cpu=45, rss_cap=1 << 30, text_cap=1 << 20,
              file_cap=4 << 20, total_cap=8 << 20)
require, sha = guards.require, owner_api.sha


def git(*args):
    return subprocess.check_output(['git', '-C', str(P), *args], timeout=20)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def sources(rev):
    pins = {}
    for name in INPUTS:
        data = git('show', rev + ':' + name)
        require(data == (P / name).read_bytes(), 'uncommitted input: ' + name)
        pins[name] = hashlib.sha256(data).hexdigest()
    return pins


def environment():
    return dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        LD_LIBRARY_PATH='/usr/lib/x86_64-linux-gnu', ZE_ENABLE_ALT_DRIVERS=str(NEO),
        ZES_ENABLE_SYSMAN='0', NEOReadDebugKeys='1', EnableDirectSubmission='0',
        UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', STRATA_SYCL_PREFILL_SAME_QUEUE='1')


def commands():
    return [[str(CC), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-fPIE',
        '-MD', '-MF', str(ROOT/'actual.d'), '-o', str(ROOT/'inventory'), str(SOURCE),
        str(LOADER), '-pie', '-Wl,-z,relro,-z,now']]


def verify(row):
    require(row['source_inputs'] == sources(row['source_commit']), 'source mismatch')
    require(row['environment'] == environment() and row['commands'] == commands(), 'recipe mismatch')
    require(row['driver'] == guards.driver_guard() and row['boot'] ==
        Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'driver/boot changed')
    require(all(sha(p) == h for p, h in row['external_inputs'].items()), 'external input changed')
    compiler = row['compiler_manifest']
    require(sha(compiler['path']) == compiler['sha256'], 'compiler manifest changed')
    files = json.loads(Path(compiler['path']).read_text())['files']
    require(str(CC) in files and all(sha(p) == h for p, h in files.items()), 'original C compiler closure changed')
    manifest = row['runtime_manifest']
    require(sha(manifest['path']) == manifest['sha256'], 'runtime manifest changed')
    providers = json.loads(Path(manifest['path']).read_text())
    require(all(sha(p) == h for p, h in providers.items()), 'runtime provider changed')
    if row.get('binary'):
        require(sha(row['binary']['path']) == row['binary']['sha256'], 'ELF changed')
    return providers


def mappings(pid, row, providers):
    paths = set()
    for line in Path('/proc', str(pid), 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and 'x' in fields[1] and fields[5].startswith('/'):
            require(not fields[5].endswith(' (deleted)'), 'deleted executable mapping')
            paths.add(str(Path(fields[5]).resolve()))
    allowed = {str(Path(p).resolve()): h for p, h in providers.items()}
    allowed[row['binary']['path']] = row['binary']['sha256']
    require(paths and all(p in allowed and sha(p) == allowed[p] for p in paths),
        'unfrozen executable mapping: ' + repr(sorted(paths - allowed.keys())))
    return {p: allowed[p] for p in sorted(paths)}


def main():
    require(sys.argv[1:] in [['prepare'], ['build'], ['run']], 'prepare|build|run')
    action = sys.argv[1]
    record = json.loads(RESULT.read_text())
    committed = json.loads(git('show', 'HEAD:' + str(RESULT.relative_to(P))))
    require(record.get(KEY) == committed.get(KEY), 'commit exact previous stage')
    require(not any(v.get('active') for v in record.values() if isinstance(v, dict)), 'active CPU observer')
    fused = json.loads(FUSED.read_text())
    require(not any(v.get('active') for v in fused.values() if isinstance(v, dict)), 'active native job')
    guards.no_model_owners([])
    if action == 'prepare':
        require(KEY not in record and not ROOT.exists(), 'no implicit retry')
        rev = git('rev-parse', 'HEAD').decode().strip()
        pins = sources(rev)
        ROOT.mkdir(mode=0o700)
        for name in ['tmp', 'cache', 'control']: (ROOT/name).mkdir(mode=0o700)
        manifest = fused['coherent_prefill_speed_v1']['runtime_manifest']
        require(sha(manifest['path']) == manifest['sha256'], 'original manifest mismatch')
        row = dict(stage='prepared', source_commit=rev, source_inputs=pins,
            commands=commands(), environment=environment(), runtime_manifest=manifest,
            compiler_manifest=fused['metric_capability']['dependencies'],
            driver=guards.driver_guard(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            external_inputs={str(CC): sha(CC), str(LOADER): sha(LOADER)}, limits=LIMITS,
            plan=dict(question='Current B570 queue-group capabilities for a bounded prefill event-lane audit',
                gate='Normal0,exact providers,all three protocol gates,unchanged boot/driver,no fault and closed owner',
                repetitions=1, GPU_work_submitted=False, performance_or_overlap_claim=False,
                retained_generations='Two unchanged engine references plus this temporary metadata-only C probe',
                retirement='After capability extraction and original source/recipe recovery; no retained new engine'))
        record[KEY] = row
    else:
        row = record[KEY]
        require(row['stage'] == ('prepared' if action == 'build' else 'built'), 'stage mismatch')
    commands_record = []
    supervisor = owner_api.Owner(ROOT/'control', lambda: write(ROOT/'ownership.json', commands_record))
    commands_record = supervisor.commands

    def run(label, argv, interaction=None):
        entry, out, err = supervisor.run(label, argv, environment(), P, interaction=interaction, **LIMITS)
        require(owner_api.closed(entry) and entry['exit_code'] == 0, label + ' failed; inspect existing owner')
        return entry, out, err

    try:
        providers = verify(row)
        if action == 'prepare':
            _, out, _ = run('dependencies', [str(CC), '-M', str(SOURCE)])
            names = out.read_text().replace('\\\n', ' ').split(':', 1)[1].split()
            for name in names:
                p = Path(name).resolve()
                if p != SOURCE: row['external_inputs'][str(p)] = sha(p)
            row['compiler_version'] = subprocess.check_output([str(CC), '--version'], timeout=10).decode().splitlines()[0]
        elif action == 'build':
            run('build', commands()[0])
            names = (ROOT/'actual.d').read_text().replace('\\\n', ' ').split(':', 1)[1].split()
            actual = {str(Path(n).resolve()): sha(Path(n).resolve()) for n in names}
            expected = {str(SOURCE): row['source_inputs'][str(SOURCE.relative_to(P))], **row['external_inputs']}
            require(all(expected.get(p) == h for p, h in actual.items()), 'compiler dependency mismatch')
            row['actual_header_inputs'] = len(actual)
            row['binary'] = dict(path=str(ROOT/'inventory'), sha256=sha(ROOT/'inventory'))
            row['stage'] = 'built'
        else:
            row.update(stage='running', active=True, ownership_closed=False)
            write(RESULT, record)
            observations, gates, provider_maps = [], [], {}

            def interact(channel, _entry):
                while True:
                    item = json.loads(channel.line(seconds=20))
                    if item['kind'] == 'gate':
                        name = item['name']
                        require(name == ['before-runtime', 'runtime-ready', 'before-exit'][len(gates)], 'gate order')
                        provider_maps[name] = mappings(channel.proc.pid, row, providers)
                        if name != 'before-runtime':
                            require(str(NEO) in provider_maps[name], 'selected driver not actually mapped')
                        gates.append(name)
                        channel.send(b'continue\n')
                    else:
                        observations.append(item)
                        if item['kind'] == 'done':
                            channel.exit(10)
                            break
            entry, _, err = run('inventory', [row['binary']['path']], interact)
            require(gates == ['before-runtime', 'runtime-ready', 'before-exit'], 'missing gate')
            require(any(v['kind'] == 'queue_group' for v in observations), 'no queue groups')
            verify(row)
            row.update(stage='closed', passed=True, complete=True, active=False, ownership_closed=True,
                result=observations, provider_maps=provider_maps, native_exit_code=entry['exit_code'],
                stderr_bytes=err.stat().st_size, boot_unchanged=True, fresh_faults=False)
        verify(row)
    except Exception as exc:
        row.update(stage='failed', error=str(exc), active=supervisor.active is not None,
            ownership_closed=supervisor.active is None, passed=False, complete=False)
        write(RESULT, record)
        raise
    write(RESULT, record)
    print(json.dumps(dict(stage=row['stage'], source_commit=row['source_commit'], result=row.get('result'))))


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
