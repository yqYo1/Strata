#!/usr/bin/env python3
"""Recover surviving PTI artifacts directly, without retired fixture readers.

Main-authored; repository LICENSE. This is HOST admission only. It never
loads a collector, enumerates a GPU, builds or runs a native workload.
Historical failures/provider-origin holds remain unchanged. A fresh bounded
collector/native/map/clock probe and matched overhead controls are still needed.
"""
import argparse
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import coherent_original_reuse as native
import unitrace_profile as pti
from request_profile_reader import PTI_REVISION

CHECKOUT = Path(__file__).resolve().parents[2]
RESULT = CHECKOUT / 'bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json'
FUSED = native.P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
THIS = 'tools/sycl/current_pti_admission.py'
ORIGINAL_SOURCES = ('09d650542c9455454516f6ebbc4597f69ec72502',
    '8fba6f3aaa22578bf0b7a3ae0cc945b6d3de89c9',
    'e57b5ee7fce1de6391c793f1014c44cae3b85d9c')
ARTIFACT_SHAS = {
    'libunitrace_tool.so': 'e7626aedf7e5bdb983c74f9d3c2e00c52468756853afb69beb37c12b671a2f5c',
    'unitrace': '6faac1e74f18ce481581f3ed112611ec3e0a1f81eac7990b57b622f212294244'}
HOST_CHECK = dict(unit_command=['/home/yayoi/.local/share/strata-sycl/venv/bin/python3',
    '-B', '-m', 'unittest', 'tools.test_current_pti_admission'],
    admission_command=['/home/yayoi/.local/share/strata-sycl/venv/bin/python3',
    '-B', THIS, '--out', '{root}/admission.json'],
    wall_seconds=120, CPU_seconds=90, RSS_bytes=1 << 30,
    text_bytes=128 << 10, total_bytes=4 << 20, GPU_executed=False,
    lock_policy='Parent locks only the unit stage;admission child owns its own exclusive lock.')


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def blob(repo, revision, path):
    return subprocess.check_output(['git', '-C', str(repo), 'show', revision + ':' + path], timeout=30)


def committed_json(path, repo):
    raw = Path(path).read_bytes()
    require(raw == blob(repo, 'HEAD', str(path.relative_to(repo))), 'commit exact admission evidence')
    return json.loads(raw)


def recover_inputs(record):
    pins = record['input_sha256']
    require(type(pins) is dict and pins, 'original source pins required')
    for name, expected in pins.items():
        require(hashlib.sha256(blob(native.P, record['source_commit'], name)).hexdigest() == expected,
                'historical controller source cannot be recovered: ' + name)
    return len(pins)


def verify_manifest(pin):
    require(digest(pin['path']) == pin['sha256'], 'original compiler manifest changed')
    rows = json.loads(Path(pin['path']).read_text())
    require(type(rows) is list and len(rows) == pin['files']
            and len({row['path'] for row in rows}) == len(rows), 'complete unique compiler inputs required')
    for row in rows:
        require(Path(row['path']).is_file() and digest(row['path']) == row['sha256'],
                'missing or changed original compiler input: ' + row['path'])
    return rows


def recover_launcher(owner):
    source = blob(native.P, owner['source_commit'], 'tools/sycl/unitrace_launch.py')
    require(hashlib.sha256(source).hexdigest() == owner['input_sha256']['tools/sycl/unitrace_launch.py'],
            'original launcher recipe source changed')
    assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'PATCHES' for target in node.targets)]
    require(len(assignments) == 1, 'one literal owned launcher overlay required')
    patches = ast.literal_eval(assignments[0].value)
    require(type(patches) is tuple and len(patches) == 3, 'exact three pre-exec launcher fixes required')
    generated = blob(pti.PTI, PTI_REVISION, 'tools/unitrace/src/unitrace.cc').decode()
    for before, after in patches:
        require(type(before) is str and type(after) is str and generated.count(before) == 1,
                'immutable official launcher patch context changed')
        generated = generated.replace(before, after)
    require(hashlib.sha256(generated.encode()).hexdigest() == owner['generated_source_sha256'],
            'recover exact original launcher source,not a substitute')
    return owner['generated_source_sha256']


def verify_tools(container):
    base, owner, indexed = [container[key] for key in
        ('unitrace_build', 'unitrace_session_owner', 'unitrace_indexed_collector')]
    require(tuple(record['source_commit'] for record in (base, owner, indexed)) == ORIGINAL_SOURCES,
            'admit the exact historical builds,not refreshed records')
    require({name: artifact['sha256'] for name, artifact in indexed['artifacts'].items()} == ARTIFACT_SHAS,
            'admit the exact surviving artifacts,not refreshed hashes')
    for record in (base, owner, indexed):
        require(record['stage'] == 'built' and record['passed'] and record['ownership_closed']
                and not record['active'] and record['official_revision'] == PTI_REVISION,
                'original closed same-revision PTI builds required')
        recover_inputs(record)
    require(pti.official_source() == base['official_source'], 'independent official PTI source changed')
    compiler = verify_manifest(base['dependencies'])
    indexed_compiler = verify_manifest(indexed['compiler_inputs'])
    root = Path(indexed['artifacts']['libunitrace_tool.so']['path']).parent
    for name, expected in indexed['generated_sha256'].items():
        require(digest(root / name) == expected, 'original indexed collector source changed')
    for artifact in indexed['artifacts'].values():
        require(Path(artifact['path']).stat().st_size == artifact['bytes']
                and digest(artifact['path']) == artifact['sha256'], 'surviving PTI artifact changed')
    require({k: indexed['artifacts']['unitrace'][k] for k in ('sha256', 'bytes')}
            == {k: owner['binary'][k] for k in ('sha256', 'bytes')},
            'surviving launcher must be exact original owned launcher bytes')
    launcher_source = recover_launcher(owner)
    require(Path(owner['command'][0]).resolve() in {Path(row['path']) for row in compiler},
            'original launcher compiler identity missing from closure')
    return dict(artifacts=indexed['artifacts'], collector_source_commit=indexed['source_commit'],
        launcher_source_commit=owner['source_commit'], recovered_launcher_source_sha256=launcher_source,
        original_compiler_inputs=len(compiler), indexed_compiler_inputs=len(indexed_compiler),
        original_manifest=base['dependencies'], indexed_manifest=indexed['compiler_inputs'],
        collector_recipe_sha256=pti.sha(json.dumps(indexed['command']).encode()),
        launcher_recipe_sha256=pti.sha(json.dumps(owner['command']).encode()),
        retired_artifact_or_fixture_readers_called=False)


def admit():
    tools = verify_tools(committed_json(RESULT, CHECKOUT))
    row, proof = native.verify_original()
    qualified = native.build.validation.committed(FUSED, ('phase_kv_split_device_v1',))
    require(qualified['stage'] == 'closed' and not qualified['active']
            and all(qualified[key] for key in ('passed', 'complete', 'ownership_closed'))
            and qualified['native_validation_probe'] == row['native_validation_probe'],
            'current independent provider/native qualification required')
    manifest = qualified['runtime_manifest']
    require(digest(manifest['path']) == manifest['sha256'], 'current qualified runtime manifest changed')
    runtime = json.loads(Path(manifest['path']).read_text())
    require(runtime and all('xestrata' not in str(Path(path).resolve()).lower()
            and digest(path) == expected for path, expected in runtime.items()),
            'use only the exact current independent provider inventory')
    require(pti.guards.driver_guard() == qualified['driver']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == qualified['boot'],
            'qualified boot/driver changed')
    return dict(schema='strata-current-pti-host-admission-v1', scope='host_source_asset_admission_only',
        controller_commit=pti.git(CHECKOUT, 'rev-parse', 'HEAD').decode().strip(),
        controller_sha256=digest(CHECKOUT / THIS), tools=tools,
        native_source_commit=proof['original_native_source'], binary=row['binaries']['strata'],
        native_build_proof_sha256=pti.sha(json.dumps(proof, sort_keys=True).encode()),
        runtime_manifest=manifest, runtime_files=len(runtime), boot=qualified['boot'], driver=qualified['driver'],
        required_pre_exec_controls=dict(UNITRACE_SystemTime='1', ZES_ENABLE_SYSMAN='0',
            UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1', EnableDirectSubmission='0'),
        source_and_artifact_recovery_passed=True, GPU_executed=False, collector_loaded=False,
        fresh_collector_probe_passed=False, observer_overhead_qualified=False,
        request_drain_qualified=False, physical_saturation_qualified=False,
        historical_provider_origin_hold_cleared=False,
        pending='Commit this direct admission;fresh bounded known-work/current-map/RAW-clock probe;'
                'then prospective32Krequest collection and matched clean observer-overhead controls. '
                'Keep original1.576/1.538 failures and old provider-origin hold unchanged.')


def check_host(output):
    root = output.parent
    require(not root.exists(), 'fresh owned host root required;inspect prior attempt')
    root.mkdir(mode=0o700); (root / 'tmp').mkdir(mode=0o700)
    owner = pti.supervision.Owner(root / 'owned')
    environment = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C',
        TMPDIR=str(root / 'tmp'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    result = dict(source_commit=pti.git(CHECKOUT, 'rev-parse', 'HEAD').decode().strip(),
        source_sha256={name: digest(CHECKOUT / name) for name in
            (THIS, 'tools/test_current_pti_admission.py', 'tools/sycl/request_profile_reader.py')},
        recipe=HOST_CHECK, passed=False, complete=False, GPU_executed=False, collector_loaded=False)
    def run(label, argv):
        entry, out, err = owner.run(label, argv, environment, CHECKOUT,
            wall=HOST_CHECK['wall_seconds'], cpu=HOST_CHECK['CPU_seconds'],
            rss_cap=HOST_CHECK['RSS_bytes'], text_cap=HOST_CHECK['text_bytes'],
            file_cap=1 << 20, total_cap=HOST_CHECK['total_bytes'])
        text = out.read_text() + err.read_text()
        require(pti.supervision.closed(entry) and entry['exit_code'] == 0, label + ': ' + text[-8000:])
        return text
    try:
        # Releasing this open-file-description lock before exec is essential:
        # the child opens the same lock file and must acquire it independently.
        with (pti.B / 'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); pti.guards.no_model_owners([])
            text = run('unit', HOST_CHECK['unit_command'])
            require(re.search(r'Ran 4 tests in [0-9.]+s', text) and text.rstrip().endswith('OK'),
                    'all four host controls required')
            result['unit_tests'] = 4
        run('admit', [value.replace('{root}', str(root)) for value in HOST_CHECK['admission_command']])
        result['admission'] = json.loads((root / 'admission.json').read_text())
        result.update(passed=True, complete=True)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        result['ownership_closed'] = owner.active is None and all(pti.supervision.closed(c) for c in owner.commands)
        result['commands'] = owner.commands
        with output.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    require(result['passed'] and result['ownership_closed'], result.get('error', 'host ownership failed'))
    print('Owned host controls and current source/asset admission close;no collector/GPUexecuted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--host-check', action='store_true')
    args = parser.parse_args()
    require((CHECKOUT / THIS).read_bytes() == blob(CHECKOUT, 'HEAD', THIS), 'commit this controller first')
    if args.host_check:
        check_host(args.out)
        sys.exit(0)
    with (pti.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); pti.guards.no_model_owners([])
        result = admit()
        with args.out.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print('Current source/assets admitted;GPUprobe,request clock/drain and overhead remain pending.')
