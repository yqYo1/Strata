#!/usr/bin/env python3
"""Main-owned frozen activation/extraction for the current QSA native observer.

Repository LICENSE. Never qualifies the rejected staging candidate for a model.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_qsa_k_staging as c
b = c.b
P, RESULT = b.P, b.RESULT
ROOT = b.ROOT / 'native-image-v1'
KEY = 'native_image_diagnostic'
SELF = str(Path(__file__).resolve().relative_to(P))
require = b.guards.require


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def original_host_plan(data):
    ref = data['host_plan_ref']
    plan = json.loads(b.gen.git('show', ref['commit'] + ':' + ref['path']))
    for key in ref['key']:
        plan = plan[key]
    require(fingerprint(plan) == ref['canonical_json_sha256'], 'original host plan changed')
    return plan


def observer(record):
    data = record[KEY]
    host = data['host_check']
    require(host['passed'] and host['complete'] and host['ownership_closed'] and
        not host['active'] and len(host['samples']) == 21 and not host['GPU_executed'],
        'closed 21-case host observer gate required')
    require(not record['decision_summary']['performance_gate_passed'] and
        not record['decision_summary']['advance_to_model'], 'preserve packed performance rejection')
    plan = original_host_plan(data)
    path = data['source']
    raw = b.gen.git('show', plan['source_commit'] + ':' + path)
    require(hashlib.sha256(raw).hexdigest() == data['source_sha256'] == b.digest(P / path),
        'original compiled observer source changed')
    artifact = host['artifact_inventory']['libstrata_ur_kernel_binary.so']
    require(b.digest(artifact['path']) == artifact['sha256'] and
        Path(artifact['path']).stat().st_size == artifact['bytes'], 'observer binary changed')
    deps = shlex.split((ROOT / 'observer.d').read_text().replace('\\\n', ' ').split(':', 1)[1])
    actual = {str(Path(p).resolve()): b.digest(p) for p in deps}
    require(all(plan['dependencies'].get(p) == sha for p, sha in actual.items()),
        'original observer actual compiler input changed')
    census = host['actual_dependency_census']['observer']
    require(len(actual) == census['files'] and fingerprint(actual) == census['sha256'],
        'observer actual compiler closure changed')
    for path, sha in plan['compiler_tools'].items():
        require(b.digest(path) == sha, 'observer compiler tool changed: ' + path)
    framework = plan['framework']
    require(b.digest(framework['path']) == framework['sha256'], 'XPTI dispatcher changed')
    return dict(source_commit=plan['source_commit'], source_sha256=data['source_sha256'],
        library=artifact, framework=framework, compiler_dependencies=len(actual),
        actual_dependency_sha256=fingerprint(actual), host_plan=data['host_plan_ref'])


def admission(record):
    rt = c.run_admission(record, record['actual_id_plan'])
    b.qualified_runtime.current(rt)
    info = observer(record)
    original = original_host_plan(record[KEY])
    require(original['runtime_files'] == rt['runtime_files'], 'host observer runtime qualification changed')
    return rt, info


def controller(revision, record):
    shared = {**b.inputs(revision), **c.inputs(revision)}
    expected = {**record['input_sha256'], **record['actual_id_plan']['controller_inputs']}
    require(shared == expected, 'shared native/controller source changed')
    raw = b.gen.git('show', revision + ':' + SELF)
    require(raw == (P / SELF).read_bytes(), 'commit exact native-image controller')
    return dict(source_commit=revision, path=SELF, sha256=hashlib.sha256(raw).hexdigest(),
        shared_source_files=len(shared), shared_source_sha256=fingerprint(shared))


def prepare_activation():
    record = b.gen.committed(RESULT);data = record[KEY]
    folder = ROOT / 'activation'
    require('activation_plan' not in data and not folder.exists(), 'fresh activation only;no retry')
    rt, info = admission(record)
    revision = b.gen.git('rev-parse', 'HEAD').decode().strip()
    code = controller(revision, record)
    canary = rt['preload']['canary']
    require(b.digest(canary['path']) == canary['sha256'], 'qualified no-device canary changed')
    folder.mkdir(mode=0o700)
    for name in ['captures', 'tmp', 'cache']:
        (folder / name).mkdir(mode=0o700)
    library = info['library']['path']
    env = dict(rt['environment'], TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
        XPTI_TRACE_ENABLE='1', XPTI_FRAMEWORK_DISPATCHER=info['framework']['path'],
        XPTI_SUBSCRIBERS=library, STRATA_NATIVE_CAPTURE_SUBSCRIBER=library,
        STRATA_NATIVE_CAPTURE_ROOT=str(folder / 'captures'), LD_DEBUG='files',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
    env.update({'STRATA_NATIVE_KERNEL_' + str(i): name for i, name in enumerate(c.kernel_ids(record))})
    plan = dict(stage='prepared', controller=code, observer=info, canary=canary,
        runtime_qualification_sha256=b.digest(b.qualified_runtime.RESULT),
        command=[canary['path']], environment=env, output_root=str(folder),
        budgets=dict(wall=30, cpu=30, rss_cap=1 << 30, text_cap=2 << 20,
            file_cap=2 << 20, total_cap=8 << 20),
        criteria='One owned no-device canary closes0. Exact provider/pre-main validation proof,expected46/unchanged version,one same-PID observer load/subscribe,expected namespace0 library/dispatcher init and no other subscriber. Trace has null-platform API version check and no device/context/queue/enqueue/module/kernel work;zero captures,quiet validation,no freshfault/bootchange. This qualifies only subscriber startup,not native getters/extraction/teardown/performance/model.',
        retirement='Keep compact activation/provenance/fault/closure result;delete successful loader/API trace and duplicate ownership output after extraction. Same observer generation remains for immediate three-query extraction only.')
    data['activation_plan'] = plan;b.write(folder / 'plan.json', plan);b.write(RESULT, record)
    print('Prepared no-device actual subscriber activation;commit exact plan before run.')


def run_activation():
    record = b.gen.committed(RESULT);data = record[KEY];plan = data['activation_plan']
    folder = Path(plan['output_root'])
    require(plan['stage'] == 'prepared' and folder == ROOT / 'activation' and
        plan == json.loads((folder / 'plan.json').read_text()) and 'activation_check' not in data,
        'fresh committed activation plan required')
    rt, info = admission(record)
    require(controller(plan['controller']['source_commit'], record) == plan['controller'] and
        info == plan['observer'] and b.digest(b.qualified_runtime.RESULT) == plan['runtime_qualification_sha256']
        and b.digest(plan['canary']['path']) == plan['canary']['sha256'], 'activation inputs changed')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursors = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor'], text=True, timeout=15), re.M)
    require(len(cursors) == 1, 'fresh journal cursor required')
    result = dict(passed=False, complete=False, active=False, ownership_closed=False,
        controller=plan['controller'], GPU_executed=False, native_getters_qualified=False)
    owner = None
    def persist():
        result['active'] = bool(owner and owner.active is not None)
        b.write(folder / 'owned.json', dict(result, commands=owner.commands if owner else []))
    owner = b.supervision.Owner(folder / 'owned', persist)
    try:
        entry, out, err = owner.run('activation', plan['command'], plan['environment'], P, **plan['budgets'])
        require(b.supervision.closed(entry) and entry['exit_code'] == 0, 'actual subscriber canary failed')
        body = json.loads(out.read_text());text = err.read_text()
        require(body == dict(pid=entry['pid'], initialized=0, checked=46, expected=46,
            version=4294967295, closed=0), 'actual canary result differs')
        preload = c.validation.preload_evidence(text, entry['pid'])
        require(preload['loader'] == rt['preload']['loader'], 'actual UR provider differs')
        rows = [json.loads(x) for x in (folder / 'captures/captures.jsonl').read_text().splitlines()]
        require(rows == [dict(kind='load', pid=entry['pid']), dict(kind='subscribe', pid=entry['pid'])],
            'exact no-device observer activation receipt required')
        initialized = re.findall(r'calling init:\s*(\S+)', text)
        for expected in [info['library']['path'], info['framework']['path']]:
            require(sum(Path(x).resolve() == Path(expected).resolve() for x in initialized) == 1,
                'expected single observer/framework initialization: ' + expected)
            family = 'libxptifw.so' if 'libxptifw.so' in expected else Path(expected).name
            objects = [(name, namespace) for name, namespace in
                re.findall(r'file=(\S+)\s+\[(\d+)\]', text) if Path(name).name.startswith(family)]
            require(objects and all(namespace == '0' and
                (not name.startswith('/') or Path(name).resolve() == Path(expected).resolve())
                for name, namespace in objects), 'observer/framework loader namespace differs')
        require(not any('libstrata_ur_' in x and str(Path(x).resolve()) not in
            [rt['preload']['library']['path'], info['library']['path']] for x in initialized),
            'unqualified co-loaded subscriber')
        trace = (folder / 'api.trace').read_text()
        require('urPlatformGetApiVersion' in trace and 'UR_RESULT_ERROR_INVALID_NULL_HANDLE' in trace and
            not re.search(r'ur(?:Enqueue|Kernel|Queue|Device|Context|Program|USM|Mem|PlatformGet\()', trace),
            'activation trace missing or unexpected native device work')
        require(not (folder / 'validation.log').exists() or not (folder / 'validation.log').stat().st_size,
            'activation validation diagnostic')
        require({p.name for p in (folder / 'captures').iterdir()} == {'captures.jsonl'},
            'no-device activation unexpectedly captured binary')
        result.update(host_probe_accepted=True, canary=body, preload=preload,
            observer_load_and_subscription=True, sole_subscriber=True,
            observed_loader_namespace=0,
            actual_observer=info['library'], dispatcher=info['framework'],
            API_trace_bytes=(folder / 'api.trace').stat().st_size)
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        c.validation.finish(result, owner, cursors[0], boot)
        result['complete'] = result['ownership_closed'];persist()
        data['activation_check'] = result;data['activation_plan']['stage'] = 'closed'
        if result['passed']:data['stage'] = 'actual_subscriber_startup_passed_native_extraction_pending'
        b.write(RESULT, record)
    require(result['passed'], 'no-device activation failed final gates')
    print('PASS actual no-device subscriber startup;native getters/GPU extraction still unqualified.')


if __name__ == '__main__':
    with (b.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare-activation': prepare_activation, 'run-activation': run_activation}[sys.argv[1]]()
