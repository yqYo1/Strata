#!/usr/bin/env python3
"""R809 Main-only qualification controller. Source delivery UNRUN; no benchmark.

Imports the pinned project Owner only when Main explicitly executes a mode.
Never configures CMake, modifies archives, installs dependencies, retries, resets,
changes a running process environment, or signals an unrelated process.
"""
import argparse
import difflib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
A = PACKAGE.parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
SCHEMA = 'strata-r809-onednn-qualification-v1'
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
ERROR = re.compile(r'(?:UR|ZE)_RESULT_ERROR_|status=FAIL|terminate called|what\(\):|Strata SYCL cleanup:|R809 drain failed|oneDNN BF16 destructor close failed')
MODEL = re.compile(r'(^|/)(strata(?:[-_.][^/]*)?|sycl_[^/]*|paired_gemm|onednn_qualified|bench\.sh|serve\.py|engine[-_]wrapper\.py|run_server_job\.py)$')
CHECKS = ['--negative', '--numerics', '--chain', '--lifecycle', '--scratch']
TEXT_CAP, JOB_CAP, RECEIPT_CAP = 64 << 20, 512 << 20, 512 << 10


def require(value, why):
    if not value:
        raise RuntimeError(why)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size < 2 << 20, 'bounded JSON file: ' + str(path))
    return json.loads(path.read_text())


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def pinned(path, pin):
    path = Path(path)
    require(path.is_file() and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
            'file identity changed: ' + str(path))
    if 'realpath' in pin:
        require(str(path.resolve()) == pin['realpath'], 'file resolution changed: ' + str(path))


def receipt_gate(module, path, expected_hash, current_boot):
    """Schema-neutral owner gate: accepts quality-screen receipts with commands.

    Reads only; a historical FAIL is never changed or accepted as qualification.
    Latestness is the caller's explicit assertion under the common lock, not a
    directory glob or an assertion this controller can independently establish.
    """
    path = Path(path).resolve()
    require(re.fullmatch(r'[0-9a-f]{64}', expected_hash) and sha(path) == expected_hash, 'explicit receipt hash')
    value = load(path)
    require(value.get('complete') is True and value.get('active') is False and value.get('passed') is True,
            'receipt must be closed and passing; original failures remain failures: ' + str(path))
    require(value.get('boot_id') == current_boot and value.get('boot_unchanged', True) is True, 'receipt current boot')
    require(not value.get('fresh_faults') and not value.get('driver_dump_exists'), 'receipt fault/dump')
    require(isinstance(value.get('commands'), list) and value['commands'], 'receipt owned commands required')
    for command in value['commands']:
        require(module.closed(command) and command.get('exit_code') == 0 and command.get('owners'),
                'receipt command normal closure and identities required')
        for old in command['owners']:
            now = module.identity(old['pid'])
            require(now is None or now['start_ticks'] != old['start_ticks'], 'receipt PID/start still present')
    require(sha(path) == expected_hash, 'receipt changed while checking closure')
    return value, dict(path=str(path), sha256=expected_hash, boot_id=current_boot, normally_closed=True)


def no_model_owners(model_files):
    """EXEC-only same-UID read-only scan; no signals, adoption or worker scan."""
    paths = list(Path('/proc').iterdir())
    require(len(paths) < 131072, 'finite proc population')
    found = []
    for path in paths:
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = [v.decode('utf-8', errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v]
            hits = [v for v in args if not v.startswith('-') and MODEL.search(v)]
            hits += [v for v in args if str(P/'serve') in v and v.endswith('.py')]
            hits += [v for v in args if v in model_files or v.startswith('serve.') or v.endswith('.gguf')]
            if hits:
                found.append(dict(pid=int(path.name), matching_arguments=hits))
        except (FileNotFoundError, ProcessLookupError):
            continue
        # Same-UID permission/observation ambiguity is an exception, never PASS.
    require(not found, 'model/fixture owner present: ' + json.dumps(found))
    return dict(same_uid_model_owners=[], scope='bounded cmdline scan; uncooperative foreign GPU use cannot be disproved')


def driver_guard():
    require((DEVICE/'driver').resolve().name == 'xe', 'expected xe driver')
    require((DEVICE/'vendor').read_text().strip() == '0x8086', 'expected Intel PCI vendor')
    dumps = list(Path('/sys/class/devcoredump').glob('devcd*'))
    require(not (DEVICE/'devcoredump').exists() and not dumps, 'driver dump exists; never remove or reset')
    return dict(pci=str(DEVICE), driver='xe', vendor='0x8086', devcoredump=False,
                device=(DEVICE/'device').read_text().strip(), kernel=Path('/proc/sys/kernel/osrelease').read_text().strip())


def generated_fixture(manifest):
    transform = manifest['fixture_transform']
    source = Path(transform['source_path']).read_text()
    require(hashlib.sha256(source.encode()).hexdigest() == transform['input_sha256'], 'fixture input SHA')
    original = source
    for edit in transform['edits']:
        require(source.count(edit['anchor']) == edit['count'], 'unexpected fixture anchor: ' + edit['name'])
        source = source.replace(edit['anchor'], edit['replacement'])
    diff = ''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
        fromfile='R804/proposal/tests/sycl/onednn_faithful_bf16.cpp', tofile='private/onednn_qualified.cpp'))
    require(hashlib.sha256(source.encode()).hexdigest() == transform['output_sha256'], 'generated fixture SHA')
    require(diff == transform['unified_diff'] and hashlib.sha256(diff.encode()).hexdigest() == transform['diff_sha256'],
            'exact fixture diff')
    return source, diff


def evidence(stdout, stderr, mode, manifest):
    """Capture original stdout and qualified DSOs; unsupported APIs stay FAIL."""
    require(len(stdout.encode()) <= 256 << 10, 'compact fixture stdout budget')
    require(not ERROR.search(stdout + '\n' + stderr), 'fixture/UR/ZE error text')
    markers = {
        '--smoke': 'numerical_cases=1 faithful_BF16=PASS model_quality=UNESTABLISHED',
        '--numerics': 'numerical_cases=20 faithful_BF16=PASS model_quality=UNESTABLISHED',
        '--negative': 'negative_contract=PASS',
        '--chain': 'BF16_producer_consumer_chain=PASS internal_FP32_storage=UNKNOWN',
        '--lifecycle': 'cache_queue_close_destructor=PASS executes=1065',
    }
    if mode == '--scratch':
        # Zero actual scratch is a truthful unexercised subcase, never a PASS
        # for rejection coverage. An unsupported API/exception exits nonzero.
        match = re.findall(r'^actual_nonzero_scratch_zero_budget_reject=(PASS|NOT_EXERCISED)(?: actual_scratch_bytes=0)?$', stdout, re.M)
        require(len(match) == 1, 'scratch original outcome missing')
        scratch_coverage = 'exercised' if match[0] == 'PASS' else 'NOT_EXERCISED: actual scratch=0'
        require(match[0] == 'PASS', 'scratch rejection NOT_EXERCISED; qualification remains unestablished, never skip-to-PASS')
    else:
        require(markers[mode] in stdout, 'original fixture completion missing: ' + mode)
        scratch_coverage = None
    plans = []
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith('shape='):
            continue
        require(i + 7 < len(lines), 'truncated plan')
        block = lines[i:i+8]
        require(' implementation=' in line and line.split(' implementation=', 1)[1].strip(), 'selected implementation missing')
        require(block[3] == 'library_hash=' + manifest['onednn']['source_revision'], 'actual oneDNN source identity')
        require(all(v in block[5] for v in ['strict_accumulation=1', 'strict_fpmath=1', 'user_scratchpad=1',
                'environment_rounding=1', 'exact_bf16_descriptors=1']), 'actual strict attributes')
        require(re.fullmatch(r'scratchpad_bytes=\d+ primitive_owned_memory_bytes=\d+ full_fp32_output_equivalent_bytes=\d+', block[6]), 'actual memory accounting')
        require(block[7] == 'internal_fp32_partition=UNKNOWN native_XMX=UNESTABLISHED GPU_arithmetic=UNQUALIFIED', 'arithmetic claim changed')
        plans.append('\n'.join(block))
    require(len(plans) == (20 if mode == '--numerics' else 0 if mode == '--negative' else 1), 'all actual plans required')
    loaded = []
    for name in re.findall(r'^R809_DSO=(.+)$', stdout, re.M):
        require(not name.endswith(' (deleted)'), 'deleted mapped library')
        path = str(Path(name).resolve())
        require(path in manifest['loaded_library_pins'], 'unqualified mapped library: ' + name)
        pin = manifest['loaded_library_pins'][path]
        pinned(path, pin)
        loaded.append(dict(reported_path=name, **pin))
    require(len(loaded) > 0 and len(loaded) < 256, 'bounded actual loaded-library proof')
    paths = {v['realpath'] for v in loaded}
    for path in manifest['required_loaded_libraries']:
        require(path in paths, 'required library absent from actual loaded proof: ' + path)
    named = re.findall(r'^R809_DNNL=(.+)$', stdout, re.M)
    require(len(named) == 1 and str(Path(named[0]).resolve()) == manifest['onednn']['library_realpath'], 'dnnl_version dladdr identity')
    cpu = re.findall(r'^R809_CPU=(.+)$', stdout, re.M)
    require(len(cpu) == 1, 'process and submitting-thread CPU time missing')
    fields = dict(v.split('=', 1) for v in cpu[0].split())
    require(set(fields) == {'process_user_s', 'process_system_s', 'thread_user_s', 'thread_system_s'}, 'CPU field coverage')
    require(all(re.fullmatch(r'\d+(?:\.\d+)?', v) for v in fields.values()), 'finite nonnegative CPU values')
    if mode == '--smoke':
        require('urEnqueueKernelLaunch' in stderr, 'small smoke lacks UR kernel launch trace')
    return dict(stdout_original=stdout, stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(), plans=plans,
                loaded_libraries=loaded, loaded_identity_proved=True, CPU_seconds=fields,
                CPU_scope='process/submitting-thread deltas before Runtime creation to final Runtime::check; excludes loader startup, evidence collection and Runtime destruction; includes setup/JIT/waits; no host bottleneck attribution',
                scratch_rejection_coverage=scratch_coverage,
                arithmetic_placement='UNQUALIFIED', native_XMX='UNESTABLISHED', internal_FP32_partition='UNKNOWN')


def failure_excerpt(path):
    """Bounded relevant lines; the original bounded raw log stays untouched."""
    if not path.exists():
        return dict(missing=True)
    lines = path.read_text(errors='replace').splitlines()
    selected = set()
    for index, line in enumerate(lines):
        if ERROR.search(line) or 'error:' in line.lower() or 'failed' in line.lower():
            selected.update(range(max(0, index-6), min(len(lines), index+7)))
    chunks, size = [], 0
    for index in sorted(selected):
        line = str(index+1) + ': ' + lines[index] + '\n'
        if size + len(line.encode()) > 256 << 10:
            break
        chunks.append(line)
        size += len(line.encode())
    return dict(raw_path=str(path), raw_sha256=sha(path), raw_bytes=path.stat().st_size,
                excerpt=''.join(chunks), relevant_lines=len(selected), retained_lines=len(chunks),
                truncated=len(chunks) < len(selected), rule='error line plus six neighboring lines; 256 KiB cap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'smoke', 'check'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--manifest-sha256', required=True, help='Main pins the reviewed immutable controller package')
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--prior-sha256', required=True)
    parser.add_argument('--confirm-latest-closed-prior', action='store_true', required=True,
                        help='Main asserts --prior is its latest normally closed passing owner receipt')
    parser.add_argument('--build-record', type=Path)
    parser.add_argument('--build-sha256')
    parser.add_argument('--smoke-record', type=Path)
    parser.add_argument('--smoke-sha256')
    options = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', options.run_id), 'bounded fresh run label')
    require(options.mode == 'build' or (options.build_record and options.build_sha256), 'private build receipt and hash required')
    require(options.mode != 'check' or (options.smoke_record and options.smoke_sha256), 'passing smoke receipt and hash required')
    manifest = load(PACKAGE/'manifest.json')
    manifest_sha = sha(PACKAGE/'manifest.json')
    require(re.fullmatch(r'[0-9a-f]{64}', options.manifest_sha256) and manifest_sha == options.manifest_sha256, 'reviewed manifest SHA required')
    require(Path(sys.executable).resolve() == Path('/usr/bin/python3').resolve(), 'pinned system Python interpreter required')
    require((PACKAGE/'manifest.json').stat().st_size < 100 << 10, 'manifest budget')
    require(set(v.name for v in PACKAGE.iterdir()) == {'main_controller.py', 'manifest.json', 'README.md'}, 'exact package payload')
    require(sum(v.stat().st_size for v in PACKAGE.iterdir()) < 160 << 10, 'package budget')
    for name, pin in manifest['package_files'].items():
        pinned(PACKAGE/name, pin)
    owner_path = Path(manifest['direct_owner']['path'])
    pinned(owner_path, manifest['direct_owner'])
    spec = importlib.util.spec_from_file_location('r809_pinned_direct_owner', owner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = B/('main-r809-onednn-' + options.run_id + '-' + options.mode)
    target = A/('own-native16-onednn-' + options.mode + '-' + options.run_id + '.json')
    require(not root.exists() and not target.exists(), 'fresh private root and final receipt; no retry/overwrite')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not root.exists() and not target.exists(), 'fresh names under exclusive lock')
        root.mkdir(mode=0o700)
        record = dict(schema=SCHEMA, mode=options.mode, run_id=options.run_id, boot_id=boot(),
            passed=False, complete=False, active=False, GPU_executed=False, status='INCOMPLETE', commands=[],
            manifest_sha256=manifest_sha, source_only_delivery='UNRUN', private_root=str(root),
            frozen_source_commit=manifest['qualified_archive_build_commit'], fault_boundaries=[],
            model_quality='UNESTABLISHED', model_performance='UNESTABLISHED', physical_262144='UNESTABLISHED',
            unchanged_68_GEMM_test=True, benchmark_executed=False, cases={},
            controller_invocation=[sys.executable] + sys.argv,
            source_pin_receipt=dict(path=str(A/'own-native16-prefill-build-v2.json'),
                sha256=manifest['input_pins'][str(A/'own-native16-prefill-build-v2.json')]['sha256']),
            limits=manifest['limits'], unresolved_static_dependencies=manifest['static_dependencies']['unresolved'])
        def persist():
            record['active'] = owner.active is not None
            (root/'record.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        owner = module.Owner(root/'owned', persist)
        record['commands'] = owner.commands
        persist()
        buildref = load(A/'own-native16-prefill-build-v2.json')
        cursor, preflight_done, workflow_done = None, False, False
        deadline = time.monotonic() + (600 if options.mode == 'build' else 480 if options.mode == 'smoke' else 1200)
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')

        def pin_all():
            require(boot() == record['boot_id'] == buildref['boot_id'], 'current/v2 boot differs; Main requalification required')
            require(sha(PACKAGE/'manifest.json') == manifest_sha, 'manifest mutated')
            for name, pin in manifest['package_files'].items():
                pinned(PACKAGE/name, pin)
            for group in ('input_pins', 'external_pins', 'loaded_library_pins'):
                for path, pin in manifest[group].items():
                    pinned(path, pin)
            pinned(owner_path, manifest['direct_owner'])
            require(buildref['source_commit'] == manifest['qualified_archive_build_commit'], 'v2 source revision')
            for name, value in buildref['source_pins'].items():
                require(name == 'CMakeLists.txt' or name.startswith(('src/', 'include/', 'cmake/', 'tests/')), 'source pin path scope')
                require(sha(P/name) == value, 'frozen v2 source changed: ' + name)
            for path, value in record.get('dynamic_pins', {}).items():
                pinned(path, value)
            proof = record.setdefault('pin_verification', dict(checks_completed=0, first_verified_utc=module.utc(),
                manifest_sha256=manifest_sha, source_pin_count=len(buildref['source_pins']),
                external_pin_count=len(manifest['external_pins']), library_pin_count=len(manifest['loaded_library_pins'])))
            proof.update(checks_completed=proof['checks_completed']+1, last_verified_utc=module.utc())

        def remember(path, expected_hash=None):
            path = Path(path).resolve()
            value = dict(sha256=sha(path), bytes=path.stat().st_size, realpath=str(path))
            require(expected_hash is None or value['sha256'] == expected_hash, 'dynamic receipt pin changed')
            record.setdefault('dynamic_pins', {})[str(path)] = value
            return value

        def run(label, argv, env, *, building=False, native=False):
            pin_all()
            wall = min(300 if building else 180 if native else 20, deadline-time.monotonic())
            require(wall > 0, 'whole-job wall budget exhausted')
            entry, out, err = owner.run(label, list(map(str, argv)), env, root,
                wall=wall, cpu=1200 if building else 120 if native else 20,
                rss_cap=(32 if building else 8 if native else 1) << 30,
                text_cap=TEXT_CAP, total_cap=JOB_CAP, file_cap=TEXT_CAP)
            pin_all()
            require(module.closed(entry) and entry.get('exit_code') == 0, label + ': original owned command failed; no retry')
            return out, err

        def journal(label, after=None):
            argv = ['/usr/bin/journalctl', '-k', '-b', '--no-pager', '-o', 'json', '--show-cursor']
            argv += ['--after-cursor', after] if after else ['-n', '0']
            out, err = run(label, argv, hostenv)
            text = out.read_text()
            tail = re.search(r'^-- cursor: (\S+)\s*$', text, re.M)
            entries = [json.loads(line) for line in text.splitlines() if line and not line.startswith('-- cursor: ') and line != '-- No entries --']
            require(tail is not None or (after is not None and not entries), 'journal cursor coverage')
            end = tail[1] if tail else after
            faults = [v for v in entries if FAULT.search(str(v.get('MESSAGE', '')))]
            record['fault_boundaries'].append(dict(label=label, after_cursor=after, end_cursor=end,
                entry_count=len(entries), fault_filter=FAULT.pattern, fault_entries=faults, fault_count=len(faults),
                empty_window=not entries, cursor_reused=tail is None, boot_id=boot(), check_exit_code=0,
                stdout_sha256=sha(out), stderr_sha256=sha(err)))
            persist()
            require(boot() == record['boot_id'], 'journal boot changed')
            driver_guard()
            require(not faults, 'fresh fault window; no dump deletion, reset or retry')
            return end

        def environment(label, *, traced=False, building=False):
            env = dict(manifest['build_environment'] if building else manifest['execution_environment'])
            cache, tmp = root/('cache-' + label), root/('tmp-' + label)
            cache.mkdir(); tmp.mkdir()
            env.update(XDG_CACHE_HOME=str(cache), TMPDIR=str(tmp))
            if traced:
                env.update(manifest['first_smoke_trace_environment'])
            # Entire environment frozen before exec. No os.environ mutation.
            return env

        try:
            pin_all()
            _, record['prior'] = receipt_gate(module, options.prior, options.prior_sha256, record['boot_id'])
            remember(options.prior, options.prior_sha256)
            for name in ['own-native16-prefill-build-v2.json', 'own-native16-gemm-correctness-v2.json']:
                path = A/name
                value, proof = receipt_gate(module, path, manifest['input_pins'][str(path)]['sha256'], record['boot_id'])
                record.setdefault('qualification_gates', []).append(proof)
                if name.endswith('gemm-correctness-v2.json'):
                    require(json.loads(value['checks']['gemm16-first']['stdout'])['numerical_cases'] == 68, 'unchanged original 68 GEMM PASS required')
            record['owner_scan'] = no_model_owners(manifest['known_model_files'])
            record['device_sysfs'] = driver_guard()
            cursor = journal('journal-fresh-n0')
            preflight_done = True
            require(not manifest['static_dependencies']['unresolved'],
                    'static dependencies unresolved under the authorized fixed environment: ' + json.dumps(manifest['static_dependencies']['unresolved']))
            require(manifest['static_dependencies']['sycl_soname_compatible'], 'libsycl.so.9 soname incompatibility')
            if options.mode == 'build':
                source, diff = generated_fixture(manifest)
                fixture, diffpath = root/'onednn_qualified.cpp', root/'fixture-transform.diff'
                fixture.write_text(source); diffpath.write_text(diff)
                remember(fixture); remember(diffpath)
                generated = dict(source=str(fixture), sha256=sha(fixture), diff=str(diffpath), diff_sha256=sha(diffpath),
                    original=manifest['fixture_transform']['source_path'], original_sha256=manifest['fixture_transform']['input_sha256'])
                record['generated_fixture'] = generated
                objects = [root/'component.o', root/'fixture.o']
                inputs = [Path(manifest['component_source']), fixture]
                compile_argvs = []
                env = environment('build', building=True)
                for i, (src, obj) in enumerate(zip(inputs, objects)):
                    argv = list(manifest['compile_template'])
                    argv = [str(src) if v == '@SOURCE@' else str(obj) if v == '@OBJECT@' else v for v in argv]
                    compile_argvs.append(argv)
                    run('compile-' + str(i), argv, env, building=True)
                    remember(obj)
                exe = root/'onednn_qualified'
                replacements = {'@COMPONENT@': str(objects[0]), '@FIXTURE@': str(objects[1]), '@BINARY@': str(exe)}
                link = [replacements.get(v, v) for v in manifest['link_template']]
                record.update(actual_compile_argvs=compile_argvs, actual_link_argv=link)
                persist()
                run('link-private', link, env, building=True)
                remember(exe)
                record.update(binary=str(exe), binary_sha256=sha(exe),
                    executable_consumer=dict(executable=str(exe), SHA256=sha(exe), purpose='R809 smoke/check only; no model or benchmark',
                        archives=manifest['linked_files'], generated_fixture=generated, component_sha256=sha(inputs[0])))
            else:
                built, record['private_build_gate'] = receipt_gate(module, options.build_record, options.build_sha256, record['boot_id'])
                remember(options.build_record, options.build_sha256)
                require(built.get('schema') == SCHEMA and built.get('mode') == 'build' and built.get('manifest_sha256') == manifest_sha, 'this package private build receipt')
                exe = Path(built['binary'])
                builtroot = Path(built['private_root'])
                require(builtroot.parent == B and builtroot.name.startswith('main-r809-onednn-') and builtroot.name.endswith('-build') and
                        builtroot.resolve() == builtroot and exe.parent == builtroot and exe.name == 'onednn_qualified', 'owned private executable scope')
                require(sha(exe) == built['binary_sha256'], 'private executable SHA')
                require(built['generated_fixture']['sha256'] == manifest['fixture_transform']['output_sha256'], 'qualified generated fixture')
                for path, pin in built['dynamic_pins'].items():
                    pinned(path, pin)
                    record.setdefault('dynamic_pins', {})[path] = pin
                record.update(binary=str(exe), binary_sha256=sha(exe), executable_consumer=built['executable_consumer'])
                if options.mode == 'check':
                    smoke, record['smoke_gate'] = receipt_gate(module, options.smoke_record, options.smoke_sha256, record['boot_id'])
                    remember(options.smoke_record, options.smoke_sha256)
                    require(smoke.get('schema') == SCHEMA and smoke.get('mode') == 'smoke' and smoke.get('manifest_sha256') == manifest_sha and
                            smoke.get('binary_sha256') == sha(exe) and smoke.get('completed_modes') == ['--smoke'], 'same binary passing first smoke')
                    require(smoke['cases']['--smoke']['loaded_identity_proved'] is True, 'mandatory smoke loaded identity proof')
                for mode in ['--smoke'] if options.mode == 'smoke' else CHECKS:
                    label = mode[2:]
                    pin_all()
                    no_model_owners(manifest['known_model_files'])
                    driver_guard()
                    cursor = journal(label + '-before', cursor)
                    env = environment(label, traced=mode == '--smoke')
                    record.setdefault('case_environments', {})[mode] = env
                    record['GPU_executed'] = True
                    persist()
                    native_error = None
                    try:
                        out, err = run(label, [exe, mode], env, native=True)
                    except BaseException as exc:
                        native_error = exc
                        out, err = owner.output/(label + '.stdout'), owner.output/(label + '.stderr')
                    # Preserve actual stdout on all outcomes; never turn API
                    # unsupported, skip, timeout or nonzero exit into PASS.
                    case = dict(original_exit_code=owner.commands[-1].get('exit_code'), validation_status='FAIL')
                    if out.exists() and out.stat().st_size <= 256 << 10:
                        case['stdout_original'] = out.read_text(errors='replace')
                    if native_error is not None:
                        case['stderr_failure_evidence'] = failure_excerpt(err)
                    record['cases'][mode] = case
                    persist()
                    require(owner.active is None, 'native owner not closed; no next process')
                    cursor = journal(label + '-after', cursor)
                    if native_error is not None:
                        raise native_error
                    proof = evidence(out.read_text(), err.read_text(), mode, manifest)
                    case.update(proof, validation_status='PASS')
                    record.setdefault('completed_modes', []).append(mode)
                    persist()
            pin_all()
            cursor = journal('journal-final', cursor)
            driver_guard()
            workflow_done = True
        except BaseException as exc:
            record.update(error=repr(exc), status='FAIL')
            if owner.commands:
                label = owner.commands[-1]['label']
                record['last_command_failure_evidence'] = failure_excerpt(owner.output/(label + '.stderr'))
        finally:
            # Serialize all final checks with the same lock. Only pinned Owner
            # handles its own cleanup; this controller never deletes artifacts.
            if cursor and owner.active is None and not workflow_done:
                try:
                    cursor = journal('journal-final-after-failure', cursor)
                except BaseException as exc:
                    record.setdefault('closure_errors', []).append(repr(exc))
            try:
                pin_all()
                record['after_pins_verified'] = True
                driver_guard()
            except BaseException as exc:
                record.setdefault('closure_errors', []).append(repr(exc))
            record.update(active=owner.active is not None, finished_utc=module.utc(), boot_unchanged=boot() == record['boot_id'])
            record['fresh_faults'] = [v for boundary in record['fault_boundaries'] for v in boundary['fault_entries']]
            record['driver_dump_exists'] = (DEVICE/'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
            record['ownership_closed'] = not record['active'] and all(module.closed(c) for c in owner.commands)
            record['normally_closed'] = record['ownership_closed'] and all(c.get('exit_code') == 0 for c in owner.commands)
            record['private_bytes'] = sum(v.stat().st_size for v in root.rglob('*') if v.is_file())
            complete = (preflight_done and record['ownership_closed'] and record['boot_unchanged'] and
                        not record.get('closure_errors') and record.get('after_pins_verified') and record['private_bytes'] <= JOB_CAP)
            record['complete'] = bool(complete)
            record['passed'] = bool(complete and workflow_done and record['normally_closed'] and not record['fresh_faults'] and
                                    not record['driver_dump_exists'] and not record.get('error'))
            record['status'] = 'PASS' if record['passed'] else 'FAIL'
            record['retention'] = dict(raw_text_root=str(owner.output), raw_text_cap=TEXT_CAP,
                cleanup='NONE; only Main may retire successful full traces after verifying extracted evidence and deletion manifest',
                current_consumer='Main review of this exact qualification receipt; failure logs until mechanism review',
                review_point='Main review before any next qualification, benchmark or adoption')
            persist()
            # Publish once, only after collection/ownership closure is complete.
            # Failed but closed diagnostics preserve their original FAIL status.
            record['durable_published'] = False
            if record['complete']:
                payload = json.dumps(record, indent=2, allow_nan=False) + '\n'
                if len(payload.encode()) > RECEIPT_CAP:
                    record.update(passed=False, complete=False, status='FAIL',
                        durable_publication_blocked='compact receipt budget exceeded; private evidence retained')
                else:
                    try:
                        # Set the value before serialization; a write error
                        # restores FAIL in the private record, without rewriting
                        # a possibly partial public file or an earlier receipt.
                        record['durable_published'] = True
                        payload = json.dumps(record, indent=2, allow_nan=False) + '\n'
                        with target.open('x') as stream:
                            stream.write(payload)
                            stream.flush()
                            os.fsync(stream.fileno())
                    except BaseException as exc:
                        record.update(passed=False, complete=False, status='FAIL', durable_published=False,
                            durable_publication_blocked='public write failed; do not consume possible partial receipt: ' + repr(exc))
                persist()
            else:
                record['durable_publication_blocked'] = 'incomplete closure; private record only'
                persist()
        print(json.dumps(dict(status=record['status'], passed=record['passed'], complete=record['complete'],
            active=record['active'], GPU_executed=record['GPU_executed'], private_record=str(root/'record.json'),
            durable_receipt=str(target) if record['durable_published'] else None,
            durable_sha256=sha(target) if record['durable_published'] else None)), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
