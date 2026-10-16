#!/usr/bin/env python3
"""Main-only natural32863 capture: FP32-a, FP32-b, vLLM BF16, vLLM BF16+QSA.

Adapted project controller under repository LICENSE. No build, service, retry,
cleanup, environment mutation, runtime replacement or automatic quality adoption.
"""
import argparse
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
A = D.parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'main-vllm-quality-v1'
TARGET = A / 'own-vllm-quality-screen-v1.json'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
OWNER_SHA = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
HOST_ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
NATIVE_ERROR = re.compile(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|retirement failed|Strata SYCL cleanup:|\b(?:NaN|Inf)\b', re.I)
FILE_CAP = 64 << 20
RAW_CAP = 384 << 20
DISK_CAP = 1 << 30
WALL = 4200
QUALIFIED = dict(ONEAPI_DEVICE_SELECTOR='level_zero:gpu', OMP_NUM_THREADS='1',
    NEOReadDebugKeys='1', EnableDirectSubmission='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
    SYCL_CACHE_PERSISTENT='0', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0', STRATA_SYCL_PREFILL_SAME_QUEUE='1',
    STRATA_PREFILL_LAYER_MAJOR='1', STRATA_PREFILL_RING='8', STRATA_SYCL_PROMPT_DIRECT='1',
    STRATA_ROPE_TABLE='1', STRATA_TRACE='1', UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
    UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
    UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
    UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')


def utc():
    return datetime.datetime.now(datetime.timezone.utc)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def pinned_module(name, path, digest):
    with Path(path).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
            raise RuntimeError('module pin mismatch: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_value(args, flag, value):
    require(args.count(flag) == 1, 'one explicit CLI flag: ' + flag)
    args[args.index(flag) + 1] = str(value)


def allowed_pin(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(B / 'prefill-gemm-only-contract-root-v1'), 'protected path forbidden')
    # This one location holds the already qualified official LLVM toolchain.
    # It never licenses access to any external engine source/helper/test tree.
    llvm = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
    require('XeStrata' not in str(path) or path.is_relative_to(llvm), 'external engine implementation forbidden')
    return path


def check_file_pin(path, pin):
    actual = allowed_pin(path)
    require(actual.is_file() and str(actual) == pin['realpath'] and
            actual.stat().st_size == pin['bytes'] and sha(actual) == pin['sha256'], 'build file pin: ' + str(path))


def source_pin_check(build):
    pins = build['source_pins']
    require(len(pins) == 432 and 'src/program/generate.cpp' in pins and
            'src/prefill/layer_major.inc' in pins and 'src/prefill/sycl/gemm16.cpp' in pins,
            'complete integrated432-source build snapshot')
    for name, digest in pins.items():
        path = P / name
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(P) and
                sha(path) == digest, 'compiled/build source changed: ' + name)
    # Report-only commits may advance. Preserve all build pins, including old
    # source/recipe/library evidence; never substitute HEAD equality for bytes.
    for path, pin in build['pins'].items():
        check_file_pin(path, pin)


def receipt_gate(module, path, digest, current_boot, *, gpu=False):
    path = allowed_pin(path)
    require(re.fullmatch('[0-9a-f]{64}', digest or '') and sha(path) == digest, 'explicit immutable receipt SHA')
    value = C.load_json(path)
    require(value.get('passed') is True and value.get('complete') is True and value.get('active') is False and
            value.get('ownership_closed') is True, 'only normally closed PASS receipts qualify')
    require(value.get('boot_id') == current_boot and value.get('fresh_faults') == [] and
            value.get('driver_dump_exists') is False and value.get('boot_unchanged', True) is True,
            'same boot and clean prior fault/dump evidence')
    require(not value.get('error') and not value.get('closure_errors') and not value.get('final_fault_check_error'),
            'failed correctness/closure cannot qualify')
    for field in ('correctness_passed', 'correctness_capture_passed', 'numerical_correctness_passed'):
        require(field not in value or value[field] is True, 'failed correctness flag cannot qualify: ' + field)
    if gpu:
        require(value.get('GPU_executed') is True and value.get('boot_unchanged') is True, 'healthy closed GPU prior')
    require(value.get('commands'), 'all original owner commands required')
    for command in value['commands']:
        require(C.L.command_closed(command) and module.closed(command) and
                command.get('session_anchored') and command.get('owners') and command.get('direct_identity'),
                'normal zero-exit/reaped/empty owned session')
        for old in command['owners'] + [command['direct_identity']]:
            require(isinstance(old.get('pid'), int) and isinstance(old.get('start_ticks'), int), 'PID/start identity')
            now = module.identity(old['pid'])
            require(now is None or now['start_ticks'] != old['start_ticks'], 'prior owner still present')
    return value


def validate_native(stdout, stderr, tokens, arm):
    require(not NATIVE_ERROR.search(stderr), 'native/runtime/numerical/retirement error')
    batched, chunks = len(tokens) - 1, (len(tokens) - 2 + 8192) // 8192
    done = [tuple(map(int, row)) for row in re.findall(
        r'^strata trace: layer-major layer (\d+) of (\d+): completed (\d+) chunks$', stderr, re.M)]
    require(done == [(i, 48, chunks) for i in range(48)], 'actual all48/5chunk completion, no dropped layer')
    native = arm not in C.ARMS[:2]
    dtype, element = ('bf16', 2) if native else ('fp32', 4)
    banner = [v for v in stderr.splitlines() if v.startswith('strata prefill: layer-major candidate,')]
    require(len(banner) == 1 and f'tokens={batched} chunk=8192 residual_dtype={dtype} ' in banner[0] and
            f'residual_element_bytes={element} ' in banner[0] and 'residual_gpu_rows=32768 ' in banner[0],
            'actual requested residual format/placement, no row/chunk backoff')
    native_line = [v for v in stderr.splitlines() if v.startswith('strata prefill: native16 BF16 HC residual/projection IO')]
    require(len(native_line) == int(native), 'actual native16 producer/consumer selection')
    route = [v for v in stderr.splitlines() if v.startswith('[native16-route]')]
    expected = ('[native16-route] engine=bf16 backend=vllm tile=auto(T<128:compact;else:wide) '
                'aot=bmg-g21-a0 isolated=1 experimental') if native else '[native16-route] engine=0 backend=own tile=own'
    require(route == [expected], 'actual public backend/tile selection')
    require(re.findall(r'^strata generate: prefill (\d+) tokens in (\d+) chunks,', stderr, re.M) ==
            [(str(batched), str(chunks))], 'actual initial-prefill coverage')
    prompt = re.findall(r'^prompt  :(.*)$', stdout, re.M)
    output = re.findall(r'^output  :(.*)$', stdout, re.M)
    require(len(prompt) == len(output) == 1 and [int(v) for v in prompt[0].split()] == tokens, 'actual frozen input IDs')
    generated = [int(v) for v in output[0].split()]
    require(len(generated) == 1 and 0 <= generated[0] < C.VOCAB and
            re.findall(r'^decode\s+(\d+) tokens in ', stdout, re.M) == ['1'], 'one normal generated token')
    return dict(placement=banner[0], native16_evidence=native_line, backend_evidence=route,
                completed_layers=48, chunks_per_layer=chunks, batched_tokens=batched, prompt_tokens=len(tokens),
                generated_tokens=generated, text_equality_checked=False, timing_eligible=False,
                QSA_evidence='Frozen per-process env plus pinned integrated dispatch source; no added runtime instrumentation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-receipt', required=True, type=Path)
    parser.add_argument('--build-sha256', required=True)
    parser.add_argument('--prior-receipt', required=True, type=Path, help='Main supplies the latest normally closed PASS GPU cohort')
    parser.add_argument('--prior-sha256', required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--retention-review-utc', required=True, help='explicit UTC raw review deadline within48h')
    options = parser.parse_args()
    manifest_path = D / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    global C, require, sha
    C = pinned_module('vllm_screen_comparator', D / 'compare_screen.py', manifest['package_files']['compare_screen.py']['sha256'])
    require, sha = C.require, C.sha
    package_pins = {str(D / name): pin for name, pin in manifest['package_files'].items()}
    for path, pin in package_pins.items():
        require(Path(path).stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'reviewed package source pin')
    manifest_sha = sha(manifest_path)
    require(options.build_sha256 == C.BUILD_SHA and options.input_sha256 == manifest['input_receipt']['sha256'],
            'fixed integrated build and existing natural input')
    review = datetime.datetime.fromisoformat(options.retention_review_utc.replace('Z', '+00:00'))
    require(review.tzinfo is not None and review.utcoffset() == datetime.timedelta(0) and
            datetime.timedelta(0) < review - utc() <= datetime.timedelta(hours=48), 'finite UTC retention deadline')
    module = pinned_module('vllm_screen_direct_owner', OWNER, OWNER_SHA)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not ROOT.exists() and not ROOT.is_symlink() and not TARGET.exists() and not TARGET.is_symlink(),
                'fresh exact private root/immutable result; no resume, overwrite or cleanup')
        ROOT.mkdir(mode=0o700)
        start, deadline = time.monotonic(), time.monotonic() + WALL
        record = dict(schema=C.SCHEMA, passed=False, complete=False, active=False, GPU_executed=False,
            capture_complete=False, closure_validated=False, ownership_closed=False, cases={}, commands=[],
            declared_order=list(C.ARMS), started_utc=module.utc(), boot_id=boot_id(), native_byteorder=sys.byteorder,
            controller=dict(identity=module.identity(os.getpid()), argv=list(sys.orig_argv),
                source=C.artifact(__file__), python=C.artifact(Path(sys.executable).resolve())),
            host=dict(kernel_release=os.uname().release, machine=os.uname().machine, pci_bdf=DEVICE.name,
                pci_identifiers={n: (DEVICE / n).read_text().strip()
                                 for n in ('vendor', 'device', 'revision', 'subsystem_vendor', 'subsystem_device')}),
            manifest=C.artifact(manifest_path), comparator=C.artifact(D / 'compare_screen.py'),
            legacy_oracle=C.artifact(C.ORACLE), owner=dict(path=str(OWNER), sha256=OWNER_SHA), references={},
            budget=dict(native_wall_seconds=900, whole_wall_seconds=WALL, shutdown_reserve_seconds=60,
                native_cpu_seconds=18000, native_RSS_bytes=88 << 30, command_text_bytes=FILE_CAP,
                aggregate_owner_text_bytes=FILE_CAP, raw_capture_bytes_strictly_less_than=RAW_CAP,
                whole_private_disk_bytes=DISK_CAP,
                exception_reason='Four64x248320 FP32 sparse arrays plus dense NLL and bounded owner text; default64MiB total cannot hold the requested evidence.'),
            retention=dict(owner='Main', consumer='compare_screen.py:compare_record and Main practical precision decision',
                raw_paths=[], review_deadline_utc=review.isoformat(), cleanup_performed=False,
                next_review='At comparison closure/supersession, no later than explicit review deadline; no automatic cleanup.'),
            model_quality_accepted=False, model_quality_qualified=False, quality_accepted=False, performance_qualified=False,
            physical_262144_qualified=False, restore_qualified=False, scope=C.LIMITATION, FP32_controls_identical=False,
            fresh_faults=[], journal_windows=[])

        def persist():
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

        owner = module.Owner(ROOT / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        build = smoke = natural = profile = None
        cursor = None
        final_error = None

        def remaining(cap):
            C.deadline_check(deadline)
            return min(cap, max(0.01, deadline - time.monotonic()))

        def frozen():
            C.deadline_check(deadline)
            require(sha(manifest_path) == manifest_sha and sha(OWNER) == OWNER_SHA, 'controller manifest/owner frozen')
            for path, pin in package_pins.items():
                require(sha(path) == pin['sha256'], 'controller/comparator/review source frozen')
            for path, digest in manifest['unchanged_sources'].items():
                require(sha(P / path) == digest, 'unchanged legacy source/provenance pin: ' + path)
            for value in record['references'].values():
                C.check_artifact(value, cap=16 << 20)
            require(boot_id() == record['boot_id'], 'boot changed')
            if build is None:
                return
            source_pin_check(build)
            require(sha(build['binary']) == build['binary_sha256'], 'integrated binary pin')
            if smoke is None:
                return
            for item in smoke['runtime_files']:
                require(sha(allowed_pin(item['path'])) == item['sha256'], 'qualified runtime library changed')
            for field in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
                require(all(sha(allowed_pin(path)) == digest for path, digest in smoke[field].items()), 'pack/MTP metadata pin')
            for path, pin in smoke['model_identity'].items():
                stat = allowed_pin(path).stat()
                require(stat.st_size == pin['bytes'] and stat.st_mtime_ns == pin['mtime_ns'], 'model shard stat identity')
            if natural is not None:
                require(sha(natural['token_file']) == natural['file_sha256'] and
                        all(sha(path) == digest for path, digest in natural['source_pins'].items()), 'input/preparer source pins')
            if profile is not None:
                require(sha(profile) == smoke['external_expert_profile_sha256'], 'external expert profile unchanged')

        def budget_check():
            files = [p for p in ROOT.rglob('*') if p.is_file()]
            disk = sum(p.stat().st_size for p in files)
            raw = sum(p.stat().st_size for p in files if p.name in ('selected.bin', 'nll.bin') or
                      p.suffix in ('.stdout', '.stderr', '.meta') or p.name.endswith('.meta.partial'))
            record.update(private_bytes=disk, raw_capture_bytes=raw)
            require(disk <= DISK_CAP and raw < RAW_CAP, 'whole private disk/raw capture budgets')

        def utility(label, argv):
            C.deadline_check(deadline)
            command, out, err = owner.run(label, argv, HOST_ENV, P, wall=remaining(30), cpu=120,
                rss_cap=2 << 30, text_cap=8 << 20, total_cap=DISK_CAP, file_cap=FILE_CAP)
            require(C.L.command_closed(command) and module.closed(command), label + ': normal owned closure required')
            budget_check()
            return out

        def driver_guard():
            require(boot_id() == record['boot_id'] and (DEVICE / 'driver').resolve().name == 'xe' and
                    not (DEVICE / 'devcoredump').exists(), 'fresh boot/driver/dump guard')

        def journal(label):
            nonlocal cursor
            driver_guard()
            out = utility(label, ['/usr/bin/journalctl', '-k', '-b', '--after-cursor', cursor,
                                  '--no-pager', '-o', 'json', '--show-cursor'])
            lines = out.read_text().splitlines()
            ends = [v.split(': ', 1)[1] for v in lines if v.startswith('-- cursor: ')]
            require(len(ends) == 1, 'explicit fresh journal end cursor')
            rows = [json.loads(v) for v in lines if v and not v.startswith('-- cursor: ')]
            require(all(v.get('_BOOT_ID') == record['boot_id'].replace('-', '') for v in rows), 'journal boot identity')
            faults = [v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))]
            record['journal_windows'].append(dict(label=label, after_cursor=cursor, end_cursor=ends[0],
                entries=len(rows), artifact=C.artifact(out), faults=faults, filter=FAULT.pattern))
            record['fresh_faults'].extend(faults)
            cursor = ends[0]
            record.update(boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=(DEVICE / 'devcoredump').exists())
            persist()
            require(not record['fresh_faults'] and record['boot_unchanged'] and not record['driver_dump_exists'],
                    'fresh faults accumulate; never clear a failed window')

        def expired(signum, frame):
            raise TimeoutError(deadline_phase)

        deadline_phase = '4140-second work deadline; final60seconds reserved for owned closure/fault evidence'
        previous_alarm = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, WALL - 60)
        try:
            build = receipt_gate(module, options.build_receipt, options.build_sha256, record['boot_id'])
            require(build['schema'] == 'strata-vllm-prefill-build-v1' and build['source_commit'] == C.SOURCE_COMMIT,
                    'exact integrated build source')
            record['references']['build'] = C.artifact(options.build_receipt)
            smoke_path = A / 'own-vllm-model-smoke-v1.json'
            smoke = receipt_gate(module, smoke_path, C.SMOKE_SHA, record['boot_id'], gpu=True)
            record['references']['model_smoke'] = C.artifact(smoke_path)
            require(build['source_pins'] == smoke['source_pins'] and build['source_commit'] == smoke['source_commit'] and
                    build['binary_sha256'] == smoke['binary_sha256'] and smoke['build_gate']['sha256'] == options.build_sha256,
                    'same integrated model mechanics/build/source chain')
            require(set(smoke['cases']) == {'fp32', 'bf16-vllm', 'bf16-vllm-qsa'}, 'three actual prior public-hook arms')
            for case in smoke['cases'].values():
                require(len(case['all128_target_NLLs']) == 128 and len(case['independent_selected_checks']) == 3,
                        'complete prior all128/3selected evidence')
                require(all(math.isfinite(v['loss']) and v['loss'] >= 0 for v in case['all128_target_NLLs']), 'prior finite dense NLL')
            prior = receipt_gate(module, options.prior_receipt, options.prior_sha256, record['boot_id'], gpu=True)
            require(prior['finished_utc'] >= smoke['finished_utc'], 'latest Main-selected PASS must postdate model mechanics')
            require(prior.get('source_commit', C.SOURCE_COMMIT) == C.SOURCE_COMMIT and
                    prior.get('binary_sha256', build['binary_sha256']) == build['binary_sha256'], 'current integrated prior')
            record['references']['prior'] = C.artifact(options.prior_receipt)
            record['prior_selection'] = 'Main explicitly supplies latest normally closed PASS; controller does not inspect active/private cohorts or discover replacements.'
            input_path = A / 'own-native16-natural-input-v1.json'
            natural, tokens = C.load_input(input_path, options.input_sha256)
            require(natural['recipe_sha256'] == manifest['preparer_sha256'], 'same original frozen natural-input recipe')
            record['input_receipt'] = C.artifact(input_path)
            inp = ROOT / 'tokens.txt'
            with inp.open('xb') as stream:
                stream.write(Path(natural['token_file']).read_bytes())
            require(sha(inp) == natural['file_sha256'], 'private identical natural input')
            profile = P / 'data/expert-profile.bin'
            profile_sha = smoke['external_expert_profile_sha256']
            frozen()
            local = ROOT / 'expert-profile.bin'
            args = list(smoke['args'])
            for flag, value in (('--max-context', 65536), ('--kv-resident', 65536), ('--prefill', 8192),
                                ('--max-new', 1), ('--tokens-file', inp), ('--expert-profile', local)):
                replace_value(args, flag, value)
            require(args.count('--no-token-graph') == 1 and '--serve' not in args and '--prefill-until' not in args and
                    '--dump-logits' not in args, 'standalone observer route, unchanged engine stream')
            base_env = dict(smoke['environment'])
            require(not any(k.startswith(('STRATA_PREFILL_NLL', 'STRATA_MEASUREMENT_', 'UNITRACE_', 'PTI_', 'IGC_', 'ZEL_')) or
                            k in ('LD_PRELOAD', 'STRATA_PREFILL_LOGITS', 'STRATA_PREFILL_LOGITS_POSITIONS') for k in base_env),
                    'no inherited observer, service output, injection or profiler')
            require(all(base_env.get(k) == v for k, v in QUALIFIED.items()), 'qualified frozen single-queue environment')
            base_env['STRATA_PREFILL_LAYER_MAJOR_R_GPU'] = '32768'
            compiler_flags = [v.split('=', 1)[1] for v in build['configure_argv'] if v.startswith('-DCMAKE_CXX_COMPILER=')]
            require(len(compiler_flags) == 1 and compiler_flags[0] in build['pins'], 'explicit compiler/build pin')
            binding = dict(source_commit=build['source_commit'], source_pins=build['source_pins'], build_file_pins=build['pins'],
                build_receipt_sha256=options.build_sha256, binary=C.artifact(build['binary']),
                compiler=C.artifact(compiler_flags[0]), runtime_files=smoke['runtime_files'],
                model_identity=smoke['model_identity'], model_identity_method='shard bytes+mtime_ns, not full-shard cryptographic hash',
                pack_metadata_sha256=smoke['pack_metadata_sha256'], MTP_metadata_sha256=smoke['MTP_metadata_sha256'],
                input_receipt_sha256=options.input_sha256, input_ids_sha256=natural['ids_le_u32_sha256'],
                input_file_sha256=natural['file_sha256'], input_source_pins=natural['source_pins'],
                external_profile=C.artifact(profile), controller=record['controller'], diagnostic_environment_origin=C.SMOKE_SHA)
            record.update(binding=binding, binding_sha256=C.canonical_sha(binding), args=args,
                source_commit=build['source_commit'], source_pins=build['source_pins'], head_ggml_type=14,
                same_queue_only=True, original_full_regression_passed=False,
                regression_limit='Original separate-queue FAIL remains immutable and ineligible. Only independently qualified single-queue route.',
                environment_changes_from_smoke=dict(STRATA_PREFILL_LAYER_MAJOR_R_GPU='32768',
                    arm='Explicit native16/dtype/backend/tile/QSA; fresh selected/dense observer+cache per arm'))
            driver_guard()
            before = utility('journal-start', ['/usr/bin/journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            match = re.search(r'-- cursor: (\S+)', before.read_text())
            require(match, 'fresh starting kernel cursor')
            cursor = match[1]
            record['journal_start_cursor'] = cursor
            for arm in C.ARMS:
                frozen()
                journal('journal-before-' + arm)
                arm_root = ROOT / arm
                arm_root.mkdir(mode=0o700)
                shutil.copyfile(profile, local)
                require(sha(local) == profile_sha, 'reset each private profile input')
                raw, nll = arm_root / 'selected.bin', arm_root / 'nll.bin'
                env = dict(base_env, **C.arm_settings(arm), XDG_CACHE_HOME=str(arm_root / 'cache'),
                    STRATA_PREFILL_LOGITS=str(raw), STRATA_PREFILL_LOGITS_POSITIONS=','.join(map(str, C.POSITIONS)),
                    STRATA_PREFILL_NLL=str(nll))
                argv = [build['binary'], *args]
                request_path = arm_root / 'request.json'
                request = dict(schema='strata-selected-logits-request-v1', arm=arm,
                    input_ids_sha256=natural['ids_le_u32_sha256'], input_receipt_sha256=options.input_sha256,
                    positions=C.POSITIONS, targets=[tokens[p + 1] for p in C.POSITIONS], vocab=C.VOCAB,
                    head='native', head_ggml_type=14, batched=len(tokens) - 1, raw_path=str(raw),
                    metadata_path=str(raw) + '.meta', expected_raw_bytes=C.RAW_BYTES, native_schema='strata-prefill-logits-v1')
                with request_path.open('x') as stream:
                    json.dump(request, stream, indent=2, allow_nan=False); stream.write('\n')
                recipe_path = arm_root / 'recipe.json'
                with recipe_path.open('x') as stream:
                    json.dump(dict(schema=C.SCHEMA, arm=arm, argv=argv, environment=env, binding=binding,
                        binding_sha256=record['binding_sha256'], request=C.artifact(request_path),
                        local_profile_initial_sha256=sha(local), raw_consumer=record['retention']),
                        stream, indent=2, allow_nan=False); stream.write('\n')
                # The recipe excludes only its own digest, avoiding a hash cycle.
                # This complete effective map is passed once, before native init.
                effective = dict(env, STRATA_PREFILL_NLL_RECIPE_SHA256=sha(recipe_path))
                case = dict(recipe=C.artifact(recipe_path), request=C.artifact(request_path), effective_environment=effective,
                    command_index=len(owner.commands), complete=False, local_profile_initial_sha256=sha(local))
                record['cases'][arm] = case
                record['retention']['raw_paths'].extend(map(str, (raw, nll)))
                record.update(active=True, GPU_executed=True)
                persist()
                command, out, err = owner.run('screen-' + arm, argv, effective, P, wall=remaining(900), cpu=18000,
                    rss_cap=88 << 30, text_cap=FILE_CAP, total_cap=DISK_CAP, file_cap=FILE_CAP)
                record['active'] = owner.active is not None
                # Always collect the fresh post-arm window after the owned child
                # retires, including native validation or numerical failures.
                if owner.active is None:
                    journal('journal-after-' + arm)
                require(C.L.command_closed(command) and module.closed(command), 'native normal closure: ' + arm)
                require(command['direct_identity']['ppid'] == record['controller']['identity']['pid'] and
                        command['direct_identity']['sid'] == command['pid'], 'true direct native controller ownership')
                case.update(validate_native(out.read_text(), err.read_text(), tokens, arm),
                    stdout=C.artifact(out), stderr=C.artifact(err), raw=C.artifact(raw), metadata=C.artifact(str(raw) + '.meta'),
                    dense_raw=C.artifact(nll), dense_metadata=C.artifact(str(nll) + '.meta'), local_profile_final_sha256=sha(local))
                require(case['raw']['bytes'] == C.RAW_BYTES, 'exact64 full-vocabulary rows')
                dense = C.parse_dense(nll, str(nll) + '.meta', tokens, sha(recipe_path))
                with ExitStack() as stack:
                    selected, metadata = C.open_capture(stack, raw, str(raw) + '.meta', tokens, C.POSITIONS, C.VOCAB, 14)
                    checked = []
                    for position in C.POSITIONS:
                        C.deadline_check(deadline)
                        stats = C.read_row(selected, C.VOCAB, tokens[position + 1])[1]
                        difference = stats['NLL'] - dense['all_target_NLLs'][position]
                        require(abs(difference) <= C.NLL_TOLERANCE, 'selected independent FP64/dense oracle')
                        checked.append(dict(position=position, target=tokens[position + 1], independent_minus_dense=difference, **stats))
                    require(not selected.read(1), 'selected exact EOF')
                case.update(independent_rows=checked, parsed_metadata=metadata, dense_rows=dense['rows'], complete=True)
                require(sha(inp) == natural['file_sha256'] and sha(recipe_path) == case['recipe']['sha256'] and
                        sha(request_path) == case['request']['sha256'], 'input/request/recipe frozen through arm')
                frozen()
                budget_check()
                if arm == 'fp32-b':
                    controls = [record['cases'][a] for a in C.ARMS[:2]]
                    selected_equal = controls[0]['raw']['sha256'] == controls[1]['raw']['sha256']
                    dense_equal = controls[0]['dense_raw']['sha256'] == controls[1]['dense_raw']['sha256']
                    equal = selected_equal and dense_equal
                    record['control_gate'] = dict(checked_before_candidates=True, selected_byte_identical=selected_equal,
                        dense_byte_identical=dense_equal,
                        selected_sha256=[v['raw']['sha256'] for v in controls], dense_sha256=[v['dense_raw']['sha256'] for v in controls],
                        generated_tokens_compared=False)
                    record['FP32_controls_identical'] = equal
                    require(equal, 'FP32 controls differ; no candidate launch or capture PASS')
                persist()
            record['capture_complete'] = True
        except BaseException as error:
            final_error = error
            record['error'] = repr(error)
        finally:
            # Keep the final fault guards/comparison inside the whole deadline.
            # The early work alarm leaves60seconds for this bounded closure.
            deadline_phase = '4200-second whole-screen deadline during final closure/comparison'
            signal.setitimer(signal.ITIMER_REAL, max(0.01, deadline - time.monotonic()))
            try:
                if cursor and owner.active is None:
                    journal('journal-final')
                frozen()
                record.update(active=owner.active is not None, finished_utc=module.utc())
                closed = (not record['active'] and bool(owner.commands) and
                    all(C.L.command_closed(c) and module.closed(c) for c in owner.commands) and
                    record.get('fresh_faults') == [] and record.get('boot_unchanged') is True and
                    record.get('driver_dump_exists') is False)
                record['ownership_closed'] = not record['active'] and all(module.closed(c) for c in owner.commands)
                require(closed, 'all-command normal ownership/kernel/boot closure required')
                record['closure_validated'] = True
                if final_error is None and record['capture_complete']:
                    record['comparison'] = C.compare_record(record, deadline=deadline)
                    journal('journal-after-comparison')
                    frozen()
                    budget_check()
                    C.deadline_check(deadline)
                    record.update(passed=True, complete=True, correctness_capture_passed=True)
            except BaseException as error:
                record.setdefault('closure_errors', []).append(repr(error))
                record.update(passed=False, complete=False)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
                record.update(active=owner.active is not None, finished_utc=module.utc(),
                    elapsed_seconds=time.monotonic() - start)
                record['ownership_closed'] = not record['active'] and all(module.closed(c) for c in owner.commands)
                if final_error is not None or record['elapsed_seconds'] >= WALL:
                    record.update(passed=False, complete=False)
                record['correctness_capture_passed'] = record['passed']
                persist()
                with TARGET.open('x') as stream:
                    json.dump(record, stream, indent=2, allow_nan=False); stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], active=record['active'],
            receipt=str(TARGET), receipt_sha256=sha(TARGET), model_quality_accepted=False, performance_qualified=False)), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
