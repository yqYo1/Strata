#!/usr/bin/env python3
"""Main-only selected-only smoke/screen. SOURCE-ONLY delivery, repository LICENSE.

No engine changes, dense-NLL observer, automatic health discovery, retry, reset,
service, cleanup, build, environment mutation after exec, or quality adoption.
"""
import argparse
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import sys
import time

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
A = D.parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
OWNER_SHA = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
HOST_ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
NATIVE_ERROR = re.compile(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|retirement failed|Strata SYCL cleanup:|\b(?:NaN|Inf)\b', re.I)
MODEL = re.compile(r'(^|/)(strata(?:[-_.][^/]*)?|sycl_[^/]*|paired_gemm|strata_vllm_xe2_numerics|bench\.sh|serve\.py|engine[-_]wrapper\.py|run_server_job\.py)$')
FILE_CAP, RAW_CAP, DISK_CAP = 64 << 20, 384 << 20, 1 << 30
WALL, RESERVE, NATIVE_WALL = 1800, 45, 300
NATIVE_TEXT_CAP = 62 << 20  # Leave2MiB of the Owner64MiB aggregate for closure.
NATIVE_FILE_CAP = 62 << 20  # Still exceeds the63,569,928-byte selected capture.
QUALIFIED = dict(ONEAPI_DEVICE_SELECTOR='level_zero:gpu', OMP_NUM_THREADS='1',
    NEOReadDebugKeys='1', EnableDirectSubmission='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
    SYCL_CACHE_PERSISTENT='0', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0', STRATA_SYCL_PREFILL_SAME_QUEUE='1',
    STRATA_PREFILL_LAYER_MAJOR='1', STRATA_PREFILL_RING='8', STRATA_SYCL_PROMPT_DIRECT='1',
    STRATA_ROPE_TABLE='1', STRATA_TRACE='1', STRATA_PREFILL_NATIVE_DENSE_BANK='1',
    STRATA_PREFILL_CANONICAL_DENSE_BANK='1', STRATA_SYCL_PREFILL_SCRATCH_ALIAS='1',
    UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
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
    require(args.count(flag) == 1 and args.index(flag) + 1 < len(args), 'one explicit CLI value: ' + flag)
    args[args.index(flag) + 1] = str(value)


def allowed_pin(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(B / 'prefill-gemm-only-contract-root-v1'), 'protected path forbidden')
    # Official compiler/runtime binary identities only; no engine source access.
    llvm = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
    require('XeStrata' not in str(path) or path.is_relative_to(llvm), 'external engine implementation forbidden')
    return path


def check_file_pin(path, pin):
    actual = allowed_pin(path)
    require(actual.is_file() and str(actual) == pin['realpath'] and actual.stat().st_size == pin['bytes'] and
            sha(actual) == pin['sha256'], 'build file pin: ' + str(path))


def source_pin_check(build):
    pins = build['source_pins']
    require(len(pins) == 432 and len(build['pins']) == 314 and 'src/program/generate.cpp' in pins and
            'src/prefill/layer_major.inc' in pins and 'src/prefill/sycl/gemm16.cpp' in pins, 'entire source432/build314 snapshot')
    for name, digest in pins.items():
        path = P / name
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(P) and sha(path) == digest,
                'compiled source changed: ' + name)
    for path, pin in build['pins'].items():
        check_file_pin(path, pin)
    for product in build['products'].values():
        check_file_pin(product['path'], product)


def original_absence(module, commands):
    result = []
    for command in commands:
        require(command.get('owners') and command.get('direct_identity'), 'original PID/start evidence required')
        for old in command['owners'] + [command['direct_identity']]:
            require(isinstance(old.get('pid'), int) and isinstance(old.get('start_ticks'), int), 'PID/start identity')
            now = module.identity(old['pid'])
            absent = now is None or now['start_ticks'] != old['start_ticks']
            result.append(dict(original=old, current=now, original_absent=absent))
            require(absent, 'original PID/start still present; never signal/adopt an unknown process')
    return result


def no_model_owners(model_files):
    """Project helper contract: bounded read-only same-UID scan; no adoption/signals."""
    found = []
    population = list(Path('/proc').iterdir())
    require(len(population) < 131072, 'finite proc population')
    for path in population:
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = [v.decode(errors='replace') for v in (path / 'cmdline').read_bytes().split(b'\0') if v]
            matches = [v for v in args if not v.startswith('-') and MODEL.search(v)]
            matches += [v for v in args if str(P / 'serve') in v and v.endswith('.py')]
            matches += [v for v in args if v in model_files or v.startswith('serve.') or v.endswith('.gguf')]
            if matches:
                found.append(dict(pid=int(path.name), matching_arguments=matches))
        except (FileNotFoundError, ProcessLookupError):
            continue
    require(not found, 'unrecognized model/fixture owner: ' + json.dumps(found))
    return dict(same_uid_model_owners=[], scope='bounded same-UID cmdline scan; Main excludes hidden/foreign GPU workloads')


def validate_native(stdout, stderr, tokens, arm, cfg):
    require(not NATIVE_ERROR.search(stderr), 'native/runtime/numerical/retirement error')
    batched, chunk = len(tokens) - 1, cfg['chunk']
    chunks = (batched + chunk - 1) // chunk
    done = [tuple(map(int, row)) for row in re.findall(
        r'^strata trace: layer-major layer (\d+) of (\d+): completed (\d+) chunks$', stderr, re.M)]
    require(done == [(i, 48, chunks) for i in range(48)], 'actual all48-layer/allchunk completion')
    native = arm.startswith('bf16')
    dtype, element = ('bf16', 2) if native else ('fp32', 4)
    banner = [v for v in stderr.splitlines() if v.startswith('strata prefill: layer-major candidate,')]
    require(len(banner) == 1 and f'tokens={batched} chunk={chunk} residual_dtype={dtype} ' in banner[0] and
            f'residual_element_bytes={element} ' in banner[0] and f'residual_gpu_rows={cfg["R_GPU"]} ' in banner[0],
            'actual requested placement/type/chunk; no backoff')
    lines = [v for v in stderr.splitlines() if v.startswith('strata prefill: native16 BF16 HC residual/projection IO')]
    require(len(lines) == int(native), 'actual native16 producer/consumer route')
    route = [v for v in stderr.splitlines() if v.startswith('[native16-route]')]
    expected = ('[native16-route] engine=bf16 backend=vllm tile=auto(T<128:compact;else:wide) '
                'aot=bmg-g21-a0 isolated=1 experimental') if native else '[native16-route] engine=0 backend=own tile=own'
    require(route == [expected], 'actual public backend/tile route')
    require(re.findall(r'^strata generate: prefill (\d+) tokens in (\d+) chunks,', stderr, re.M) ==
            [(str(batched), str(chunks))], 'actual initial-prefill coverage')
    prompts, outputs = re.findall(r'^prompt  :(.*)$', stdout, re.M), re.findall(r'^output  :(.*)$', stdout, re.M)
    require(len(prompts) == len(outputs) == 1 and [int(v) for v in prompts[0].split()] == tokens, 'actual conditioning IDs')
    generated = [int(v) for v in outputs[0].split()]
    require(len(generated) == 1 and 0 <= generated[0] < C.VOCAB and
            re.findall(r'^decode\s+(\d+) tokens in ', stdout, re.M) == ['1'], 'one normal generated token')
    return dict(placement=banner[0], native16_evidence=lines, backend_evidence=route, completed_layers=48,
        chunks_per_layer=chunks, batched_tokens=batched, prompt_tokens=len(tokens), generated_tokens=generated,
        text_equality_checked=False, timing_eligible=False,
        QSA_evidence='Explicit immutable child environment plus build-pinned dispatch source; no additional QSA runtime stamp')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('smoke', 'screen'))
    parser.add_argument('--run-id', required=True, help='fresh vN; no retry/resume/overwrite')
    parser.add_argument('--build-receipt', required=True, type=Path)
    parser.add_argument('--build-sha256', required=True)
    parser.add_argument('--health-receipt', required=True, type=Path, help='Main explicitly supplies latest normally closed sameboot health')
    parser.add_argument('--health-sha256', required=True)
    parser.add_argument('--retention-review-utc', required=True, help='explicit UTC deadline within48h')
    parser.add_argument('--trace-first-smoke', action='store_true', help='optional first synthetic FP32 process only; total text64MiB')
    options = parser.parse_args()
    if options.trace_first_smoke:
        parser.error('Whole-model full API tracing is quarantined; trace a tiny component instead')
    manifest_path = D / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    global C, require, sha
    C = pinned_module('selected_quality_comparator', D / 'compare_screen.py', manifest['package_files']['compare_screen.py']['sha256'])
    require, sha = C.require, C.sha
    require(re.fullmatch(r'v[1-9][0-9]{0,3}', options.run_id) and (not options.trace_first_smoke or options.mode == 'smoke'), 'finite run ID/trace mode')
    require(options.build_sha256 == C.BUILD_SHA, 'exact frozen integrated build receipt')
    for name, pin in manifest['package_files'].items():
        require((D / name).stat().st_size == pin['bytes'] and sha(D / name) == pin['sha256'], 'complete reviewed package source pin')
    manifest_sha = sha(manifest_path)
    review = datetime.datetime.fromisoformat(options.retention_review_utc.replace('Z', '+00:00'))
    require(review.tzinfo is not None and review.utcoffset() == datetime.timedelta(0) and
            datetime.timedelta(0) < review - utc() <= datetime.timedelta(hours=48), 'explicit finite UTC retention review')
    module = pinned_module('selected_quality_direct_owner', OWNER, OWNER_SHA)
    cfg = C.contract(options.mode)
    root = B / f'main-vllm-selected-{options.mode}-{options.run_id}'
    target = A / f'own-vllm-selected-{options.mode}-{options.run_id}.json'
    require(B.resolve() == B and D == P / 'bench/results/2026-10-15-pr6-recovery-audit/own-vllm-selected-quality-v2', 'canonical workspace/state roots')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not root.exists() and not root.is_symlink() and not target.exists() and not target.is_symlink(), 'fresh canonical root/receipt only')
        root.mkdir(mode=0o700)
        start = time.monotonic()
        deadline, work_deadline = start + WALL, start + WALL - RESERVE
        phase_deadline = work_deadline
        record = dict(schema=C.SCHEMA, mode=options.mode, run_id=options.run_id, selected_only=True,
            passed=False, complete=False, active=False, GPU_executed=False, capture_complete=False,
            closure_validated=False, ownership_closed=False, native_byteorder=sys.byteorder,
            cases={a: dict(status='not_started', complete=False) for a in cfg['arms']}, declared_order=cfg['arms'],
            commands=[], started_utc=module.utc(), boot_id=boot_id(), boot_unchanged=False, driver_dump_exists=None,
            trace_first_smoke=options.trace_first_smoke, fresh_faults=[], journal_windows=[], owner_scans=[], driver_snapshots=[], pin_checkpoints=[],
            controller=dict(identity=module.identity(os.getpid()), argv=list(sys.orig_argv),
                source=C.artifact(__file__), python=C.artifact(Path(sys.executable).resolve())),
            manifest=C.artifact(manifest_path), comparator=C.artifact(D / 'compare_screen.py'), legacy_oracle=C.artifact(C.ORACLE),
            owner=dict(path=str(OWNER), sha256=OWNER_SHA), references={},
            budget=dict(native_wall_seconds=NATIVE_WALL, native_CPU_seconds=18000, native_RSS_bytes=88 << 30,
                whole_wall_seconds=WALL, work_seconds=WALL - RESERVE, closure_reserve_seconds=RESERVE,
                text_aggregate_owner_bytes=FILE_CAP, native_text_bytes=NATIVE_TEXT_CAP, native_file_bytes=NATIVE_FILE_CAP, tensor_bytes=256 << 20,
                raw_capture_bytes_strictly_less_than=RAW_CAP, whole_private_disk_bytes=DISK_CAP,
                exception_reason='Four64x248320 FP32 selected captures require254279712 bytes before bounded64MiB text; default64MiB job total cannot hold required evidence.'),
            retention=dict(owner='Main', consumer='compare_screen.py:compare_record and Main selected-position numerical review',
                raw_paths=[], raw_byte_budget=RAW_CAP, review_deadline_utc=review.isoformat(), cleanup_performed=False,
                canonical_reference=None, next_review='At comparison closure/supersession, no later than explicit UTC deadline within48h'),
            scope=C.LIMITATION, original_failure_status_preserved=True, dense_NLL_quarantine_preserved=True,
            **{k: False for k in C.QUALIFICATION_KEYS})

        def persist():
            # Owner callbacks publish real process state, including failed
            # launches; scheduling an arm alone never claims GPU execution.
            record['active'] = owner.active is not None
            if any(c.get('direct_identity') and c.get('label') in
                   [options.mode + '-' + a for a in cfg['arms']] for c in record['commands']):
                record['GPU_executed'] = True
            (root / 'record.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        build = historical = natural = profile = None
        cursor, active_arm, final_error = None, None, None

        def remaining(cap):
            C.deadline_check(phase_deadline)
            return min(cap, max(0.01, phase_deadline - time.monotonic()))

        def frozen(label):
            C.deadline_check(phase_deadline)
            require(boot_id() == record['boot_id'] and sha(manifest_path) == manifest_sha and sha(OWNER) == OWNER_SHA, 'boot/manifest/owner frozen')
            for name, pin in manifest['package_files'].items():
                require(sha(D / name) == pin['sha256'], 'reviewed package source frozen: ' + name)
            for name, digest in manifest['unchanged_sources'].items():
                require(sha(P / name) == digest, 'legacy source/provenance frozen: ' + name)
            for value in record['references'].values():
                C.check_artifact(value, cap=16 << 20)
            if build is not None:
                source_pin_check(build)
                require(sha(allowed_pin(build['binary'])) == C.BINARY_SHA == build['binary_sha256'], 'actual engine6354 binary pin')
            if historical is not None:
                for item in historical['runtime_files']:
                    require(sha(allowed_pin(item['path'])) == item['sha256'], 'runtime library identity')
                for key in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
                    require(all(sha(allowed_pin(p)) == h for p, h in historical[key].items()), 'pack/MTP metadata identity')
                for path, pin in historical['model_identity'].items():
                    st = allowed_pin(path).stat()
                    require(st.st_size == pin['bytes'] and st.st_mtime_ns == pin['mtime_ns'], 'model shard stat identity; not full SHA')
            if natural is not None:
                require(all(sha(p) == h for p, h in natural['source_pins'].items()), 'unchanged tokenizer/template/document pins')
            if profile is not None:
                require(sha(profile) == historical['external_expert_profile_sha256'], 'external expert profile unchanged')
            if 'input_artifact' in record:
                C.check_artifact(record['input_artifact'], cap=1 << 20)
                C.check_artifact(record['input_receipt'], cap=1 << 20)
            record['pin_checkpoints'].append(dict(label=label, checked_utc=module.utc(),
                source_count=len(build['source_pins']) if build is not None else 0,
                build_file_count=len(build['pins']) if build is not None else 0,
                actual_engine_sha256=build['binary_sha256'] if build is not None else None))

        def budget_check():
            files = [p for p in root.rglob('*') if p.is_file()]
            disk = sum(p.stat().st_size for p in files)
            raw = sum(p.stat().st_size for p in files if p.name == 'selected.bin' or p.suffix in ('.stdout', '.stderr', '.meta') or p.name.endswith('.meta.partial'))
            tensor = sum(p.stat().st_size for p in files if p.name == 'selected.bin')
            record.update(private_bytes=disk, raw_capture_bytes=raw, tensor_bytes=tensor)
            require(disk <= DISK_CAP and raw < RAW_CAP and tensor <= 256 << 20, 'finite whole-job disk/raw/tensor limits')

        def utility(label, argv):
            command, out, err = owner.run(label, argv, HOST_ENV, P, wall=remaining(12), cpu=60,
                rss_cap=2 << 30, text_cap=8 << 20, total_cap=DISK_CAP, file_cap=FILE_CAP)
            require(C.L.command_closed(command) and module.closed(command), 'normal utility ownership: ' + label)
            original_absence(module, [command])
            budget_check()
            return out

        def driver_guard(label):
            require(boot_id() == record['boot_id'] and (DEVICE / 'driver').resolve().name == 'xe' and
                    (DEVICE / 'vendor').read_text().strip() == '0x8086', 'current sameboot Intel xe driver')
            dumps = [str(p) for p in Path('/sys/class/devcoredump').glob('devcd*')]
            record.update(boot_unchanged=True, driver_dump_exists=(DEVICE / 'devcoredump').exists() or bool(dumps))
            snapshot = dict(label=label, kernel=os.uname().release, pci_bdf=DEVICE.name, driver='xe',
                pci={n: (DEVICE / n).read_text().strip() for n in ('vendor', 'device', 'revision', 'subsystem_vendor', 'subsystem_device')},
                devcoredumps=dumps)
            record['driver_snapshots'].append(snapshot)
            require(not record['driver_dump_exists'], 'driver dump exists; no removal/reset authorized')
            if len(record['driver_snapshots']) > 1:
                old = record['driver_snapshots'][0]
                require(snapshot['pci'] == old['pci'] and snapshot['kernel'] == old['kernel'], 'driver/PCI/kernel identity stable')

        def journal(label, *, initial=False):
            nonlocal cursor
            driver_guard(label)
            argv = ['/usr/bin/journalctl', '-k', '-b', '--no-pager', '-o', 'json', '--show-cursor']
            argv += ['-n', '0'] if initial else ['--after-cursor', cursor]
            out = utility(label, argv)
            lines = out.read_text().splitlines()
            ends = [v.split(': ', 1)[1] for v in lines if v.startswith('-- cursor: ')]
            rows = [json.loads(v) for v in lines if v and not v.startswith('-- cursor: ') and v != '-- No entries --']
            require(len(ends) == 1 or (not initial and not rows and not ends and cursor), 'explicit kernel cursor boundary')
            end = ends[0] if ends else cursor
            require(all(v.get('_BOOT_ID') == record['boot_id'].replace('-', '') for v in rows), 'journal boot identity')
            faults = [v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))]
            record['journal_windows'].append(dict(label=label, after_cursor=cursor, end_cursor=end, entries=len(rows),
                faults=faults, artifact=C.artifact(out), filter=FAULT.pattern))
            record['fresh_faults'].extend(faults)
            cursor = end
            persist()
            require(not record['fresh_faults'], 'faults accumulate permanently; a later empty window never clears a fault')

        def expired(signum, frame):
            raise TimeoutError(deadline_phase)

        deadline_phase = '1755-second work deadline;45seconds reserved for closure/fresh faults/persistence'
        previous_alarm = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, WALL - RESERVE)
        try:
            require(re.fullmatch('[0-9a-f]{64}', options.health_sha256 or ''), 'Main supplies explicit health SHA')
            for name, path, digest in (
                ('build', options.build_receipt, C.BUILD_SHA),
                ('historical_recipe', A / 'own-vllm-model-smoke-v1.json', C.HISTORICAL_SMOKE_SHA),
                ('original_failure', A / 'own-vllm-quality-screen-v1.json', C.FAILURE_SHA),
                ('health', options.health_receipt, options.health_sha256)):
                path = allowed_pin(path)
                require(sha(path) == digest, 'explicit immutable receipt: ' + name)
                record['references'][name] = C.artifact(path)
            build = C.load_json(record['references']['build']['path'])
            historical = C.load_json(record['references']['historical_recipe']['path'])
            failure = C.load_json(record['references']['original_failure']['path'])
            health = C.load_json(record['references']['health']['path'])
            C.closed_receipt(build)
            C.closed_receipt(historical)
            require(build['schema'] == 'strata-vllm-prefill-build-v1' and build['source_commit'] == historical['source_commit'] == C.SOURCE_COMMIT and
                    build['source_pins'] == historical['source_pins'] and build['binary_sha256'] == historical['binary_sha256'] == C.BINARY_SHA and
                    historical['build_gate']['sha256'] == C.BUILD_SHA, 'strict original build/historical model identity chain')
            require(failure['passed'] is False and failure['complete'] is False and failure['active'] is False and
                    failure['ownership_closed'] is False, 'latest original FAIL/incomplete/formal-unclosed status remains unchanged')
            require(all(c.get('direct_child_reaped') is True and c.get('session_empty') is True and not c.get('survivors') for c in failure['commands']),
                    'original failed sessions must be physically retired without waiving formal FAIL')
            record['original_failure_owner_absence'] = original_absence(module, failure['commands'])
            record['health_scope'] = C.health_contract(health, build, record['boot_id'], failure)
            record['health_owner_absence'] = original_absence(module, health['commands'])
            record['health_selection'] = 'Main explicitly supplies latest normally closed sameboot health; no discovery/substitution or failure waiver. Historical PASS is only recipe/pin provenance.'
            if health.get('schema') == C.SCHEMA:
                record['verified_health_selected_smoke'] = C.compare_record(health, deadline=work_deadline)
            if options.mode == 'screen':
                inp_source = B / 'main-vllm-quality-v1/tokens.txt'
                input_receipt = A / 'own-native16-natural-input-v1.json'
                natural, tokens = C.load_natural(input_receipt, inp_source)
                record['references']['canonical_tokens'] = C.artifact(inp_source)
                record['input_receipt'] = C.artifact(input_receipt)
                data = inp_source.read_bytes()
                ids_sha, input_sources = C.IDS_SHA, natural['source_pins']
                record['input_provenance'] = 'Original R837 A receipt and retained B/main-vllm-quality-v1/tokens.txt exact hashes; no input.json fabricated, retokenization or padding'
            else:
                tokens = list(range(129))
                require(historical['tokens'] == tokens and historical['positions'] == cfg['positions'], 'original synthetic129 recipe')
                data = (','.join(map(str, tokens)) + '\n').encode('ascii')
                record['input_receipt'] = C.artifact(record['references']['historical_recipe']['path'])
                ids_sha = hashlib.sha256(struct.pack('<129I', *tokens)).hexdigest()
                input_sources = {}
            inp = root / 'tokens.txt'
            with inp.open('xb') as stream:
                stream.write(data)
            record['input_artifact'] = C.artifact(inp)
            profile = P / 'data/expert-profile.bin'
            local = root / 'expert-profile.bin'
            args = list(historical['args'])
            for flag, value in (('--max-context', cfg['context']), ('--kv-resident', cfg['resident']),
                                ('--prefill', cfg['chunk']), ('--max-new', 1), ('--tokens-file', inp), ('--expert-profile', local)):
                replace_value(args, flag, value)
            require(args.count('--no-token-graph') == 1 and '--serve' not in args and '--prefill-until' not in args and
                    '--dump-logits' not in args and args[args.index('--spec') + 1] == '4', 'standalone same MTP4 recipe')
            # Start solely from the frozen project recipe, never os.environ.
            # Strip every NLL-prefixed key before constructing any native map.
            base_env = {k: v for k, v in historical['environment'].items() if not k.startswith('STRATA_PREFILL_NLL')}
            record['NLL_environment_removed_from_recipe'] = [k for k in historical['environment'] if k.startswith('STRATA_PREFILL_NLL')]
            require(not any(k.startswith(('STRATA_MEASUREMENT_', 'UNITRACE_', 'PTI_', 'IGC_', 'ZEL_')) or
                            k in ('LD_PRELOAD', 'STRATA_PREFILL_LOGITS', 'STRATA_PREFILL_LOGITS_POSITIONS', 'UR_LOG_TRACING') for k in base_env),
                    'no inherited observer/injection/profiler')
            require(all(base_env.get(k) == v for k, v in QUALIFIED.items()) and
                    Path(base_env['UR_ADAPTERS_FORCE_LOAD']).name == 'libur_adapter_level_zero_v2.so.0', 'safe frozen URv2/queue/copyoffload/bank/scratch recipe')
            base_env['STRATA_PREFILL_LAYER_MAJOR_R_GPU'] = str(cfg['R_GPU'])
            C.no_nll(base_env)
            compilers = [v.split('=', 1)[1] for v in build['configure_argv'] if v.startswith('-DCMAKE_CXX_COMPILER=')]
            require(len(compilers) == 1 and compilers[0] in build['pins'], 'actual compiler build pin')
            binding = dict(source_commit=build['source_commit'], source_pins=build['source_pins'], build_file_pins=build['pins'],
                build_receipt_sha256=C.BUILD_SHA, binary=C.artifact(build['binary']), compiler=C.artifact(compilers[0]),
                runtime_files=historical['runtime_files'], model_identity=historical['model_identity'],
                model_identity_method='model shard bytes+mtime_ns, not full cryptographic shard hashes',
                pack_metadata_sha256=historical['pack_metadata_sha256'], MTP_metadata_sha256=historical['MTP_metadata_sha256'],
                input_receipt_sha256=record['input_receipt']['sha256'], input_file_sha256=record['input_artifact']['sha256'],
                input_ids_sha256=ids_sha, input_source_pins=input_sources, tokens=tokens,
                external_profile=C.artifact(profile), controller=record['controller'], diagnostic_environment_origin=C.HISTORICAL_SMOKE_SHA)
            record.update(binding=binding, binding_sha256=C.canonical_sha(binding), source_commit=build['source_commit'],
                source_pins=build['source_pins'], binary_sha256=build['binary_sha256'], args=args, head_ggml_type=14,
                same_queue_only=True, original_full_regression_passed=False,
                regression_limit='Original separate-queue FAIL and latest dense-NLL FAIL remain immutable; selected-only single-queue diagnostic scope')
            frozen('preflight')
            journal('journal-start', initial=True)
            for arm in cfg['arms']:
                active_arm = arm
                frozen('before-' + arm)
                journal('journal-before-' + arm)
                record['owner_scans'].append(dict(arm=arm, before=no_model_owners(list(historical['model_identity']))))
                arm_root = root / arm
                arm_root.mkdir(mode=0o700)
                shutil.copyfile(profile, local)
                require(sha(local) == historical['external_expert_profile_sha256'], 'identical private profile reset')
                raw = arm_root / 'selected.bin'
                env = dict(base_env, **C.arm_settings(arm), XDG_CACHE_HOME=str(arm_root / 'cache'),
                    STRATA_PREFILL_LOGITS=str(raw), STRATA_PREFILL_LOGITS_POSITIONS=','.join(map(str, cfg['positions'])))
                if options.trace_first_smoke and arm == 'fp32':
                    env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                        UR_LOG_TRACING='level:info;flush:info;output:stderr')
                C.no_nll(env)
                request_path, recipe_path = arm_root / 'request.json', arm_root / 'recipe.json'
                with request_path.open('x') as stream:
                    json.dump(C.request_for(arm, binding, cfg, raw), stream, indent=2, allow_nan=False)
                    stream.write('\n')
                with recipe_path.open('x') as stream:
                    json.dump(dict(schema=C.SCHEMA, mode=options.mode, arm=arm, argv=[build['binary'], *args],
                        environment=env, binding=binding, binding_sha256=record['binding_sha256'], request=C.artifact(request_path),
                        local_profile_initial_sha256=sha(local), raw_consumer=record['retention']), stream, indent=2, allow_nan=False)
                    stream.write('\n')
                case = record['cases'][arm]
                case.update(status='starting', recipe=C.artifact(recipe_path), request=C.artifact(request_path),
                    effective_environment=env, command_index=len(owner.commands), kernel_start_cursor=cursor,
                    local_profile_initial_sha256=sha(local), complete=False)
                record['retention']['raw_paths'].extend([str(raw), str(raw) + '.meta'])
                persist()
                C.no_nll(env)  # last explicit check immediately before exec; no environment hash cycle
                command, out, err = owner.run(options.mode + '-' + arm, [build['binary'], *args], env, P,
                    wall=remaining(NATIVE_WALL), cpu=18000, rss_cap=88 << 30, text_cap=NATIVE_TEXT_CAP,
                    total_cap=DISK_CAP, file_cap=NATIVE_FILE_CAP)
                record['active'] = owner.active is not None
                case.update(status='native_exited', process_outcome={k: command.get(k) for k in
                    ('exit_code', 'observation_complete', 'ownership_status', 'direct_child_reaped', 'session_empty', 'survivors', 'error', 'errors', 'cleanup')})
                for key, path in (('stdout', out), ('stderr', err)):
                    if path.is_file():
                        case[key] = C.artifact(path)
                persist()  # preserve original native outcome even if a fresh fault/parse check fails
                if owner.active is None:
                    if command.get('direct_identity') and command.get('owners'):
                        case['post_exit_PID_start_absence'] = original_absence(module, [command])
                    journal('journal-after-' + arm)
                    case['kernel_end_cursor'] = cursor
                require(C.L.command_closed(command) and module.closed(command), 'native normal zero-exit/reaped/empty closure: ' + arm)
                require(command['direct_identity']['ppid'] == record['controller']['identity']['pid'] and
                        command['direct_identity']['sid'] == command['pid'], 'true direct native ownership')
                record['owner_scans'][-1]['after'] = no_model_owners(list(historical['model_identity']))
                case.update(validate_native(out.read_text(), err.read_text(), tokens, arm, cfg))
                # Callback completion is established only from the published .meta
                # after process exit, never from elapsed progress or dense-NLL rows.
                case.update(raw=C.artifact(raw), metadata=C.artifact(str(raw) + '.meta'), local_profile_final_sha256=sha(local))
                with ExitStack() as stack:
                    selected, metadata = C.open_capture(stack, raw, str(raw) + '.meta', tokens, cfg['positions'], C.VOCAB, 14)
                    checked = []
                    for position in cfg['positions']:
                        C.deadline_check(work_deadline)
                        stats = C.read_row(selected, C.VOCAB, tokens[position + 1])[1]
                        checked.append(dict(position=position, target=tokens[position + 1], **stats))
                    require(not selected.read(1), 'selected exact EOF')
                case.update(independent_rows=checked, parsed_metadata=metadata, complete=True, status='complete',
                    callback_completion_source='Published exact strata-prefill-logits-v1 .meta parsed after native exit; partial metadata does not qualify')
                if arm == cfg['arms'][0]:
                    record['retention']['canonical_reference'] = dict(arm=arm, raw=case['raw'], metadata=case['metadata'],
                        consumer='Current selected-only control for compare_screen.py and Main numerical review; retain through closure/supersession review')
                require(sha(recipe_path) == case['recipe']['sha256'] and sha(request_path) == case['request']['sha256'], 'request/recipe frozen through exit')
                frozen('after-' + arm)
                budget_check()
                persist()
                active_arm = None
            record['capture_complete'] = True
            journal('journal-before-comparison')
            record.update(active=owner.active is not None,
                ownership_closed=owner.active is None and all(module.closed(c) for c in owner.commands), closure_validated=True)
            record['comparison'] = C.compare_record(record, deadline=work_deadline)
            frozen('after-comparison')
            budget_check()
            C.deadline_check(work_deadline)
            record.update(passed=True, complete=True)
        except BaseException as error:
            final_error = error
            record['error'] = repr(error)
            if active_arm is not None:
                record['cases'][active_arm].update(status='failed', failure=repr(error))
        finally:
            phase_deadline = deadline
            deadline_phase = '1800-second whole-screen limit during45second closure reserve'
            signal.setitimer(signal.ITIMER_REAL, max(0.01, deadline - time.monotonic()))

            def closure_step(label, operation):
                # A fault or pin error must not suppress the independent
                # PID/start, capture-identity and other closure observations.
                try:
                    C.deadline_check(deadline)
                    operation()
                except BaseException as error:
                    record.setdefault('closure_errors', []).append(dict(step=label, error=repr(error)))
                    record.update(passed=False, complete=False)

            def final_journal():
                if cursor and owner.active is None:
                    journal('journal-final')
                else:
                    driver_guard('final-without-journal')
                    require(cursor is not None and owner.active is None, 'final fault/ownership observation unavailable')

            def failure_captures():
                if active_arm is not None:
                    case = record['cases'][active_arm]
                    case['failure_capture_artifacts'] = []
                    for name in ('selected.bin', 'selected.bin.meta', 'selected.bin.meta.partial'):
                        path = root / active_arm / name
                        if path.is_file():
                            require(path.stat().st_size <= FILE_CAP, 'failure capture file cap')
                            case['failure_capture_artifacts'].append(C.artifact(path))

            def final_absence():
                record['final_PID_start_absence'] = original_absence(module, owner.commands) if owner.commands else []

            def final_owner_scan():
                record['final_owner_scan'] = no_model_owners(list(historical['model_identity'])) if historical is not None else None

            def normal_closure():
                require(owner.active is None and bool(owner.commands) and
                        all(module.closed(c) and C.L.command_closed(c) for c in owner.commands), 'every original command must close normally')
                require(record['fresh_faults'] == [] and record['boot_unchanged'] is True and record['driver_dump_exists'] is False,
                        'complete final kernel/driver/boot evidence')

            try:
                for label, operation in (('fresh-final-journal', final_journal), ('all-frozen-pins', lambda: frozen('final')),
                        ('failure-capture-identities', failure_captures), ('PID-start-absence', final_absence),
                        ('unrecognized-model-owners', final_owner_scan), ('raw-disk-budgets', budget_check),
                        ('normal-ownership-and-fault-closure', normal_closure)):
                    closure_step(label, operation)
                record['closure_validated'] = not record.get('closure_errors')
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
                record.update(active=owner.active is not None, finished_utc=module.utc(), elapsed_seconds=time.monotonic() - start)
                record['ownership_closed'] = bool(owner.commands) and owner.active is None and all(module.closed(c) for c in owner.commands)
                record['normal_commands_closed'] = bool(owner.commands) and all(C.L.command_closed(c) for c in owner.commands)
                if (final_error is not None or record['elapsed_seconds'] >= WALL or record['fresh_faults'] or
                    not record['boot_unchanged'] or record['driver_dump_exists'] is not False or not record['ownership_closed']):
                    record.update(passed=False, complete=False)
                record['correctness_capture_passed'] = record['passed']
                record['original_process_outcomes'] = {a: dict(status=c['status'], command_index=c.get('command_index'),
                    process_outcome=c.get('process_outcome'), complete=c['complete']) for a, c in record['cases'].items()}
                persist()
                with target.open('x') as stream:
                    json.dump(record, stream, indent=2, allow_nan=False)
                    stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], active=record['active'],
            receipt=str(target), receipt_sha256=sha(target), quality_accepted=False, performance_qualified=False)), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
