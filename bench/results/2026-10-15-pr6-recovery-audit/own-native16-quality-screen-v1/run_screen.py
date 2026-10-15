#!/usr/bin/env python3
"""Main-only serialized diagnostic: two FP32 controls then native BF16.

Original project code under the repository LICENSE. This file never builds,
starts a service, changes os.environ, retries, cleans captures, or adopts PIDs.
"""
import argparse
import datetime
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
A = D.parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
OWNER_SHA = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
NATIVE_ERROR = re.compile(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|retirement failed|Strata SYCL cleanup:|\b(?:NaN|Inf)\b', re.I)
HOST_ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
FILE_CAP = 64 << 20
TEXT_CAP = 64 << 20
TENSOR_CAP = 256 << 20
JOB_CAP = 384 << 20
COMMON_OVERRIDES = dict(STRATA_PREFILL_LAYER_MAJOR_R_GPU='32768')
QUALIFIED = dict(ONEAPI_DEVICE_SELECTOR='level_zero:gpu', OMP_NUM_THREADS='1',
                 NEOReadDebugKeys='1', EnableDirectSubmission='0',
                 UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', SYCL_CACHE_PERSISTENT='0',
                 STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0', STRATA_SYCL_PREFILL_SAME_QUEUE='1',
                 STRATA_PREFILL_LAYER_MAJOR='1', STRATA_PREFILL_RING='8',
                 STRATA_SYCL_PROMPT_DIRECT='1', STRATA_TRACE='1',
                 UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                 UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                 UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                 UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')


def utc():
    return datetime.datetime.now(datetime.timezone.utc)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def load_pinned_module(name, path, digest):
    # Bootstrap is stdlib-only; no mutable unpinned project helper executes.
    import hashlib
    with Path(path).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != digest:
        raise RuntimeError('module pin mismatch: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_value(args, flag, value):
    require(args.count(flag) == 1, 'one explicit CLI flag: ' + flag)
    index = args.index(flag)
    require(index + 1 < len(args), 'CLI flag value: ' + flag)
    args[index + 1] = str(value)


def source_pin_check(build):
    pins = build['source_pins']
    require(pins and 'src/program/generate.cpp' in pins and 'src/prefill/layer_major.inc' in pins,
            'complete build source receipt')
    require(all(n == 'CMakeLists.txt' or n.startswith(('src/', 'include/', 'cmake/', 'tests/'))
                for n in pins), 'compiled/test source-only freeze scope')
    require(all(sha(P / n) == h for n, h in pins.items()), 'compiled/test source changed')
    # Report-only commits may advance. Never substitute HEAD equality for these pins.


def receipt_gate(module, value, *, gpu=False):
    require(value['complete'] is True and value['active'] is False and
            value['passed'] is True, 'prior immutable gate not complete/closed')
    require(not value.get('fresh_faults') and not value.get('driver_dump_exists'), 'prior GPU fault')
    if gpu:
        require(value['boot_unchanged'] is True and value['GPU_executed'] is True, 'prior healthy GPU receipt')
    require(value['commands'], 'prior owner evidence absent')
    for command in value['commands']:
        require(module.closed(command) and command['exit_code'] == 0, 'prior command normal closure')
        for old in command['owners']:
            now = module.identity(old['pid'])
            require(now is None or now['start_ticks'] != old['start_ticks'], 'prior owner still present')


def known_failed_gate(module, value):
    # Exact original v2 separate-queue timeout remains FAIL. It cannot satisfy a
    # normal gate; the independent followup2 receipt supplies the same-queue gate.
    require(value['passed'] is False and value['active'] is False and
            value['fresh_faults'] == [] and value['boot_unchanged'] is True and
            value['driver_dump_exists'] is False, 'original known failure changed')
    for command in value['commands']:
        require(command.get('direct_child_reaped') and command.get('session_empty') and
                not command.get('survivors'), 'failed cohort not retired')
        for old in command['owners']:
            now = module.identity(old['pid'])
            require(now is None or now['start_ticks'] != old['start_ticks'], 'failed owner present')


def runtime_checks(build, smoke, natural, profile, profile_sha):
    source_pin_check(build)
    require(sha(build['binary']) == build['binary_sha256'] and
            sha(build['compiler']) == build['compiler_sha256'], 'binary/toolchain pins')
    for item in smoke['runtime_files']:
        require(sha(item['path']) == item['sha256'], 'runtime library pin: ' + item['path'])
    for field in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
        require(all(sha(n) == h for n, h in smoke[field].items()), 'pack/MTP pin: ' + field)
    for name, identity in smoke['model_identity'].items():
        stat = Path(name).stat()
        require(stat.st_size == identity['bytes'] and stat.st_mtime_ns == identity['mtime_ns'], 'model stat identity')
    require(sha(profile) == profile_sha, 'external expert profile changed')
    require(all(sha(n) == h for n, h in natural['source_pins'].items()), 'input source/template pins')
    require(sha(natural['token_file']) == natural['file_sha256'], 'original conditioning input changed')


def validate_native(stdout, stderr, tokens, arm):
    require(not NATIVE_ERROR.search(stderr), 'native/runtime/retirement/numerical error')
    batched = len(tokens) - 1
    chunks = (batched + 8191) // 8192
    done = [tuple(map(int, row)) for row in re.findall(
        r'^strata trace: layer-major layer (\d+) of (\d+): completed (\d+) chunks$', stderr, re.M)]
    require(done == [(i, 48, chunks) for i in range(48)], 'actual all48-layer/chunk completion')
    banner = [line for line in stderr.splitlines() if line.startswith('strata prefill: layer-major candidate,')]
    dtype = 'bf16' if arm == 'bf16' else 'fp32'
    element = 2 if arm == 'bf16' else 4
    require(len(banner) == 1 and f'tokens={batched} chunk=8192 residual_dtype={dtype} ' in banner[0] and
            f'residual_element_bytes={element} ' in banner[0] and 'residual_gpu_rows=32768 ' in banner[0],
            'actual matched placement/format, no chunk or row backoff')
    native_line = [line for line in stderr.splitlines() if line.startswith('strata prefill: native16 BF16 HC residual/projection IO')]
    require(len(native_line) == (1 if arm == 'bf16' else 0), 'actual native16 producer/consumer path')
    summary = re.findall(r'^strata generate: prefill (\d+) tokens in (\d+) chunks,', stderr, re.M)
    require(summary == [(str(batched), str(chunks))], 'actual prefill token/chunk counts')
    prompt = re.findall(r'^prompt  :(.*)$', stdout, re.M)
    output = re.findall(r'^output  :(.*)$', stdout, re.M)
    require(len(prompt) == len(output) == 1 and [int(t) for t in prompt[0].split()] == tokens,
            'actual CLI conditioning IDs match frozen input')
    generated = [int(t) for t in output[0].split()]
    require(len(generated) == 1 and 0 <= generated[0] < C.VOCAB, 'one finite normal generated token ID')
    require(re.findall(r'^decode\s+(\d+) tokens in ', stdout, re.M) == ['1'], 'normal generated count')
    return dict(placement=banner[0], native16_evidence=native_line, completed_layers=48,
                chunks_per_layer=chunks, batched_tokens=batched, prompt_tokens=len(tokens),
                generated_tokens=generated, text_equality_checked=False,
                timing_eligible=False, normal_retirement='normal owned exit + no cleanup/runtime error')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True, choices=('v2', 'v3'))
    parser.add_argument('--run-id', required=True, help='fresh vN; never resume or overwrite')
    parser.add_argument('--input-sha256', required=True, help='Main pins completed natural-input receipt')
    parser.add_argument('--closed-cohort', nargs=2, action='append', required=True, metavar=('RECEIPT', 'SHA256'))
    parser.add_argument('--retention-review-utc', required=True, help='UTC ISO timestamp within 48h')
    parser.add_argument('--gate-pins', type=Path, help='v3 only: Main-reviewed name-to-SHA JSON')
    parser.add_argument('--gate-pins-sha256', help='v3 only: immutable gate-pin file hash')
    parser.add_argument('--skip-distributions', action='store_true')
    options = parser.parse_args()
    # The manifest is review material, not a claim that review or execution ran.
    manifest_path = D / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    global C, require, sha
    C = load_pinned_module('natural_screen_comparator', D / 'compare_screen.py',
                           manifest['proposal_files']['compare_screen.py']['sha256'])
    require, sha = C.require, C.sha
    for name, pin in manifest['proposal_files'].items():
        require((D / name).stat().st_size == pin['bytes'] and sha(D / name) == pin['sha256'], 'reviewed proposal file: ' + name)
    require(re.fullmatch(r'v[1-9][0-9]{0,3}', options.run_id), 'finite fresh versioned run ID')
    review = datetime.datetime.fromisoformat(options.retention_review_utc.replace('Z', '+00:00'))
    require(review.tzinfo is not None and review.utcoffset() == datetime.timedelta(0) and
            datetime.timedelta(0) < review - utc() <= datetime.timedelta(hours=48), 'finite UTC raw-retention review deadline')
    module = load_pinned_module('screen_direct_owner', OWNER, OWNER_SHA)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = B / f'main-own-native16-quality-screen-root-{options.version}-{options.run_id}'
        target = A / f'own-native16-quality-screen-{options.version}-{options.run_id}.json'
        require(not root.exists() and not target.exists(), 'fresh root and immutable receipt required')
        root.mkdir(mode=0o700)
        record = dict(schema=C.SCHEMA, passed=False, complete=False, active=False, GPU_executed=False,
                      capture_complete=False, closure_validated=False, cases={}, commands=[],
                      declared_order=list(C.ARMS), build_version=options.version, run_id=options.run_id,
                      started_utc=module.utc(), boot_id=boot_id(), native_byteorder=sys.byteorder,
                      controller=dict(identity=module.identity(os.getpid()), argv=list(sys.orig_argv),
                                      source=C.artifact(__file__), python=C.artifact(Path(sys.executable).resolve())),
                      host=dict(kernel_release=os.uname().release, machine=os.uname().machine,
                                pci_bdf=DEVICE.name,
                                pci_identifiers={n: (DEVICE / n).read_text().strip()
                                                 for n in ('vendor', 'device', 'revision', 'subsystem_vendor', 'subsystem_device')}),
                      manifest=C.artifact(manifest_path), comparator=C.artifact(D / 'compare_screen.py'),
                      owner=dict(path=str(OWNER), sha256=OWNER_SHA), references={}, closed_cohorts=[],
                      budget=dict(per_file_bytes=FILE_CAP, job_text_bytes=TEXT_CAP,
                                  tensor_bytes=TENSOR_CAP, whole_job_bytes=JOB_CAP,
                                  whole_job_reason='Three 63569928-byte sparse captures plus 64MiB text, local profile/cache/metadata; 256MiB total is too tight for this bounded job.',
                                  native_wall_seconds=1800, native_cpu_seconds=21600, native_RSS_bytes=88 << 30),
                      retention=dict(owner='Main', consumer='compare_screen.py:compare_record and Main numerical review',
                                     raw_paths=[], review_deadline_utc=review.isoformat(),
                                     next_review='At comparison closure or supersession, no later than review_deadline_utc; Main serializes retention review under the measurement lock.',
                                     cleanup_performed=False),
                      model_quality_qualified=False, performance_qualified=False,
                      physical_262144_qualified=False, restore_qualified=False,
                      quality_accepted=False, scope=C.LIMITATION)

        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        cursor = None
        final_error = None
        build = smoke = natural = profile = None

        def utility(label, argv):
            command, out, err = owner.run(label, argv, HOST_ENV, P, wall=60, cpu=120,
                                          rss_cap=2 << 30, text_cap=8 << 20,
                                          total_cap=JOB_CAP, file_cap=FILE_CAP)
            require(module.closed(command) and command['exit_code'] == 0, label + ': ' + err.read_text()[-2000:])
            return out

        def journal(label):
            out = utility(label, ['/usr/bin/journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
            rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
            require(all(v.get('_BOOT_ID') == record['boot_id'].replace('-', '') for v in rows), 'journal boot identity')
            faults = [v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))]
            record.setdefault('journal_windows', []).append(dict(label=label, after_cursor=cursor,
                entries=len(rows), artifact=C.artifact(out), faults=faults, filter=FAULT.pattern))
            record.update(fresh_faults=faults, boot_unchanged=boot_id() == record['boot_id'],
                          driver_dump_exists=(DEVICE / 'devcoredump').exists())
            persist()
            require(not faults and record['boot_unchanged'] and not record['driver_dump_exists'], 'fresh GPU/boot fault guard')

        def reference(name, pins):
            require(name in pins and re.fullmatch(r'[A-Za-z0-9_.-]+\.json', name), 'pinned named gate')
            path = A / name
            require(sha(path) == pins[name], 'immutable gate hash: ' + name)
            record['references'][name] = C.artifact(path)
            return C.load_json(path)

        try:
            pins = manifest['base_receipt_pins']
            if options.version == 'v3':
                require(options.gate_pins is not None and options.gate_pins_sha256 is not None and
                        sha(options.gate_pins) == options.gate_pins_sha256, 'v3 requires Main-reviewed explicit gate-pin file')
                pins = C.load_json(options.gate_pins)
                record['gate_pins'] = C.artifact(options.gate_pins)
            else:
                require(options.gate_pins is None and options.gate_pins_sha256 is None, 'v2 pins fixed in reviewed proposal')
            build = reference(f'own-native16-prefill-build-{options.version}.json', pins)
            gemm = reference(f'own-native16-gemm-correctness-{options.version}.json', pins)
            smoke = reference(f'own-native16-model-smoke-{options.version}.json', pins)
            original = reference(f'own-native16-regression-correctness-{options.version}.json', pins)
            receipt_gate(module, build)
            receipt_gate(module, gemm, gpu=True)
            receipt_gate(module, smoke, gpu=True)
            gemm_result = json.loads(gemm['checks']['gemm16-first']['stdout'])
            require(gemm_result['status'] == 'PASS' and gemm_result['numerical_cases'] == 68, 'all68 typed GEMM gate')
            if original['passed']:
                receipt_gate(module, original, gpu=True)
                record['regression_gate'] = f'own-native16-regression-correctness-{options.version}.json'
            else:
                require(options.version == 'v2', 'only pinned known v2 separate-queue failure can use narrow gate')
                known_failed_gate(module, original)
                narrow_name = 'own-native16-regression-followup2-correctness-v2.json'
                receipt_gate(module, reference(narrow_name, pins), gpu=True)
                record['regression_gate'] = narrow_name
            record['original_full_regression_passed'] = original['passed']
            record['regression_limit'] = 'Same-queue only; original separate-queue failure is immutable and remains unresolved.'
            require(gemm['source_commit'] == smoke['source_commit'] == build['source_commit'] and
                    gemm['source_pins'] == smoke['source_pins'] == build['source_pins'] and
                    smoke['binary_sha256'] == build['binary_sha256'], 'same-source/binary gate chain')
            require(set(smoke['cases']) == {'fp32', 'bf16'}, 'qualified same-binary model mechanics')
            for case in smoke['cases'].values():
                losses = case['all128_target_NLLs']
                require(len(losses) == 128 and all(math.isfinite(v['loss']) and v['loss'] >= 0 for v in losses),
                        'prior model mechanics all128 finite NLL gate')
            for name, digest in options.closed_cohort:
                require(sha(name) == digest, 'explicit closed-cohort receipt hash')
                prior = C.load_json(name)
                receipt_gate(module, prior, gpu=True)
                record['closed_cohorts'].append(C.artifact(name))
            input_path = A / 'own-native16-natural-input-v1.json'
            natural, tokens = C.load_input(input_path, options.input_sha256)
            preparer = A / 'prepare_owned_native16_natural_input_v1.py'
            require(sha(preparer) == manifest['natural_preparer_sha256'] == natural['recipe_sha256'], 'frozen natural-input preparer')
            record['input_receipt'] = C.artifact(input_path)
            inp = root / 'tokens.txt'
            with inp.open('xb') as stream:
                stream.write(Path(natural['token_file']).read_bytes())
            require(sha(inp) == natural['file_sha256'], 'private exact input copy')
            profile = P / 'data/expert-profile.bin'
            profile_sha = smoke['external_expert_profile_sha256']
            runtime_checks(build, smoke, natural, profile, profile_sha)
            record['external_profile'] = C.artifact(profile)
            local = root / 'expert-profile.bin'
            base_args = list(smoke['args'])
            for flag, value in (('--max-context', 262144), ('--kv-resident', 32768), ('--prefill', 8192),
                                ('--max-new', 1), ('--tokens-file', inp), ('--expert-profile', local)):
                replace_value(base_args, flag, value)
            require(base_args.count('--no-token-graph') == 1 and '--serve' not in base_args and
                    '--prefill-until' not in base_args and '--dump-logits' not in base_args,
                    'qualified standalone sparse CLI only')
            base_env = dict(smoke['environment'])
            require(not any(k.startswith('STRATA_PREFILL_NLL') or k.startswith(('UNITRACE_', 'PTI_', 'IGC_', 'ZEL_')) or
                            k in ('LD_PRELOAD', 'STRATA_PREFILL_LOGITS', 'STRATA_PREFILL_LOGITS_POSITIONS')
                            for k in base_env), 'qualified environment cannot inherit observers/injection/profilers')
            for name, value in QUALIFIED.items():
                require(base_env.get(name) == value, 'qualified frozen native environment: ' + name)
            require(not any(k.startswith('STRATA_MEASUREMENT_') for k in base_env), 'stale service measurement output')
            base_env.update(COMMON_OVERRIDES)
            record['environment_changes_from_smoke'] = dict(COMMON_OVERRIDES,
                arm='STRATA_SYCL_PREFILL_NATIVE16 + STRATA_PREFILL_LAYER_MAJOR_R_DTYPE',
                isolation='fresh per-arm XDG_CACHE_HOME + sparse observer path/positions')
            # The model shards retain the established stat identity (not full
            # shard SHA). Metadata/runtime/compiler/binary/input are SHA-pinned.
            binding = dict(source_commit=build['source_commit'], source_pins=build['source_pins'],
                           build_receipt_sha256=record['references'][f'own-native16-prefill-build-{options.version}.json']['sha256'],
                           binary=C.artifact(build['binary']), compiler=C.artifact(build['compiler']),
                           runtime_files=smoke['runtime_files'], model_identity=smoke['model_identity'],
                           model_identity_method='shard bytes + mtime_ns; not full-shard cryptographic hashes',
                           pack_metadata_sha256=smoke['pack_metadata_sha256'], MTP_metadata_sha256=smoke['MTP_metadata_sha256'],
                           input_receipt_sha256=options.input_sha256, input_ids_sha256=natural['ids_le_u32_sha256'],
                           input_file_sha256=natural['file_sha256'], input_source_pins=natural['source_pins'],
                           external_profile=record['external_profile'], controller=record['controller'],
                           diagnostic_environment_origin=record['references'][f'own-native16-model-smoke-{options.version}.json']['sha256'])
            record.update(binding=binding, binding_sha256=C.canonical_sha(binding), args=base_args,
                          source_commit=build['source_commit'], source_pins=build['source_pins'],
                          head_ggml_type=14, same_queue_only=True)
            require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'GPU preflight')
            before = utility('journal-before', ['/usr/bin/journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            match = re.search(r'-- cursor: (\S+)', before.read_text())
            require(match, 'fresh journal cursor')
            cursor = match[1]
            record['journal_start_cursor'] = cursor
            for arm in C.ARMS:
                require(boot_id() == record['boot_id'], 'boot changed before arm')
                runtime_checks(build, smoke, natural, profile, profile_sha)
                arm_root = root / arm
                arm_root.mkdir(mode=0o700)
                shutil.copyfile(profile, local)
                require(sha(local) == profile_sha, 'reset private profile input')
                raw = arm_root / 'selected.bin'
                env = dict(base_env, XDG_CACHE_HOME=str(arm_root / 'cache'),
                           STRATA_SYCL_PREFILL_NATIVE16='bf16' if arm == 'bf16' else '0',
                           STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='bf16' if arm == 'bf16' else 'fp32',
                           STRATA_PREFILL_LOGITS=str(raw), STRATA_PREFILL_LOGITS_POSITIONS=','.join(map(str, C.POSITIONS)))
                argv = [build['binary'], *base_args]
                recipe_path = arm_root / 'recipe.json'
                with recipe_path.open('x') as stream:
                    json.dump(dict(schema=C.SCHEMA, arm=arm, argv=argv, environment=env,
                                   binding=binding, binding_sha256=record['binding_sha256'],
                                   local_profile_initial_sha256=sha(local), raw_consumer=record['retention']),
                              stream, indent=2, allow_nan=False)
                    stream.write('\n')
                case = dict(recipe=C.artifact(recipe_path), command_index=len(owner.commands), complete=False,
                            planned_raw_path=str(raw), local_profile_initial_sha256=sha(local))
                record['cases'][arm] = case
                record['retention']['raw_paths'].append(str(raw))
                record.update(active=True, GPU_executed=True)
                persist()
                command, out, err = owner.run('screen-' + arm, argv, env, P, wall=1800, cpu=21600,
                                              rss_cap=88 << 30, text_cap=TEXT_CAP,
                                              total_cap=JOB_CAP, file_cap=FILE_CAP)
                record['active'] = owner.active is not None
                require(module.closed(command) and command['exit_code'] == 0, 'native normal closure: ' + arm)
                require(command['direct_identity']['ppid'] == record['controller']['identity']['pid'] and
                        command['direct_identity']['sid'] == command['pid'], 'true direct controller ownership')
                evidence = validate_native(out.read_text(), err.read_text(), tokens, arm)
                case.update(evidence, stdout=C.artifact(out), stderr=C.artifact(err),
                            raw=C.artifact(raw), metadata=C.artifact(str(raw) + '.meta'),
                            local_profile_final_sha256=sha(local))
                require(case['raw']['bytes'] == C.RAW_BYTES and case['raw']['bytes'] < FILE_CAP, 'exact 64-row tensor budget')
                require(sum(v.get('raw', {}).get('bytes', 0) for v in record['cases'].values()) <= TENSOR_CAP, 'cohort tensor cap')
                require(sha(inp) == natural['file_sha256'], 'private conditioning input unchanged')
                runtime_checks(build, smoke, natural, profile, profile_sha)
                # Parse every vocabulary row after each arm, before launching the
                # next one. Reopening during comparison is an independent pass.
                from contextlib import ExitStack
                with ExitStack() as stack:
                    stream, metadata = C.open_capture(stack, raw, str(raw) + '.meta', tokens, C.POSITIONS,
                                                      C.VOCAB, record['head_ggml_type'])
                    case['independent_rows'] = [dict(position=p, target=tokens[p + 1], **C.read_row(stream, C.VOCAB, tokens[p + 1])[1])
                                                for p in C.POSITIONS]
                    require(not stream.read(1), 'exact sparse EOF')
                case.update(complete=True, parsed_metadata=metadata)
                persist()
                journal('journal-after-' + arm)
            record['capture_complete'] = True
        except BaseException as error:
            final_error = error
            record['error'] = repr(error)
        finally:
            try:
                if cursor and owner.active is None:
                    journal('journal-final')
                if build is not None and smoke is not None and natural is not None and profile is not None:
                    runtime_checks(build, smoke, natural, profile, profile_sha)
                record.update(active=owner.active is not None, finished_utc=module.utc())
                closed = (not record['active'] and all(module.closed(c) and c['exit_code'] == 0 for c in owner.commands)
                          and record.get('fresh_faults') == [] and record.get('boot_unchanged') is True
                          and record.get('driver_dump_exists') is False)
                require(closed, 'complete normal ownership/kernel/boot closure required')
                record['closure_validated'] = True
                if final_error is None and record['capture_complete']:
                    record['comparison'] = C.compare_record(record, distributions=not options.skip_distributions)
                    record.update(passed=True, complete=True)
            except BaseException as error:
                record.setdefault('closure_errors', []).append(repr(error))
                record.update(passed=False, complete=False)
            finally:
                record.update(active=owner.active is not None, finished_utc=module.utc())
                if final_error is not None:
                    record.update(passed=False, complete=False)
                persist()
                with target.open('x') as stream:
                    json.dump(record, stream, indent=2, allow_nan=False)
                    stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], active=record['active'],
                              receipt=str(target), receipt_sha256=sha(target), quality_accepted=False)), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
