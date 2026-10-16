#!/usr/bin/env python3
# Authored for Strata; repository LICENSE (MIT) applies.
"""R846 SOURCE ONLY / UNRUN. Main reviews and executes after the quality job closes.

Compile the unchanged R833 public fixture once, then run four isolated children.
No CMake/Ninja, source transformation, old-ABI link, retry, adoption or deletion.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
import traceback

sys.dont_write_bytecode = True
P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
A = P / 'bench/results/2026-10-15-pr6-recovery-audit'
D = A / 'own-vllm-public-qualification-v3'
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'main-vllm-public-hook-v3'
BUILD = B / 'main-vllm-prefill-build-v1/build'
TARGET = A / 'own-vllm-public-hook-v3.json'
LLVM = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
COMMIT = '6354f0b4ff3731466795410e11ca9ef785c9d326'
BUILD_SHA = '990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639'
R816_SHA = '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
OWNER_SHA = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
FIXTURE_SHA = '7e1d5a45526eb367e1062d46135c2e99a8fa3a12f4e3db9684de9f526b204716'
SCHEMA = 'strata-r846-public-vllm-qualification-v3'
ORDER = [('compact', 'smoke'), ('wide', 'smoke'), ('compact', 'full'), ('wide', 'full')]
FORMAT = [(a, b) for a in ('fp16', 'bf16') for b in ('fp16', 'bf16')]
TEXT_CAP, DISK_CAP, RECEIPT_CAP = 64 << 20, 512 << 20, 4 << 20
HOST_ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
ERROR = re.compile(r'(?:UR|ZE)_RESULT_ERROR_|PUBLIC_HOOK_FAILED|CHECKED_DRAIN_FAILED|terminate called|what\(\):|retirement failed|Strata SYCL cleanup:|\b(?:fallback|emulat(?:ion|ed))\b', re.I)


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def allowed(path):
    path = Path(path)
    require(path.is_absolute(), 'absolute input path required')
    resolved = path.resolve()
    require(not path.is_relative_to(B / 'prefill-gemm-only-contract-root-v1') and
            not resolved.is_relative_to(B / 'prefill-gemm-only-contract-root-v1'), 'protected path forbidden')
    require('XeStrata' not in str(resolved) or resolved.is_relative_to(LLVM), 'only official v711 install may use historical toolchain path')
    return path


def sha(path):
    with allowed(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    path = allowed(path)
    return dict(sha256=sha(path), bytes=path.stat().st_size, realpath=str(path.resolve()))


def check_pin(path, expected):
    require(pin(path) == {k: expected[k] for k in ('sha256', 'bytes', 'realpath')}, 'immutable file changed: ' + str(path))


def load(path):
    path = allowed(path)
    require(path.is_file() and path.stat().st_size <= 32 << 20, 'bounded JSON input')
    return json.loads(path.read_text())


def imported(name, path, expected):
    check_pin(path, expected)  # Main-time import only, after the sealed manifest check.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utc():
    return datetime.datetime.now(datetime.timezone.utc)


def canonical(path):
    require(path.is_absolute() and path == path.resolve(), 'canonical stable path required: ' + str(path))
    for component in (path, *path.parents):
        require(not component.is_symlink(), 'symlink in controlled path: ' + str(component))


def evidence(stdout, stderr, tile, mode, expected_rejects):
    require(not ERROR.search(stdout + '\n' + stderr), 'native/API/cleanup/fallback error')
    lines = stdout.splitlines()
    begin = f'PUBLIC_HOOK_BEGIN suite={mode} backend=vllm tile={tile} device=0'
    marker = ('PUBLIC_VLLM_NATIVE16_SUITE_PASS; correctness_component_only; '
              'loaded_image_ISA/model_quality/full262144/performance UNRUN')
    require(lines.count(begin) == 1 and lines[0] == begin and lines.count(marker) == 1 and lines[-1] == marker,
            'actual public begin and post-Gemm-destructor final marker required')
    route = f'[native16-route] engine=bf16 backend=vllm tile={tile} aot=bmg-g21-a0 isolated=1 experimental'
    require([v for v in stderr.splitlines() if v.startswith('[native16-route]')] == [route], 'actual experimental GPU-only public route')
    rows = re.findall(r'^PUBLIC_HOOK_COUNTS positive_cases=(\d+) dyadic=(\d+) edge=(\d+) cancellation=(\d+) chain=(\d+) successful_public_calls=(\d+) rejected_calls=(\d+) peak_test_device_bytes=(\d+)$', stdout, re.M)
    require(len(rows) == 1, 'exact fixture census')
    counts = list(map(int, rows[0]))
    require(counts[:7] == ([48, 32, 8, 4, 4, 52, 160] if mode == 'smoke' else [52, 36, 8, 4, 4, 56, 160]) and
            0 < counts[7] < 128 << 20, 'unchanged smoke/full counts and device payload bound')
    dyadic = re.findall(r'^dyadic PASS t=(\d+) n=(\d+) k=(\d+) ld=(\d+) ldy_arg=(\d+) (fp16|bf16)->(fp16|bf16)$', stdout, re.M)
    expected = []
    for inp, out in FORMAT:
        for t in (1, 3, 17, 160):
            expected += [(str(t), '64', '32', '64', '0', inp, out), (str(t), '192', '96', '200', '200', inp, out)]
        if mode == 'full':
            expected.append(('8192', '128', '64', '136', '136', inp, out))
    require(dyadic == expected, 'all formats, T/N tails, pitch and full-span dyadic cases')
    per_format = []
    for inp, out in FORMAT:
        fmt = inp + '->' + out
        edges = re.findall(r'^edge row=(\d+) ' + re.escape(fmt) + r' expected=0x([0-9a-f]+) got=0x([0-9a-f]+) expected_class=(\S+) observed_class=(\S+)$', stdout, re.M)
        require([int(v[0]) for v in edges] == list(range(17)) * 2, 'both strict edge K32/K64 cases per format')
        rejects = re.findall(r'^PUBLIC_REJECT_PASS ' + re.escape(fmt) + r' label=(.*)$', stdout, re.M)
        require(rejects == expected_rejects, 'all 39 labeled reject calls plus counted uninitialized Gemm rejection')
        require(lines.count('PUBLIC_GUARDS_PASS ' + fmt + ' rejected=40 input/output/guards immutable') == 1, '40 rejects per format')
        require(lines.count('PUBLIC_KTILE_PASS ' + fmt + ' K=96 terms_at=0,32,64 expected=+/-0.25') == 1, 'FP32 K-tile accumulation gate')
        require(lines.count('chain PASS ' + inp + '->bf16->' + out + ' direct same-queue device intermediate') == 1, 'device-resident same-queue chain')
        per_format.append(dict(input=inp, output=out, dyadic=9 if mode == 'full' else 8,
            edge=2, strict_edge_rows=edges, cancellation=1, chain=1, rejects=40, labeled_rejects=rejects,
            public_calls=14 if mode == 'full' else 13))
    require(lines.count('edges PASS strict gradual-underflow comparison; hardware contract still requires review') == 8, 'all strict edge completions')
    require(lines.index(begin) < next(i for i, v in enumerate(lines) if v.startswith('PUBLIC_HOOK_COUNTS ')) < len(lines)-1,
            'completion marker follows counts and actual Gemm destruction')
    return dict(positive_cases=counts[0], dyadic=counts[1], edge=counts[2], cancellation=counts[3], chain=counts[4],
        successful_public_calls=counts[5], rejected_calls=counts[6], peak_test_device_bytes=counts[7],
        per_format=per_format, begin=begin, postdestructor_marker=marker, route=route,
        strict_oracle_unchanged=True, loaded_image_ISA_qualified=False)


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--prior-receipt', required=True, type=Path, help='Main chooses latest normally closed PASS GPU receipt after active quality job closes')
    parser.add_argument('--prior-sha256', required=True)
    parser.add_argument('--root', required=True, type=Path, choices=[ROOT])
    parser.add_argument('--retention-review-utc', required=True, help='UTC deadline within 48 hours; earlier review at this qualification boundary')
    args = parser.parse_args()
    canonical(args.prior_receipt)
    require(Path(__file__).absolute() == D / 'controller.py', 'execute canonical reviewed source only')
    for path in (P, A, D, B, BUILD, ROOT, TARGET):
        canonical(path)
    require(re.fullmatch('[0-9a-f]{64}', args.manifest_sha256) and sha(D / 'manifest.json') == args.manifest_sha256, 'explicit reviewed manifest SHA')
    manifest = load(D / 'manifest.json')
    require(manifest['status'] == 'UNRUN' and manifest['source_commit'] == COMMIT, 'source-only package identity')
    for name, expected in manifest['package_files'].items():
        check_pin(D / name, expected)
    require(manifest['input_pins'][str(A / 'own-vllm-engine-build-v2.json')]['sha256'] == BUILD_SHA and
            manifest['r816_manifest']['sha256'] == R816_SHA and manifest['direct_owner']['sha256'] == OWNER_SHA and
            manifest['fixture']['sha256'] == FIXTURE_SHA, 'fixed reviewed dependency identities')
    review = datetime.datetime.fromisoformat(args.retention_review_utc.replace('Z', '+00:00'))
    require(review.tzinfo is not None and review.utcoffset() == datetime.timedelta(0) and
            datetime.timedelta(0) < review - utc() <= datetime.timedelta(hours=48), 'finite UTC retention boundary')
    lockpath = B / 'owned-v0141-measurement.lock'
    require(not lockpath.is_symlink(), 'measurement lock symlink forbidden')
    fd = os.open(lockpath, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as lock:
        require(stat.S_ISREG(os.fstat(lock.fileno()).st_mode), 'regular exclusive measurement lock')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not ROOT.exists() and not ROOT.is_symlink() and not TARGET.exists() and not TARGET.is_symlink(), 'fresh root/result required; no overwrite/resume/retry')
        for path, expected in manifest['input_pins'].items():
            check_pin(path, expected)
        helpers = imported('r846_reviewed_r816_helpers', manifest['r816_helper']['path'], manifest['r816_helper'])
        module = imported('r846_reviewed_direct_owner', manifest['direct_owner']['path'], manifest['direct_owner'])
        ROOT.mkdir(mode=0o700)
        deadline, work_deadline = start + 900, start + 855
        record = dict(schema=SCHEMA, status='FAIL', passed=False, complete=False, active=False, GPU_executed=False,
            boot_id=helpers.boot(), started_utc=module.utc(), source_commit=COMMIT, source_pins=manifest['source_pins'],
            private_root=str(ROOT), manifest_sha256=args.manifest_sha256, controller=pin(__file__),
            invocation=list(sys.orig_argv), controller_identity=module.identity(os.getpid()), python=pin(Path(sys.executable).resolve()),
            build_receipt=dict(path=str(A / 'own-vllm-engine-build-v2.json'), sha256=BUILD_SHA),
            fixture=manifest['fixture'], API_contract_pins=manifest['contract_pins'],
            compiled_existing_build_root=str(BUILD), declared_order=ORDER, commands=[], cases={},
            fresh_faults=[], fault_boundaries=[], dynamic_pins={}, references={},
            limits=manifest['limits'], original_results_preserved=True,
            component_qualified=False, performance_qualified=False, performance_adopted=False,
            model_quality_qualified=False, physical_262144_qualified=False, restore_qualified=False,
            loaded_image_ISA_qualified=False, original_separate_queue_failure_overridden=False,
            scope='R833 public Gemm component correctness only; experimental default OFF. No model/quality/performance/fullcontext/RESTORE adoption.',
            retention=dict(owner='Main R846', consumer='Main complete controller review and exact R833 public Gemm qualification decision',
                raw_root=str(ROOT / 'owned'), byte_budget=TEXT_CAP, private_byte_budget=DISK_CAP,
                review_deadline_utc=review.isoformat(), next_review='At this qualification closure, before any next GPU/build/retention boundary, no later than deadline',
                cleanup_performed=False, raw_trace_exception='First compact smoke needs the full flushed API history until Main verifies compact evidence; no other full-history consumer.'))

        owner = None

        def persist():
            record['active'] = owner is not None and owner.active is not None
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

        # Pinned Owner calls this callback during owned polling and final reap.
        owner = module.Owner(ROOT / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        cursor, workflow_done, gates_done = None, False, False

        def remaining(cap, final=False):
            left = (deadline if final else work_deadline) - time.monotonic()
            require(left > 0, 'whole qualification deadline exhausted')
            return min(cap, left)

        def remember(path):
            record['dynamic_pins'][str(path)] = pin(path)

        def frozen():
            require(helpers.boot() == record['boot_id'], 'boot changed')
            require(sha(D / 'manifest.json') == args.manifest_sha256, 'manifest changed')
            for name, expected in manifest['package_files'].items():
                check_pin(D / name, expected)
            for group in ('input_pins', 'build_file_pins', 'linked_file_pins', 'runtime_library_pins', 'contract_pins'):
                for path, expected in manifest[group].items():
                    check_pin(path, expected)
            for name, digest in manifest['source_pins'].items():
                path = P / name
                require(not Path(name).is_absolute() and path.resolve().is_relative_to(P) and sha(path) == digest, 'compiled432 source drift: ' + name)
            for path, expected in manifest['header_trees'].items():
                allowed(path)
                require(helpers.tree_pin(path) == expected, 'exact qualified header tree changed: ' + path)
            for path, expected in manifest['metadata_pins'].items():
                require(sha(path) == expected, 'pack/MTP metadata changed')
            for path, expected in manifest['model_stat_identity'].items():
                actual = allowed(path).stat()
                require(actual.st_size == expected['bytes'] and actual.st_mtime_ns == expected['mtime_ns'], 'model stat identity changed')
            for path, expected in record['dynamic_pins'].items():
                check_pin(path, expected)
            record['pin_checks'] = record.get('pin_checks', 0) + 1
            record['last_pin_check_utc'] = module.utc()
            persist()

        def budget():
            files = [p for p in ROOT.rglob('*') if p.is_file()]
            record['private_bytes'] = sum(p.stat().st_size for p in files)
            record['raw_text_bytes'] = sum(p.stat().st_size for p in files if p.suffix in ('.stdout', '.stderr'))
            require(record['private_bytes'] <= DISK_CAP and record['raw_text_bytes'] <= TEXT_CAP, 'aggregate private/text budget')

        def guard():
            require(helpers.boot() == record['boot_id'], 'fresh boot guard')
            record['driver_latest'] = helpers.driver_guard()
            record['owner_scan'] = helpers.no_model_owners(list(manifest['model_stat_identity']))
            # Match actual executable identities too, including renamed argv[0].
            blocked = {v['realpath'] for v in manifest['build_products'].values()} | {str(ROOT / 'vllm_public_hook')}
            matches, unreadable = [], []
            population = list(Path('/proc').iterdir())
            require(len(population) < 131072, 'finite actual-executable proc population')
            for proc in population:
                if not proc.name.isdecimal() or int(proc.name) == os.getpid():
                    continue
                try:
                    if proc.stat().st_uid == os.getuid() and str((proc / 'exe').resolve()) in blocked:
                        matches.append(dict(pid=int(proc.name), identity=module.identity(int(proc.name))))
                except PermissionError as exc:
                    # Non-dumpable same-UID user services legitimately deny exe
                    # reads. The independent cmdline scan above still ran.
                    # Record visibility limits; never claim hidden owners absent.
                    unreadable.append(dict(pid=int(proc.name), identity=module.identity(int(proc.name)),
                        observation='same-UID executable visibility denied', error=repr(exc)))
                except (FileNotFoundError, ProcessLookupError):
                    continue
            record['actual_executable_visibility'] = dict(matches=matches, unreadable=unreadable,
                complete=not unreadable, scope='visible same-UID executable scan plus independent bounded cmdline scan; cannot prove hidden foreign GPU use absent')
            require(not matches, 'actual model/fixture executable owner present: ' + json.dumps(matches))

        def owned(label, argv, env, wall, rss, *, final=False):
            require(owner.active is None, 'serialize owned children')
            command, out, err = owner.run(label, list(map(str, argv)), dict(env), ROOT,
                wall=remaining(wall, final), cpu=720 if rss == 8 << 30 else 240,
                rss_cap=rss, text_cap=TEXT_CAP, total_cap=DISK_CAP, file_cap=128 << 20)
            budget()
            require(module.closed(command) and command.get('exit_code') == 0 and command.get('session_anchored'), label + ': owned failure; no retry')
            direct = command['direct_identity']
            require(direct['ppid'] == record['controller_identity']['pid'] and direct['sid'] == command['pid'], 'direct PID/start/session ownership')
            return out, err

        def journal(label, *, final=False):
            nonlocal cursor
            old = cursor
            argv = ['/usr/bin/journalctl', '-k', '-b', '--no-pager', '-o', 'json', '--show-cursor']
            argv += ['--after-cursor', old] if old else ['-n', '0']
            out, err = owned(label, argv, HOST_ENV, 15, 1 << 30, final=final)
            text = out.read_text()
            ends = re.findall(r'^-- cursor: (\S+)\s*$', text, re.M)
            rows = [json.loads(line) for line in text.splitlines() if line and not line.startswith('-- cursor: ') and line != '-- No entries --']
            require(len(ends) == 1 or (old is not None and not rows and not ends), 'fresh journal cursor coverage')
            cursor = ends[0] if ends else old
            faults = [r for r in rows if helpers.FAULT.search(str(r.get('MESSAGE', '')))]
            record['fresh_faults'].extend(faults)  # Never replace or clear an earlier fault.
            record['fault_boundaries'].append(dict(label=label, after_cursor=old, end_cursor=cursor,
                entry_count=len(rows), faults=faults, fault_filter=helpers.FAULT.pattern,
                explicit_empty_window=not rows, cursor_reused=not ends, stdout=pin(out), stderr=pin(err)))
            persist()
            require(all(r.get('_BOOT_ID') == record['boot_id'].replace('-', '') for r in rows), 'kernel journal boot identity')
            require(helpers.boot() == record['boot_id'], 'journal boot changed')
            helpers.driver_guard()
            require(not record['fresh_faults'], 'fresh xe/GPU/GuC fault; no reset/removal/retry')

        def operate(label, argv, env, wall, rss, *, native=False):
            frozen()
            remaining(wall)
            guard()
            journal(label + '-before')
            # Recheck all pins immediately before the actual compiler/native exec.
            frozen()
            result, errors = None, []
            if native:
                record['GPU_executed'] = True
                persist()
            try:
                result = owned(label, argv, env, wall, rss)
            except BaseException as exc:
                errors.append(repr(exc))
            finally:
                if owner.active is None:
                    try:
                        journal(label + '-after', final=True)
                    except BaseException as exc:
                        errors.append('post-operation fault check: ' + repr(exc))
                try:
                    frozen()
                except BaseException as exc:
                    errors.append('post-operation pins: ' + repr(exc))
            require(not errors, json.dumps(dict(operation=label, errors=errors)))
            return result

        def receipt(path, digest, *, gpu=False):
            allowed(path)
            value, proof = helpers.receipt_gate(module, path, digest, record['boot_id'], qualification=True)
            require(value.get('ownership_closed') is True and not value.get('error') and not value.get('closure_errors'), 'PASS normal ownership closure')
            require(value.get('boot_unchanged', True) is True and value.get('normally_closed', True) is True, 'normal prior closure')
            if gpu:
                require(value.get('GPU_executed') is True and value.get('boot_unchanged') is True, 'closed same-boot GPU prior')
            for key in ('correctness_passed', 'correctness_capture_passed', 'numerical_correctness_passed', 'quality_passed'):
                require(key not in value or value[key] is True, 'failed correctness/quality gate cannot qualify: ' + key)
            require(not value.get('quality_failure') and not value.get('quality_failure_override') and not value.get('failure_override'), 'no quality-failure override')
            remember(path)
            return value, proof

        def environment(label, tile=None, traced=False):
            env = dict(manifest['build_environment'] if tile is None else manifest['native_environment'])
            cache, tmp = ROOT / ('cache-' + label), ROOT / ('tmp-' + label)
            cache.mkdir(mode=0o700); tmp.mkdir(mode=0o700)
            env.update(XDG_CACHE_HOME=str(cache), TMPDIR=str(tmp))
            if tile:
                env.update(STRATA_SYCL_PREFILL_NATIVE16='bf16', STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
                           STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE=tile)
                if traced:
                    env.update(manifest['first_smoke_trace_environment'])
            # Explicit copied map; never mutate os.environ or a running child.
            return env

        def timeout(signum, frame):
            raise TimeoutError('whole900s qualification work boundary; final45s reserved for closure')

        old_alarm = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, max(.001, work_deadline - time.monotonic()))
        try:
            frozen()
            build, record['references']['build'] = receipt(A / 'own-vllm-engine-build-v2.json', BUILD_SHA)
            require(build['source_commit'] == COMMIT and build['build_path'] == str(BUILD) and
                    build['source_pins'] == manifest['source_pins'] and len(build['source_pins']) == 432 and
                    build['pins'] == manifest['build_file_pins'] and len(build['pins']) == 314,
                    'current integrated432/source314 build, never old9c ABI')
            smoke, record['references']['model_smoke'] = receipt(A / 'own-vllm-model-smoke-v1.json', manifest['model_smoke_sha256'], gpu=True)
            require(smoke['source_commit'] == COMMIT and smoke['source_pins'] == build['source_pins'] and
                    smoke['binary_sha256'] == build['binary_sha256'] and smoke['build_gate']['sha256'] == BUILD_SHA, 'qualified same integrated model identity')
            prior, record['references']['latest_prior'] = receipt(args.prior_receipt, args.prior_sha256, gpu=True)
            binding = prior.get('binding', {})
            require(prior.get('source_commit') == COMMIT and prior.get('source_pins') == build['source_pins'], 'latest exact compiled source identity')
            binary = binding.get('binary')
            if binary is not None:
                require(binding['source_commit'] == COMMIT and binding['source_pins'] == build['source_pins'] and
                        binding['build_receipt_sha256'] == BUILD_SHA and binary['path'] == build['binary'] and
                        binary['sha256'] == build['binary_sha256'], 'new four-arm binding.binary identity')
                canonical(Path(binary['path']))
                check_pin(binary['path'], dict(binary, realpath=str(Path(binary['path']).resolve())))
            else:
                require(prior.get('binary_sha256') == build['binary_sha256'] and
                        prior.get('binary', build['binary']) == build['binary'], 'latest exact integrated binary')
            require(prior['finished_utc'] >= smoke['finished_utc'], 'Main-selected latest prior predates model qualification')
            record['prior_selection'] = 'Main explicitly supplies latest closed PASS after the active quality job; no discovery, failure waiver or quality adoption'
            record['prior_quality_scope'] = {k: prior[k] for k in ('quality_accepted', 'model_quality_qualified', 'quality_acceptance') if k in prior}
            # Independent read-only text scan; configured experiment ON is separate
            # from the repository's user-facing default, which must remain OFF.
            cmake = (P / 'CMakeLists.txt').read_text()
            defaults = re.findall(r'(?m)^option\(STRATA_SYCL_VLLM_XE2_NATIVE16\s+"[^"]*"\s+(OFF|ON)\)', cmake)
            require(defaults == ['OFF'], 'experimental backend default must remain OFF')
            record['experimental_default_scan'] = dict(path=str(P / 'CMakeLists.txt'), sha256=sha(P / 'CMakeLists.txt'), default='OFF', configured_build='ON')
            require([str(LLVM / 'bin/clang++')] == [manifest['compiler']], 'official exact v711 compiler only')
            for section, expected in manifest['ninja_sections'].items():
                require(expected in (BUILD / 'build.ninja').read_text(), 'exact target build.ninja section changed: ' + section)
            guard()
            journal('journal-start')
            gates_done = True
            obj, exe = ROOT / 'public_fixture.o', ROOT / 'vllm_public_hook'
            replacements = {'@SOURCE@': manifest['fixture']['path'], '@OBJECT@': str(obj), '@BINARY@': str(exe), '@DEPFILE@': str(ROOT / 'public_fixture.o.d')}
            compile_argv = [replacements.get(v, v) for v in manifest['compile_template']]
            link_argv = [replacements.get(v, v) for v in manifest['link_template']]
            record.update(actual_compile_argv=compile_argv, actual_link_argv=link_argv,
                linkage='unchanged final native image+host slice, then current6354 archives; original system library order; generic fsycl link')
            env = environment('compile')
            operate('compile-private-fixture', compile_argv, env, 180, 8 << 30)
            remember(obj)
            if (ROOT / 'public_fixture.o.d').exists():
                remember(ROOT / 'public_fixture.o.d')
            env = environment('link')
            operate('link-private-fixture', link_argv, env, 180, 8 << 30)
            remember(exe)
            record.update(binary=str(exe), binary_sha256=sha(exe), binary_pin=pin(exe))
            for index, (tile, mode) in enumerate(ORDER):
                label = tile + '-' + mode
                env = environment(label, tile, traced=index == 0)
                case = dict(tile=tile, mode=mode, status='FAIL', effective_environment=env, argv=[str(exe), '--device', '0', '--' + mode])
                record['cases'][label] = case
                persist()
                try:
                    out, err = operate(label, case['argv'], env, 120, 4 << 30, native=True)
                    if index == 0:
                        require('urEnqueueKernelLaunch' in err.read_text(), 'first flushed UR launch trace absent')
                        case['flushed_UR_launch_trace_present'] = True
                    case['proof'] = evidence(out.read_text(), err.read_text(), tile, mode, manifest['labeled_rejects'])
                    case['status'] = 'PASS'
                finally:
                    # Preserve original native failure evidence even if a later
                    # journal/pin check is the final owned command.
                    out, err = owner.output / (label + '.stdout'), owner.output / (label + '.stderr')
                    case['owned_command'] = next((c for c in owner.commands if c['label'] == label), None)
                    if out.exists():
                        case['stdout'] = pin(out)
                        if out.stat().st_size < 128 << 10:
                            case['stdout_original'] = out.read_text(errors='replace')
                    if err.exists():
                        case['stderr'] = pin(err)
                        if case['status'] != 'PASS':
                            with err.open('rb') as stream:
                                stream.seek(max(0, err.stat().st_size - (32 << 10)))
                                case['stderr_failure_tail'] = stream.read().decode(errors='replace')
                    persist()
            frozen()
            journal('journal-final')
            guard()
            workflow_done = True
        except BaseException as exc:
            record['error'] = repr(exc)
            record['error_traceback'] = traceback.format_exc(limit=12)
        finally:
            # The Owner alone signals its PID/start-pinned children. No deletion,
            # external-owner adoption, numeric killpg or driver recovery here.
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
            try:
                if cursor and owner.active is None and not workflow_done:
                    try:
                        journal('journal-final-after-failure', final=True)
                    except BaseException as exc:
                        record.setdefault('closure_errors', []).append('final journal: ' + repr(exc))
                try:
                    frozen()
                    record['after_pins_verified'] = True
                    guard()
                    budget()
                except BaseException as exc:
                    record.setdefault('closure_errors', []).append(repr(exc))
                record.update(finished_utc=module.utc(), boot_unchanged=helpers.boot() == record['boot_id'],
                    driver_dump_exists=(DEVICE / 'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*'))))
                record['ownership_closed'] = owner.active is None and all(module.closed(c) for c in owner.commands)
                record['normally_closed'] = record['ownership_closed'] and all(c.get('exit_code') == 0 for c in owner.commands)
                record['complete'] = bool(gates_done and record['ownership_closed'] and record['boot_unchanged'] and
                    record.get('after_pins_verified') and not record.get('closure_errors') and time.monotonic() <= deadline)
                record['passed'] = bool(record['complete'] and workflow_done and record['normally_closed'] and
                    not record['fresh_faults'] and not record['driver_dump_exists'] and not record.get('error') and
                    list(record['cases']) == [t + '-' + m for t, m in ORDER] and all(v['status'] == 'PASS' for v in record['cases'].values()))
                record.update(status='PASS' if record['passed'] else 'FAIL', component_qualified=record['passed'], elapsed_seconds=time.monotonic() - start)
                if owner.commands:
                    label = owner.commands[-1]['label']
                    record['failure_log_location'] = str(owner.output / (label + '.stderr')) if not record['passed'] else None
                persist()
                # Publish original PASS or FAIL once; no previous receipt rewrite.
                payload = json.dumps(record, indent=2, allow_nan=False) + '\n'
                require(len(payload.encode()) <= RECEIPT_CAP, 'compact public receipt cap; private original remains')
                with TARGET.open('x') as stream:
                    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
                print(json.dumps(dict(status=record['status'], complete=record['complete'], active=record['active'],
                    receipt=str(TARGET), receipt_sha256=sha(TARGET), private_record=str(ROOT / 'record.json'))), flush=True)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_alarm)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
