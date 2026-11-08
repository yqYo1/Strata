#!/usr/bin/env python3
"""Main's finite candidate-input/CLI gate; no model, device or queue work.

Repository LICENSE. Uses the owned measurement and validation helpers; native
source is the independently committed cache-loan child, never a relabelled build.
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
import build_engine as source
import build_upstream_features as build
import engine_once
import measurement_guards as guards
import measurement_owner as ownership
import ur_validation as validation

P, B = source.P, source.B
ROOT = B / 'dflash2-cache-loan-cli-v1'
NATIVE = B / 'dflash2-cache-loan-native-v1'
STATUS = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'dflash2_cache_loan_cli_v1'
EVIDENCE = 'f2d679316bf454158ed99c2cc12276bbfac7ad41'
BUILD_KEY = 'dflash2_cache_loan_native_build'
INPUTS = ['tools/sycl/' + name for name in (
    'check_cache_loan_cli.py', 'build_engine.py', 'build_upstream_features.py',
    'engine_once.py', 'measurement_guards.py', 'measurement_owner.py',
    'ur_validation.py', 'measurement_environment.json')]
ENV_KEYS = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'PYTHONDONTWRITEBYTECODE',
    'LD_LIBRARY_PATH', 'MKLROOT', 'ONEAPI_DEVICE_SELECTOR', 'UR_ADAPTERS_FORCE_LOAD',
    'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
    'STRATA_SYCL_VERIFY_NATIVE_CAPTURE', 'STRATA_SYCL_PREFILL_SAME_QUEUE',
    'SYCL_CACHE_PERSISTENT', 'ZES_ENABLE_SYSMAN', 'ZE_ENABLE_ALT_DRIVERS')
require, digest, write = guards.require, source.digest, build.write


def native():
    row = json.loads((NATIVE / 'record.json').read_text())
    proof = json.loads(source.git(P, 'show', EVIDENCE + ':' + str(
        build.STATUS.relative_to(P))))['current_sync'][BUILD_KEY]
    require(proof['record']['path'] == str(NATIVE / 'record.json') and
        digest(NATIVE / 'record.json') == proof['record']['sha256'] and
        row['stage'] == 'built' and row['passed'] and row['ownership_closed'] and
        proof['passed'] and proof['ownership_closed'] and
        proof['source_commit'] == row['project']['revision'] and
        proof['binaries'] == row['binaries'], 'original committed candidate build differs')
    build.check_inputs(row)
    for value in row['binaries'].values():
        require(digest(value['path']) == value['sha256'] and
            Path(value['path']).stat().st_size == value['bytes'], 'same-build output changed')
    return row, proof


def compact(row):
    out = {k: v for k, v in row.items() if k not in (
        'controller', 'environment', 'runtime_files', 'commands')}
    out['controller'] = {k: row['controller'][k] for k in (
        'revision', 'count', 'aggregate_sha256')}
    out['environment_sha256'] = build.signature(row['environment'])
    out['commands'] = [dict(label=x['label'], exit_code=x.get('exit_code'),
        ownership_closed=ownership.closed(x), errors=x.get('errors', []),
        cleanup=x.get('cleanup', [])) for x in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    status = json.loads(STATUS.read_text())
    status[KEY] = compact(row)
    write(STATUS, status)


def verify(row):
    require(validation.committed(STATUS, (KEY,)) == compact(
        json.loads((ROOT / 'record.json').read_text())), 'commit exact CLI plan first')
    pin = row['controller']
    require(build.signature(pin['source_sha256']) == pin['aggregate_sha256'] and
        {p: digest(P / p) for p in pin['source_sha256']} == pin['source_sha256'] and
        {p: hashlib.sha256(raw).hexdigest() for p, raw in source.blobs(
            P, pin['revision'], pin['source_sha256'])} == pin['source_sha256'],
        'controller inputs changed or unrecoverable')
    n, proof = native()
    require(proof['record'] == row['native_build']['record'] and
        n['project']['revision'] == row['native_build']['source_commit'], 'native identity changed')
    for path, sha in row['audit_inputs'].items():
        require(digest(path) == sha, 'original compilation metadata changed: ' + path)
    require(validation.preload_qualification() == row['native_validation_probe'], 'preload qualification changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'],
        'runtime manifest changed')
    for path, sha in row['runtime_files'].items():
        require(digest(path) == sha, 'runtime input changed: ' + path)
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'driver/boot changed')
    require(not any((ROOT / x).exists() for x in (
        'absent-target', 'absent-target.gguf', 'absent-draft.df2p')), 'sentinel assets must stay absent')
    guards.no_model_owners([])


def cases(binary):
    base = [binary, '--pack', str(ROOT / 'absent-target'), '--native', str(ROOT / 'absent-target.gguf'),
        '--draft', 'dflash2', '--draft-model', str(ROOT / 'absent-draft.df2p'),
        '--draft-fallback', 'none', '--spec', '4', '--prefill', '8192',
        '--no-coupled-draft', '--no-token-graph', '--serve']
    reject = 'DFlash2 cache scratch requires serve, native dense bank, borrowed buffers, spec4 and no token graphs/stages'
    result = []
    for label, extra, remove, env, expected in (
        ('owned', ['--no-prefill-borrow'], [], {}, reject),
        ('not_serve', [], ['--serve'], {}, reject),
        ('token_graph', [], ['--no-token-graph'], {}, reject),
        ('stages', ['--gpu-stages'], [], {}, reject),
        ('not_native_bank', [], [], {'STRATA_PREFILL_NATIVE_DENSE_BANK': '0'}, reject),
        ('width2', ['--spec', '2'], [], {}, reject),
        ('width8', ['--spec', '8'], [], {}, reject),
        ('cap3', ['--draft-cap', '3'], [], {}, reject),
        ('mtp_expert_release', [], [], {'STRATA_PREFILL_RELEASE_MTP_EXPERTS': '1'},
            'DFlash2 requires STRATA_PREFILL_RELEASE_MTP_EXPERTS=0'),
        ('admit_width4', [], [], {}, 'dflash2 pack: cannot open pack'),
        ('admit_width8_cap4', ['--spec', '8', '--draft-cap', '4'], [], {}, 'dflash2 pack: cannot open pack'),
        ('admit_lease0_owned', ['--no-prefill-borrow'], [], {'STRATA_SYCL_CACHE_SCRATCH_LEASE': '0'},
            'dflash2 pack: cannot open pack')):
        result.append(dict(label=label, argv=[x for x in base if x not in remove] + extra,
            environment=env, expected=expected, expected_exit=2))
    return result


def prepare():
    require(not ROOT.exists(), 'fresh finite CLI plan only; no automatic retry')
    n, proof = native()
    reference = B / 'decode-iq3-audit-speed-v2'
    old = json.loads((reference / 'record.json').read_text())
    env = {k: old['environment_template'][k] for k in ENV_KEYS}
    env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        CUDA_MODULE_LOADING='EAGER', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
        UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
        UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr', STRATA_TRACE='1',
        STRATA_SYCL_CACHE_SCRATCH_LEASE='1', STRATA_PREFILL_NATIVE_DENSE_BANK='1',
        STRATA_PREFILL_RELEASE_MTP_EXPERTS='0')
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0' and
        env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0' and env['ZES_ENABLE_SYSMAN'] == '0', 'qualified safety settings')
    row = dict(stage='prepared', passed=False, commands=[], ownership_closed=True,
        controller=source.pin(P, INPUTS), native_build=dict(evidence_commit=EVIDENCE,
            key='current_sync.' + BUILD_KEY, record=proof['record'], source_commit=n['project']['revision'],
            binaries=n['binaries']), native_validation_probe=validation.preload_qualification(),
        environment=env, runtime_manifest=dict(path=str(reference / 'runtime-inputs.json'),
            sha256=digest(reference / 'runtime-inputs.json')),
        runtime_files=json.loads((reference / 'runtime-inputs.json').read_text()),
        audit_inputs={str(path): digest(path) for path in (
            NATIVE / 'build/.ninja_deps', NATIVE / 'build/build.ninja',
            NATIVE / 'build/compile_commands.json', NATIVE / 'owned-2/compile.stdout')},
        dependency_command=['/usr/bin/ninja', '-C', str(NATIVE / 'build'), '-t', 'deps'],
        cases=cases(n['binaries']['strata']['path']), driver=guards.driver_guard(),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        budgets=dict(wall=25, cpu=20, rss_cap=2 << 30, text_cap=2 << 20, file_cap=2 << 20, total_cap=16 << 20),
        criteria='First audit actual Ninja compiler deps/GNU ld -t inputs against original prebuild pins;'
            'only original owned generated outputs may be outside external pins. Then twelve separate immutable'
            'processes in listed order;all normal exit2 with the exact expected rejection and same-PID qualified'
            'pre-main proof. No platform/device/context/queue/allocation/enqueue calls or model stages.'
            'Positive cases only reach the missing draft-pack CPU inspection;not device/model admission.'
            'Stop first failure;no automatic retry,reset,weights,quality,speed or physical-context qualification.',
        retention='Keep compact closure/individual case results and source/recipe identities. Retire closed normal'
            'traces after evidence commit;retain native90f and5fixtures only for imminent same-build device/model gate.'
            ' Preserve f633/552b/e1b until feature-generation selection;review extra builds at first model decision.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir(mode=0o700)
    save(row)
    print('Prepared actual-input audit plus12 pre-GPU CLI cases;commit before run.', flush=True)


def audit(row, output):
    n, _ = native()
    allowed = {}
    for pin in (n['project'], *n['dependencies']):
        allowed.update({str((Path(pin['root']) / p).resolve()): sha for p, sha in pin['source_sha256'].items()})
    allowed.update({str(Path(p).resolve()): sha for p, sha in json.loads(Path(n['external_inputs']['path']).read_text()).items()})
    allowed.update({str(Path(p).resolve()): sha for p, sha in n['toolchain'].items()})
    found = {str((Path(x.strip()) if Path(x.strip()).is_absolute() else NATIVE / 'build' / x.strip()).resolve())
        for x in output.read_text().splitlines() if x.startswith('    ') and x.strip()}
    generated, compiled = {}, {}
    for p in sorted(found):
        sha = digest(p)
        if Path(p).is_relative_to(NATIVE / 'build'):
            generated[p] = sha
        else:
            require(allowed.get(p) == sha, 'unfrozen actual compiler input: ' + p)
            compiled[p] = sha
    require(len(compiled) > 100 and str(Path(n['project']['root']) / 'src/program/generate.cpp') in compiled,
        'actual compiler census incomplete')
    frozen_links, owned_links = {}, set()
    for line in (NATIVE / 'owned-2/compile.stdout').read_text(errors='replace').splitlines():
        v = line.strip(); p = Path(v)
        if not v or any(c.isspace() for c in v) or not (p.is_absolute() or p.suffix in ('.o', '.a') or '.so' in p.name):
            continue
        p = (p if p.is_absolute() else NATIVE / 'build' / p).resolve()
        if any(p.is_relative_to(NATIVE / name) for name in ('build', 'tmp')):
            owned_links.add(str(p)); continue
        require(p.is_file() and allowed.get(str(p)) == digest(p), 'unfrozen actual linker input: ' + str(p))
        frozen_links[str(p)] = digest(p)
    require(len(frozen_links) > 8, 'actual linker trace incomplete')
    database = json.loads((NATIVE / 'build/compile_commands.json').read_text())
    production = [x for x in database if x['file'] == str(Path(n['project']['root']) / 'src/program/generate.cpp')
        and '-DSTRATA_SYCL_FULL_CONTEXT_DIAGNOSTIC' not in x['command']]
    require(len(production) == 1, 'one actual production generator compile command required')
    argv = shlex.split(production[0]['command'])
    require('-DSTRATA_ENABLE_SYCL=1' in argv and '-O3' in argv and '-fsycl' in argv and
        not any(x in argv for x in ('-Ofast', '-ffast-math')), 'production compile mode changed')
    proof = dict(compiled=compiled, generated=generated, frozen_links=frozen_links,
        owned_link_products=sorted(owned_links), production_compile=production[0])
    write(ROOT / 'actual-inputs.json', proof)
    row['actual_inputs'] = dict(path=str(ROOT / 'actual-inputs.json'), sha256=digest(ROOT / 'actual-inputs.json'),
        compiler_files=len(compiled), owned_generated_files=len(generated), external_link_files=len(frozen_links),
        owned_link_products=len(owned_links), uncovered=0,
        scope='Post-build census against original prebuild pins;owned generated outputs are derived from the pinned recipe,not new external inputs. No compilation/GPU work.')


def kernel_log(*args):
    p = subprocess.run(['/usr/bin/journalctl', '-k', '-b', *args, '--no-pager'], capture_output=True, text=True, timeout=15)
    require(p.returncode == 0 and len(p.stdout) < 1 << 20 and 'permission' not in p.stderr.lower(), 'kernel log unavailable/oversized')
    return p.stdout


def run():
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared', 'fresh prepared plan only;no automatic retry')
    verify(row)
    cursor = next(x.split(': ', 1)[1] for x in kernel_log('-n', '0', '--show-cursor').splitlines() if x.startswith('-- cursor: '))
    owner = None
    def persist():
        write(ROOT / 'active.json', dict(active=bool(owner and owner.active is not None),
            commands=owner.commands if owner else []))
    try:
        owner = ownership.Owner(ROOT / 'owned', persist)
        entry, stdout, _ = owner.run('dependencies', row['dependency_command'], row['environment'], P, **row['budgets'])
        require(ownership.closed(entry) and entry['exit_code'] == 0, 'dependency metadata query failed')
        audit(row, stdout)
        row['case_results'] = []
        for case in row['cases']:
            env = engine_once.native_environment(row, dict(row['environment'], **case['environment']))
            entry, stdout, stderr = owner.run(case['label'], case['argv'], env, P, **row['budgets'])
            text = stderr.read_text(errors='replace')
            require(ownership.closed(entry) and entry['exit_code'] == case['expected_exit'], 'CLI closure/exit failed: ' + case['label'])
            proof = validation.preload_evidence(text, entry['pid'])
            require(proof['loader'] == row['native_validation_probe']['loader']['path'] and
                text.splitlines().count(case['expected']) == 1 and stdout.stat().st_size == 0,
                'specific CLI rejection/probe result missing: ' + case['label'])
            require(not re.search(r'ur(?:Enqueue|Kernel|Queue|Device|ContextCreate|USM|PlatformGet\()', text) and
                'UR_RESULT_ERROR_INVALID_NULL_HANDLE' in text and
                not re.search(r'^strata (?:generate: expert cache|serve: prompt|trace:)', text, re.M), 'unexpected GPU/model work')
            row['case_results'].append(dict(label=case['label'], expected=case['expected'], exit_code=2,
                ownership_closed=True, native_pre_main_validation=True, GPU_calls_observed=False))
            print('PASS', case['label'], flush=True)
        verify(row)
        row.update(stage='passed', passed=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row['commands'] = owner.commands if owner else []
        row['ownership_closed'] = owner is None or owner.active is None and all(ownership.closed(x) for x in owner.commands)
        try:
            after = kernel_log('--after-cursor', cursor, '-n', '2048')
            require(len(after.splitlines()) < 2048, 'kernel window incomplete')
            row['fresh_faults'] = [x for x in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
            require(not row['fresh_faults'] and guards.driver_guard() == row['driver'] and
                Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'fresh fault or driver/boot change')
        except BaseException as error:
            row.update(stage='failed', passed=False, fault_error=repr(error))
        if not row['ownership_closed']: row.update(stage='failed', passed=False)
        save(row)
    require(row['passed'], row.get('error', row.get('fault_error', 'CLI gate failed')))
    print('PASS actual-input census and12 CLI cases;no GPU/model qualification.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use prepare or run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prepare if sys.argv[1] == 'prepare' else run)()
