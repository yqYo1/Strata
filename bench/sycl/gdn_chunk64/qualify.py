#!/usr/bin/env python3
"""Build or smoke-check the current independent GDN candidate; Main owns execution."""
import fcntl
import hashlib
import json
import re
import stat
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(P / 'tools/sycl'))
sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as owner_tools

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'gdn-chunk64-current'
PACKAGE = P / 'bench/sycl/gdn_chunk64'
ENVIRONMENT = P / 'tools/sycl/measurement_environment.json'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/gdn-chunk64-current.json'
ENGINE = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-score-gemm-entry-engine-build-v2.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_identity():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
    paths = [*PACKAGE.glob('*.cpp'), *PACKAGE.glob('*.hpp'), Path(__file__),
             P / 'tools/sycl/measurement_owner.py', P / 'tools/sycl/measurement_guards.py', ENVIRONMENT]
    # All project headers must be tracked and match this revision, including
    # transitive inputs not mentioned individually by the recipe.
    paths += list((P / 'include').rglob('*.hpp')) + list((P / 'include').rglob('*.h'))
    pins = {}
    for path in paths:
        name = str(path.relative_to(P))
        recorded = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=P)
        guards.require(path.read_bytes() == recorded, 'uncommitted build input: ' + name)
        pins[name] = hashlib.sha256(recorded).hexdigest()
    return commit, pins


def main():
    guards.require(__debug__ and len(sys.argv) == 2 and sys.argv[1] in ('build', 'smoke'),
                   'use qualify.py build or qualify.py smoke')
    action = sys.argv[1]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        commit, pins = source_identity()
        settings = json.loads(ENVIRONMENT.read_text())
        base_env = settings['environment']
        compiler = Path(base_env['LD_LIBRARY_PATH'].split(':')[0]).parent / 'bin/clang++'
        guards.require(digest(compiler) == settings['compiler_sha256'], 'compiler identity')
        engine = json.loads(ENGINE.read_text())
        library = Path(engine['binary']['path']).parent / 'libstrata_sycl_runtime.a'
        binary = ROOT / 'sycl_gdn_chunk64'
        if action == 'build':
            guards.require(not ROOT.exists(), 'retire the previous candidate before replacing it')
            ROOT.mkdir(mode=0o700)
            (ROOT / 'tmp').mkdir()
            (ROOT / 'cache').mkdir()
            record = dict(source_commit=commit, project_input_sha256=pins,
                          runtime_archive=dict(path=str(library), sha256=digest(library),
                                               source_commit=engine['source_commit'],
                                               build_receipt=str(ENGINE.relative_to(P)),
                                               build_receipt_sha256=digest(ENGINE)),
                          compiler=dict(path=str(compiler), sha256=digest(compiler)),
                          model_quality_qualified=False, physical262144_RESTORE_qualified=False,
                          production_selected=False, performance_measured=False)
        else:
            record = json.loads((ROOT / 'record.json').read_text())
            guards.require(record['source_commit'] == commit and record['project_input_sha256'] == pins,
                           'rebuild after source changes')
            guards.require(record.get('build', {}).get('passed') and 'smoke' not in record,
                           'a successful build and a fresh smoke are required')
            guards.require(digest(binary) == record['binary_sha256'] and
                           digest(library) == record['runtime_archive']['sha256'], 'binary/runtime changed')
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            if owner:
                record['current_commands'] = owner.commands
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = owner_tools.Owner(ROOT / 'owned', persist)
        phase = dict(passed=False, complete=False, GPU_executed=False)
        record[action] = phase
        persist()
        try:
            if action == 'build':
                env = dict(PATH=str(compiler.parent) + ':/usr/bin:/bin', HOME='/home/yayoi',
                           LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1',
                           LD_LIBRARY_PATH=str(compiler.parent.parent / 'lib'),
                           TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
                argv = [str(compiler), '-std=c++20', '-O3', '-DNDEBUG', '-DSTRATA_ENABLE_SYCL=1',
                        '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat'),
                        '-fno-fast-math', '-ffp-contract=off', '-fsycl',
                        '-fsycl-id-queries-range=size_t', '-Wall', '-Wextra', '-Werror',
                        str(PACKAGE / 'candidate.cpp'), str(PACKAGE / 'fixture.cpp'),
                        str(library), '-o', str(binary)]
                record['build_command'] = argv
                command, stdout, stderr = owner.run('build', argv, env, P, wall=300, cpu=900,
                                                     rss_cap=8 << 30, file_cap=128 << 20,
                                                     total_cap=1 << 30)
                phase.update(complete=owner_tools.closed(command), exit_code=command.get('exit_code'))
                phase['passed'] = phase['complete'] and phase['exit_code'] == 0
                if phase['passed']:
                    record['binary_sha256'] = digest(binary)
                else:
                    phase['error_excerpt'] = stderr.read_text(errors='replace')[-8192:]
            else:
                env = dict(base_env, XDG_CACHE_HOME=str(ROOT / 'cache'), TMPDIR=str(ROOT / 'tmp'))
                boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                journal_env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8')
                journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0',
                                                   '--show-cursor', '--no-pager'], env=journal_env, text=True)
                cursor = next(line.split(': ', 1)[1] for line in journal.splitlines()
                              if line.startswith('-- cursor: '))

                allowed = {str(Path(path).resolve()): item['sha256']
                           for path, item in settings['runtime_elfs'].items()}
                allowed[str(binary.resolve())] = record['binary_sha256']
                # Check the pinned assets before admitting a new runtime, then
                # check the actual loaded ELF set at each stopped protocol gate.
                for path, expected in allowed.items():
                    guards.require(digest(path) == expected, 'runtime asset changed: ' + path)
                observed = set()

                def mapped_elfs(pid):
                    paths = set()
                    for line in Path('/proc', str(pid), 'maps').read_text().splitlines():
                        fields = line.split(None, 5)
                        if len(fields) != 6 or not fields[5].startswith('/'):
                            continue
                        # Device/anonymous mappings are not files to inspect:
                        # opening a DRM node and reading it waits for events.
                        if fields[5].endswith(' (deleted)'):
                            guards.require('x' not in fields[1], 'deleted executable mapping')
                            continue
                        path = str(Path(fields[5]).resolve())
                        if not stat.S_ISREG(Path(path).stat().st_mode):
                            guards.require('x' not in fields[1], 'non-file executable mapping')
                            continue
                        with Path(path).open('rb') as stream:
                            if stream.read(4) != b'\x7fELF':
                                continue
                        guards.require(path in allowed, 'unqualified mapped ELF: ' + path)
                        paths.add(path)
                    observed.update(paths)
                    phase['observed_runtime_elf_count'] = len(observed)

                def interaction(child, _entry):
                    for marker in ('before-runtime', 'runtime-ready', 'after-cleanup'):
                        while True:
                            line = child.line(seconds=150)
                            if line.strip() == 'GDN_CHUNK64_READY ' + marker:
                                break
                        guards.driver_guard()
                        mapped_elfs(child.proc.pid)
                        if marker == 'runtime-ready':
                            phase['GPU_work_admitted'] = True
                            phase['GPU_executed'] = None  # unknown until a completed device case
                        child.send(b'X')
                    status = child.line(seconds=20)
                    guards.require(status.startswith('fixture_status=PASS '), 'missing final fixture status')
                    child.exit(20)

                record['smoke_command'] = [str(binary), '--smoke']
                phase['launch_attempted'] = True
                command, stdout, stderr = owner.run('smoke', [str(binary), '--smoke'], env, P,
                                                     interaction=interaction, wall=180, cpu=180,
                                                     rss_cap=2 << 30, file_cap=32 << 20,
                                                     text_cap=32 << 20, total_cap=1 << 30)
                text = stdout.read_text(errors='replace')
                phase['case_results'] = [line for line in text.splitlines()
                                         if line.startswith(('case T=', 'fixture_status='))]
                if any(line.startswith('case T=') for line in phase['case_results']):
                    phase['GPU_executed'] = True
                phase['complete'] = owner_tools.closed(command)
                phase['exit_code'] = command.get('exit_code')
                after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor,
                                                 '--no-pager'], env=journal_env, text=True)
                phase['fresh_faults'] = [line for line in after.splitlines() if re.search(
                    r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout)|page fault)', line)]
                phase['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                guards.driver_guard()
                phase['passed'] = (phase['complete'] and phase['exit_code'] == 0 and
                                   len([line for line in phase['case_results'] if line.startswith('case T=')]) == 2
                                   and 'fixture_status=PASS' in text and phase['boot_unchanged']
                                   and not phase['fresh_faults'])
                if not phase['passed']:
                    phase['error_excerpt'] = stderr.read_text(errors='replace')[-8192:]
        except BaseException as error:
            phase['error'] = repr(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(owner_tools.closed(c) for c in owner.commands)
            if not record['ownership_closed']:
                phase['passed'] = False
            persist()
            # Commit only useful results/inputs, not per-process transport/maps.
            compact = {k: v for k, v in record.items() if k not in ('current_commands', 'project_input_sha256')}
            compact['project_input_count'] = len(pins)
            compact['environment_reference'] = dict(path=str(ENVIRONMENT.relative_to(P)), sha256=digest(ENVIRONMENT))
            compact['command'] = owner.commands[-1]['argv'] if owner.commands else []
            RESULT.write_text(json.dumps(compact, indent=2) + '\n')
        print(json.dumps(dict(action=action, passed=phase['passed'], complete=phase['complete'],
                              result=str(RESULT), error=phase.get('error'))))
        return 0 if phase['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
