#!/usr/bin/env python3
"""One reviewed 64-byte allocation/fill/drain/free probe, never a failed-run retry."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
BUILD = A / 'own-vllm-engine-build-v2.json'
BUILD_SHA = '990635731be1198706abeb682817fdab247f8f51d31bab971d3d4f4db53cb639'
FAILURE = A / 'own-vllm-quality-screen-v1.json'
FAILURE_SHA = 'bf9d6f0036bff3d6c8f49a384ecc03d849453a41e3ab93b2d90fd4c9833a4f18'
MANIFEST = A / 'own-vllm-xe2-qualification-v1/manifest.json'
MANIFEST_SHA = '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    need(len(sys.argv) == 1, 'fixed reviewed probe; no overrides or retries')
    need(sha(MANIFEST) == MANIFEST_SHA, 'reviewed helper manifest')
    manifest = json.loads(MANIFEST.read_text())
    owner_path = Path(manifest['direct_owner']['path'])
    need(sha(owner_path) == manifest['direct_owner']['sha256'], 'reviewed direct Owner')
    helper_path = MANIFEST.parent / 'main_controller.py'
    need(sha(helper_path) == manifest['package_files']['main_controller.py']['sha256'], 'reviewed helper')
    owner_module = module('postfault_owner', owner_path)
    h = module('postfault_helpers', helper_path)
    root = B / 'main-postfault-allocation-health-v1'
    target = A / 'own-postfault-allocation-health-v1.json'
    need(B.resolve() == B and not root.exists() and not root.is_symlink() and not target.exists(), 'fresh canonical output')
    self_sha = sha(__file__)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        record = dict(schema='strata-postfault-allocation-health-v1', passed=False, complete=False,
            active=False, GPU_executed=False, commands=[], boot_id=h.boot(), boot_unchanged=False,
            fresh_faults=[], driver_dump_exists=False, fault_windows=[],
            original_failure_sha256=FAILURE_SHA, original_failure_status_preserved=True,
            source_sha256=self_sha, started_utc=owner_module.utc(),
            scope='64-byte allocation/fill/drain/free and process teardown only; no model/head/GEMM/NLL retry or source exoneration',
            quarantine='Long dense-NLL diagnostic remains failed and quarantined; this distinct bounded probe cannot qualify that path',
            model_quality_qualified=False, quality_accepted=False, performance_qualified=False,
            physical_262144_qualified=False, restore_qualified=False,
            retention=dict(owner='Main', consumer='postfault independent adapter-health gate',
                raw_byte_budget=8 << 20, review='2026-10-17T03:00:00Z; retire successful API trace after compact operation proof'))
        receipt = root / 'record.json'

        def save():
            receipt.write_text(json.dumps(record, indent=2) + '\n')

        owner = owner_module.Owner(root / 'owned', save)
        record['commands'] = owner.commands
        save()
        cursor = None
        deadline = time.monotonic() + 180
        build = None

        def verify():
            need(time.monotonic() < deadline, 'whole probe180s limit')
            need(h.boot() == record['boot_id'] and sha(__file__) == self_sha, 'boot/controller unchanged')
            need(sha(BUILD) == BUILD_SHA and sha(FAILURE) == FAILURE_SHA and sha(MANIFEST) == MANIFEST_SHA, 'immutable receipts')
            if build is not None:
                need(all(sha(P / p) == value for p, value in build['source_pins'].items()), 'frozen compiled source')
                for path, value in build['pins'].items():
                    h.pinned(path, value)
                for value in build['products'].values():
                    h.pinned(value['path'], value)
            for path, value in manifest['loaded_library_pins'].items():
                h.pinned(path, value)

        host = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')

        def run(label, argv, env, wall):
            verify()
            c, out, err = owner.run(label, argv, env, root, wall=min(wall, deadline-time.monotonic()),
                cpu=60, rss_cap=2 << 30, text_cap=8 << 20, total_cap=16 << 20, file_cap=8 << 20)
            verify()
            need(owner_module.closed(c) and c.get('exit_code') == 0, 'normal owned closure required: ' + label)
            return out, err

        def journal(label, after=None):
            argv = ['/usr/bin/journalctl', '-k', '-b', '--no-pager', '-o', 'json', '--show-cursor']
            argv += ['--after-cursor', after] if after else ['-n', '0']
            out, _ = run(label, argv, host, 20)
            text = out.read_text()
            match = re.search(r'^-- cursor: (\S+)\s*$', text, re.M)
            rows = [json.loads(line) for line in text.splitlines()
                    if line and not line.startswith('-- cursor: ') and line != '-- No entries --']
            need(match or (after and not rows), 'fresh journal cursor required')
            faults = [row for row in rows if h.FAULT.search(str(row.get('MESSAGE', '')))]
            record['fresh_faults'].extend(faults)
            record['fault_windows'].append(dict(label=label, after=after, end=match[1] if match else after,
                entry_count=len(rows), fault_filter=h.FAULT.pattern, fault_entries=faults))
            save()
            need(not record['fresh_faults'], 'new faults accumulate; never clear failed window')
            h.driver_guard()
            return match[1] if match else after

        try:
            verify()
            failure = json.loads(FAILURE.read_text())
            need(failure['passed'] is False and failure['complete'] is False and failure['active'] is False,
                 'preserved failed diagnostic must already be inactive')
            need(failure['boot_id'] == record['boot_id'], 'same-boot failure context')
            record['original_owners_absence'] = []
            for command in failure['commands']:
                need(command.get('direct_child_reaped') is True and command.get('session_empty') is True and not command.get('survivors'),
                     'prior commands must have reaped leaders and empty final sessions; formal FAIL is not waived')
                for old in command.get('owners', []) + [command['direct_identity']]:
                    now = owner_module.identity(old['pid'])
                    absent = now is None or now['start_ticks'] != old['start_ticks']
                    record['original_owners_absence'].append(dict(original=old, current=now, original_absent=absent))
                    need(absent, 'failed-run PID/start still present; no adoption or signal')
            need(sha(BUILD) == BUILD_SHA, 'exact original integrated build')
            build = json.loads(BUILD.read_text())
            need(build['passed'] is True and build['complete'] is True and build['active'] is False, 'qualified build')
            need(build['source_commit'] == '6354f0b4ff3731466795410e11ca9ef785c9d326', 'compiled source revision')
            record.update(source_commit=build['source_commit'], source_pins=build['source_pins'],
                build_receipt_sha256=BUILD_SHA, build_reference=str(BUILD))
            binary = build['products']['sycl_allocation_teardown_test']
            record.update(binary=binary['path'], binary_sha256=binary['sha256'], binary_pin=binary)
            verify()
            record['unrecognized_owner_scan'] = h.no_model_owners(manifest['known_model_files'])
            record['driver_before'] = h.driver_guard()
            cursor = journal('journal-before')
            env = dict(manifest['execution_environment'])
            env.update(STRATA_SYCL_PREFILL_NATIVE16='0', STRATA_SYCL_PREFILL_NATIVE16_BACKEND='own',
                STRATA_SYCL_PREFILL_NATIVE16_QSA='0', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:stderr', XDG_CACHE_HOME=str(root / 'cache'), TMPDIR=str(root / 'tmp'))
            (root / 'cache').mkdir(); (root / 'tmp').mkdir()
            record['GPU_executed'] = True
            out, err = run('allocation-early', [binary['path'], '--early'], env, 45)
            need(out.read_text() == 'allocation_teardown early main complete\n', 'explicit completion marker')
            stderr = err.read_text()
            need(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', stderr), 'API/runtime error')
            operations = ['urUSMDeviceAlloc', 'urEnqueueUSMFill', 'urUSMFree']
            need(all(op in stderr for op in operations), 'actual allocation/fill/free API proof')
            record['operation_proof'] = dict(bytes=64, operations=operations, normal_process_exit=True,
                stdout_sha256=sha(out), stderr_sha256=sha(err), stderr_bytes=err.stat().st_size)
            cursor = journal('journal-after', cursor)
            verify()
            record['passed'] = True
        except BaseException as exc:
            record['error'] = repr(exc)
        finally:
            if cursor and owner.active is None and record['fault_windows'][-1]['label'] != 'journal-after':
                try:
                    journal('journal-after-failure', cursor)
                except BaseException as exc:
                    record['final_fault_check_error'] = repr(exc)
            record.update(active=owner.active is not None, boot_unchanged=h.boot() == record['boot_id'],
                ownership_closed=bool(owner.commands) and owner.active is None and all(owner_module.closed(c) for c in owner.commands),
                driver_dump_exists=(h.DEVICE / 'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*'))),
                finished_utc=owner_module.utc())
            record['private_bytes'] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            record['complete'] = bool(record['ownership_closed'] and record['boot_unchanged'] and
                record.get('operation_proof') and not record.get('final_fault_check_error'))
            record['passed'] &= bool(record['complete'] and not record['fresh_faults'] and not record['driver_dump_exists'] and record['private_bytes'] <= 16 << 20)
            save()
            with target.open('x') as stream:
                json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], active=record['active'],
            receipt=str(target), sha256=sha(target), scope=record['scope'])))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
