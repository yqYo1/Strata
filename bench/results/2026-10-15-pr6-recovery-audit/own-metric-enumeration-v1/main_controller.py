#!/usr/bin/env python3
"""Main-only bounded build OR enumeration; never prepares or refreshes its seal.
Independently authored for Strata, repository LICENSE. SOURCE_ONLY_UNRUN.
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
import sys
import time

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parents[3]
STATE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
BASE = D.parent / 'own-vllm-resource-timeline-v2/manifest.json'
BASE_SHA = 'fe0bc066c00a8aa9693678ef924060ed6430ad237c77dc1f1b03787a59b6855f'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
OWNER_SHA = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
LLVM = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
HOST_ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
RAW, PRIVATE = 32 << 20, 128 << 20


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def pin(p, v):
    p = Path(p).resolve(strict=True)
    require('prefill-gemm-only-contract-root-v1' not in p.parts, 'protected path')
    require('XeStrata' not in p.parts or p.is_relative_to(LLVM), 'excluded implementation')
    require(str(p) == v['realpath'] and p.stat().st_size == v['bytes'] and sha(p) == v['sha256'], 'original pin drift: ' + str(p))


def module(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--manifest-sha256', required=True)
    ap.add_argument('--reviewed-package-sha256', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--mode', choices=('build', 'enumerate'), required=True)
    args = ap.parse_args()
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{7,79}', args.run_id), 'fresh bounded run ID')
    require(sha(args.manifest) == args.manifest_sha256, 'external Main manifest SHA')
    require(sha(D / 'SHA256SUMS') == args.reviewed_package_sha256, 'external complete Main source review SHA')
    names = set()
    for line in (D / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(re.fullmatch('[a-f0-9]{64}', digest) and not Path(name).is_absolute() and '..' not in Path(name).parts and name not in names, 'strict package row')
        require(sha(D / name) == digest and not (D / name).is_symlink(), 'reviewed package bytes')
        names.add(name)
    require(names == {str(p.relative_to(D)) for p in D.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}, 'complete snapshot')
    manifest = json.loads(Path(args.manifest).read_text())
    require(manifest['status'] == 'MAIN_REVIEWED_SEALED' and manifest['reviewed_package_sha256'] == args.reviewed_package_sha256 and
            manifest['run_id'] == args.run_id and manifest['mode'] == args.mode, 'explicit seal/mode/fresh identity')
    require(sha(BASE) == BASE_SHA and sha(OWNER) == OWNER_SHA, 'original base and qualified owner')
    base = json.loads(BASE.read_text())
    candidate = json.loads((D / 'manifest-candidate.json').read_text())
    require(manifest['original_file_pins'] == candidate['original_file_pins'] and
            manifest['original_elf_pins'] == base['elf_pins'] and
            manifest['input_pins'] == candidate['input_pins'], 'no original inventory rewrite or live re-enrollment')
    require(manifest['additional_dependency_review']['approved'] is True and
            manifest['additional_dependency_review']['report_sha256'] and
            manifest['additional_dependency_review']['report_path'], 'separate Main dependency closure review')
    require(sha(manifest['additional_dependency_review']['report_path']) == manifest['additional_dependency_review']['report_sha256'], 'dependency review report seal')
    extra = manifest['additional_dependency_review']['pins']
    require(not set(extra).intersection(base['elf_pins']), 'additional dependencies cannot replace originals')
    frozen_pins = dict(manifest['original_file_pins'])
    for p, v in {**manifest['input_pins'], **extra}.items():
        require(p not in frozen_pins or frozen_pins[p] == v, 'no conflicting original pins')
        frozen_pins[p] = v
    owner_mod = module(OWNER, 'metric_owned_child')
    root = STATE / ('main-metric-' + args.run_id)
    require(not root.exists(), 'fresh private output; no overwrite')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(boot == manifest['boot_id'], 'explicit sameboot Main selection')
    started = time.monotonic()
    deadline = started + (240 if args.mode == 'build' else 120)
    with (STATE / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.mkdir(root, 0o700)
        record = dict(status='INCOMPLETE', passed=False, complete=False, active=False, commands=[],
            mode=args.mode, run_id=args.run_id, boot_id=boot, started_utc=utc(),
            manifest_sha256=args.manifest_sha256, reviewed_package_sha256=args.reviewed_package_sha256,
            execution_kind='HOST_LEVEL_ZERO_ENUMERATION' if args.mode == 'enumerate' else 'HOST_COMPILATION',
            collection_qualified=False, no_gpu_commands_authored=True, ownership_closed=False,
            budgets=dict(raw_bytes=RAW, private_bytes=PRIVATE, whole_seconds=deadline-started,
                child_cpu_seconds=90 if args.mode == 'build' else 30, child_rss_bytes=2 << 30),
            retention=dict(owner='Main', consumer='one bounded actual metric-enumeration qualification',
                next_review='After Main verifies compact API/identity/fault evidence; no automatic cleanup'))
        owner = None
        settling = False
        closing = False

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            files = [p for p in root.rglob('*') if p.is_file()]
            exceeded = (sum(p.stat().st_size for p in files) > PRIVATE or
                sum(p.stat().st_size for p in files if p.suffix in ('.stdout', '.stderr')) > RAW)
            if exceeded:
                record.update(budget_error='private/raw byte budget', status='INCOMPLETE', passed=False, complete=False)
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
            if exceeded and not settling:
                raise RuntimeError('private/raw byte budget')

        owner = owner_mod.Owner(root / 'owned', persist)
        record['commands'] = owner.commands

        def freeze():
            require(time.monotonic() < deadline - 40, 'whole deadline with closure reserve')
            require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot, 'boot drift')
            require(sha(args.manifest) == args.manifest_sha256 and sha(D / 'SHA256SUMS') == args.reviewed_package_sha256, 'seal drift')
            for line in (D / 'SHA256SUMS').read_text().splitlines():
                digest, name = line.split('  ', 1)
                require(sha(D / name) == digest, 'reviewed source drift')
            for p, v in frozen_pins.items():
                pin(p, v)
                require(time.monotonic() < deadline - 40, 'pinning work deadline')

        def driver():
            require((DEVICE / 'driver').resolve().name == 'xe', 'exact driver')
            require((DEVICE / 'vendor').read_text().strip() == '0x8086' and
                    (DEVICE / 'device').read_text().strip() == '0xe20c', 'exact PCI target')
            require(not (DEVICE / 'devcoredump').exists() and not list(Path('/sys/class/devcoredump').glob('devcd*')), 'dump exists; no reset/deletion')

        def run(label, argv, env, interaction=None, wall=12, cpu=30):
            end = deadline if closing else deadline - 40
            require(time.monotonic() < end, 'work/closure deadline')
            c, out, err = owner.run(label, argv, env, root, interaction=interaction,
                wall=min(wall, end-time.monotonic()), cpu=cpu, rss_cap=2 << 30,
                text_cap=RAW, total_cap=PRIVATE, file_cap=RAW)
            require(owner_mod.closed(c), 'owned child closure')
            for old in c['owners'] + [c['direct_identity']]:
                now = owner_mod.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'owned PID/start survives')
            return c, out, err

        def journal(label, argv):
            c, out, _ = run(label, ['/usr/bin/journalctl', *argv], HOST_ENV)
            require(c['exit_code'] == 0, 'journal visibility/closure')
            return out.read_text()

        cursor = None
        try:
            freeze(); driver()
            before = journal('journal-before', ['-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            match = re.search(r'-- cursor: (\S+)', before)
            require(match, 'journal initial cursor'); cursor = match[1]
            if args.mode == 'build':
                compiler = str(LLVM / 'bin/clang++')
                header = str(STATE / 'main-primary-unitrace-build-root-v1/build/level_zero')
                binary = root / 'metric-enumeration'
                argv = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-fno-sycl',
                    '-I' + header, str(D / 'enumerate.cpp'), '/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0',
                    '-ldl', '-o', str(binary)]
                env = dict(HOST_ENV, TMPDIR=str(root), LD_LIBRARY_PATH=str(LLVM / 'lib'))
                c, _, _ = run('host-build', argv, env, wall=120, cpu=90)
                require(c['exit_code'] == 0, 'host build result')
                record['binary_candidate'] = dict(path=str(binary), realpath=str(binary.resolve()),
                    bytes=binary.stat().st_size, sha256=sha(binary))
                record['binary_review_required_before_enumeration'] = True
                record['candidate_complete'] = True
            else:
                binary = manifest['binary']; pin(binary['path'], binary)
                build = manifest['build_receipt']; pin(build['path'], build)
                receipt = json.loads(Path(build['path']).read_text())
                require(receipt['mode'] == 'build' and receipt['passed'] and receipt['complete'] and
                        receipt['ownership_closed'] and receipt['boot_id'] == boot and
                        receipt['reviewed_package_sha256'] == args.reviewed_package_sha256 and
                        receipt['binary_candidate'] == binary and receipt['run_id'] != args.run_id, 'separately reviewed completed build')
                frozen_pins[binary['path']] = binary
                frozen_pins[build['path']] = build
                env = dict(HOST_ENV, ZET_ENABLE_METRICS='1', NEOReadDebugKeys='1', EnableDirectSubmission='0')
                allowed = {**base['elf_pins'], **extra, binary['realpath']: binary}

                def interact(io, entry):
                    rows = []
                    for _ in range(40000):
                        row = json.loads(io.line(seconds=30)); rows.append(row)
                        if row.get('kind') == 'main_snapshot_hold':
                            break
                    else:
                        raise RuntimeError('row budget')
                    summaries = [v for v in rows if v.get('kind') == 'summary']
                    require(len(summaries) == 1, 'one native summary')
                    record['enumeration_summary'] = summaries[0]
                    ident = owner_mod.identity(io.proc.pid)
                    require(ident and ident['start_ticks'] == entry['direct_identity']['start_ticks'], 'actual PID/start')
                    proc = Path('/proc', str(io.proc.pid))
                    require((proc / 'exe').resolve() == Path(binary['realpath']), 'actual native executable')
                    effective = dict(v.decode().split('=', 1) for v in (proc / 'environ').read_bytes().split(b'\0') if v)
                    require(effective == env, 'exact child environment before exec')
                    mappings = []
                    for line in (proc / 'maps').read_text().splitlines():
                        f = line.split(None, 5)
                        if len(f) != 6 or 'x' not in f[1] or not f[5].startswith('/'):
                            continue
                        require(not f[5].endswith(' (deleted)'), 'deleted mapping')
                        p = Path(f[5]).resolve(strict=True)
                        require(str(p) in allowed, 'unreviewed loaded ELF dependency: ' + str(p))
                        v = allowed[str(p)]; pin(p, v); st = p.stat()
                        major, minor = [int(x, 16) for x in f[3].split(':')]
                        require(st.st_ino == int(f[4]) and os.major(st.st_dev) == major and os.minor(st.st_dev) == minor, 'mapped dev/inode')
                        mappings.append(dict(mapping=line, pin=v))
                    required = [binary['realpath'], '/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0']
                    if summaries[0]['status'] == 'ENUMERATION_PASS':
                        required.append('/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395')
                    seen = {v['pin']['realpath'] for v in mappings}
                    require(all(p in seen for p in required), 'actual binary/loader/driver mappings')
                    after = owner_mod.identity(io.proc.pid)
                    require(after and after['start_ticks'] == ident['start_ticks'], 'stable owned mapping snapshot')
                    record['native_snapshot'] = dict(identity=ident, environment=effective, mappings=mappings,
                        scope='File-backed executable mappings at post-enumeration hold only; unloaded/anonymous code not attested')
                    io.send(b'X'); io.exit(seconds=5)

                c, _, _ = run('native-enumeration', [binary['path'], '--main-owned-hold'], env, interact, wall=60)
                record['native_exit_code'] = c['exit_code']
                summary = record['enumeration_summary']
                require(c['exit_code'] in (0, 2), 'normal protocol exit')
                record['candidate_complete'] = True
                record['candidate_status'] = summary['status']
                require((c['exit_code'] == 0) == (summary['status'] == 'ENUMERATION_PASS'), 'status/exit agreement')
            freeze()
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            closing = True
            try:
                require(owner.active is None, 'ownership closure before journal')
                require(cursor, 'initial journal boundary required')
                end = journal('journal-end', ['-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
                match = re.search(r'-- cursor: (\S+)', end); require(match, 'final cursor')
                raw = journal('journal-after', ['-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                rows = [json.loads(v) for v in raw.splitlines() if v.strip()]
                require(all(v.get('_BOOT_ID') == boot.replace('-', '') for v in rows), 'sameboot journal rows')
                fresh = [v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))]
                record['fault_window'] = dict(before_cursor=cursor, after_cursor=match[1], boot_id=boot,
                    row_count=len(rows), fault_filter=FAULT.pattern, fresh_faults=fresh, complete=True)
                require(not fresh, 'fresh GPU fault'); driver()
                require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot, 'sameboot closure')
                record['ownership_closed'] = all(owner_mod.closed(c) and c.get('exit_code') in (0, 2) for c in owner.commands)
                require(record['ownership_closed'], 'all owned sessions closed')
                record['complete'] = bool(record.get('candidate_complete') and not record.get('error') and not record.get('budget_error'))
                record['status'] = record.get('candidate_status', 'BUILD_COMPLETE') if record['complete'] else 'INCOMPLETE'
                record['passed'] = record['complete'] and record['status'] in ('BUILD_COMPLETE', 'ENUMERATION_PASS')
            except BaseException as error:
                record['closure_error'] = repr(error)
                record.update(status='INCOMPLETE', passed=False, complete=False)
            record.update(finished_utc=utc(), elapsed_seconds=time.monotonic()-started)
            settling = True
            persist()
        print(json.dumps(dict(receipt=str(root / 'record.json'), sha256=sha(root / 'record.json'), status=record['status'])))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
