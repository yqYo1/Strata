#!/usr/bin/env python3
"""Main-only source proposal: one original BF16 --smoke MetricQuery diagnostic.
Repository LICENSE. No launcher, collector build, model mode, retry or cleanup.
Requires separately Main-sealed patched collector; exact statuses must reconcile.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import selectors
import sys
import time
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
A = D.parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
V1 = A / 'own-metric-query-tiny-v1'
V1_PACKAGE_SHA = '080316bd939dd5a34934dfc3b7ece2cf3871bf84d5195379773b3c733ff6987c'
BASE = A / 'own-vllm-resource-timeline-v3'
BASE_MANIFEST_SHA = '61da9469e3f975c72d770beee77d45118c988eb6d3d82b6dbe1b568cd9ea3f1c'
IMPORTS = {
    BASE / 'controller.py': 'b81a06f1e0866d571748e7cf767d07930fa165339c152cfbf7d8b3f386005a38',
    BASE / 'timeline.py': '2121482d114f1de1fb087a6bc1425314022af1198a3242986b571a2664ed2c12',
    P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py':
        'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'}
ENUM_RECEIPT = B / 'main-metric-enum-query-v1/record.json'
ENUM_SHA = '5f9f2debab68c87ddcf6346874a9edcf132b1d85cf6731144f9ccb6827ed1af5'
STRICT_HOLD = 'NO_QUALIFICATION_WITHOUT_EXACT_PATCHED_COLLECTOR_API_EVIDENCE'
RAW, WORK_RAW, PRIVATE = 64 << 20, 62 << 20, 1 << 30
CONTROLS = dict(UNITRACE_MetricQuery='1', UNITRACE_MetricGroup='ComputeBasic',
    UNITRACE_HostTiming='1', UNITRACE_DeviceTiming='1', UNITRACE_FollowChildProcess='1',
    UNITRACE_LogToFile='1', ZE_ENABLE_TRACING_LAYER='1', ZET_ENABLE_METRICS='1',
    ZE_FLAT_DEVICE_HIERARCHY='FLAT', CUDA_MODULE_LOADING='EAGER')


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package(digest, directory=D):
    need(sha(directory / 'SHA256SUMS') == digest, 'externally reviewed full source package')
    names = set()
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        h, name = line.split('  ', 1)
        need(re.fullmatch('[a-f0-9]{64}', h) and not Path(name).is_absolute() and
             '..' not in Path(name).parts and name not in names, 'strict source snapshot')
        need(not (directory / name).is_symlink() and sha(directory / name) == h, 'reviewed package bytes')
        names.add(name)
    need(names == {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}, 'complete source snapshot')


def main():
    controller_started = time.monotonic()
    deadline = controller_started + 300
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True, help='Main-authored external review seal; never a live pin refresher')
    parser.add_argument('--seal-sha256', required=True)
    parser.add_argument('--reviewed-package-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    need(D == Path('/home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-15-pr6-recovery-audit/own-metric-query-tiny-v2'), 'canonical source package')
    need(os.geteuid() != 0, 'non-root controller/native only')
    need(re.fullmatch('v[1-9][0-9]{0,3}', args.run_id), 'fresh finite run identity')
    need(not any(k.startswith(('UNITRACE_', 'PTI_', 'XPTI_', 'ZE_ENABLE_', 'ZEL_', 'ZET_')) or
                 k == 'LD_PRELOAD' for k in os.environ), 'unpreloaded controller without collector controls')
    package(args.reviewed_package_sha256); package(V1_PACKAGE_SHA, V1)
    need(sha(args.seal) == args.seal_sha256, 'external Main seal')
    seal = json.loads(args.seal.read_text())
    need(seal['status'] == 'MAIN_REVIEWED_SEALED' and seal['run_id'] == args.run_id and
         seal['reviewed_package_sha256'] == args.reviewed_package_sha256 and
         seal['review_scope'] == 'ONE_TINY_PATCHED_QUERY_EXACT_API_STATUS', 'Main review of this separate collector only')
    for path, digest in IMPORTS.items():
        need(not path.is_symlink() and sha(path) == digest, 'qualified helper import source')
    need(sha(BASE / 'manifest.json') == BASE_MANIFEST_SHA, 'unchanged original inventory')
    base = imported(BASE / 'controller.py', 'query_original_guards')
    owner_module = imported(base.OWNER, 'query_original_owner')
    # The base import adds its own timeline directory. Add this package only
    # afterward, with all package bytes already reviewed, for these offline readers.
    sys.path.insert(0, str(V1))
    from metrics import decode, finite_text
    from summary import parse_summary, ERROR
    sys.path.insert(0, str(D))
    from status_reader import read as read_status
    from collector_identity import sealed_replacement, native_snapshot
    manifest = json.loads((BASE / 'manifest.json').read_text())
    manifest['_sha256'] = BASE_MANIFEST_SHA
    extra = seal['additional_dependency_review']
    base.check_pin(extra['report']['path'], extra['report'])
    review = json.loads(Path(extra['report']['path']).read_text())
    need(review['approved'] is True and review['elf_pins'] == extra['elf_pins'] and
         review['file_pins'] == extra['file_pins'], 'separate Main review of additional dependencies')
    need(not set(extra['elf_pins']).intersection(manifest['elf_pins']) and
         not set(extra['file_pins']).intersection(manifest['file_pins']), 'additional review cannot replace original pins')
    pins = dict(manifest['file_pins'])
    for path, v in {**extra['file_pins'], **extra['elf_pins']}.items():
        need(path not in pins or pins[path] == v, 'no conflicting additional pins')
        pins[path] = v
    allowed = {**manifest['elf_pins'], **extra['elf_pins']}
    replacement, build_inputs = sealed_replacement(seal, manifest, base, owner_module, args.reviewed_package_sha256)
    collector = replacement['binary']
    # Independent replacement identity slot; all original pins remain frozen.
    need(collector['realpath'] not in allowed, 'no replacement allowlist collision')
    allowed[collector['realpath']] = collector
    need(set(seal['catalog']) == {'receipt', 'build_receipt', 'binary', 'stdout'}, 'exact catalog artifact schema')
    inputs = [extra['report'], seal['health_receipt'], *seal['catalog'].values(), *build_inputs]
    for v in inputs:
        need(v['path'] not in pins or pins[v['path']] == v, 'no conflicting receipt pins')
        pins[v['path']] = v
    need(base.boot_id() == seal['boot_id'], 'explicit sameboot selection')
    retention = datetime.datetime.fromisoformat(seal['retention_review_utc'].replace('Z', '+00:00'))
    now = datetime.datetime.now(datetime.timezone.utc)
    need(retention.tzinfo and retention.utcoffset() == datetime.timedelta(0) and
         datetime.timedelta(0) < retention-now <= datetime.timedelta(hours=48), 'finite UTC review point')
    root = B / ('main-metric-query-tiny-v2-' + args.run_id)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not root.exists() and not root.is_symlink(), 'fresh private directory; no retry/overwrite')
        root.mkdir(mode=0o700)
        sides = root / 'collector'; sides.mkdir(mode=0o700)
        started = controller_started; work_end = deadline - 100
        record = dict(schema='strata-metric-query-tiny-v2', status='INCOMPLETE', passed=False,
            complete=False, active=False, collection_qualified=False, diagnostic_complete=False,
            qualification_hold=STRICT_HOLD, source_commit=base.SOURCE, source_pins=manifest['source_pins'],
            original_manifest_sha256=BASE_MANIFEST_SHA, binary=manifest['tiny'],
            original_reference_collector=manifest['collector'], replacement_collector=replacement,
            reviewed_package_sha256=args.reviewed_package_sha256, seal_sha256=args.seal_sha256,
            boot_id=seal['boot_id'], commands=[], fresh_faults=[], fault_windows=[],
            GPU_executed=False, ownership_closed=False, original_failure_status_preserved=True,
            budget=dict(native_seconds=90, whole_seconds=300, raw_bytes=RAW, private_bytes=PRIVATE,
                        decode_seconds=15, metric_rows=4096, line_bytes=65536),
            retention=dict(owner='Main', consumer='Main one original BF16 tiny MetricQuery; exact status/CSV/sample reconciliation',
                next_review=retention.isoformat(), cleanup_performed=False),
            model_quality_qualified=False, performance_qualified=False, physical_262144_qualified=False)
        owner = None; closing = settling = False; cursor = None

        def budget():
            files = list(root.rglob('*'))
            need(len(files) <= 4096 and all(not p.is_symlink() for p in files), 'bounded private regular artifacts')
            private = sum(p.stat().st_size for p in files if p.is_file())
            raw = sum(p.stat().st_size for p in files if p.is_file() and
                      (p.is_relative_to(sides) or p.suffix in ('.stdout', '.stderr')))
            record.update(private_bytes=private, raw_bytes=raw)
            need(private <= PRIVATE and raw <= (RAW if closing else WORK_RAW), 'live total/side/partial raw budget')

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            problem = None
            try:
                budget()
            except BaseException as error:
                record['budget_error'] = repr(error); problem = error
                record.update(diagnostic_complete=False, passed=False, complete=False, collection_qualified=False,
                              status='HOLD_API_EVIDENCE', qualification_hold=STRICT_HOLD)
            (root / 'record.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
            if problem and record['active'] and not settling:
                raise problem

        owner = owner_module.Owner(root / 'owned', persist); record['commands'] = owner.commands

        def remaining(cap):
            end = deadline if closing else work_end
            need(time.monotonic() < end, 'finite controller/closure deadline')
            return min(cap, end-time.monotonic())

        def frozen():
            remaining(1)
            need(base.boot_id() == record['boot_id'] and sha(args.seal) == args.seal_sha256, 'sameboot/frozen seal')
            package(args.reviewed_package_sha256); package(V1_PACKAGE_SHA, V1)
            need(sha(BASE / 'manifest.json') == BASE_MANIFEST_SHA and sha(ENUM_RECEIPT) == ENUM_SHA, 'original inventory/enumeration receipt unchanged')
            for path, digest in IMPORTS.items():
                need(sha(path) == digest, 'pinned imports unchanged')
            for path, v in pins.items():
                base.check_pin(path, v); remaining(1)
            for path, v in manifest['elf_pins'].items():
                base.check_pin(path, v); remaining(1)
            for path, digest in manifest['source_pins'].items():
                need(sha(P / path) == digest, 'original product source freeze'); remaining(1)

        def driver():
            need(base.boot_id() == record['boot_id'] and (base.DEVICE / 'driver').resolve().name == 'xe', 'sameboot xe')
            pci = {n: (base.DEVICE / n).read_text().strip() for n in ('vendor', 'device', 'revision', 'subsystem_vendor', 'subsystem_device')}
            need(pci['vendor'] == '0x8086' and pci['device'] == '0xe20c', 'exact B570 PCI05:00.0 target')
            dump = (base.DEVICE / 'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
            need(not dump, 'driver dump exists; no reset/deletion')
            record.update(driver_dump_exists=dump, boot_unchanged=True)
            previous = record.setdefault('driver_snapshots', [])
            snap = dict(kernel=os.uname().release, pci=pci)
            need(not previous or previous[0] == snap, 'stable driver/PCI identity')
            previous.append(snap)

        def utility(label, argv):
            c, out, _ = owner.run(label, argv, base.HOST_ENV, P, wall=remaining(12), cpu=30,
                rss_cap=2 << 30, text_cap=1 << 19, file_cap=1 << 19, total_cap=PRIVATE)
            need(owner_module.closed(c) and c['exit_code'] == 0, 'normal owned utility closure')
            base.absence([c]); budget()
            return out.read_text()

        try:
            frozen(); driver()
            for v in manifest['component_receipts']:
                base.check_pin(v['path'], v)
                r = json.loads(Path(v['path']).read_text())
                need(r['passed'] and r['complete'] and not r['active'] and r['ownership_closed'] and
                     r['boot_id'] == record['boot_id'] and r['source_pins'] == manifest['source_pins'] and
                     r['source_commit'] == base.SOURCE and r['products'] == manifest['products'] and
                     not r['fresh_faults'] and r['driver_dump_exists'] is False and
                     all(owner_module.closed(c) and c['exit_code'] == 0 for c in r['commands']), 'original sameboot component gates')
                base.absence(r['commands'])
            h = seal['health_receipt']
            record['health_selection'] = base.health_gate(h['path'], h['sha256'], owner_module, record['boot_id'], manifest)
            record['owner_scan_before'] = base.owner_scan({})
            cat = seal['catalog']; r = json.loads(Path(cat['receipt']['path']).read_text())
            need(r['passed'] and r['complete'] and not r['active'] and r['ownership_closed'] and
                 r['boot_id'] == record['boot_id'] and r['mode'] == 'enumerate' and
                 r['enumeration_summary']['status'] == 'ENUMERATION_PASS' and
                 r['fault_window']['complete'] and not r['fault_window']['fresh_faults'] and
                 all(owner_module.closed(c) and c['exit_code'] == 0 for c in r['commands']), 'closed actual full-catalog enumeration')
            base.absence(r['commands'])
            build = json.loads(Path(cat['build_receipt']['path']).read_text())
            need(build['passed'] and build['complete'] and not build['active'] and build['ownership_closed'] and
                 build['mode'] == 'build' and build['binary_candidate'] == cat['binary'] and
                 build['boot_id'] == record['boot_id'] and
                 build['reviewed_package_sha256'] == V1_PACKAGE_SHA and
                 all(owner_module.closed(c) and c['exit_code'] == 0 for c in build['commands']), 'reviewed separately closed full-catalog build')
            builds = [c for c in build['commands'] if c['label'] == 'host-build']
            need(len(builds) == 1 and str(V1 / 'enumerate_full.cpp') in builds[0]['argv'], 'full-catalog source path in reviewed build recipe')
            base.absence(build['commands'])
            native = [c for c in r['commands'] if c['label'] == 'native-enumeration']
            need(len(native) == 1 and native[0]['argv'] == [cat['binary']['path'], '--main-owned-hold'] and
                 r['native_snapshot']['identity']['pid'] == native[0]['pid'] and
                 r['native_snapshot']['identity']['start_ticks'] == native[0]['direct_identity']['start_ticks'] and
                 r['native_snapshot']['environment']['ZET_ENABLE_METRICS'] == '1' and
                 r['native_snapshot']['environment']['ZE_FLAT_DEVICE_HIERARCHY'] == 'FLAT', 'actual full-catalog binary/env/PID binding')
            need(any(v['pin'] == cat['binary'] for v in r['native_snapshot']['mappings']), 'actual full-catalog executable mapping pin')
            logpin = native[0]['logs']['native-enumeration.stdout']
            need(logpin['sha256'] == cat['stdout']['sha256'] and logpin['bytes'] == cat['stdout']['bytes'], 'actual enumeration stdout hash/size')
            rows = [json.loads(l) for l in finite_text(cat['stdout']['path'], time.monotonic()+10, 4 << 20).splitlines()]
            need(len(rows) <= 40000 and all(v['code'] == 0 for v in rows if v.get('kind') == 'result'), 'all enumeration API results preserved/successful')
            groups = [v for v in rows if v.get('kind') == 'group' and v['name'] == 'ComputeBasic' and v['query_event_based']]
            need(len(groups) == 1 and groups[0]['target'] and groups[0]['metric_count'] == 83 and
                 groups[0]['sampling_flags'] == 1, 'unambiguous ComputeBasic EVENT_BASED group')
            g = groups[0]
            catalog = [v for v in rows if v.get('kind') == 'metric' and
                       (v['driver'], v['root'], v['group']) == (g['driver'], g['root'], g['group'])]
            need(len(catalog) == 83 and [m['metric'] for m in catalog] == list(range(83)), 'full ordered schema; sampled v1 is insufficient')
            record['catalog_selection'] = dict(receipt=cat['receipt'], group=g, metrics=catalog)
            env = dict(manifest['profile']['environment'])
            for k in ('STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'):
                env.pop(k, None)
            need(not any(k.startswith(('UNITRACE_', 'PTI_', 'XPTI_', 'ZEL_', 'IGC_', 'ZE_ENABLE_', 'ZET_')) or k == 'LD_PRELOAD' for k in env), 'no inherited trace/filter/programmatic controls')
            env.update(base.WARNINGS, **CONTROLS, LD_PRELOAD=collector['path'],
                UNITRACE_LogFilename=str(sides / 'query.csv'), XDG_CACHE_HOME=str(root / 'cache'),
                STRATA_SYCL_PREFILL_NATIVE16_BACKEND='own')
            record['native_environment'] = env
            start = utility('journal-before', ['/usr/bin/journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            match = re.search(r'-- cursor: (\S+)', start); need(match, 'initial journal boundary'); cursor = match[1]
            record['fault_windows'].append(dict(before_cursor=cursor, boot_id=record['boot_id'], complete=False))
            proofs = []

            def smoke_io(io, entry):
                record['GPU_executed'] = True; io.proc.stdin.close()
                next_map_check = 0
                with selectors.DefaultSelector() as ready:
                    ready.register(io.proc.stdout, selectors.EVENT_READ)
                    while not io.exit_watch.poll(0) or not io.eof:
                        io.poll(); remaining(1)
                        if not io.exit_watch.poll(0) and time.monotonic() >= next_map_check:
                            raw_maps = Path('/proc', str(io.proc.pid), 'maps').read_text()
                            signature = '\n'.join(l for l in raw_maps.splitlines() if len(l.split(None, 5)) == 6 and
                                'x' in l.split(None, 5)[1] and l.split(None, 5)[5].startswith('/'))
                            digest = hashlib.sha256(signature.encode()).hexdigest()
                            if not proofs or proofs[-1]['executable_maps_sha256'] != digest:
                                snap = native_snapshot(base, io.proc.pid, manifest['tiny'], env, allowed, io.proc.pid, collector, manifest['collector'])
                                if snap:
                                    need(len(proofs) < 64, 'bounded actual loaded-map changes'); proofs.append(snap)
                                    record['mapping_snapshots'] = proofs
                            next_map_check = time.monotonic()+.05
                        if ready.select(.002):
                            io.read(); io.pending.clear()

            c, out, err = owner.run('native-smoke', [manifest['tiny']['path'], '--smoke'], env, sides,
                interaction=smoke_io, wall=remaining(90), cpu=90, rss_cap=8 << 30,
                text_cap=WORK_RAW, total_cap=PRIVATE, file_cap=WORK_RAW)
            record['native_exit_code'] = c['exit_code']; base.absence([c])
            need(owner_module.closed(c) and c['exit_code'] == 0, 'normal original tiny native closure')
            need(proofs and all(p['native']['pid'] == c['pid'] and
                 p['native']['start_ticks'] == c['direct_identity']['start_ticks'] and
                 p['argv'] == [manifest['tiny']['path'], '--smoke'] for p in proofs), 'actual original argv/PID/start/maps/env')
            stdout, stderr = out.read_text(), err.read_text()
            record['oracle_stdout'] = stdout
            cases = [l for l in stdout.splitlines() if l.startswith('case dtype=bf16 ')]
            need(len(cases) == 2 and all('oracle=FP64_full ' in l and
                 'finite=PASS guards=PASS inputs=PASS qualification=COMPONENT_ONLY_UNQUALIFIED' in l for l in cases) and
                 stdout.count('fixture=PASS model_quality=UNRUN whole_prefill=UNRUN physical262144=UNRUN faults=EXTERNAL_REQUIRED') == 1 and
                 not ERROR.search(stderr), 'unchanged two full FP64 oracle/finite/guards/input/tolerance gates')
            fields = [dict(v.split('=', 1) for v in line.split() if '=' in v) for line in cases]
            need([(v['nq'], v['shape'], v['page'], v['masked'], v['width'], v['profile']) for v in fields] ==
                 [('1', '1,24,256', '4', '0', '33', 'regular'), ('3', '3,24,256', '4', '1', '33', 'regular')] and
                 all(math.isfinite(float(v['max_budget_ratio'])) and 0 <= float(v['max_budget_ratio']) <= 1 for v in fields),
                 'original case identities and predetermined numerical budget')
            summary_path = sides / ('query.' + str(c['pid']) + '.csv')
            metrics_path = sides / ('query.metrics.' + str(c['pid']) + '.csv')
            need(sorted(p for p in sides.rglob('*') if p.is_file()) == sorted([summary_path, metrics_path]), 'exact original native PID-named files; unknown/partial sides fail')
            decode_end = min(work_end, time.monotonic()+15)
            summary = parse_summary(summary_path, decode_end)
            record['summary'] = summary
            record['query'] = decode(metrics_path, catalog, summary['expected_compute_calls'], decode_end)
            record['exact_metric_api'] = read_status(err, c['pid'], record['query'], g, catalog, decode_end)
            # The unchanged v1 decoder carries its historical API-status warning.
            # Its original result stays intact; v2 exact evidence is a separate key.
            frozen(); budget()
            record['candidate_diagnostic_complete'] = True
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            closing = True
            try:
                need(owner.active is None and cursor, 'closed ownership and initial journal cursor required')
                end = utility('journal-end', ['/usr/bin/journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
                match = re.search(r'-- cursor: (\S+)', end); need(match, 'final journal boundary')
                raw = utility('journal-after', ['/usr/bin/journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                journal = [json.loads(l) for l in raw.splitlines() if l.strip()]
                need(all(v.get('_BOOT_ID') == record['boot_id'].replace('-', '') for v in journal), 'sameboot journal records')
                faults = [v for v in journal if base.FAULT.search(str(v.get('MESSAGE', '')))]
                record['fresh_faults'] = faults
                record['fault_windows'][-1].update(after_cursor=match[1], row_count=len(journal),
                    last_returned_cursor=journal[-1].get('__CURSOR') if journal else None,
                    fault_filter=base.FAULT.pattern, fresh_faults=faults, complete=True)
                need(not faults, 'fresh GPU faults'); driver(); frozen()
                record['final_PID_start_absence'] = base.absence(owner.commands)
                record['owner_scan_after'] = base.owner_scan({})
                record['ownership_closed'] = owner.active is None and all(owner_module.closed(c) for c in owner.commands)
                need(record['ownership_closed'] and all(c['exit_code'] == 0 for c in owner.commands), 'normal commands/reap/session-empty closure')
                budget(); remaining(1)
                record['diagnostic_complete'] = bool(record.get('candidate_diagnostic_complete') and
                    not record.get('error') and not record.get('budget_error'))
            except BaseException as error:
                record['closure_error'] = repr(error); record['diagnostic_complete'] = False
            exact = bool(record.get('exact_metric_api', {}).get('proved'))
            qualified = record['diagnostic_complete'] and exact
            record.update(passed=qualified, complete=qualified, collection_qualified=qualified,
                          qualification_hold=None if qualified else STRICT_HOLD,
                          status='TINY_QUERY_EXACT_STATUS_PASS' if qualified else 'HOLD_API_EVIDENCE')
            record['finished_utc'] = base.utc(); record['elapsed_seconds'] = time.monotonic()-started
            record['collector_sides'] = []
            if owner.active is None:
                files = sorted(p for p in sides.rglob('*') if p.is_file() or p.is_symlink())
                can_hash = (time.monotonic() < deadline-2 and len(files) <= 4096 and
                    all(not p.is_symlink() for p in files) and
                    sum(p.stat().st_size for p in files) <= RAW)
                record['collector_sides'] = ([base.artifact(p) for p in files] if can_hash else
                    [dict(path=str(p), bytes=p.lstat().st_size,
                          hash_status='DEFERRED_MAIN_REVIEW_BUDGET_OR_UNSAFE_FILE') for p in files[:4096]])
                if not can_hash:
                    record.update(status='HOLD_API_EVIDENCE', passed=False, complete=False, collection_qualified=False,
                                  diagnostic_complete=False, side_inventory_incomplete=True, qualification_hold=STRICT_HOLD)
            settling = True; persist()
        print(json.dumps(dict(status=record['status'], receipt=str(root / 'record.json'),
            sha256=sha(root / 'record.json'), collection_qualified=record['collection_qualified'])), flush=True)
        return 0 if record['collection_qualified'] else 2  # Tiny collection only; no model pipeline.


if __name__ == '__main__':
    sys.exit(main())
