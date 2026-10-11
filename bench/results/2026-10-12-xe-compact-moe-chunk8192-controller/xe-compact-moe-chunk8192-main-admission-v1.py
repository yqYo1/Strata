"""Main CPU-only resolution from actual closed controller and six GPU prerequisites."""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
H = B / 'xe-compact-moe-chunk8192-controller-source-v1'
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
WS = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
s = importlib.util.spec_from_file_location('chunk_admission_owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
sys.path.insert(0, str(H))
import admission as a

ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-compact-moe-chunk8192-main-admission-root-v1'
    m.require(not out.exists(), 'new immutable Main converter output')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             owner_sha256=m.sha(O), recipe_sha256=m.sha(__file__), gpu_executed=False,
             model_executed=False, queue_created=False, performance_eligible=False,
             original_prerequisite_results_modified=False)

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands

    def read(path):
        a.safe_record_path(path)
        m.require(path.stat().st_size <= 8 << 20, 'bounded exact retained prerequisite')
        return json.loads(path.read_text())

    def run(label, argv):
        c, so, _ = owner.run(label, argv, ENV, W, wall=30, cpu=60, rss_cap=2 << 30,
                            text_cap=2 << 20, total_cap=16 << 20, file_cap=64 << 20)
        m.require(a.strong_closed_command(c) and c['exit_code'] == 0, 'normal owned proof ' + label)
        return so.read_text()

    save()
    try:
        original_path = B / 'xe-compact-moe-main-admission-converter-root-v1/qualification.resolved.json'
        original = read(original_path)
        m.require(m.sha(original_path) == '9a00ad8dbfa2ad5982c4745555ef6b9c86d44d0cd982c1a870aff1d50b2834e8', 'actual immutable original admission')
        a.require_closed_prerequisite(read(original_path.parent / 'record.json'))
        a.require_qualification(original, original['production_source_pins'], a.ORIGINAL_CODE_NAMES)
        cpu_path = B / 'xe-compact-moe-chunk8192-controller-cpu-root-v1/record.json'
        review_path = B / 'xe-compact-moe-chunk8192-controller-main-review-v1.json'
        cpu, review = read(cpu_path), read(review_path)
        a.require_closed_prerequisite(cpu)
        m.require(review['main_reviewed'] and review['source_pins'] == cpu['source_pins'] and cpu['source_stable'] and
            cpu['owner_sha256'] == a.OWNER_SHA256 and cpu['main_review_sha256'] == m.sha(review_path), 'actual reviewed twelve-file CPU qualification')
        m.require(set(cpu['source_pins']) == a.CODE_NAMES and
            all(m.sha(H / name) == h for name, h in review['all_fifteen_file_pins'].items()), 'all fifteen source-only files stable')
        a.require_original_controller_proofs(read(Path(original['controller_cpu_receipt'])),
            read(Path(original['controller_review'])), original)
        m.require(all(m.sha(Path(p)) == h for p, h in original['receipt_sha256'].items()), 'all original prerequisite identities stable')
        m.require(all(m.sha(W / p) == h for p, h in original['production_source_pins'].items()), 'actual published42 content stable')
        head = run('source-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        status = run('source-status', ['/usr/bin/git', 'status', '--porcelain', '--untracked-files=all'])
        diff = run('source-diff', ['/usr/bin/git', 'diff', 'HEAD'])
        a.require_published_state(original['production_source_pins'], original['production_source_pins'], head,
            a.PUBLISHED_REVISION, status, diff, m.sha(Path(original['free_release_binary']['path'])), original['free_release_binary']['sha256'])
        compiler = read(Path(original['compiler_receipt']))
        a.require_compiler_prerequisite(compiler)
        for value in compiler['installed_files'].values():
            m.require(Path(value['path']).stat().st_size == value['bytes'] and m.sha(Path(value['path'])) == value['sha256'], 'current actual compiler/runtime')
        host_path = Path(original['host_fixture_receipt'])
        host = read(host_path)
        a.require_host_prerequisite(host, original['production_source_pins'], W, original['compiler_receipt_sha256'])
        for value in host['binaries'].values():
            m.require(Path(value['path']).stat().st_size == value['bytes'] and m.sha(Path(value['path'])) == value['sha256'], 'current actual host fixtures')
        for mode, path in original['build_receipts'].items():
            build = read(Path(path))
            value = build['binary']
            m.require(Path(value['path']).stat().st_size == value['bytes'] and m.sha(Path(value['path'])) == value['sha256'], 'current actual four-mode binary')
            for recipe in build['build_recipe'].values():
                m.require(m.sha(Path(recipe['path'])) == recipe['sha256'], 'actual retained build recipe')
            cache = Path(build['build_recipe']['CMakeCache.txt']['path']).read_text()
            a.require_build_prerequisite(build, mode, original['production_source_pins'], W, m.sha(host_path),
                original['compiler_receipt_sha256'], original['qualified_build_binaries'][mode], cache)
        lint = read(Path(original['lint_receipt']))
        original_lint_path = Path(lint['original_failed_comparison']['path'])
        source_review = read(B / 'xe-compact-moe-applied-main-review-v2.json')
        a.require_lint_prerequisite(lint, original['production_source_pins'], source_review['changed_source_pins'],
            original['build_receipt_sha256'], read(original_lint_path), m.sha(original_lint_path))
        manifest = read(Path(cpu['matrix_manifest']['path']))
        pins = manifest['six_case_receipts']
        matrix = {key: read(Path(pin['path'])) for key, pin in pins.items()}
        m.require(all(m.sha(Path(pin['path'])) == pin['sha256'] for pin in pins.values()), 'actual all-six original model hashes')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        r['six_case_verification'] = a.require_completed_matrix(matrix, pins, original, m.sha(original_path), boot, B)
        bridge_path = out / 'publication.bridge.resolved.json'
        q = copy.deepcopy(original)
        q.update(published_revision=a.PUBLISHED_REVISION, compact='on', sequence=[list(x) for x in a.SEQUENCE],
            original_qualification=str(original_path), original_qualification_sha256=m.sha(original_path),
            six_case_receipts=pins, controller_source_pins=cpu['source_pins'], controller_cpu_receipt=str(cpu_path),
            controller_review=str(review_path), controller_review_sha256=m.sha(review_path),
            first_predecessor=str(a.matrix_path(B, 32768, 'on')), publication_bridge=str(bridge_path),
            notes='Actual closed CPU qualification and original six6144 diagnostics support bounded changed-chunk measurement only. No8192 fit,64K quality,full262144,RESTORE or performance is inferred.')
        bridge = copy.deepcopy(read(Path(original['publication_bridge'])))
        bridge.update(controller_source_pins=cpu['source_pins'], original_admission_publication_bridge=dict(
            path=original['publication_bridge'], sha256=original['publication_bridge_sha256']),
            new_controller_cpu_receipt=dict(path=str(cpu_path), sha256=m.sha(cpu_path)),
            new_controller_main_review=dict(path=str(review_path), sha256=m.sha(review_path)),
            gpu_executed=False, model_executed=False)
        bridge_path.write_text(json.dumps(bridge, indent=2) + '\n')
        q['publication_bridge_sha256'] = m.sha(bridge_path)
        for p in [original_path, cpu_path, review_path, bridge_path, Path(cpu['matrix_manifest']['path'])]:
            q['receipt_sha256'][str(p)] = m.sha(p)
        for pin in pins.values():
            q['receipt_sha256'][pin['path']] = pin['sha256']
        a.require_chunk_qualification(q, original['production_source_pins'], original)
        a.require_publication(bridge, q, q['production_source_pins'], W)
        qpath = out / 'qualification.resolved.json'
        qpath.write_text(json.dumps(q, indent=2) + '\n')
        r.update(qualification=dict(path=str(qpath), sha256=m.sha(qpath)), publication_bridge=dict(path=str(bridge_path), sha256=m.sha(bridge_path)),
            controller_source_pins=cpu['source_pins'], source_pins=original['production_source_pins'], receipt_pins=q['receipt_sha256'], boot_id=boot)
        # Preserve a repeated, superseding R149 return separately, restoring the exact committed original.
        report_name = 'research-xe64k-teacher-forced-first-distribution-hook-source-contract-round149-v1.txt'
        private = B / 'research-20261009' / report_name
        archived = WS / 'bench/results/2026-10-12-parallel-round149' / report_name
        repeat = B / 'research-20261009/research-xe64k-quality-hook-r149-repeated-return-20261012-v1.txt'
        m.require(m.sha(private) == '15956ef3587c2176a42358882985b3cdbf0b3e56a0a4c1aa2bbd6386c8fa6e61' and
            m.sha(archived) == 'a7283662d0b25f947b8700b3a1454cf5cc79fa62425583708cbd909dd8e9fd3b' and not repeat.exists(), 'exact repeated return and original committed report')
        repeat.write_bytes(private.read_bytes())
        m.require(m.sha(repeat) == m.sha(private), 'preserve repeated return before restoration')
        private.write_bytes(archived.read_bytes())
        m.require(m.sha(private) == m.sha(archived), 'restore authoritative original exact report')
        r['report_original_restoration'] = dict(original_path=str(private), original_sha256=m.sha(private),
            repeated_return_path=str(repeat), repeated_return_sha256=m.sha(repeat), original_committed_archive=str(archived),
            reason='Completed follow-up mistakenly returned R149 again and overwrote its private path; both texts retained separately, published original and its statuses unchanged.', research_count_increment=0)
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'qualification')}))
    raise SystemExit(0 if r['passed'] else 1)
