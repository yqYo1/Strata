"""Main CPU-only separate coarse-phase proof/recipe/front-door admission from actual closed controller, six plus three GPU prerequisites."""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
H = B / 'xe-compact-moe-phase-observer-controller-source-v1'
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
    out = B / 'xe-compact-moe-phase-observer-main-admission-root-v1'
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
        chunk_path = B / 'xe-compact-moe-chunk8192-main-admission-root-v1/qualification.resolved.json'
        chunk_q = read(chunk_path)
        m.require(m.sha(chunk_path) == '1d077ebe12abcd8e2d5f18c630f06c82a9b9dea17868f66b3836e9f9cecf351e', 'actual closed chunk admission')
        a.require_closed_prerequisite(read(chunk_path.parent / 'record.json'))
        a.require_chunk_qualification(chunk_q, original['production_source_pins'], original)
        a.require_original_controller_proofs(read(Path(chunk_q['controller_cpu_receipt'])), read(Path(chunk_q['controller_review'])), chunk_q)
        m.require(all(m.sha(Path(p)) == h for p,h in chunk_q['receipt_sha256'].items()), 'all retained chunk prerequisite pins stable')
        cpu_path = B / 'xe-compact-moe-phase-observer-controller-cpu-root-v1/record.json'
        review_path = B / 'xe-compact-moe-phase-observer-controller-main-review-v1.json'
        cpu, review = read(cpu_path), read(review_path)
        a.require_closed_prerequisite(cpu)
        m.require(review['main_reviewed'] and review['source_pins'] == cpu['source_pins'] and cpu['source_stable'] and
            cpu['owner_sha256'] == a.OWNER_SHA256 and cpu['main_review_sha256'] == m.sha(review_path), 'actual reviewed fourteen-file CPU qualification')
        m.require(set(cpu['source_pins']) == a.CODE_NAMES and
            all(m.sha(H / name) == h for name, h in review['all_nineteen_file_pins'].items()), 'all nineteen source-only files stable')
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
        triple_manifest = read(Path(cpu['three_case_manifest']['path']))
        triple_pins = triple_manifest['three_case_receipts']
        m.require(all(m.sha(Path(pin['path'])) == pin['sha256'] for pin in triple_pins.values()), 'actual three GPU closed receipt hashes')
        diagnostics = {key:read(Path(pin['path'])) for key,pin in triple_pins.items()}
        r['three_case_verification'] = a.require_closed_chunk_diagnostics(diagnostics, triple_pins, chunk_q, m.sha(chunk_path), boot, B)
        analysis_path = B/'xe-compact-moe-quiet64k-analysis-root-v1/result.json'
        analysis = read(analysis_path)
        m.require(analysis['passed'] and analysis['complete'] and not analysis['active'] and analysis['all_six_exact16_ID_LP5'], 'Main completed quiet campaign analysis')
        quiet_q_path=B/'xe-compact-moe-quiet64k-main-admission-root-v1/qualification.resolved.json'
        quiet_q=read(quiet_q_path)
        m.require(m.sha(quiet_q_path)=='f67cee5cd040aa03076ef8dc4dd48f9d76c6670cba8753f3f5e53316e61c9c09','actual quiet predecessor admission')
        a.require_quiet_qualification(quiet_q,original['production_source_pins'],chunk_q,original)
        for sample,t in a.QUIET_SEQUENCE:
            p=a.quiet_root_path(B,sample,t)/'record.json'
            m.require(m.sha(p)==analysis['receipt_pins'][str(p)], 'all quiet actual closed pins')
            a.require_prior_quiet(read(p),boot,sample,t,quiet_q['controller_source_pins']['run_root.py'],m.sha(quiet_q_path),triple_pins[f'n65536-t{t}'])
        predecessor_path=a.quiet_root_path(B,3,8192)/'record.json'
        a.require_normal_model(read(predecessor_path),boot)
        m.require(not (B/'xe-compact-moe-phase-observer-root-v1').exists(),'phase namespace not launched')
        bridge_path = out / 'publication.bridge.resolved.json'
        q = copy.deepcopy(chunk_q)
        q.update(published_revision=a.PUBLISHED_REVISION, compact='on', phase_sequence=[list(x) for x in a.PHASE_SEQUENCE], quiet_arm=False, timing_environment_delta={'STRATA_PREFILL_TIMING':'1'},
            chunk_qualification=str(chunk_path), chunk_qualification_sha256=m.sha(chunk_path), three_case_receipts=triple_pins,
            original_qualification=str(original_path), original_qualification_sha256=m.sha(original_path),
            six_case_receipts=pins, controller_source_pins=cpu['source_pins'], controller_cpu_receipt=str(cpu_path),
            controller_review=str(review_path), controller_review_sha256=m.sha(review_path),
            first_predecessor=str(predecessor_path), publication_bridge=str(bridge_path),
            notes='Actual closed own phase CPU/review and original six plus three diagnostics; complete quiet campaign precedes N64 timestamp sanity thenN32768 fixed8192 diagnostics. No performance/default adoption/broad quality/full262144/RESTORE claim; compute marker spans are not exclusive H2D/kernel/criticalpath.')
        bridge = copy.deepcopy(read(Path(chunk_q['publication_bridge'])))
        bridge.update(controller_source_pins=cpu['source_pins'], original_admission_publication_bridge=dict(
            path=original['publication_bridge'], sha256=original['publication_bridge_sha256']),
            new_controller_cpu_receipt=dict(path=str(cpu_path), sha256=m.sha(cpu_path)),
            new_controller_main_review=dict(path=str(review_path), sha256=m.sha(review_path)),
            gpu_executed=False, model_executed=False)
        bridge_path.write_text(json.dumps(bridge, indent=2) + '\n')
        q['publication_bridge_sha256'] = m.sha(bridge_path)
        for p in [original_path, chunk_path, cpu_path, review_path, bridge_path, Path(cpu['matrix_manifest']['path']), Path(cpu['three_case_manifest']['path']), quiet_q_path, analysis_path, *map(Path,analysis['receipt_pins'])]:
            q['receipt_sha256'][str(p)] = m.sha(p)
        for pin in [*pins.values(), *triple_pins.values()]:
            q['receipt_sha256'][pin['path']] = pin['sha256']
        # Separate proof first; neither proof nor recipe hashes the front-door qualification.
        proof_path=out/'phase-proof.resolved.json'
        a.require_qualification(q,original['production_source_pins'])
        a.require_original_controller_proofs(cpu,review,q)
        a.require_publication(bridge,q,q['production_source_pins'],W)
        proof_path.write_text(json.dumps(q,indent=2)+'\n')
        proof=read(proof_path)
        recipe=read(H/'phase.recipe.pending.json')
        recipe.update(status='passed',main_reviewed=True,cpu_qualified=True,qualification_sha256=m.sha(proof_path),
            source_pins=proof['production_source_pins'],binary=proof['free_release_binary'],all_three_diagnostics_closed_pass=True,
            required_controller_cpu_receipt=str(cpu_path),required_main_phase_admission=str(out/'qualification.resolved.json'),
            before_launch='Main-reviewed fourteen-file bounded phase controller; actual own CPU qualification and separate proof/recipe/front-door converter. N64 first thenN32768. No queue property or source/kernel modification; finite owner and normal QUIT/source/runtime/fault/health/output gates.')
        from phase_observer import require_phase_admission
        require_phase_admission(recipe,proof,m.sha(proof_path))
        recipe_path=out/'phase-recipe.resolved.json'
        recipe_path.write_text(json.dumps(recipe,indent=2)+'\n')
        q=copy.deepcopy(proof)
        q.update(phase_proof_qualification=str(proof_path),phase_proof_qualification_sha256=m.sha(proof_path),
            resolved_phase_recipe=str(recipe_path),resolved_phase_recipe_sha256=m.sha(recipe_path))
        q['receipt_sha256'][str(proof_path)]=m.sha(proof_path)
        q['receipt_sha256'][str(recipe_path)]=m.sha(recipe_path)
        a.require_phase_qualification(q,original['production_source_pins'],chunk_q,original)
        a.require_phase_proof(proof,q,original['production_source_pins'])
        qpath = out / 'qualification.resolved.json'
        qpath.write_text(json.dumps(q, indent=2) + '\n')
        r.update(qualification=dict(path=str(qpath), sha256=m.sha(qpath)), publication_bridge=dict(path=str(bridge_path), sha256=m.sha(bridge_path)),
            controller_source_pins=cpu['source_pins'], source_pins=original['production_source_pins'], receipt_pins=q['receipt_sha256'], boot_id=boot)
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
