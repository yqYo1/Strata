import fcntl, hashlib, json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
R = B / 'research-20261009'
WX = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
WS = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
OUT = WS / 'bench/results/2026-10-11-xe-native-dequant-prefill-batch'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def require(value, message):
    if not value:
        raise RuntimeError(message)

def git(*args):
    return subprocess.run(['/usr/bin/git', '-C', str(WX), *args], capture_output=True,
                          check=True, timeout=20).stdout

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (OUT / 'archive-manifest.json').exists(), 'boundary not finalized')
    require(git('rev-parse', 'HEAD').decode().strip() == 'fa3e20e842fbed6986a69bade8afe3edabedee07', 'published caller')
    require(not git('status', '--porcelain'), 'clean published source')
    require(sha(WX / 'src/prefill/prefill.cpp') == 'c5cfa727302d334e2f6a8aaad8c1bf59e7d581dd8c2cec2af0d031faf807272d', 'exact caller')
    roots = [
        'xe-e32-free-full-build-root-v19',
        'xe-e32-free-full-build-root-v20', 'xe-e32-contrib-llvm-full-build-root-v20',
        'xe-e32-free-asan-full-build-root-v17', 'xe-e32-contrib-llvm-asan-full-build-root-v17',
        'xe-e32-lints-baseline-compare-root-v15',
        'xe-e32-free-full-build-root-v21', 'xe-e32-contrib-llvm-full-build-root-v21',
        'xe-e32-free-asan-full-build-root-v18', 'xe-e32-contrib-llvm-asan-full-build-root-v18',
        'xe-e32-lints-baseline-compare-root-v16',
        'xe-native-dequant-batch-census-observer-cpu-root-v2',
        'xe-native-dequant-prefill-model-controller-cpu-root-v2',
        'xe-native-dequant-prefill-model-controller-cpu-root-v3',
        'xe-native-dequant-prefill-model-root-v2/n64-g16-off',
        'xe-native-dequant-prefill-model-root-v3/n64-g16-off',
        'xe-native-dequant-prefill-post-startup-health-root-v1',
        'xe-native-dequant-prefill-source-publish-root-v1',
    ]
    receipts = {}
    for root in roots:
        p = B / root / 'record.json'
        d = json.loads(p.read_text())
        require(not d['active'], 'closed receipt ' + root)
        require(all(c.get('direct_child_reaped') and c.get('session_empty') and not c.get('survivors')
                    for c in d['commands']), 'all owners absent ' + root)
        receipts[root] = (p, d)
    for root in ['xe-e32-free-full-build-root-v21', 'xe-e32-contrib-llvm-full-build-root-v21',
                 'xe-e32-free-asan-full-build-root-v18', 'xe-e32-contrib-llvm-asan-full-build-root-v18',
                 'xe-e32-lints-baseline-compare-root-v16',
                 'xe-native-dequant-batch-census-observer-cpu-root-v2',
                 'xe-native-dequant-prefill-model-controller-cpu-root-v3',
                 'xe-native-dequant-prefill-post-startup-health-root-v1',
                 'xe-native-dequant-prefill-source-publish-root-v1']:
        require(receipts[root][1]['passed'] and receipts[root][1]['complete'], 'actual qualified ' + root)
    OUT.mkdir(exist_ok=True)
    manifest = []
    def copy(p, destination):
        q = OUT / destination
        q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists():
            require(q.read_bytes() == p.read_bytes(), 'unchanged draft archive ' + str(q))
        else:
            q.write_bytes(p.read_bytes())
        require(sha(q) == sha(p), 'exact archive ' + str(p))
        manifest.append(dict(source=str(p), archive=str(q.relative_to(WS)), sha256=sha(q), bytes=q.stat().st_size))
        return str(q.relative_to(WS))
    def write(name, value):
        p = OUT / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        return p

    for root, (p, _) in receipts.items():
        copy(p, 'receipts/' + root.replace('/', '--') + '.json')
    reviews = ['xe-native-dequant-prefill-batch-main-review-v1.json']
    reviews += ['xe-native-dequant-prefill-batch-applied-main-review-v' + str(n) + '.json' for n in [1, 2, 3, 4]]
    reviews += ['xe-native-dequant-batch-census-observer-main-review-v' + str(n) + '.json' for n in [1, 2]]
    reviews += ['xe-native-dequant-prefill-model-controller-main-review-v' + str(n) + '.json' for n in [2, 3]]
    for name in reviews:
        copy(B / name, 'reviews/' + name)
    recipes = [
        'xe-e32-full-build-v21/run_root.py', 'xe-e32-host-asan-build-v18/run_root.py',
        'xe-e32-lints-baseline-compare-v16/run_root.py',
        'xe-native-dequant-batch-census-observer-main-selfcheck-v2.py',
        'xe-native-dequant-prefill-model-controller-main-cpu-v2.py',
        'xe-native-dequant-prefill-model-controller-main-cpu-v3.py',
        'xe-native-dequant-prefill-post-startup-health-v1.py',
        'xe-native-dequant-prefill-source-publish-v1.py',
    ]
    for name in recipes:
        copy(B / name, 'recipes/' + name.replace('/', '--'))
    for v in [2, 3]:
        source = B / ('xe-native-dequant-prefill-model-controller-source-v' + str(v))
        for name in ['run_root.py', 'main-adaptation.patch']:
            copy(source / name, 'controller-source/v' + str(v) + '/' + name)
    source = B / 'xe-native-dequant-prefill-model-controller-source-v3'
    for p in sorted(source.glob('*.py')):
        if p.name != 'run_root.py':
            copy(p, 'controller-source/shared/' + p.name)
    for name in ['batch_census.py', 'selfcheck.py', 'candidate.patch', 'manifest.json']:
        copy(B / 'xe-native-dequant-batch-census-observer-source-v2' / name, 'observer-source/' + name)
    copy(B / 'xe-native-dequant-prefill-batch-source-v4/checked-expert-count.patch', 'source/checked-expert-count.patch')
    patch = OUT / 'source/prefill-batch.patch'
    patch.write_bytes(git('diff', 'b72640fff05098f1762cb3bf9c427a5f8d7db31c', 'HEAD', '--', 'src/prefill/prefill.cpp'))
    source = WX / 'src/prefill/prefill.cpp'
    old = git('show', 'b72640fff05098f1762cb3bf9c427a5f8d7db31c:src/prefill/prefill.cpp').decode()
    new = source.read_text()
    # The ON arm was inserted before the original grouped path; its OFF body is literal.
    default_start = old.index('                    // Stage ahead: the copy stream moves blobs')
    default_end = old.index('                    pt.mark(kPfCombine, cs);', default_start)
    original = old[default_start:default_end]
    require(new.count(original) == 1, 'unique literal original grouped path retained')
    new_start = new.index(original)
    require(new[:new_start].rstrip().endswith('} else {'), 'literal body is in OFF branch')
    require(len(original.encode()) == 20088 and hashlib.sha256(original.encode()).hexdigest() ==
            '9d6c91d570e1df2d14048995feec108280899e27d423f478f7d057f4b8104d75', 'known reviewed OFF bytes')
    write('source/default-path-proof.json', dict(base='b72640fff05098f1762cb3bf9c427a5f8d7db31c',
          published='fa3e20e842fbed6986a69bade8afe3edabedee07', prefill_sha256=sha(source),
          original_grouped_body_bytes=len(original.encode()), original_grouped_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
          original_grouped_body_literal_present=True, adopted=False, model_qualified=False))
    failure = receipts['xe-native-dequant-prefill-model-root-v3/n64-g16-off'][1]
    error_file = B / 'xe-native-dequant-prefill-model-root-v3/n64-g16-off/commands/model.stderr'
    lines = error_file.read_text(errors='replace').splitlines()
    regex = re.compile(r'UR_RESULT_ERROR_[A-Z_]+|ZE_RESULT_ERROR_[A-Z_]+|\[(?:ERROR|FATAL)\]|GPU completion unconfirmed|terminate called|Assertion .*failed|CAT error|fault response|out of resources', re.I)
    matches = [i for i, v in enumerate(lines) if regex.search(v)]
    require(len(matches) == 6, 'complete startup matcher findings')
    indices = set()
    for i in matches:
        indices.update(range(max(0, i - 8), min(len(lines), i + 42)))
    longjmp = [i for i, v in enumerate(lines) if 'longjmp causes uninitialized stack frame' in v]
    require(longjmp, 'cleanup symptom')
    indices.update(range(max(0, longjmp[0] - 12), min(len(lines), longjmp[0] + 4)))
    selected = '\n'.join(str(i + 1) + ': ' + lines[i] for i in sorted(indices)) + '\n'
    require(len(selected.encode()) < 256 << 10, 'bounded representative')
    (OUT / 'startup-error-excerpt.txt').write_text(selected)
    write('startup-error-analysis.json', dict(original_path=str(error_file), original_bytes=error_file.stat().st_size,
          original_sha256=sha(error_file), matching_line_numbers=[i + 1 for i in matches],
          module_build_failures=3, ur_program_build_failures=3, longjmp_messages=len(longjmp),
          first_longjmp_line=longjmp[0] + 1, GEN_sent=False, native_batch='off', ERROR_matcher_weakened=False,
          model_status_unchanged='FAIL', model_cleanup='owned SIGTERM; exit -11; child reaped; session empty',
          kernel_fault_messages=failure['new_fault_messages'], post_stop_integer_health='PASS',
          open_questions=['Determine actually selected versus speculative preload specializations',
                          'Exact compiler/driver local-matrix lowering contract',
                          'Identify longjmp/termination origin; temporal order alone does not establish cause'],
          main_source_finding='Runtime::preload_kernels iterates all get_kernel_ids and catches build exceptions locally; thread is joinable, not detached'))
    reports = [
        'implementation-xe-native-dequant-batch-census-observer-v1.txt',
        'implementation-xe-native-dequant-batch-census-observer-v2.txt',
        'research-native-batch-compiled-call-path-v1.txt',
        'research-native-batch-group-memory-admission-v1.txt',
        'research-native-batch-driver-lowering-v1.txt',
        'research-native-batch-prefill-service-bound-v1.txt',
        'research-xe-long-prefill-stream-selection-next-v1.txt',
        'implementation-native-batch-model-controller-v1.txt',
        'research-xe-avx2-native-decode-pool-next-v1.txt',
        'research-xe-long-prefill-stream-selection-correction-v2.txt',
        'research-xe-startup-precompile-error-lifecycle-next-v1.txt',
        'research-xe-joint-matrix-packed-a-lowering-next-v1.txt',
        'research-xe-failed-startup-kernel-reachability-next-v1.txt',
        'implementation-xe-optional-kernel-preload-control-v1.txt',
        'research-xe-failed-startup-binary-provenance-correction-v2.txt',
    ]
    previous = WS / 'bench/results/2026-10-11-xe-native-dequant-group-v2/report-registry-v129.json'
    prior = json.loads(previous.read_text())
    entries = []
    for name in reports:
        p = R / name
        archive = copy(p, name)
        implementation = name.startswith('implementation')
        entries.append(dict(path=str(p), archive=archive, sha256=sha(p), bytes=p.stat().st_size,
                       model='gpt-6.1-sol' if implementation else 'gpt-6-luna', read_only=not implementation,
                       status='completed', main_full_report_review=True,
                       qualification='Source-only; actual Main checks recorded separately' if implementation else 'Read-only research; no GPU/test/build execution'))
    write('report-registry-v130.json', dict(registry_version=130, research_completed=prior['research_completed'] + len(entries),
          created_utc=datetime.now(timezone.utc).isoformat(), registry_format='incremental; follow previous_committed_registry',
          previous_committed_registry=dict(path=str(previous.relative_to(WS)), sha256=sha(previous), reports=prior['research_completed']),
          new_reports=entries, reports=entries,
          duplicate_assignment_result=dict(agent='/root/research_native_batch_compiled_call_path',
              scope='Follow-up decode assignment', returned_report='research-native-batch-compiled-call-path-v1.txt',
              disposition='Duplicate original report; does not constitute a completed new decode investigation'),
          provenance_notes=['Original stream-selection report used live source; separate v2 addendum verifies fixed snapshot and withdraws priority claim',
                            'Startup lifecycle report uses detached once; source explicitly joins preload_ and Main records this correction',
                            'Prior Luna quota blocker superseded by actual successful research turns; selected models unchanged'],
          live_agent_snapshot=dict(at_utc=datetime.now(timezone.utc).isoformat(),
              research_agents=[dict(agent='/root/research_decode_avx2_pool_delta', status='completed', model='gpt-6-luna',
                                   scope='Separate correction of binary provenance in reachability report'),
                               dict(agent='/root/research_native_batch_driver_lowering', status='running', model='gpt-6-luna',
                                   scope='Exact PackedA builtin contract in fixed LLVM/IGC primary source')],
              implementation_agents=[dict(agent='/root/implement_optional_kernel_preload_control', status='completed', model='gpt-6.1-sol',
                                          scope='Source-only diagnostic preload disable candidate; unapplied and unexecuted')]),
          next_research_scopes=['Resolve exact unused-versus-selected module compilation and normal close',
                                'Official independent prefill/decode parameter search after safe qualification']))
    write('environment.json', dict(hardware=dict(cpu='Ryzen 5 5600X, 6 cores/12 threads, AVX2', gpu='Intel Arc B570, PCI 0000:05:00.0, 10GiB'),
          boot_id=failure['boot_id'], kernel='7.0.0-38-generic', source_base=failure['candidate_revision'],
          source_published_revision='fa3e20e842fbed6986a69bade8afe3edabedee07',
          source_pins=failure['source_pins'], runtime_libraries=failure['observed_libraries'],
          effective_settings=failure['effective_settings'], model_environment=failure['environment'],
          compiler_receipt='xe-llvm-v711-source-root-v5/record.json',
          model_failure_receipt_sha256=sha(receipts['xe-native-dequant-prefill-model-root-v3/n64-g16-off'][0]),
          historical_timing=dict(prefill_tokens_per_second=269.159148725, cold_decode_tokens_per_second=8.27129859,
              status='Prior diagnostic only; not current candidate performance'),
          new_model_prefill_measurement=False, new_model_decode_measurement=False, full262144_qualified=False,
          performance_eligible=False, model_qualified=False, adopted=False))
    retire = []
    for root, (receipt, data) in receipts.items():
        # Successful logs have complete compact evidence; failure logs await Main diagnostic review.
        if not data['passed']:
            continue
        directory = receipt.parent / 'commands'
        for p in sorted(directory.glob('*')):
            if p.is_file() and p.suffix in {'.stdout', '.stderr'}:
                retire.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p), allocated_bytes=p.stat().st_blocks * 512,
                              retained_receipt=str((OUT / 'receipts' / (root.replace('/', '--') + '.json')).relative_to(WS)),
                              reason='Closed successful API/build/test/lint/poll output represented by exact structured receipt'))
    write('retention-v1.json', dict(status='planned; delete only after committed archive is verified', files=retire,
          logical_bytes=sum(x['bytes'] for x in retire), allocated_bytes=sum(x['allocated_bytes'] for x in retire),
          retained_failures=[dict(path=str(error_file), bytes=error_file.stat().st_size, sha256=sha(error_file),
              budget_bytes=32 << 20, owner='Main startup JIT/error lifecycle investigation',
              consumer='Active read-only reachability and exact PackedA-contract agents may refer to original line windows',
              next_review='On return of both current research reports; retain compact representative and retire full history unless allocation/command causality needs it')],
          reset_executed=False, deletion_changes_status=False))
    write('archive-manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), byte_exact_copies=manifest,
          source_patch_sha256=sha(patch), archive_budget_bytes=16 << 20,
          generated_note='Original receipts retain historical source, outcome and paths. New reports do not rewrite old status.'))
    require(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < 16 << 20, 'archive size bound')
    print(json.dumps(dict(archive=str(OUT), exact_copies=len(manifest), receipts=len(receipts),
                         new_reports=len(entries), completed_reports=prior['research_completed'] + len(entries),
                         successful_logs_to_retire=len(retire), logical_bytes_to_retire=sum(x['bytes'] for x in retire))))
