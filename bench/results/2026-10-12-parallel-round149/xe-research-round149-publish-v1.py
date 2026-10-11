"""Main: preserve reviewed source-only R149/R150 findings and their limits in Git."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
S = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('r149_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(B / 'xe-compact-moe-model-controller-source-v2'))
import admission as a

ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']
REPORTS = {
    'research-xe64k-teacher-forced-first-distribution-hook-source-contract-round149-v1.txt':
        'a7283662d0b25f947b8700b3a1454cf5cc79fa62425583708cbd909dd8e9fd3b',
    'research-xe-full262144-zero-output-physical-cell-probe-contract-round149-v1.txt':
        'c33713c03cc98675ace22aab2848ced9e68098eafa73c5d97920165fe9ac5f90',
    'research-xe-compact-chunk-repeat-work-compute-memory-census-round150-v1.txt':
        '482c360fadc796f654ec00217df2362d713ad0c65fa8de280affa3470b8a964a',
    'research-xe-safe-partial-multichunk-weight-reuse-reference-map-round150-v1.txt':
        '842f83e35aa5f1437019e904b9a4ef70195cc12b3b1e06a2d0c6fec790a9d488',
}
SOURCE = {
    'src/program/generate.cpp': 'c9b60ef079694302dfa1f8099ddffd5375d7dae503f6edec454c1d7de3d941ee',
    'src/core/verify.cpp': '44d8667e973c44d889afac2387cf7686a1b3785d31fa22f22cc5c7097927e27a',
    'include/strata/core/verify.hpp': '452c85060e68e94b7d9e0f039d25f3b9696972a50292e08bf6d69b9b96e454a7',
    'src/core/native_head.cpp': '49911d2ebc83518964e07520f3271818f39be9e75eb8c7c4127fa2786549a345',
    'src/kernels/xe/native_mmvq.cpp': 'a351f613dc8e157f416d60939651c60280df889af6e6378b05d88012ab96d457',
    'src/prefill/prefill.cpp': 'c7a427dc58fd5f03732b4a807274f2cc5d406cd2f6c39d64a620c0cd3518cc67',
    'src/prefill/gemm.cpp': 'bfa40846fddb46a3d03d8c58ef5209b181c46674939beeefe512db0073c9d648',
    'src/kernels/xe/iq_kernels.cpp': '75f2ac1dfdb337c424bba19e62c34159a1bde547a776b9b5d6cf7975dfc29d50',
    'src/kernels/xe/xmx_gemm.cpp': '2d2144e7331b3dd66a2289a9c67b541b1c136332be53f31c935c9176880085d1',
    'include/strata/kernels/iq_kernels.hpp': '61513ccb71831d1065bd25829541c508adf3ae58e1e3df6d3bc0c9681cb54f6d',
    'include/strata/prefill/compact_moe.hpp': '433897666aa6c93d645ead09eba04d872e67bdbce3b7aa232f553168595fac72',
    'src/core/layer.cpp': 'fefc457de582573a8f1132c02139a61161d90fe50dcacda46c8c64cf0b581c97',
    'src/core/session.cpp': 'd914f75a19022edd29f6fec0fd4df85f1b24b09d56bf014744be79d12e914ab6',
}

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-research-round149-publish-root-v1'
    dest = W / 'bench/results/2026-10-12-parallel-round149'
    m.require(not out.exists() and not dest.exists(), 'new immutable publication')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             owner_sha256=m.sha(O), recipe_sha256=m.sha(__file__), gpu_executed=False,
             model_executed=False, research_only=True, original_reports_modified=False)

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands

    def run(label, argv, cwd=W):
        c, so, _ = owner.run(label, argv, ENV, cwd, wall=60, cpu=120, rss_cap=2 << 30,
                            text_cap=2 << 20, total_cap=16 << 20, file_cap=64 << 20)
        m.require(a.strong_closed_command(c) and c['exit_code'] == 0, 'normal closure ' + label)
        return so.read_text().strip()

    def write(name, value):
        (dest / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    save()
    try:
        m.require(not run('status', ['/usr/bin/git', 'status', '--porcelain']), 'clean report worktree')
        identity = run('identity', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(identity and all(x.startswith('file:/home/yayoi/.config/git/config') for x in identity.splitlines()), 'global identity')
        m.require(run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin']) == 'git@github.com:yqYo1/Strata.git', 'own fork')
        m.require(run('source-head', ['/usr/bin/git', 'rev-parse', 'HEAD'], S) == '39e395a103af36323262cdc0dbaea27cba81f3e4', 'immutable source')
        m.require(not run('source-status', ['/usr/bin/git', 'status', '--porcelain'], S), 'clean source')
        m.require(all(m.sha(S / p) == h for p, h in SOURCE.items()), 'Main verified report source pins')
        dest.mkdir()
        entries = []
        for name, wanted in REPORTS.items():
            original = B / 'research-20261009' / name
            digest = m.sha(original)
            m.require(digest == wanted, 'original report identity')
            (dest / name).write_bytes(original.read_bytes())
            m.require(m.sha(dest / name) == digest, 'byte-exact report archive')
            entries.append(dict(path=str(original), archive=str((dest / name).relative_to(W)), sha256=digest,
                                model='gpt-6-luna', main_full_report_review=True, implemented=False))
        prior = W / 'bench/results/2026-10-12-xe-compact-moe-gpu-parity/report-registry-v139.json'
        m.require(json.loads(prior.read_text())['research_completed'] == 586, 'committed registry count')
        write('report-registry-v140.json', dict(research_completed=590, prior_registry_path=str(prior.relative_to(W)),
              prior_registry_sha256=m.sha(prior), new_reports=entries, source_revision='39e395a103af36323262cdc0dbaea27cba81f3e4',
              source_pins=SOURCE, measured=False))
        write('main-review-and-limits.json', dict(original_reports_modified=False,
              full_reports_read=True, relevant_source_main_read=True, source_pins_independently_verified=True,
              first_distribution=dict(harness_unimplemented=True,
                  current_strict16_lp5_is_not_full_distribution_quality=True,
                  numeric_threshold_note='The source report does not establish an acceptance threshold. Its suggestion is not a permission requirement; existing user authorization covers measuring deltas and evaluating near-zero quality changes.'),
              full_context=dict(stock_gen_zero_output_refused=True,
                  legal_262143_plus1_last_processed_position=262142,
                  diagnostic_intends_exact262144_batched_rows=True,
                  physical_last_cell_write_readback_unmeasured=True,
                  full_extent_restore_unmeasured=True,
                  suggested_deadlines_unmeasured=True,
                  actual_memtotal_bytes=134986948608,
                  exact_memtotal_gib=134986948608 / (1 << 30),
                  configured_context262144_host_kv_bytes=6643777536),
              repeat_work=dict(native_dense_full_matrix_dequant_repeats_per_actual_invocation=True,
                  logical_bytes_not_measured_physical_traffic=True,
                  correction='R150 incorrectly combines route-skipping with H2D skipping in full_nonresident_stream. The sequence contains every nonresident expert; issue_until and the threaded issuer submit copies for sequence entries regardless of routed row count. Route-zero skips can avoid expert dequant/product without avoiding an already issued copy. Routed mode is distinct.',
                  source_evidence='src/prefill/prefill.cpp:1410 sequence construction, :1452 issue_until, :1505 threaded issuer; all at pinned39e source.',
                  native_ple_subbatch_note='Count actual Gemm::native calls inside PLE subbatches as well; chunk count alone does not establish PLE projection call count.',
                  dispatch_matched_microbenchmark_unimplemented=True),
              window_schedule=dict(proposed_bounded_window_unimplemented=True,
                  full_active_layer_bank_fit_unmeasured=True,
                  recurrent_layer_order_and_callback_transaction_unqualified=True,
                  raw_reference_links_mutable_commits_unpinned=True),
              gpu_executed=False, quality_qualified=False, full262144_qualified=False, restore_qualified=False))
        (dest / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        (dest / 'README.md').write_text('''# Read-only research, rounds149 and150

The first two reports propose a full-distribution teacher-forced quality hook and distinguish
ordinary zero-output admission from the full262144 diagnostic. Main read both reports
and checked their relevant source. Neither proposed contract is a runtime qualification.
[Main limits](main-review-and-limits.json) preserve the original reports and distinguish
unmeasured physical last-cell write/readback and RESTORE from source intent.
The next two count repeated weight transforms and propose a bounded layer-major window.
Main's correction distinguishes full-stream H2D copies from routed dequant/product visits;
route-zero skips do not establish skipped full-stream copies. All mechanisms remain unmeasured.

Research continues in parallel on repeated per-chunk work and safe partial weight reuse.
Main owns all controller tests and GPU measurements; the next8192/64K controller remains
source preparation until Main review and actual CPU qualification.
''')
        scan = out / 'gitleaks.json'
        run('secret-scan', [str(S.parent / 'fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks'),
                           'dir', str(dest), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'clean report scan')
        run('stage', ['/usr/bin/git', 'add', '--', str(dest.relative_to(W))])
        run('staged-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'docs(sycl): preserve quality and full-context research contracts'])
        r['committed_revision'] = run('head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('push', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean published worktree')
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'committed_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
