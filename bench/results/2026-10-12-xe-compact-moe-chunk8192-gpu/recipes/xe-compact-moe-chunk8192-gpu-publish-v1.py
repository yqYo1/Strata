"""Main: preserve serial three-case8192/64K diagnostics without speed adoption."""
import base64
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
S = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
L = Path('/home/yayoi/ghq/github.com/intel/llvm/.worktree/toolchain-sycl-v711-b570-20261011')
C = B / 'xe-compact-moe-chunk8192-controller-source-v1'
R = W / 'bench/results/2026-10-12-xe-compact-moe-chunk8192-gpu'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('chunk_gpu_publish_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(C))
import admission as a
import window_protocol as wp
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']
REPORTS = {
    'research-xe-chunk8192-numerical-seams-fixedprefix-quality-matrix-round153-v1.txt':
        'a33fd440be7e4a6f903675a701ca1f7ba534ba631dfecb2c7648779a7ebec1ee',
    'research-xe-routezero-fullstream-active-sequence-lifetime-contract-round153-v1.txt':
        'aa694a90685b71702fb64919c190c761ee61da561dcd1b0f32a1e59e5793fbca',
    'research-xe-llvm-v711-profiling-tag-copy-event-source-eligibility-round154-v1.txt':
        'b7caccbda685dcda0c6007adbb4a8d3162bc5e54dbd531cc0129d296c00565e4',
    'research-xe-zero-route-host-stager-producer-pinning-contract-round154-v1.txt':
        'c23681bc5710100ffec02a57d0d29be30bfc5740c08a40d815140fa01002297a',
    'research-xe-gdn-bounded-row-scratch-liveness-capacity-seam-round155-v1.txt':
        'e468a9b734b4ea674819f6d5b4686bfc26a9be462d201c3d67dac084a42c908f',
    'research-xe-free-prefill-expert-gemm-shape-roof-external-kernels-round155-v1.txt':
        '236b732c3aa43f889622453a954ac6c0e4ea857012c47f486ce8bf139c7fa584',
}
ENGINE_PINS = {
    'src/prefill/kernels.cpp': '0e89bbb83220ad85257d17fc6b98f2332a5c17e8d2832937a98a1e3ae300ab81',
    'src/kernels/xe/mma_gemm.cpp': '540a4366709b8a2caf43aae36bbaeefd21bf4290d5555557077c463754da2238',
    'CMakeLists.txt': '577ebbd766568558ba43a64960854cdbcae712cbe961491b50ac7c204887e775',
    'src/core/expert_source.cpp': '7915f5ceb462a5c14655bbe585408f72ef9abd581b95ed14f08d9533d20f2edc',
    'src/core/pinned.cpp': '1d2beba41fc8864cae2cf433dfe618725f6455c430809622ac9dda69d7575356',
    'include/strata/core/expert_source.hpp': 'd54b627e363124c6cee733053b9c513f3493ff13f4335b2d54404d57a3d8b2eb',
    'include/strata/core/pinned.hpp': '31bf40c8c12be0246a662fa92b47c77c471bf2779a54abf51377835c83818186',
    'src/kernels/xe/kv_q8.cpp': '3294bf29b48ccfcd7c8e10b5673534aafa70d6c7f4bc3cb765e7d8546120dda6',
    'src/kernels/xe/kv_stream.cpp': 'cd3bb7877dd848497a1d7d50cbcf75ae28cffdd187b5c18b6234b917520f7fbf',
    'src/kernels/xe/qsa.cpp': 'a6bf5966f862adb188bc1ece52ebe668d02a29c4fa10a4aca33593deb2611c6a',
}
LLVM_PINS = {
    'sycl/include/sycl/ext/oneapi/experimental/profiling_tag.hpp': '8bf94c6e3c5006bdacd042c78293894823170cbd7db3930c7448c2124a9a8bd3',
    'sycl/source/detail/event_impl.cpp': 'd0abccb3f209afb46810f949efd31b33212512ab19211028179305d24c6a17a6',
    'sycl/source/detail/scheduler/commands.cpp': 'd9d46ea4a3c90868fd4fb7172d3becf8d4fd18587dbaef8af50669791768ebf6',
    'sycl/unittests/Extensions/ProfilingTag.cpp': '2dbdd1b27d452cd6e6535f500d18353ea6ce737d8d102dbb727bbf0a3eab51f5',
    'unified-runtime/source/adapters/level_zero/v2/event.cpp': '35cb322613492ac825ce2759c5c89a57df88eb5546c0d38ef28655d20aee10c2',
    'unified-runtime/source/adapters/level_zero/v2/command_list_manager.cpp': '4e0ae5b72247a196668614836adbd5f3128ae8dc5d0f2da24b3db76a083af2a0',
    'unified-runtime/source/adapters/level_zero/v2/queue_batched.cpp': '1acaeb74da378f52786f561023d04bc49cc1fb140ac53ce4faf1c341e04332a2',
    'unified-runtime/source/adapters/native_cpu/event.cpp': 'b2e517a338cfd722d5e26cbb4e44b63afe547d0f709db3a035d0b2db01d40aea',
}

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-compact-moe-chunk8192-gpu-publish-root-v1'
    m.require(not out.exists() and not R.exists(), 'new immutable publication')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             recipe_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False,
             model_executed=False, original_results_modified=False, performance_eligible=False)

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands

    def run(label, argv, cwd=W):
        c, so, _ = owner.run(label, argv, ENV, cwd, wall=60, cpu=120, text_cap=8 << 20,
                            total_cap=32 << 20, rss_cap=2 << 30, file_cap=64 << 20)
        m.require(a.strong_closed_command(c) and c['exit_code'] == 0, 'normal closed ' + label)
        return so.read_text().strip()

    def write(name, value):
        p = R / name
        p.parent.mkdir(parents=True, exist_ok=True)
        m.require(not p.exists(), 'new archive')
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def copy(p, name, wanted=None):
        digest = m.sha(p)
        m.require(wanted is None or digest == wanted, 'original report identity')
        target = R / name
        target.parent.mkdir(parents=True, exist_ok=True)
        m.require(not target.exists(), 'new byte-exact archive')
        target.write_bytes(p.read_bytes())
        m.require(m.sha(target) == digest, 'exact copied report')
        return dict(source=str(p), archive=str(target.relative_to(W)), bytes=p.stat().st_size, sha256=digest)

    save()
    try:
        m.require(not run('status', ['/usr/bin/git', 'status', '--porcelain']), 'clean report worktree')
        identity = run('identity', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(identity and all(x.startswith('file:/home/yayoi/.config/git/config') for x in identity.splitlines()), 'global identity')
        m.require(run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin']) == 'git@github.com:yqYo1/Strata.git', 'own fork')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        qpath = B / 'xe-compact-moe-chunk8192-main-admission-root-v1/qualification.resolved.json'
        m.require(m.sha(qpath) == '1d077ebe12abcd8e2d5f18c630f06c82a9b9dea17868f66b3836e9f9cecf351e', 'actual qualification')
        q = json.loads(qpath.read_text())
        m.require(all(m.sha(C / name) == digest for name, digest in q['controller_source_pins'].items()), 'qualified controller closure')
        m.require(run('source-head', ['/usr/bin/git', 'rev-parse', 'HEAD'], S) == '39e395a103af36323262cdc0dbaea27cba81f3e4' and
                  not run('source-status', ['/usr/bin/git', 'status', '--porcelain'], S), 'clean exact source')
        m.require(all(m.sha(S / p) == h for p, h in q['production_source_pins'].items()), 'actual42 compiled source bytes')
        m.require(all(m.sha(S / p) == h for p, h in ENGINE_PINS.items()) and
                  all(m.sha(L / p) == h for p, h in LLVM_PINS.items()), 'independent Main report source pins')
        records = {}
        for n, chunk in a.SEQUENCE:
            p = a.root_path(B, n, chunk) / 'record.json'
            d = json.loads(p.read_text())
            a.require_normal_model(d, boot)
            m.require(d['passed'] and d['complete'] and not d['active'] and d['qualification_sha256'] == m.sha(qpath) and
                      d['source_pins'] == q['production_source_pins'] and d['published_revision'] == '39e395a103af36323262cdc0dbaea27cba81f3e4' and
                      d['source_guard_phases'] == ['admission', 'ready', 'done', 'final'] and
                      d['validation'] == wp.validate(d['request_protocol'], n=n, chunk=chunk) and
                      d['validation']['generated_tokens'] == 16 and d['performance_eligible'] is False and
                      not d['new_fault_messages'] and d['health-before_protocol'][:6] == d['health-after_protocol'][:6] and
                      d['retry_count'] == 0 and d['reset_executed'] is False, 'exact closed serial diagnostic')
            for command in d['commands'] + d['guard_commands']:
                m.require(a.strong_closed_command(command) and command['exit_code'] == 0, 'owned normal command')
                m.require(not any('TOKEN' in key or 'PASSWORD' in key or 'SECRET' in key for key in command['environment']), 'no credential environment')
            records[n, chunk] = (p, d)
        comparisons = []
        for n in (32768, 65536):
            current = records[n, 8192][1]
            refpath = Path(current['reference']['path'])
            m.require(m.sha(refpath) == current['reference']['sha256'], 'same-N immutable reference')
            before = json.loads(refpath.read_text())
            result = wp.output_deltas(before, current)
            m.require(result == current['reference_comparison_observation'] and result['strict_continuation_passed'], 'independent exact16/LP5 comparison')
            comparisons.append(dict(input_tokens=n, reference_receipt=str(refpath), reference_sha256=m.sha(refpath),
                result=result, diagnostic_prompt_ms=dict(chunk6144=before['validation']['prompt_ms'], chunk8192=current['validation']['prompt_ms']),
                diagnostic_decode_ms=dict(chunk6144=before['validation']['decode_ms'], chunk8192=current['validation']['decode_ms']),
                diagnostic_only=True, performance_eligible=False, quiet_repeat_required=True))
        R.mkdir()
        plain = out / 'plaintext-receipts'
        plain.mkdir()
        for (n, chunk), (p, d) in records.items():
            target = plain / f'n{n}-t{chunk}-g16-on.json'
            target.write_bytes(p.read_bytes())
            m.require(m.sha(target) == m.sha(p), 'exact plaintext scan inputs')
        scanner = str(S.parent / 'fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
        scan = out / 'gitleaks-plaintext.json'
        run('secret-scan-plaintext', [scanner, 'dir', str(plain), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'no secrets in receipt plaintext')
        artifacts = []
        cases = []
        for (n, chunk), (p, d) in records.items():
            raw = p.read_bytes()
            name = f'receipts/n{n}-t{chunk}-g16-on.json.lossless.json'
            write(name, dict(encoding='gzip+base64', source=str(p), decoded_bytes=len(raw), decoded_sha256=m.sha(p),
                             original_passed=d['passed'], data_base64=base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii')))
            archived = json.loads((R / name).read_text())
            m.require(gzip.decompress(base64.b64decode(archived['data_base64'], validate=True)) == raw, 'lossless receipt archive')
            artifacts.append(dict(source=str(p), archive=str((R / name).relative_to(W)), decoded_bytes=len(raw),
                                  decoded_sha256=m.sha(p), archive_sha256=m.sha(R / name)))
            cases.append(dict(input_tokens=n, chunk=chunk, compact='on', generated_tokens=16,
                windows=wp.windows(n, chunk), route_census_rows=len(d['route_census']['rows']),
                prompt_ms=d['validation']['prompt_ms'], decode_ms=d['validation']['decode_ms'],
                ready_free_vram_mib=d['memory_admission']['ready_post_init_free_mib'],
                boot_id=boot, normal_quit=True, fresh_integer_health_before_after=True, kernel_faults=[],
                source_guards=d['source_guard_phases'], physical_peak_unmeasured=True, performance_eligible=False))
        reports = []
        for name, wanted in REPORTS.items():
            item = copy(B / 'research-20261009' / name, 'research/' + name, wanted)
            reports.append(dict(path=item['source'], archive=item['archive'], sha256=item['sha256'], model='gpt-6-luna',
                                main_full_report_review=True, implemented=False))
        prior = W / 'bench/results/2026-10-12-xe-compact-moe-chunk8192-controller/report-registry-v141.json'
        m.require(json.loads(prior.read_text())['research_completed'] == 594, 'committed registry')
        write('report-registry-v142.json', dict(research_completed=600, prior_registry_path=str(prior.relative_to(W)),
            prior_registry_sha256=m.sha(prior), new_reports=reports,
            repeated_R149_after_new_R153_assignment_count_increment=0,
            repeated_R149_original_sha256='a7283662d0b25f947b8700b3a1454cf5cc79fa62425583708cbd909dd8e9fd3b',
            scope='Six new reviewed R153/R154/R155 source-only reports; no measurements counted as research.'))
        write('main-research-corrections.json', dict(original_reports_modified=False, engine_pins=ENGINE_PINS, llvm_pins=LLVM_PINS,
            source_pins_independently_verified=True,
            R153_routezero=dict(host_staging_suppression_unqualified=True,
                correction='R154 supersedes the host-staging part of R153: skipping wait/H2D does not stop jobs already started before route counts. Source-correct job suppression requires filtering before workers claim jobs; per-layer immutable generations and explicit completion/lifetime edges are unimplemented.',
                current_arena_host_usm_direct_source=True, current_transient_stager_not_inferred=True),
            quality=dict(standalone_stop_eos_guard_confirmed=True,
                full_distribution_comparison_unimplemented=True,
                permission_note='A report suggestion about an owner-approved numeric cutoff is not a permission requirement. Existing user authorization covers measuring and evaluating near-zero changes; current16-ID/LP5 parity does not establish broad quality.'),
            R155_gdn=dict(bounded_row_implementation_unimplemented=True,
                shared_region_delta_is_difference_of_maxima=True, persistent_state_remains_fp32=True,
                short_convolution_semantics_need_preservation=True, external_fla_refs_not_commit_pinned=True),
            R155_gemm=dict(source_prefill_hash_typo_in_original=True,
                corrected_prefill_sha256=q['production_source_pins']['src/prefill/prefill.cpp'],
                top_k=10, MS=32, NS=64, WM=128, WN=256, NSG=16, local_work_items=256,
                correction='R155 grouped workgroup count is overcounted16x and128 subgroups/workgroup is incorrect. Actual NSG=(128/32)*(256/64)=16; global range is G*tiles_m*nN*16*16 workitems, local256, giving G*tiles_m*nN workgroups. Distinguish workitems from workgroups.',
                launched_grouped_workgroups='G*ceil(max_rows/128)*ceil(N/256)',
                live_matrix_row_slots='sum(ceil(rows_e/128)*128 for rows_e>0); unused grid workgroups return before matrix products',
                maximum_grid_reservation_not_actual_padded_FLOPs=True,
                dispatch_matched_measurement_pending=True,
                external_ark_pr_not_runtime_evidence=True,
                measurement_authorization='Existing user goal authorizes measurements; the report phrase separately authorized creates no new approval step.'),
            profiling=dict(actual_llvm_files_inspected=True, llvm_base='504366f4b82ff00bbac7b0c956635eed8bc599d8',
                llvm_worktree_has17_local_patches=True, inspected_profiling_files_base_unchanged=True,
                native_tag_source_eligible_without_queue_profiling=True,
                tag_start_end_are_point_not_work_duration=True,
                actual_tag_runtime_probe_pending=True, actual_memcpy_profiling_unqualified=True,
                current_compute_spans_not_exclusive_H2D_or_kernel_service=True)))
        write('results.json', dict(source_revision='39e395a103af36323262cdc0dbaea27cba81f3e4', source_pins=q['production_source_pins'],
            qualification_sha256=m.sha(qpath), cases=cases, comparisons=comparisons, artifacts=artifacts,
            physical8192_init_and_bounded_gen_succeeded=True, physical_peak_or_allocation_limit_unmeasured=True,
            default_compact_on=False, arithmetic_precision_changed=False, full262144_qualified=False,
            restore_qualified=False, broad_quality_qualified=False, performance_eligible=False,
            next_step='Main review/CPU qualify returned quiet64K+coarse-phase controller source; serial balanced quiet repeated samples and separate native-tag sanity/coarse-phase diagnostic. Keep official independent parameter optimization TODO.'))
        copy(Path(__file__), 'recipes/' + Path(__file__).name)
        (R / 'README.md').write_text('''# CompactMoE8192/64K GPU diagnostics

All three serial cases passed:32768/chunk8192/on,65536/chunk6144/on,
65536/chunk8192/on. Same-N8192 and6144 emitted identical16 IDs and displayed LP5.
Each case closed by normal QUIT, passed fresh integer GPU health before/after,
source/runtime mapping guards and a fresh kernel fault gate without reset or retry.
READY observed713MiB free for8192 and1286MiB for6144 with owned stage buffers.
This proves these bounded invocations fit; it is not a physical peak/maximum-fit test.

[Results](results.json) retain individual diagnostic times, settings, source/input/
binary/runtime provenance, exact equality and fault/closure evidence. All timings
include diagnostic logging/validation and are performance-ineligible. Compact
remains defaultOFF; quiet repeated performance, broad full-distribution quality,
full262144 physical occupancy and RESTORE remain unqualified.

Receipts use byte-verified gzip+base64, scanned as exact plaintext before encoding.
Original private receipts stay for current reference/controller consumers. Successful
full logs should be retired after extraction/publication and review; no deletion
or active-run mutation is performed here. Researchers continue distinct source/
external-kernel and bounded-GDN questions; a separate Sol implementer prepares
quiet64K/coarse-phase controller source only. Main owns all tests and GPU work.

本家単体の最適設定探索・独立検証は未完了。
[TODO](../2026-10-11-upstream-parameter-audit/TODO.md)を維持する。
''')
        scan = out / 'gitleaks-archive.json'
        run('secret-scan-archive', [scanner, 'dir', str(R), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'no archive secrets')
        run('stage', ['/usr/bin/git', 'add', '--', str(R.relative_to(W))])
        run('staged-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'docs(sycl): record compact8192 and64K GPU parity'])
        r['committed_revision'] = run('head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('push', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean published boundary')
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
