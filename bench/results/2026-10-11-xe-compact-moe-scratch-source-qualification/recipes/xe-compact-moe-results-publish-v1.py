"""Main: preserve actual source/CPU qualification and reviewed conditional research."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
S = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
P = S.parent / 'fix-xestrata-e32-host-handoff-20261011'
R = W / 'bench/results/2026-10-11-xe-compact-moe-scratch-source-qualification'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
with O.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('report_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'original owner identity')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']
REPORTS = [
    'research-official-xe-phase-separated-parameter-search-contract-round141-v1.txt',
    'research-xe-layer-major-streamed-int8-qsa-contract-round141-v1.txt',
    'research-xe-layer-major-current-port-insertion-design-round142-v1.txt',
    'research-xe-grouped-gemm-row-tiles-padding-diagnostics-round142-v1.txt',
    'research-xe-post-compact-gdn-qsa-owned-arena-reduction-round143-v1.txt',
    'research-xe-v711-b570-gpu-profiler-admission-critical-path-round143-v2.txt',
    'research-xe-gdn-qkv-y-alias-variant-lifetime-proof-round144-v1.txt',
    'research-xe-vtune-phase-marker-stdio-supervision-contract-round144-v1.txt',
    'research-xe-qsa-kv-projection-attention-scratch-lifetime-matrix-round145-v1.txt',
    'research-xe-native-kernel-eligible-roofline-accounting-gap-round145-v1.txt',
]


def aligned(payload):
    return (payload + 256 + 255) & ~255


with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-compact-moe-results-publish-root-v1'
    m.require(not out.exists() and not R.exists(), 'immutable new report boundary')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             recipe_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False,
             model_executed=False, performance_eligible=False, adopted=False)

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands

    def run(label, argv, cwd=W):
        c, so, _ = owner.run(label, argv, ENV, cwd, wall=60, cpu=120, text_cap=8 << 20,
                             total_cap=32 << 20, rss_cap=2 << 30, file_cap=64 << 20)
        m.require(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] == 0, 'closed ' + label)
        return so.read_text().strip()

    def copy(source, target):
        m.require(not target.exists(), 'new archive ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        m.require(m.sha(target) == m.sha(source), 'byte-identical archive')
        return dict(source=str(source), archive=str(target.relative_to(W)), bytes=target.stat().st_size,
                    sha256=m.sha(source))

    def write(name, value):
        target = R / name
        m.require(not target.exists(), 'new compact evidence ' + name)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    save()
    try:
        m.require(not run('initial-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean report worktree')
        identity = run('identity', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(identity and all(v.startswith('file:/home/yayoi/.config/git/config') for v in identity.splitlines()), 'global identity only')
        m.require(run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin']) == 'git@github.com:yqYo1/Strata.git', 'own fork only')
        pub_path = B / 'xe-compact-moe-source-publish-root-v2/record.json'
        pub = json.loads(pub_path.read_text())
        m.require(pub['passed'] and pub['complete'] and not pub['active'] and
                  all(m.closed(c) and c.get('session_anchored') is True for c in pub['commands']), 'actual source publication')
        m.require(run('source-head', ['/usr/bin/git', 'rev-parse', 'HEAD'], S) == pub['published_head'] and
                  not run('source-status', ['/usr/bin/git', 'status', '--porcelain'], S), 'clean published source')
        R.mkdir(parents=True)
        preserved = []
        roots = [('host', 1, False), ('host', 2, True), ('free', 2, True), ('contrib-llvm', 2, True),
                 ('free-asan', 2, True), ('contrib-llvm-asan', 2, True), ('lint', 1, False), ('lint', 2, True)]
        for phase, version, passed in roots:
            source = B / f'xe-compact-moe-{phase}-qualification-root-v{version}/record.json'
            v = json.loads(source.read_text())
            m.require(v['passed'] is passed and not v['active'], 'preserve original result ' + str(source))
            if passed:
                m.require(v['complete'], 'actual completed qualification')
                if phase not in ('host', 'lint'):
                    m.require(v['source_stable'] and v['source_pins'] == pub['source_pins'] and
                              m.sha(Path(v['binary']['path'])) == v['binary']['sha256'], 'current qualified build bytes')
            m.require(all(m.closed(c) and c.get('session_anchored') is True for c in v['commands']), 'all historical command closures')
            preserved.append(dict(**copy(source, R / 'receipts' / (source.parent.name + '.json')), original_passed=passed))
        for version, passed in [(1, False), (2, True)]:
            source = B / f'xe-compact-moe-model-controller-cpu-root-v{version}/record.json'
            value = json.loads(source.read_text())
            m.require(value['passed'] is passed and not value['active'] and all(m.closed(c) and
                      c.get('session_anchored') is True for c in value['commands']), 'original controller CPU status and closure')
            preserved.append(dict(**copy(source, R / 'receipts' / (source.parent.name + '.json')), original_passed=passed))
        for source in [pub_path, B / 'xe-compact-moe-applied-main-review-v1.json', B / 'xe-compact-moe-applied-main-review-v2.json',
                       B / 'xe-compact-moe-lint-output-main-review-v2.json', B / 'xe-compact-moe-source-publication-main-review-v2.json',
                       B / 'xe-compact-moe-model-controller-main-review-v1.json']:
            preserved.append(copy(source, R / 'receipts' / (source.name if source.name != 'record.json' else source.parent.name + '.json')))
        recipes = ['xe-compact-moe-main-qualification-v1.py', 'xe-compact-moe-main-qualification-v2.py',
                   'xe-compact-moe-lint-qualification-v1.py', 'xe-compact-moe-lint-qualification-v2.py',
                   'xe-compact-moe-source-publish-v1.py', 'xe-compact-moe-source-publish-v2.py',
                   'xe-compact-moe-controller-cpu-qualification-v1.py', 'xe-compact-moe-controller-cpu-qualification-v2.py']
        for name in recipes:
            preserved.append(copy(B / name, R / 'recipes' / name))
        preserved.append(copy(Path(__file__), R / 'recipes' / Path(__file__).name))
        for name in ['implementation-xe-prefill-row-bounded-scratch-source-v1.txt', 'implementation-xe-compact-moe-model-controller-source-v1.txt',
                     'implementation-xe-compact-moe-model-controller-source-v2.txt']:
            preserved.append(copy(B / 'research-20261009' / name, R / 'implementation' / name))
        for version in [1, 2]:
            for source in sorted((B / f'xe-compact-moe-model-controller-source-v{version}').iterdir()):
                m.require(source.is_file(), 'immutable source-only controller directory')
                preserved.append(copy(source, R / f'implementation/controller-source-v{version}' / source.name))
        review = json.loads((B / 'xe-compact-moe-applied-main-review-v2.json').read_text())
        m.require(review['source_pins'] == pub['source_pins'] and all(m.sha(S / p) == h for p, h in pub['source_pins'].items()), 'same 42 published/built source pins')
        r['source_publication'] = dict(path=str(pub_path), sha256=m.sha(pub_path), published_revision=pub['published_head'])
        run('published-source-diff', ['/usr/bin/git', 'diff', '--binary', pub['historical_build_base'], pub['published_head']], S)
        preserved.append(copy(out / 'commands/published-source-diff.stdout', R / 'source.patch'))
        write('source-and-binaries.json', dict(historical_build_base=pub['historical_build_base'], published_revision=pub['published_head'],
            source_pins=pub['source_pins'], source_tree=str(S), original_build_banner_not_rebuilt_after_commit=True,
            opt_in_environment='STRATA_PREFILL_COMPACT_MOE_SCRATCH', default_on=False, performance_eligible=False,
            gpu_correctness_qualified=False, physical_vram_qualified=False, full262144_qualified=False, restore_qualified=False,
            original_failed_host_v1='Main command used C++17 for a C++20 publication header; original result stays FAILED',
            original_failed_lint_v1='Shared compiler warning canonical path compared inconsistently; original result stays FAILED',
            controller_source_v1='SOURCE ONLY; actual lint consumer used wrong source_review_sha256 field; separately corrected source-v2 passed actual CPU-root-v2',
            original_failed_controller_cpu_v1='Outer Main RLIMIT_FSIZE8MiB prevented nested unchanged owner from setting64MiB hard limit; original FAILED. CPU-root-v2 changes only outer file cap, not controller source or text/artifact budgets.',
            artifacts=preserved))
        reports = []
        for name in REPORTS:
            entry = copy(B / 'research-20261009' / name, R / 'research' / name)
            reports.append(dict(path=entry['source'], archive=entry['archive'], sha256=entry['sha256'],
                                main_full_report_review=True, model='gpt-6-luna', implemented=False))
        T, C, ZV, HV, cap, max_blocks, sel_batch, attn_batch = 8192, 10240, 6144, 48, 2051, 65538, 256, 32
        gdn = [T * C * 4, T * ZV * 4, T * 2 * HV * 4, T * HV * 4, T * HV * 4, T * C * 4, T * ZV * 4, T * ZV * 2]
        scratch_payload = attn_batch * (((cap + 63) // 64) * 24 * (256 + 2) + 64) * 4
        qsa = [T * 512 * 4, T * 512 * 4, T * 12288 * 4, T * ZV * 4, T * 128 * 4, T * 512 * 4, T * ZV * 4,
               T * ZV * 2, T * cap * 4, sel_batch * max_blocks * 4, scratch_payload]
        gdn_original, qsa_original = sum(map(aligned, gdn)), sum(map(aligned, qsa))
        gdn_alias = gdn_original - aligned(T * ZV * 4)
        qsa_alias_two_takes = qsa_original - aligned(scratch_payload)
        qsa_alias_merged = qsa_alias_two_takes - (aligned(qsa[0]) + aligned(qsa[1]) - aligned(qsa[0] + qsa[1]))
        threshold = (scratch_payload + 4095) // 4096
        m.require((gdn_original, qsa_original, gdn_alias, qsa_alias_merged, threshold) ==
                  (1180698624, 1120979712, 979371776, 1094816000, 6388), 'integer calculation matches actual Alloc take policy')
        write('main-research-corrections-v1.json', dict(main_reviewed=True, original_reports_modified=False,
            measured_results_modified=False, source_only=True, source_pins={p:m.sha(S / p) for p in
                ['src/prefill/prefill.cpp', 'src/prefill/kernels.cpp', 'src/kernels/xe/xmx_gemm.cpp', 'src/kernels/xe/qsa_decode_attn.cpp']},
            grouped_xmx_workgroups='G*ceil(max_rows/128)*ceil(N/256); NSG counts subgroups per workgroup and cancels in global/local division',
            resident_iq_scope='FUSED0 recipe makes fuse_res false and nf=0; conditional routed/resident counts in R142 are not observed resident-IQ dispatch in current samples',
            footprint=dict(allocator='A(payload)=round_up256(payload+256)', conditional_geometry=dict(T=T,C=C,ZV=ZV,HV=HV,cap=cap,max_blocks=max_blocks,sel_batch=sel_batch,attn_batch=attn_batch),
                gdn_payloads=gdn, qsa_payloads=qsa, gdn_takes=8, qsa_takes=11, gdn_original=gdn_original,
                qsa_raw_payload=sum(qsa), qsa_aligned_requested=qsa_original, y_allocation_removed=aligned(T*ZV*4), gdn_after_y_alias=gdn_alias,
                shared_after_gdn_only=qsa_original, gdn_only_shared_reduction=gdn_original-qsa_original,
                attention_scratch_payload=scratch_payload, attention_scratch_allocation=aligned(scratch_payload),
                qsa_after_scratch_alias_keep_two_takes=qsa_alias_two_takes, qsa_after_merge_two_projections_and_scratch_alias=qsa_alias_merged,
                combined_shared_reduction_with_merge=gdn_original-qsa_alias_merged, conservative_payload_only_threshold=threshold,
                scope='Conditional source-request bytes only; no physical VRAM or fit/speed measurement. All GDN/QSA aliases unimplemented.'),
            qsa_pointer_correction='R145 retaining original two takes differs from R143 merging them. Merge saves a further256B but moves Vc to Kc+T*512; retain logical extents and kernel values, not original byte addresses. Do not count allocator guards as payload.',
            profiler_campaign_scope='Eight quiet32K samples are four OFF/four ON;260.0258/7.8578 are OFF subgroup means.',
            h2d_scope='6.446834GB/s came from isolated HostUSM H2D microbenchmark, not measured H2D time in the eight quiet model samples.',
            native_bf16_scope='Recorded quiet/model recipes do not include --native-bf16. R145 flag discussion is conditional, not evidence that this CLI flag ran; inspect actual per-role weight formats and dispatch.',
            logical_byte_roof_scope='Logical dequant writes/reads divided by380GB/s is an illustrative quotient, not a guaranteed DRAM latency floor: cache-served reads and physical transactions must be established separately.'))
        prior = W / 'bench/results/2026-10-11-xe-quiet32k-balanced-campaign/report-registry-v137.json'
        m.require(json.loads(prior.read_text())['research_completed'] == 570, 'actual prior registry')
        write('report-registry-v138.json', dict(research_completed=570+len(reports), prior_registry_path=str(prior.relative_to(W)),
            prior_registry_sha256=m.sha(prior), new_reports=reports, scope='Ten R141–R145 full Main-reviewed read-only reports; corrections separate; implementation source reports not counted.'))
        snapshot_path = B / 'xe-compact-moe-qualification-agent-snapshot-v1.json'
        preserved.append(copy(snapshot_path, R / 'agent-snapshot-v1.json'))
        (R / 'README.md').write_text('''# MoE temporary-storage source and CPU qualification

The opt-in compact MoE scratch implementation was committed and pushed to the own fork.
Default remains OFF. FP16 gathered inputs/activation output and FP32 GU reuse at most T rows;
the route-sized FP32 Down output and all arithmetic precision remain unchanged.

Both free and contrib-llvm full builds passed with WERROR. Each mode's release and
address/undefined host-sanitizer CPU selection passed13 tests and skipped the AVX512-only
expert_multi_test. Standalone compact routing/count/release fixtures passed release and
ASan/UBSan. These checks did not execute a GPU or model. Actual both-mode mandatory lint
has no new findings against the unchanged parent; existing warnings remain.

Original failed C++17 host-command and original failed lint-path comparison receipts stay
FAILED. Their separate corrections change only Main's qualification commands/comparison.
The actual source content equals all42 pins of the uncommitted builds; publication did
not rebuild their banners. GPU output/lifetime parity, physical VRAM/8192 fit, performance,
RESTORE and full262144 occupancy are still unqualified. The separately corrected source-v2
diagnostic controller passed actual CPU admission/prerequisite/output/QUIT fixtures and
CLI refusal checks. The failed outer-file-limit CPU-root-v1 is preserved. Main next resolves
the actual admission/publication bridge, then compares N64,2049,32768 OFF/ON at actual6144.

[Individual actual receipts](receipts/) preserve commands, explicit environments,
CPU outputs, toolchain/binary/source identities and owners. [Separate Main corrections](main-research-corrections-v1.json)
fix research arithmetic and workload scopes without rewriting reports or measurements.
The source-only GDN/QSA lifetime aliases and layer-major proposals are not implemented.

本家の測定は箱出し設定だけではないが、本家単体の最適設定探索は未完了。
prefillとdecodeを個別に選択し、32K以上の入力で選択後の独立検証を行うTODOを維持する。
''')
        todo = W / 'bench/results/2026-10-11-upstream-parameter-audit/TODO.md'
        todo.write_text(todo.read_text() + '\n[R141のphase別設定探索案](../2026-10-11-xe-compact-moe-scratch-source-qualification/research/research-official-xe-phase-separated-parameter-search-contract-round141-v1.txt)を追加した。prefillの固定4096/6144/8192と標準auto方針、decodeの独立した設定選択と選択後の3回以上の検証を区別する。これは未実行の選定案であり、本家の最適化完了ではない。\n')
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        scan = out / 'gitleaks-redacted.json'
        run('secret-scan', [str(P / '.lint/bin/gitleaks'), 'dir', str(R), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'no credentials archived')
        run('stage', ['/usr/bin/git', 'add', '--', str(R.relative_to(W)), str(todo.relative_to(W))])
        run('staged-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'docs(sycl): record compact MoE source and CPU qualification'])
        r['committed_revision'] = run('head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('push-own-fork', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean published report boundary')
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','committed_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
