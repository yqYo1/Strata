"""Main: publish the actual CPU/admission boundary before any new GPU case."""
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
spec = importlib.util.spec_from_file_location('chunk_publish_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(B / 'xe-compact-moe-model-controller-source-v2'))
import admission as a

ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']
REPORTS = {
    'research-xe-closed32k-quant-route-repeat-work-census-round151-v1.txt':
        '4680a138c2cc0a67bb5016c90cde84930f3a1b2e49d240dc9ea43256b37c3c5f',
    'research-xe-bounded-window-layer-bank-authority-lifetime-seam-round151-v1.txt':
        '2a5d0f556c5fa7dc65dba4f3a6e215d74545f126961e2aee58360836a7f6f11e',
    'research-xe-existing-oracle-follow-native-quality-dump-contract-round152-v1.txt':
        'e55177ba72019fbfa83abc7322ccd3927bb818b713edb2e69728e47af8d132fc',
    'research-xe-current-prefill-phase-event-critical-path-observer-contract-round152-v1.txt':
        'ba85feea559cf99766befc6d3ab818618d08f7175c2c59c6bd70344715791857',
}
EXTRA_SOURCE = {
    'src/core/gpu.cpp': '9792966f6b2c84434a1dddafefb3286c3c5fcfa78e23065b455904f9cf6fcfb8',
    'src/core/device.cpp': '3e7523a2947af7c82d41fc9a7653f4f9aad0fa3cbb062999de19553659e0ec27',
    'include/strata/core/gpu.hpp': '54b1a10524c9ee0c9b42979d75bd4f775ee0c45da2839d8129c2a2bd3daa552a',
}

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-compact-moe-chunk8192-controller-publish-root-v1'
    dest = W / 'bench/results/2026-10-12-xe-compact-moe-chunk8192-controller'
    m.require(not out.exists() and not dest.exists(), 'new immutable publication')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             owner_sha256=m.sha(O), recipe_sha256=m.sha(__file__), gpu_executed=False,
             model_executed=False, original_results_modified=False, artifacts=[])

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
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def copy(p, name, wanted=None):
        digest = m.sha(p)
        if wanted is not None:
            m.require(digest == wanted, 'source identity ' + str(p))
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
        m.require(m.sha(target) == digest, 'byte-exact archive')
        entry = dict(path=str(p), archive=str(target.relative_to(W)), bytes=p.stat().st_size,
                     sha256=digest)
        r['artifacts'].append(entry)
        return entry

    save()
    try:
        m.require(not run('status', ['/usr/bin/git', 'status', '--porcelain']), 'clean report worktree')
        identity = run('identity', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(identity and all(x.startswith('file:/home/yayoi/.config/git/config') for x in identity.splitlines()), 'global identity')
        m.require(run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin']) == 'git@github.com:yqYo1/Strata.git', 'own fork')
        m.require(run('source-head', ['/usr/bin/git', 'rev-parse', 'HEAD'], S) == '39e395a103af36323262cdc0dbaea27cba81f3e4', 'immutable source')
        m.require(not run('source-status', ['/usr/bin/git', 'status', '--porcelain'], S), 'clean source')
        m.require(all(m.sha(S / p) == h for p, h in EXTRA_SOURCE.items()), 'extra report source pins')
        cpu = B / 'xe-compact-moe-chunk8192-controller-cpu-root-v1/record.json'
        cv = B / 'xe-compact-moe-chunk8192-main-admission-root-v1/record.json'
        for p in (cpu, cv):
            d = json.loads(p.read_text())
            m.require(d['passed'] and d['complete'] and not d['active'], 'actual closed CPU/admission receipt')
            m.require(all(a.strong_closed_command(c) and c['exit_code'] == 0 for c in d['commands']), 'owned CPU commands')
        q = B / 'xe-compact-moe-chunk8192-main-admission-root-v1/qualification.resolved.json'
        m.require(m.sha(q) == '1d077ebe12abcd8e2d5f18c630f06c82a9b9dea17868f66b3836e9f9cecf351e', 'actual admission identity')
        m.require(not (B / 'xe-compact-moe-chunk8192-root-v1').exists(), 'GPU namespace still unrun')
        dest.mkdir()
        review = json.loads((B / 'xe-compact-moe-chunk8192-controller-main-review-v1.json').read_text())
        m.require(review['main_reviewed'], 'Main review')
        for name, pin in review['all_fifteen_file_pins'].items():
            wanted = pin if isinstance(pin, str) else pin['sha256']
            copy(B / 'xe-compact-moe-chunk8192-controller-source-v1' / name, 'controller/' + name, wanted)
        for name in ('xe-compact-moe-chunk8192-controller-cpu-qualification-v1.py',
                     'xe-compact-moe-chunk8192-main-admission-v1.py',
                     'xe-compact-moe-chunk8192-controller-main-review-v1.json'):
            copy(B / name, name)
        for directory, names in {
            'xe-compact-moe-chunk8192-controller-cpu-root-v1': ['record.json', 'matrix.closed.json', 'negative_cli.py'],
            'xe-compact-moe-chunk8192-main-admission-root-v1': ['record.json', 'qualification.resolved.json', 'publication.bridge.resolved.json'],
        }.items():
            for name in names:
                copy(B / directory / name, directory + '/' + name)
        copy(B / 'research-20261009/implementation-xe-compact-moe-chunk8192-controller-source-v1.txt',
             'implementation-report.txt', 'b03826467f05c590d8336f18d52d5c25e7258a00326e8a329db969fc6288dbd0')
        entries = []
        for name, wanted in REPORTS.items():
            entry = copy(B / 'research-20261009' / name, 'research/' + name, wanted)
            entry.update(model='gpt-6-luna', main_full_report_review=True, implemented=False)
            entries.append(entry)
        repeated = copy(B / 'research-20261009/research-xe64k-quality-hook-r149-repeated-return-20261012-v1.txt',
                        'research/research-xe64k-quality-hook-r149-repeated-return-20261012-v1.txt',
                        '15956ef3587c2176a42358882985b3cdbf0b3e56a0a4c1aa2bbd6386c8fa6e61')
        m.require(m.sha(B / 'research-20261009/research-xe64k-teacher-forced-first-distribution-hook-source-contract-round149-v1.txt') ==
                  'a7283662d0b25f947b8700b3a1454cf5cc79fa62425583708cbd909dd8e9fd3b', 'R149 original restored unchanged')
        prior = W / 'bench/results/2026-10-12-parallel-round149/report-registry-v140.json'
        m.require(json.loads(prior.read_text())['research_completed'] == 590, 'prior committed registry')
        write('report-registry-v141.json', dict(research_completed=594, prior_registry_path=str(prior.relative_to(W)),
              prior_registry_sha256=m.sha(prior), new_reports=entries, repeated_return=repeated,
              repeated_return_count_increment=0, source_revision='39e395a103af36323262cdc0dbaea27cba81f3e4',
              next_assignments=dict(round153_routezero='research_routezero_fullstream_round153',
                                    round153_quality_seams='research_pass_reduction_scheduling_round135',
                                    model='gpt-6-luna', status='assigned_after_final_and_full_review'), measured=False))
        write('main-review-and-corrections.json', dict(original_reports_modified=False,
              report_full_reads=True, extra_source_pins=EXTRA_SOURCE,
              controller_cpu_actual_pass=True, admission_actual_pass=True, gpu_executed=False,
              qualification_sha256=m.sha(q),
              cases=['32768/8192/compact-on versus closed32768/6144/on',
                     '65536/6144/compact-on fresh control', '65536/8192/compact-on versus same-N6144 control'],
              physical8192_fit_unmeasured=True, full262144_qualified=False, restore_qualified=False,
              broad_quality_qualified=False, performance_eligible=False, compact_default='off',
              route_work=dict(logical_bytes_not_physical_traffic=True,
                  zero_route_full_stream_copy_bytes=16473728000, five_full_bank_bytes=5 * 50292326400,
                  corrected_percent_of_five_full_banks=100 * 16473728000 / (5 * 50292326400),
                  correction='R151 3.25% of five whole512-bank passes is arithmetically incorrect; it is about6.55%. Its6.59% of nonresident passes uses a different denominator.'),
              bank_reference=dict(unimplemented=True,
                  correction='39e was pushed to yqYo1/Strata. Original MistVVK/XeStrata commit links in R151 do not establish remote availability.',
                  immutable_url='https://github.com/yqYo1/Strata/tree/39e395a103af36323262cdc0dbaea27cba81f3e4'),
              quality=dict(existing_native_oracle_follow_dump_can_supply_full_rows_conditionally=True,
                  no_new_engine_hook_required_for_mtp_off=True, caller_validation_unimplemented=True,
                  correction='R152 EOS statement crosses paths: standalone native generation at generate.cpp10805 checks EOS only when o.stop_eos is true. Defaultfalse does not truncate this loop for an EOS ID. Other serve/batch checks are distinct.',
                  mtp4_qualified=False, full_distribution_runtime_unmeasured=True),
              timing=dict(current_phases_are_adjacent_compute_marker_spans=True,
                  not_per_kernel_or_H2D_service=True, current_presence_value_zero_enables=True,
                  proposed_observer_unimplemented=True)))
        (dest / 'README.md').write_text('''# Three-case compactMoE8192 controller: CPU/admission boundary

Main reviewed the returned controller and ran its actual CPU fixtures, four owned
normal-QUIT children, prerequisite/negative CLI checks and the source/admission
converter. Both closed receipts pass. No GPU case has been launched at this boundary.
The source snapshot retains pending templates unchanged; the separate resolved
qualification is the actual launch input. Existing failed and passed prerequisites
remain unchanged. Source39e keeps compact scratch off by default.

Next run exact three cases in order:32768/chunk8192/on,65536/chunk6144/on,
65536/chunk8192/on. The controller owns finite deadlines and validates progress,
normal exit, fresh GPU health/faults and exact16-ID/LP5 continuation against each
same-N6144 reference. It records numerical differences before rejecting them;
changed-chunk numerical rejection is not evidence of a GPU hang or broad quality.
8192 physical fit, quiet throughput, full262144, RESTORE and full-distribution
quality remain unmeasured. Diagnostic logs are not quiet performance samples.

R151/R152 originals are preserved with separate Main corrections. A repeated
R149 return that overwrote a private historical filename was preserved separately;
the historical private report was restored byte-exact from its committed original,
as recorded in the converter receipt. This duplicate adds zero to the registry count.
Research continues with two distinct read-only Luna R153 scopes while Main runs
the three GPU cases serially. Main owns all testing and implementation review.
''')
        copy(Path(__file__), Path(__file__).name)
        write('artifact-manifest.json', r['artifacts'])
        scan = out / 'gitleaks.json'
        run('secret-scan', [str(S.parent / 'fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks'),
                           'dir', str(dest), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'clean exact archive scan')
        run('stage', ['/usr/bin/git', 'add', '--', str(dest.relative_to(W))])
        run('staged-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'docs(sycl): qualify serial compact8192 controller on CPU'])
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
