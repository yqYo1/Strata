"""Main CPU-only conversion of actual publication/build proofs for bounded diagnostics."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
S = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
C = B / 'xe-compact-moe-model-controller-source-v2'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('admission_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(C))
import admission as a
from protocol import validate

ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           PYTHONDONTWRITEBYTECODE='1')

def load(path):
    a.safe_record_path(path)
    m.require(path.stat().st_size <= 8 << 20, 'bounded explicit receipt')
    return json.loads(path.read_text())

def stat_identity(path):
    v = path.stat()
    return dict(device=v.st_dev, inode=v.st_ino, bytes=v.st_size, mtime_ns=v.st_mtime_ns, ctime_ns=v.st_ctime_ns)

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-compact-moe-main-admission-converter-root-v1'
    m.require(not out.exists(), 'immutable new Main admission conversion')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             recipe_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False,
             model_executed=False, queue_created=False, performance_eligible=False, adopted=False,
             scope='CPU-only verification and separate schema conversion; no GPU outcome is inferred.')

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands

    def run(label, argv):
        e, so, _ = owner.run(label, argv, ENV, S, wall=30, cpu=60, text_cap=2 << 20,
            total_cap=16 << 20, rss_cap=2 << 30, file_cap=64 << 20)
        m.require(a.strong_closed_command(e) and e['exit_code'] == 0, 'normal closed CPU proof ' + label)
        return so.read_text()

    save()
    try:
        review_path = B / 'xe-compact-moe-applied-main-review-v2.json'
        m.require(m.sha(review_path) == a.SOURCE_REVIEW_SHA256, 'actual Main applied source review')
        review = load(review_path)
        pins = review['source_pins']
        a.require_source_pins(pins, pins)
        m.require(review['main_reviewed'] is True and all(m.sha(S/p) == h for p,h in pins.items()), 'current exact42 reviewed source')
        m.require((C/'source-review.expected.json').read_bytes() == review_path.read_bytes(), 'controller exact applied review copy')
        controller_review_path = B / 'xe-compact-moe-model-controller-main-review-v1.json'
        controller_review = load(controller_review_path)
        code_pins = {name:m.sha(C/name) for name in sorted(a.CODE_NAMES)}
        m.require(controller_review['main_reviewed'] is True and controller_review['source_pins'] == code_pins,
                  'actual Main full eleven-code-file review')
        cpu_path = B / 'xe-compact-moe-model-controller-cpu-root-v2/record.json'
        cpu = load(cpu_path)
        a.require_closed_prerequisite(cpu)
        m.require(cpu['source_pins'] == code_pins and cpu['source_stable'] is True and
                  cpu['main_review_sha256'] == m.sha(controller_review_path), 'actual new controller CPU proof exact source')
        for name, wanted in controller_review['all_fourteen_file_pins'].items():
            m.require(m.sha(C/name) == wanted, 'all reviewed controller files unchanged')
        compiler_path = B / 'xe-llvm-v711-source-root-v5/record.json'
        compiler = load(compiler_path)
        a.require_compiler_prerequisite(compiler)
        for identity in compiler['installed_files'].values():
            p = Path(identity['path'])
            m.require(p.stat().st_size == identity['bytes'] and m.sha(p) == identity['sha256'], 'current compiler/runtime bytes')
        host_path = B / 'xe-compact-moe-host-qualification-root-v2/record.json'
        host = load(host_path)
        a.require_host_prerequisite(host, pins, S, m.sha(compiler_path))
        for identity in host['binaries'].values():
            m.require(Path(identity['path']).stat().st_size == identity['bytes'] and
                      m.sha(Path(identity['path'])) == identity['sha256'], 'current standalone host fixture binaries')
        build_paths = {mode:B/f'xe-compact-moe-{mode}-qualification-root-v2/record.json' for mode in a.BUILD_MODES}
        builds = {mode:load(path) for mode,path in build_paths.items()}
        build_hashes = {mode:m.sha(path) for mode,path in build_paths.items()}
        for mode,value in builds.items():
            identity = value['binary']
            m.require(Path(identity['path']).stat().st_size == identity['bytes'] and
                      m.sha(Path(identity['path'])) == identity['sha256'], 'current full mode binary ' + mode)
            for recipe in value['build_recipe'].values():
                m.require(m.sha(Path(recipe['path'])) == recipe['sha256'], 'actual retained build recipe ' + mode)
            cache = Path(value['build_recipe']['CMakeCache.txt']['path']).read_text()
            a.require_build_prerequisite(value, mode, pins, S, m.sha(host_path), m.sha(compiler_path), identity, cache)
        lint_path = B / 'xe-compact-moe-lint-qualification-root-v2/record.json'
        failed_lint_path = B / 'xe-compact-moe-lint-qualification-root-v1/record.json'
        lint, failed_lint = load(lint_path), load(failed_lint_path)
        a.require_lint_prerequisite(lint, pins, review['changed_source_pins'], build_hashes, failed_lint, m.sha(failed_lint_path))
        pub_path = B / 'xe-compact-moe-source-publish-root-v2/record.json'
        pub = load(pub_path)
        a.require_closed_prerequisite(pub)
        raw_path = B / 'xe-compact-moe-source-publication-main-review-v2.json'
        raw = load(raw_path)
        m.require(raw['main_reviewed'] is True and raw['clean_tracked_working_tree'] is True and
                  raw['source_bytes_identical_to_all_qualified_builds'] is True and
                  raw['production_source_pins'] == pub['source_pins'] == pins and
                  pub['published_head'] == raw['published_head'] == '39e395a103af36323262cdc0dbaea27cba81f3e4' and
                  raw['historical_build_base'] == pub['historical_build_base'] == a.BASE_REVISION,
                  'actual clean source publication and historical built content equality')
        head = run('source-head', ['/usr/bin/git','rev-parse','HEAD'])
        status = run('source-status', ['/usr/bin/git','status','--porcelain','--untracked-files=all'])
        patch = run('source-diff', ['/usr/bin/git','diff','HEAD'])
        a.require_published_state({p:m.sha(S/p) for p in pins}, pins, head, pub['published_head'], status, patch,
                                 m.sha(Path(builds['free']['binary']['path'])), builds['free']['binary']['sha256'])
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        predecessor = B / 'xe-native-dequant-prefill-quiet-timing-root-v3/main-balanced-quiet32k-v3/sample-003/n32768-g16-off/record.json'
        a.require_normal_model(load(predecessor), boot)
        historical_paths = [B/f'xe-native-dequant-prefill-model-root-v6/n{n}-g16-off/record.json' for n in a.HISTORICAL]
        for p in historical_paths:
            value = load(p)
            n = value['input_tokens']
            m.require(m.sha(p) == a.HISTORICAL[n], 'immutable historical same-N OFF')
            a.require_normal_model(value, boot)
            m.require(validate(value['request_protocol'],n=n) == value['validation'] and
                      value['argv'][value['argv'].index('--prefill')+1] == '6144', 'retained OFF protocol and actual requested6144')
        historical = load(historical_paths[0])
        maps = {phase:{p:{k:x[k] for k in ('path','canonical_path','sha256','stat','elf_build_id')}
                       for p,x in snap['files'].items() if not x['original_runtime']}
                for phase,snap in historical['mapping_snapshots'].items()}
        m.require(set(maps) == {'ready_before_gen','after_gen'}, 'two actually observed historical mapping phases')
        checked = {}
        for phase, expected in maps.items():
            a.require_additional_mappings(historical['mapping_snapshots'][phase], expected)
            for p,identity in expected.items():
                canonical = Path(p).resolve(strict=True)
                m.require(str(canonical) == identity['canonical_path'] and stat_identity(canonical) == identity['stat'], 'stable qualified mapped file stat')
                if p not in checked:
                    digest = m.sha(canonical)
                    text = run('mapping-build-id-'+str(len(checked)), ['/usr/bin/readelf','--notes',str(canonical)])
                    ids = re.findall(r'Build ID: ([0-9a-f]+)', text)
                    m.require(ids == [identity['elf_build_id']] and digest == identity['sha256'] and
                              stat_identity(canonical) == identity['stat'], 'current actual mapped file hash and ELF build-id')
                    checked[p] = identity
                m.require(checked[p] == identity, 'same mapped file identity in both actual phases')
        required = [review_path, controller_review_path, cpu_path, compiler_path, host_path,
                    lint_path, failed_lint_path, pub_path, raw_path, predecessor, *historical_paths,
                    *build_paths.values(), B/'xe-native-dequant-prefill-model-controller-main-review-v7.json',
                    B/'xe-native-dequant-batch-census-observer-cpu-root-v2/record.json',
                    B/'xe-chunk6144-capture-and-observer-cpu-root-v1/record.json',
                    B/'xe-native-dequant-prefill-model-controller-cpu-root-v6/record.json',
                    B/'xe-source-llvm-small-health-root-v3/record.json', B/'xe-free-host-capabilities-root-v3/record.json',
                    B/'xe-source-llvm-census32k-root-v3/counts/record.json',
                    B/'xe-source-llvm-model8k-root-v1/record.json']
        # The failed hash receipt is consumed only for its independently successful complete CPU hash phase.
        hash_phase = load(required[-1])
        phase = next(c for c in hash_phase['commands'] if c['label'] == 'model-input-hashes')
        m.require(hash_phase['passed'] is False and not hash_phase['active'] and not hash_phase['model_launch_attempted'] and
                  a.strong_closed_command(phase) and phase['exit_code'] == 0, 'original failed preflight remains failed')
        for identity in hash_phase['model_inputs']:
            m.require(stat_identity(Path(identity['path'])) == identity['stat'], 'complete previously full-hashed model file stat unchanged')
        helper = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh')
        m.require(m.sha(helper) == '51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457', 'current reviewed diagnostic helper')
        bridge_path = out / 'publication.bridge.resolved.json'
        q = dict(status='passed',main_reviewed=True,base_revision=a.BASE_REVISION,source_review_sha256=a.SOURCE_REVIEW_SHA256,
            production_source_pins=pins,owner_sha256=a.OWNER_SHA256,host_fixture_receipt=str(host_path),
            build_receipts={mode:str(path) for mode,path in build_paths.items()},build_receipt_sha256=build_hashes,
            qualified_build_binaries={mode:value['binary'] for mode,value in builds.items()},free_release_binary=builds['free']['binary'],
            compiler_receipt=str(compiler_path),compiler_receipt_sha256=m.sha(compiler_path),lint_receipt=str(lint_path),
            lint_receipt_sha256=m.sha(lint_path),publication_bridge=str(bridge_path),controller_source_pins=code_pins,
            controller_cpu_receipt=str(cpu_path),controller_review=str(controller_review_path),controller_review_sha256=m.sha(controller_review_path),
            qualified_additional_mappings=maps,first_predecessor=str(predecessor),receipt_sha256={str(p):m.sha(p) for p in required},
            full262144_qualified=False,broad_quality_qualified=False,restore_qualified=False,performance_eligible=False,
            notes='Main verified actual CPU qualification and content equality. Runtime parity and VRAM fit are unqualified. Historical mapping observations are references, not a current-process mapping claim.')
        bridge = dict(status='passed',main_reviewed=True,base_revision=a.BASE_REVISION,source_review_sha256=a.SOURCE_REVIEW_SHA256,
            source_tree=str(S),published_revision=pub['published_head'],clean=True,source_pins=pins,built_source_pins=pins,
            build_receipt_sha256=build_hashes,lint_receipt_sha256=m.sha(lint_path),controller_source_pins=code_pins,
            source_publication_receipt=dict(path=str(pub_path),sha256=m.sha(pub_path)),raw_main_publication_review=dict(path=str(raw_path),sha256=m.sha(raw_path)),
            original_build_banner_not_rebuilt_after_commit=True,model_executed=False,gpu_executed=False)
        a.require_qualification(q,pins)
        a.require_publication(bridge,q,pins,S)
        bridge_path.write_text(json.dumps(bridge,indent=2)+'\n')
        q['publication_bridge_sha256'] = m.sha(bridge_path)
        q['receipt_sha256'][str(bridge_path)] = m.sha(bridge_path)
        qpath = out / 'qualification.resolved.json'
        qpath.write_text(json.dumps(q,indent=2)+'\n')
        r.update(qualification=dict(path=str(qpath),sha256=m.sha(qpath)),publication_bridge=dict(path=str(bridge_path),sha256=m.sha(bridge_path)),
                 verified_source_pins=pins,controller_source_pins=code_pins,receipt_pins=q['receipt_sha256'],
                 mapping_identities=checked,boot_id=boot,complete=True)
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','qualification')}))
    raise SystemExit(0 if r['passed'] else 1)
