"""Resolve launch inputs from actual closed evidence; no GPU or model execution."""
import ast
import copy
import fcntl
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
A = B / 'xe-native-dequant-prefill-quiet-timing-controller-source-v3'
sys.path.insert(0, str(A))
sys.dont_write_bytecode = True
import run_root as candidate

m = candidate.m
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write_new(path, value):
    m.require(not path.exists() and not path.is_symlink(), 'immutable new evidence ' + str(path))
    path.write_text(json.dumps(value, indent=2) + '\n')

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-quiet32k-campaign-main-prepare-root-v3'
    m.require(not out.exists(), 'immutable preparation output')
    out.mkdir(mode=0o700)
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=sha(Path(__file__)), gpu_executed=False, model_executed=False,
             scope='CPU provenance and actual diagnostic review plus resolved quiet launch manifest. No new speed or model quality result.')
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    try:
        names = ['xe-e32-free-full-build-root-v22', 'xe-e32-contrib-llvm-full-build-root-v22',
                 'xe-e32-free-asan-full-build-root-v19', 'xe-e32-contrib-llvm-asan-full-build-root-v19',
                 'xe-e32-lints-baseline-compare-root-v17', 'xe-native-dequant-batch-census-observer-cpu-root-v2',
                 'xe-chunk6144-capture-and-observer-cpu-root-v1',
                 'xe-native-dequant-prefill-quiet-timing-controller-cpu-root-v3',
                 'xe-preload-control-source-publish-root-v1', 'xe-source-llvm-small-health-root-v3',
                 'xe-free-host-capabilities-root-v3', 'xe-source-llvm-census32k-root-v3/counts',
                 'xe-native-dequant-prefill-model-root-v6/n32768-g16-off',
                 'xe-native-dequant-prefill-model-root-v6/n32768-g16-on']
        proofs = {}
        receipt_pins = {}
        for name in names:
            p = B / name / 'record.json'
            v = json.loads(p.read_text())
            m.require(v['passed'] is True and v['complete'] is True and v['active'] is False
                      and v['commands'] and all(candidate.closed_owned(c) for c in v['commands']), 'closed anchored proof ' + name)
            allowed = {0, 2} if ('full-build-root-v22' in name or 'controller-cpu-root-v3' in name) else {0}
            m.require(all(c['exit_code'] in allowed for c in v['commands']), 'proof expected command exits ' + name)
            proofs[name] = v
            receipt_pins[str(p)] = sha(p)
        free = proofs['xe-e32-free-full-build-root-v22']
        pins = free['source_pins']
        m.require(len(pins) == 38 and all(sha(candidate.W / p) == h for p, h in pins.items()), 'actual published38 source bytes')
        cpu = proofs['xe-native-dequant-prefill-quiet-timing-controller-cpu-root-v3']
        m.require(all(sha(A / n) == h for n, h in cpu['source_pins'].items()), 'actual CPU qualified nine inputs')
        for label, argv in [('head', ['rev-parse', 'HEAD']), ('diff', ['diff', 'HEAD']),
                            ('status', ['status', '--porcelain=v1', '--untracked-files=no'])]:
            e, so, se = owner.run(label, ['/usr/bin/git', *argv], ENV, candidate.W,
                                 wall=20, cpu=20, text_cap=1 << 20, total_cap=8 << 20, rss_cap=1 << 30)
            m.require(candidate.closed_owned(e) and e['exit_code'] == 0, 'normal anchored ' + label)
            r[label] = so.read_text().strip()
        m.require(r['head'] == '0740b43488d812117f021ccbe4bd0e15d223f955' and r['diff'] == r['status'] == '', 'published clean source')
        publish = proofs['xe-preload-control-source-publish-root-v1']
        m.require(publish['published_revision'] == r['head'] and publish['source_pins'] == pins, 'actual source publication')
        publication_path = B / 'xe-quiet32k-publication-main-review-v3.json'
        publication = dict(main_reviewed=True, published_head=r['head'], historical_build_base=free['source_base'],
                           historical_build_patch_sha256=free['source_patch_sha256'], production_source_pins=pins,
                           clean_tracked_working_tree=True, actual_publication_receipt_sha256=receipt_pins[str(B / 'xe-preload-control-source-publish-root-v1/record.json')],
                           scope='Content proof for unchanged38 inputs. Original build revision and patch remain historical.')
        write_new(publication_path, publication)
        pair = {arm: proofs['xe-native-dequant-prefill-model-root-v6/n32768-g16-' + arm] for arm in ('off', 'on')}
        pair_pins = {arm: dict(path=str(B / 'xe-native-dequant-prefill-model-root-v6' / ('n32768-g16-' + arm) / 'record.json'),
                              sha256=receipt_pins[str(B / 'xe-native-dequant-prefill-model-root-v6' / ('n32768-g16-' + arm) / 'record.json')]) for arm in pair}
        health = proofs['xe-source-llvm-small-health-root-v3']
        for arm, d in pair.items():
            m.require(candidate.validate(d['request_protocol'], n=32768) == d['validation'] and d['validation']['generated_tokens'] == 16,
                      'actual strict32K protocol ' + arm)
            m.require(d['source_pins'] == pins and d['binary'] == free['binary'] and d['model_normal_close'] is True
                      and d['kernel_gate_passed'] is True and d['boot_unchanged'] is True
                      and d['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'same boot actual diagnostic proof ' + arm)
            m.require(candidate.exact_health(d['health-before_protocol']) and candidate.exact_health(d['health-after_protocol']), 'actual exact health ' + arm)
            candidate.validate_mapping_phases(d['mapping_snapshots'], health['loaded_libraries'])
        numerical = lambda d: [x for x in d['request_protocol'] if x.startswith(('T ', 'LP '))]
        m.require(numerical(pair['off']) == numerical(pair['on']), 'actual strict16 T/LP5 pair')
        diag_review_path = B / 'xe-quiet32k-diagnostic-pair-main-review-v3.json'
        write_new(diag_review_path, dict(main_reviewed=True, caller_numerical_passed=True, diagnostic_pair=pair_pins,
                                        production_source_pins=pins, performance_qualified=False,
                                        full_model_quality_qualified=False, full262144_qualified=False,
                                        scope='Actual immutable diagnostic32K pair only. Timing below is not reused as a quiet sample.'))
        main_review = B / 'xe-native-dequant-prefill-quiet-timing-controller-main-review-v3.json'
        applied_review = B / 'xe-optional-preload-control-applied-main-review-v1.json'
        applied = json.loads(applied_review.read_text())
        q = json.loads((A / 'qualification.pending.json').read_text())
        q.update(status='passed', main_reviewed=True, campaign='main-balanced-quiet32k-v3',
                 publication_review_path=str(publication_path), publication_review_sha256=sha(publication_path),
                 diagnostic_pair=pair_pins, diagnostic_off_environment=pair['off']['environment'],
                 diagnostic_pair_review_path=str(diag_review_path), diagnostic_pair_review_sha256=sha(diag_review_path),
                 source_review_pins=applied['source_pins'], source_review_sha256=sha(applied_review),
                 production_source_pins=pins, source_patch_sha256=free['source_patch_sha256'], free_release_binary=free['binary'],
                 lint_source_pins=proofs['xe-e32-lints-baseline-compare-root-v17']['source_pins'], receipt_sha256=receipt_pins,
                 controller_review_sha256=sha(main_review))
        q['qualified_build_binaries'] = {mode + '-' + flavor: proofs[prefix + ('-full-build-root-v22' if flavor == 'release' else '-asan-full-build-root-v19')]['binary']
            for mode, prefix in [('free', 'xe-e32-free'), ('contrib-llvm', 'xe-e32-contrib-llvm')] for flavor in ('release', 'asan')}
        q['sample_order_sha256'] = hashlib.sha256(json.dumps(q['sample_order'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        # Evaluate the actual production environment-construction statements only.
        tree = ast.parse((A / 'run_root.py').read_text())
        main = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
        body = next(x for x in main.body if isinstance(x, ast.With)).body
        body = next(x for x in body if isinstance(x, ast.Try)).body
        start = next(i for i,x in enumerate(body) if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'env' for t in x.targets))
        end = next(i for i,x in enumerate(body) if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'quiet_absent' for t in x.targets))
        ns = dict(BASE=candidate.BASE, adapter=health['adapter'], batch='off')
        fragment = ast.Module(body=copy.deepcopy(body[start:end+1]), type_ignores=[])
        exec(compile(ast.fix_missing_locations(fragment), str(A / 'run_root.py'), 'exec'), ns)
        d = pair['off']
        identity = {k:d[k] for k in ('input', 'model_inputs', 'argv', 'cpu_exclusion_proof', 'transfer_expectations',
                                    'source_applied_main_review_sha256', 'health_probe')}
        identity.update(diagnostic_qualification=dict(pair=pair_pins, review_path=str(diag_review_path), review_sha256=sha(diag_review_path),
                                                     caller_numerical_passed=True, performance_qualified=False),
                        diagnostic_helper_pins=cpu['source_pins'], controller_main_review_sha256=sha(main_review),
                        quiet_switches_absent=list(ns['quiet_absent']), environment_off=ns['env'],
                        qualified_runtime_libraries=health['loaded_libraries'],
                        effective_settings_without_vram={k:v for k,v in d['effective_settings'].items() if k != 'vram_free_mib'})
        q['campaign_identity'] = identity
        q['required_receipts'] = sorted(receipt_pins)
        q['notes'] = 'Actual Main CPU-qualified controller and immutable closed proofs. Four fresh-process adjacent pairs with equal OFF-first/ON-first counts, no retries, no warmup/reset. Single16-output correctness only; full262144 and broad model quality unqualified. Every quiet sample remains performance_eligible=false pending complete campaign Main review. Prefill and cold decode evaluated independently.'
        manifest = B / 'xe-quiet32k-campaign-qualification-v3.json'
        write_new(manifest, q)
        r.update(manifest_path=str(manifest), manifest_sha256=sha(manifest), actual_receipt_pins=receipt_pins,
                 publication_review_sha256=sha(publication_path), diagnostic_pair_review_sha256=sha(diag_review_path), complete=True)
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed', 'error', 'manifest_sha256')}))
    raise SystemExit(0 if r['passed'] else 1)
