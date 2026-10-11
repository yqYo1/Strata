import fcntl
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
sp = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-native-dequant-group-api-integrate-root-v2'
    m.require(not out.exists(), 'immutable integration')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), source_base='d73288cba04f728dadfd69e9797729c1a10af3ab',
             gpu_executed=False, model_executed=False, build_executed=False, adopted=False)

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()

    def run(label, args):
        e, so, se = owner.run(label, args, ENV, W, wall=60, cpu=60,
                              text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(e) and e['exit_code'] == 0, 'closed ' + label)
        return so.read_text().strip()

    try:
        m.require(run('head', ['/usr/bin/git', 'rev-parse', 'HEAD']) == r['source_base'] and
                  not run('status', ['/usr/bin/git', 'status', '--porcelain']), 'clean reviewed committed base')
        for root in ['xe-native-dequant-group-boundary-root-v2',
                     'xe-native-dequant-group-optimized-ir-boundary-root-v1']:
            p = B / root / 'record.json'
            v = json.loads(p.read_text())
            m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),
                      'closed published evidence ' + root)
            for c in v['commands']:
                for ident in c['owners']:
                    now = m.identity(ident['pid'])
                    m.require(not now or now['start_ticks'] != ident['start_ticks'], 'prior owner absent')
            r.setdefault('prior_boundary_pins', {})[root] = m.sha(p)
        review = B / 'xe-native-dequant-group-api-main-review-v2.json'
        m.require(m.sha(review) == '4e98ad3c6c48f91c2a65303115434907a9c090f4141ee8951f182b8fbb2a8425', 'full Main source review')
        ir = B / 'xe-native-dequant-group-optimized-ir-analysis-v1/analysis.json'
        analysis = json.loads(ir.read_text())
        m.require(analysis['both_modes_actually_compiled'] and analysis['both_mode_ir_byte_identical_by_version'] and
                  len(analysis['normalized_changed_definitions']) == 2 and not analysis['gpu_executed'], 'actual scoped IR qualification')
        r['optimized_ir_analysis_sha256'] = m.sha(ir)
        pins = {'src/kernels/xe/iq_kernels.cpp': '7a109e4123bcd6c6cdf8b11ff411657eaf9cd8ed81b0d7f4fdab59285461cdf7',
                'include/strata/kernels/iq_kernels.hpp': '61513ccb71831d1065bd25829541c508adf3ae58e1e3df6d3bc0c9681cb54f6d',
                'src/prefill/prefill.cpp': '1b30e3de22b30f4eb2160756c9fdde623abe87a2360fe029f0767be33aa9d664'}
        m.require(all(m.sha(W / f) == h for f, h in pins.items()), 'exact v1 base and unchanged callers')
        patch = B / 'xe-native-dequant-group-api-source-v2/candidate.patch'
        m.require(m.sha(patch) == '3f73ced5323e7fa6561a5fcdd0086501af347555def59db8d65230364b3056b1', 'exact reviewed incremental patch')
        run('patch-check', ['/usr/bin/git', 'apply', '--check', str(patch)])
        run('patch-apply', ['/usr/bin/git', 'apply', str(patch)])
        m.require(run('changed', ['/usr/bin/git', 'diff', '--name-only', 'HEAD']) == 'src/kernels/xe/iq_kernels.cpp' and
                  not run('untracked', ['/usr/bin/git', 'ls-files', '--others', '--exclude-standard']), 'only optional group pointer selector changed')
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        pins['src/kernels/xe/iq_kernels.cpp'] = '75f2ac1dfdb337c424bba19e62c34159a1bde547a776b9b5d6cf7975dfc29d50'
        m.require(all(m.sha(W / f) == h for f, h in pins.items()), 'exact applied v2 and unchanged header/callers')
        r['source_pins'] = pins
        r['source_review_sha256'] = m.sha(review)
        r['patch_sha256'] = m.sha(patch)
        q = B / 'xe-native-dequant-group-api-applied-main-review-v2.json'
        m.require(not q.exists(), 'new immutable applied review')
        q.write_text(json.dumps(dict(main_reviewed=True, base_revision=r['source_base'], source_pins=pins,
                                    applied=True, compiler_ir_qualified=True, production_build_lint_test_gpu_qualified=False,
                                    adopted=False, original_review_path=str(review), original_review_sha256=m.sha(review),
                                    optimized_ir_analysis_sha256=m.sha(ir), exact_patch_sha256=m.sha(patch),
                                    review='Only constant-index selection in optional group kernels changed; admission/geometry/decoder/header/individual functions/callers unchanged. v1 performance rejected. Fresh production build/lint/byte/service and model gates required.'), indent=2) + '\n')
        r['applied_review_sha256'] = m.sha(q)
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ['passed', 'error', 'source_pins']}))
    raise SystemExit(0 if r['passed'] else 1)
