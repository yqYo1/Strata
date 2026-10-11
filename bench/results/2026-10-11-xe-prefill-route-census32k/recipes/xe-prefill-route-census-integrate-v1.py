import fcntl
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
O = B/'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner',O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','owner')
ENV = dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out = B/'xe-prefill-route-census-integrate-root-v1'
    m.require(not out.exists(),'immutable integration')
    out.mkdir()
    r = dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),
             controller_sha256=m.sha(__file__),source_base='9c0b291d85adecb6c1dc489c3e9965b82226b799',
             gpu_executed=False,model_executed=False,build_executed=False)
    def save():
        (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
    owner = m.Owner(out/'commands',save)
    r['commands'] = owner.commands
    save()
    def run(label,args):
        e,so,se = owner.run(label,args,ENV,W,wall=60,cpu=60,text_cap=4 << 20,total_cap=16 << 20,rss_cap=2 << 30)
        m.require(m.closed(e) and e['exit_code'] == 0,'closed '+label)
        return so.read_text().strip()
    try:
        m.require(run('head',['/usr/bin/git','rev-parse','HEAD']) == r['source_base'] and
                  not run('status',['/usr/bin/git','status','--porcelain']),'clean reviewed committed base')
        p = B/'xe-chunk6144-ring96-boundary-root-v1/record.json'
        boundary = json.loads(p.read_text())
        m.require(boundary['passed'] and boundary['complete'] and not boundary['active'] and
                  all(m.closed(c) for c in boundary['commands']),'previous measurements/retention published and closed')
        r['previous_boundary_sha256'] = m.sha(p)
        for name in ['xe-source-llvm-ring96-diagnostic32k-root-v1/counts','xe-source-llvm-chunk6144-diagnostic32k-root-v3/counts',
                     'xe-chunk6144-capture-and-observer-cpu-root-v1']:
            v = json.loads((B/name/'record.json').read_text())
            m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),'closed prior consumer')
            for c in v['commands']:
                for ident in c['owners']:
                    now = m.identity(ident['pid'])
                    m.require(not now or now['start_ticks'] != ident['start_ticks'],'prior owned identities absent')
        review_path = B/'xe-prefill-route-branch-census-main-review-v1.json'
        review = json.loads(review_path.read_text())
        m.require(review['main_reviewed'] and not review['applied'] and
                  all(m.sha(Path(p)) == h for p,h in review['artifact_pins'].items()),'full exact Main source review')
        patch = B/'xe-prefill-route-branch-census-source-v1/change.patch'
        run('patch-check',['/usr/bin/git','apply','--check',str(patch)])
        run('patch-apply',['/usr/bin/git','apply',str(patch)])
        m.require(run('changed',['/usr/bin/git','diff','--name-only','HEAD']) == 'src/prefill/prefill.cpp' and
                  not run('untracked',['/usr/bin/git','ls-files','--others','--exclude-standard']),'only reviewed census source')
        run('diff-check',['/usr/bin/git','diff','--check'])
        m.require(m.sha(W/'src/prefill/prefill.cpp') == 'a47ccad9f5ce5c307218b0540f24a3e0ae5456f8a8bf8fcf33c79ce9eb036761','exact reviewed delivered source')
        r['source_pins'] = {'src/prefill/prefill.cpp':m.sha(W/'src/prefill/prefill.cpp')}
        r['source_review_sha256'] = m.sha(review_path)
        r['patch_sha256'] = m.sha(patch)
        q = B/'xe-prefill-route-census-applied-main-review-v1.json'
        m.require(not q.exists(),'new immutable applied review')
        q.write_text(json.dumps(dict(main_reviewed=True,base_revision=r['source_base'],source_pins=r['source_pins'],
            applied=True,compile_build_lint_test_gpu_qualified=False,adopted=False,
            original_review_path=str(review_path),original_review_sha256=m.sha(review_path),exact_patch_sha256=m.sha(patch),
            review='Exact delivered diff applied without manual edits. Only optional consumer-owned host diagnostics; existing queue/copy/compute arithmetic and enablement unchanged.'),indent=2)+'\n')
        r['applied_review_sha256'] = m.sha(q)
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','source_pins')}))
    raise SystemExit(0 if r['passed'] else 1)
