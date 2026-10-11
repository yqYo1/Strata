"""Publish source only after closed matching release/host-ASan/lint qualification."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('owner',O)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ: ENV['SSH_AUTH_SOCK']=os.environ['SSH_AUTH_SOCK']

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'xe-per-layer-profile-fill-publish-root-v1'
    m.require(not out.exists(),'immutable publication output')
    out.mkdir(mode=0o700)
    r=dict(active=True,complete=False,passed=False,commands=[],created_utc=m.utc(),controller_sha256=m.sha(__file__),base_revision='51436a0fe4c9588b0d3eaae921a7eae12343ce7e',source_tree=str(W),qualifications={},model_executed=False,performance_qualified=False,full_model_quality_qualified=False,full_context_qualified=False,adopted=False,publication_target='git@github.com:yqYo1/Strata.git',publication_branch='fix/xestrata-e32-host-handoff-20261011')
    def save(): (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
    owner=m.Owner(out/'commands',save)
    r['commands']=owner.commands
    save()
    def run(label,argv,wall=30):
        e,so,_=owner.run(label,argv,ENV,W,wall=wall,cpu=int(wall)*2,text_cap=4<<20,total_cap=8<<20,rss_cap=1<<30)
        m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label)
        return so.read_text().strip()
    try:
        review_path=B/'xe-per-layer-profile-fill-main-review-v1.json'
        review=json.loads(review_path.read_text())
        m.require(review['main_reviewed'] and review['base_revision']==r['base_revision'],'Main reviewed source')
        r['source_review_sha256']=m.sha(review_path)
        paths=sorted(review['source_pins'])
        m.require(run('head',['/usr/bin/git','rev-parse','HEAD'])==r['base_revision'],'source base')
        m.require(run('changed',['/usr/bin/git','diff','--name-only','HEAD']).splitlines()==paths,'exact reviewed files only')
        m.require(not run('untracked',['/usr/bin/git','ls-files','--others','--exclude-standard']),'no unreviewed source')
        expected=None
        for mode in ('free','contrib-llvm'):
            for kind,version in (('full-build','v11'),('asan-full-build','v8')):
                p=B/('xe-e32-'+mode+'-'+kind+'-root-'+version+'/record.json')
                v=json.loads(p.read_text())
                m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),'closed qualification '+mode+' '+kind)
                m.require(all(m.sha(W/f)==h for f,h in v['source_pins'].items()),'stable qualified source '+mode+' '+kind)
                m.require(m.sha(Path(v['binary']['path']))==v['binary']['sha256'],'stable qualified binary '+mode+' '+kind)
                m.require(all(v['source_pins'].get(f)==h for f,h in review['source_pins'].items()),'reviewed source qualification')
                if expected is None: expected=v['source_pins']
                m.require(v['source_pins']==expected,'same complete qualified source in all modes')
                r['qualifications'][mode+'-'+kind]=dict(path=str(p),sha256=m.sha(p),cpu='12 passed,1 native ISA skip; includes actual reader boundary regressions')
        p=B/'xe-e32-lints-baseline-compare-root-v6/record.json'
        lint=json.loads(p.read_text())
        m.require(lint['passed'] and lint['complete'] and not lint['active'] and all(m.closed(c) for c in lint['commands']),'closed lint qualification')
        m.require(lint['source_pins']==review['source_pins'] and all(not c['new_findings'] for c in lint['comparisons'].values()),'no new lint findings on exact source')
        r['qualifications']['lint']=dict(path=str(p),sha256=m.sha(p),scope=lint['scope'])
        r['source_pins']=expected
        patch=run('patch',['/usr/bin/git','diff','--binary','HEAD'])
        (out/'source.patch').write_text(patch+'\n')
        r['source_patch_sha256']=m.sha(out/'source.patch')
        run('diff-check',['/usr/bin/git','diff','--check'])
        run('stage',['/usr/bin/git','add','--',*paths])
        run('commit',['/usr/bin/git','commit','-m','fix(cache): fill per-layer quotas from the full profile'])
        r['committed_revision']=run('committed-head',['/usr/bin/git','rev-parse','HEAD'])
        m.require(not run('clean',['/usr/bin/git','status','--porcelain']),'source committed clean')
        m.require(all(m.sha(W/f)==h for f,h in expected.items()),'same qualified source after commit')
        r['build_provenance']='Qualified base514 plus exact reviewed uncommitted three-file patch is byte-identical to this committed source; no post-build source transition.'
        save()
        run('publish',['/usr/bin/git','push',r['publication_target'],'HEAD:refs/heads/'+r['publication_branch']],wall=60)
        r['complete']=True
    except BaseException as error: r['error']=repr(error)
    finally:
        r['active']=owner.active is not None
        r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc']=m.utc()
        save()
        if r['passed']:
            manifest=B/'xe-per-layer-profile-fill-source-commit-v1.json'
            m.require(not manifest.exists(),'immutable source publication manifest')
            manifest.write_text(json.dumps({k:v for k,v in r.items() if k!='commands'},indent=2)+'\n')
    print(json.dumps({k:r.get(k) for k in ('passed','error','committed_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
