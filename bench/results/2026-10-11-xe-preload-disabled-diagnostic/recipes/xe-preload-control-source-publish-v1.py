import fcntl, hashlib, importlib.util, json, os
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
with O.open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:ENV['SSH_AUTH_SOCK']=os.environ['SSH_AUTH_SOCK']
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'xe-preload-control-source-publish-root-v1';m.require(not out.exists(),'fresh publication');out.mkdir()
    r=dict(active=True,complete=False,passed=False,commands=[],controller_sha256=m.sha(__file__),started_utc=m.utc(),gpu_executed=False,model_executed=False,reset_executed=False)
    def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
    owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
    def run(label,argv,expected=(0,)):
        c,so,se=owner.run(label,argv,ENV,W,wall=60,cpu=60,text_cap=4<<20,total_cap=16<<20,rss_cap=2<<30)
        m.require(m.closed(c) and c['exit_code'] in expected,'normal closed '+label)
        return so.read_text().rstrip('\n'),c
    try:
        qualification=B/'xe-native-dequant-prefill-model-qualification-v3.json';q=json.loads(qualification.read_text());m.require(m.sha(qualification)=='3c73760a294e1ddd46537ab108d612b274badc71804de1b8ff3c7e797a485ee6','exact Main qualification')
        r['qualification_sha256']=m.sha(qualification);r['source_pins']=q['production_source_pins'];m.require(len(r['source_pins'])==38 and all(m.sha(W/p)==h for p,h in r['source_pins'].items()),'qualified38 sources')
        r['prerequisite_receipts']={}
        for name,h in q['receipt_sha256'].items():
            p=Path(name);m.require(m.sha(p)==h,'stable exact prerequisite');v=json.loads(p.read_text())
            if p.name=='record.json':m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),'actual prerequisite PASS')
            r['prerequisite_receipts'][name]=h
        r['model_receipts']={}
        for n in (64,2049,32768):
            for arm in ('off','on'):
                p=B/f'xe-native-dequant-prefill-model-root-v6/n{n}-g16-{arm}/record.json';v=json.loads(p.read_text())
                m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),'actual closed finite model PASS')
                m.require(v['source_pins']==r['source_pins'] and v['kernel_gate_passed'] and v['validation']['generated_tokens']==16,'same qualified model sources and strict output scope')
                if arm=='on':m.require(v['reference_comparison']['output_ids_equal'] and v['reference_comparison']['selected_logprob_top5_equal'],'exact OFF/ON selected outputs')
                r['model_receipts'][str(p)]=m.sha(p)
        text,_=run('status',['/usr/bin/git','status','--porcelain']);m.require(text==' M src/core/device.cpp','only qualified guard changed')
        text,_=run('head-before',['/usr/bin/git','rev-parse','HEAD']);m.require(text==q['base_revision'],'qualified historical base')
        text,_=run('identity',['/usr/bin/git','config','--show-origin','--get-regexp','^user\\.(name|email)$']);m.require('/.config/git/config' in text and '.git/config' not in text,'global identity only')
        text,_=run('name',['/usr/bin/git','config','--global','user.name']);m.require(text=='yqYoi','user identity')
        text,_=run('email',['/usr/bin/git','config','--global','user.email']);m.require(text=='52419113+yqYo1@users.noreply.github.com','user email')
        run('diff-check',['/usr/bin/git','diff','--check']);run('stage',['/usr/bin/git','add','--','src/core/device.cpp']);run('staged-check',['/usr/bin/git','diff','--cached','--check'])
        run('commit',['/usr/bin/git','commit','-m','fix(sycl): allow disabling eager kernel preload for diagnostics'])
        r['published_revision'],_=run('published-head',['/usr/bin/git','rev-parse','HEAD'])
        text,_=run('published-device',['/usr/bin/git','show',r['published_revision']+':src/core/device.cpp']);r['published_device_sha256']=m.sha(out/'commands/published-device.stdout');m.require(r['published_device_sha256']==r['source_pins']['src/core/device.cpp'],'published exact guard source')
        run('push-own-fork',['/usr/bin/git','push','git@github.com:yqYo1/Strata.git','HEAD:refs/heads/fix/xestrata-e32-host-handoff-20261011'])
        text,_=run('clean-status',['/usr/bin/git','status','--porcelain']);m.require(not text,'clean published source')
        m.require(all(m.sha(W/p)==h for p,h in r['source_pins'].items()),'source content unchanged by publication');r['complete']=True
    except BaseException as e:r['error']=repr(e)
    finally:
        r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','published_revision')}));raise SystemExit(0 if r['passed'] else 1)
