import hashlib,json,sys
from pathlib import Path
B=Path(__file__).parent;Q=B/'clean64k-task-fixture-v2';OUT=B/'clean64k-task-fixture-cpu-root-r3';DEST=B/'clean64k-task-fixture-root-r3'
sys.path.insert(0,str(B/'xestrata-clean-64k-comparison-v4'));from direct_owner import Owner,closed,sha
assert sha(Q/'prepare_fixture.py')=='10a065c15ee785eafc0cd639e2ed3cf17edebfa7b484b5d743b1173e1a03882b'
assert not OUT.exists()and not DEST.exists();OUT.mkdir(mode=0o700)
r=dict(active=True,passed=False,complete=False,gpu_executed=False,model_executed=False,source_sha256=sha(Q/'prepare_fixture.py'),owner_sha256=sha(B/'xestrata-clean-64k-comparison-v4/direct_owner.py'))
def save():(OUT/'record.json').write_text(json.dumps(r,indent=2)+'\n')
o=Owner(OUT/'processes',save);r['commands']=o.commands;save()
env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
try:
 interpreter=B/'upstream-refresh-20261007/test-venv/bin/python'
 assert interpreter.is_file()
 r['interpreter']=dict(path=str(interpreter),executable_sha256=sha(interpreter),venv_cfg_sha256=sha(interpreter.parent.parent/'pyvenv.cfg'))
 dep='import hashlib,json,sys,regex,regex._regex,regex._regex_core;from pathlib import Path\np=[Path(regex.__file__),Path(regex._regex.__file__),Path(regex._regex_core.__file__)]\nprint(json.dumps(dict(python=sys.version,regex_version=regex.__version__,files=[dict(path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest())for x in p])))'
 e,so,se=o.run('dependency',[str(interpreter),'-c',dep],env,B,wall=10,text_cap=1<<20,rss_cap=1<<30,cpu=10)
 assert closed(e)and e['exit_code']==0
 r['dependency']=json.loads(so.read_text());save()
 e,so,se=o.run('prepare',[str(interpreter),str(Q/'prepare_fixture.py'),'--output',str(DEST)],env,B,wall=120,text_cap=8<<20,rss_cap=1<<30,cpu=90)
 assert closed(e)and e['exit_code']==0
 m=json.loads((DEST/'manifest.json').read_text());p=Path(m['fixture']['path']);v=list(map(int,p.read_text().split()))
 assert m['passed']is True and m['active']is False and m['gpu_executed']is False and len(v)==65536 and all(0<=x<248320 for x in v)
 assert sha(p)==m['fixture']['sha256']and m['suffix']['encode_exact']and m['suffix']['decode_exact']and m['suffix']['original32k_tail_exact']
 assert len(list(map(int,(DEST/'body-prefix-65487-tokens.txt').read_text().split())))==65487
 r.update(passed=True,complete=True,manifest_path=str(DEST/'manifest.json'),manifest_sha256=sha(DEST/'manifest.json'),fixture=m['fixture'],independent_persisted_hash_count_domain_passed=True)
except BaseException as ex:r.update(error_type=type(ex).__name__,error='bounded fixture CPU preparation or persisted verification failed; no GPU admission')
finally:r['active']=o.active is not None;save()
print(json.dumps(dict(passed=r['passed'],record=str(OUT/'record.json'))));raise SystemExit(0 if r['passed']else 1)
