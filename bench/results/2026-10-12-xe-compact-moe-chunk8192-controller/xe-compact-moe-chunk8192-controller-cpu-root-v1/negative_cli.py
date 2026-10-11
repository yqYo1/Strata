import hashlib,importlib.util,json
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
H=B/'xe-compact-moe-chunk8192-controller-source-v1'
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest()=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
s=importlib.util.spec_from_file_location('cli_owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
owner=m.Owner(Path(__file__).parent/'negative-cli-commands')
env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
cases=[['--input-tokens','32768','--chunk','8192','--compact','on'],
       ['--input-tokens','65536','--chunk','6144','--compact','on','--reference','/tmp/unused-no-open'],
       ['--input-tokens','32768','--chunk','6144','--compact','on'],
       ['--input-tokens','32768','--chunk','8192','--compact','off','--reference','/tmp/unused-no-open'],
       ['--input-tokens','65537','--chunk','8192','--compact','on','--reference','/tmp/unused-no-open'],
       ['--input-tokens','32768','--chunk','8192','--group','8','--compact','on','--reference','/tmp/unused-no-open']]
for i,args in enumerate(cases):
 c,so,se=owner.run('invalid-cli-'+str(i),['/usr/bin/python3',str(H/'run_root.py'),*args],env,H,wall=10,cpu=20,text_cap=65536,total_cap=2<<20,rss_cap=128<<20)
 assert m.closed(c) and c.get('session_anchored') is True and c['exit_code']==2
 assert not so.read_text() and 'error:' in se.read_text()
p=H/'qualification.pending.json'
c,so,se=owner.run('pending-refused',['/usr/bin/python3',str(H/'run_root.py'),'--input-tokens','65536','--chunk','6144','--compact','on','--qualification',str(p),'--qualification-sha256',m.sha(p)],env,H,wall=10,cpu=20,text_cap=65536,total_cap=2<<20,rss_cap=128<<20)
assert m.closed(c) and c.get('session_anchored') is True and c['exit_code']==1
assert not so.read_text() and 'actual Main qualification, no pending launch' in se.read_text()
assert not (B/'xe-compact-moe-chunk8192-root-v1').exists()
print(json.dumps(dict(passed=True,gpu_executed=False,model_executed=False,actual_cli_commands=owner.commands)))
