from pathlib import Path
import types,json,datetime,hashlib,sys,subprocess,shlex
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
base=Path(__file__).parent;mode=sys.argv[1];assert mode in ('retained-uncached','lease-uncached')
assert json.loads((base/'small-health/record.json').read_text())['healthy']
out=base/('model-'+mode);out.mkdir();out.chmod(0o700)
frozen=base/'strata-residency-candidate'
src=root/'sycl/tools/recover-xe.sh';mod=types.ModuleType('readonly');exec(compile(src.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(src),'exec'),mod.__dict__)
r=mod.Runner(out);env=mod.health_environment();env['LD_LIBRARY_PATH']+=':/opt/intel/oneapi/mkl/2026.1/lib';env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0',LogAllocationType='1',LogAllocationStdout='1',STRATA_PREFILL_COMPACT='2',STRATA_PREFILL_LAYER_MAJOR='2',STRATA_PREFILL_RELEASE_DRAFT=str(int(mode=='lease-uncached')),STRATA_PREFILL_DRAFT_VERIFY=str(int(mode=='lease-uncached')))
env.pop('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD',None)
wrapper=out/'strata-gdb';strict=root/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/first-fault-strict.gdb'
commands=['/usr/bin/gdb','-nx','-q','-batch','--return-child-result','-iex','set debuginfod enabled off','-ex','set logging file '+str(out/'gdb.txt'),'-ex','set logging overwrite on','-ex','set logging redirect on','-ex','set logging enabled on','-ex','set debug-file-directory /usr/lib/debug:/home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/runtime-symbols/extracted/usr/lib/debug','-x',str(strict),'--args',str(frozen)]
wrapper.write_text('#!/bin/bash\nexport DEBUGINFOD_URLS=""\nexec '+shlex.join(commands)+' "$@"\n');wrapper.chmod(0o700)
record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),scope='Actual current normal-MTP/checkpoints, forced prefill, DS disabled, context128; diagnostics not speed/full capacity',mode=mode,binary_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checker_sha256=hashlib.sha256((root/'bench/results/2026-10-05-sycl-upstream-arc/serve_check.py').read_bytes()).hexdigest(),healthy=False,environment={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ONEAPI_')) or k in ['NEOReadDebugKeys','EnableDirectSubmission','LogAllocationType','LogAllocationStdout','LD_LIBRARY_PATH']})
record['packages']=subprocess.check_output(['dpkg-query','-W','-f=${Package} ${Version}\n','libze-intel-gpu1','libze1','intel-igc-core-2'],text=True)
cursor=mod.journal_cursor(r,'kernel-before')
try:
 deps=r.run('link',['/usr/bin/ldd',str(frozen)],env=env);assert 'not found' not in deps
 r.run('checker',['/usr/bin/python3',str(root/'bench/results/2026-10-05-sycl-upstream-arc/serve_check.py'),'--engine',str(wrapper),'--out',str(out/'normal-mtp.json'),'--force-prefill','--dump-first-head','--protocol-timeout','120','--uncached'],seconds=240,env=env)
 measured=json.loads((out/'normal-mtp.json').read_text());record['exit_code']=measured['exit_code'];record['requests']=[dict(name=x['name'],ids=x['ids'],first_head=x['first_head']) for x in measured['requests']]
 if mode=='lease-uncached':
  reference=json.loads((base/'model-retained-uncached/normal-mtp.json').read_text());eq=[]
  for x,y in zip(measured['requests'],reference['requests']):
   eq.append(dict(name=x['name'],ids_equal=x['ids']==y['ids'],logprobs_equal=x['logprobs']==y['logprobs'],head_equal=Path(x['first_head']['file']).read_bytes()==Path(y['first_head']['file']).read_bytes()))
  record['equality']=eq;assert len(eq)==4 and all(all(v for k,v in x.items() if k!='name') for x in eq)
  text=(out/'normal-mtp.log').read_text();releases=text.count('strata mtp decode release:');restores=text.count('strata mtp decode restore:');record.update(releases=releases,restores=restores);assert releases>=3 and releases==restores
except BaseException as e:record['error']=repr(e)
finally:
 log=r.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json']);rows=[json.loads(l) for l in log.splitlines() if l.startswith('{')];(out/'kernel.json').write_text(json.dumps(rows,indent=2)+'\n')
 record['new_fault_messages']=[d.get('MESSAGE','') for d in rows if (('xe' in d.get('MESSAGE','') or '0000:05:00.0' in d.get('MESSAGE','')) and mod.FAULT.search(d.get('MESSAGE',''))) or ('segfault' in d.get('MESSAGE','') and 'strata' in d.get('MESSAGE',''))]
record['healthy']='error' not in record and not record['new_fault_messages'] and record.get('exit_code')==0
record.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),steps=r.calls);(out/'record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ['environment','steps']},indent=2))
if not record['healthy']:raise SystemExit(1)
