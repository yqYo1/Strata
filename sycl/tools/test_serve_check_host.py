"""Exercise the real serve checker with CPU protocol children, without a GPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


CHILD = r'''#!/usr/bin/python3
import os,signal,sys
mode=os.environ['STRATA_TEST_PROTOCOL_CASE']
if mode=='ignore-shutdown':signal.signal(signal.SIGTERM,signal.SIG_IGN)
print(('allocation-proof '*20+'\n')*800,end='',flush=True)
print('READY 128',flush=True)
requests=0
for line in sys.stdin:
 if line.startswith('QUIT'):
  if mode=='ignore-shutdown':continue
  print('shutdown-tail-proof\n'*16000,end='',flush=True)
  sys.exit(0)
 if not line.startswith('GEN '):continue
 requests+=1
 print('RESUME 30' if requests>1 else 'RESUME 0',flush=True)
 if requests==2:
  if mode=='exit':print('before-unexpected-exit',flush=True);sys.exit(17)
  if mode=='signal':os.kill(os.getpid(),signal.SIGTERM)
  if mode in ('timeout','ignore-shutdown'):continue
 ids=[760,10849,88851,2272]
 if mode=='different-repeat' and requests==2:ids[-1]+=1
 for token in ids:
  print('T '+str(token),flush=True)
  print('LP 760:-0.5 10:-2',flush=True)
 print('DONE 37 4',flush=True)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    checker=root/'bench/results/2026-10-05-sycl-upstream-arc/serve_check.py'
    child=args.output/'cpu-protocol-child';child.write_text(CHILD);child.chmod(0o700)
    record=dict(scope=__doc__,checker_sha256=hashlib.sha256(checker.read_bytes()).hexdigest(),
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=[],passed=False)
    for mode in ['healthy','exit','signal','different-repeat','timeout','ignore-shutdown']:
        out=args.output/mode;out.mkdir()
        env=dict(os.environ,STRATA_TEST_PROTOCOL_CASE=mode)
        run=subprocess.run([sys.executable,str(checker),'--engine',str(child),
                            '--out',str(out/'result.json'),'--force-prefill',
                            '--protocol-timeout','2'],capture_output=True,text=True,env=env,timeout=30)
        (out/'controller.stdout').write_text(run.stdout)
        (out/'controller.stderr').write_text(run.stderr)
        partial=json.loads((out/'result.partial.json').read_text())
        raw=(out/'result.stdout.raw').read_bytes()
        events=[json.loads(line) for line in (out/'result.events.jsonl').read_text().splitlines()]
        assert raw.startswith(b'allocation-proof ') and not partial['still_alive']
        assert partial['pid']>0 and partial['args'][-2:]==['--short-read','0']
        assert events[0]['kind']=='spawn' and events[-1]['kind']=='exit'
        assert max([e['raw_offset'] for e in events if e['kind']=='stdout'])==len(raw)
        if mode=='healthy':
            assert run.returncode==0 and partial['exit_code']==0 and len(partial['requests'])==4
            assert partial['cleanup']==[] and (out/'result.json').exists()
        elif mode in ('exit','signal'):
            expected=17 if mode=='exit' else -15
            assert run.returncode==1 and partial['exit_before_cleanup']==expected
            assert partial['exit_code']==expected and partial['cleanup']==[]
            assert len(partial['requests'])==1 and partial['active_request']=='repeat'
            assert 'RESUME 30' in partial['active_protocol']
        elif mode=='ignore-shutdown':
            assert run.returncode==1 and partial['exit_code']==-9
            assert partial['cleanup']==['QUIT','SIGTERM','SIGKILL']
        else:
            assert run.returncode==1 and partial['exit_code']==0 and partial['cleanup']==['QUIT']
            assert raw.endswith(b'shutdown-tail-proof\n')
        record['runs'].append(dict(mode=mode,controller_exit_code=run.returncode,
                                  engine_exit_code=partial['exit_code'],cleanup=partial['cleanup'],
                                  raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),
                                  completed_requests=len(partial['requests']),expected_result=True))
        (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        print('PASS '+mode,flush=True)
    record.update(passed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
