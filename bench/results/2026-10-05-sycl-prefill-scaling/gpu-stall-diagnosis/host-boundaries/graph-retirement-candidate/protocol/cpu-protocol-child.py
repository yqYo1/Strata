#!/usr/bin/python3
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
