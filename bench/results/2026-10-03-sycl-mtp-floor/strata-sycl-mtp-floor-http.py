import os,sys,subprocess,time,signal,urllib.request
from pathlib import Path
log=open('/tmp/strata-sycl-mtp-floor-http-server.log','w')
env=dict(os.environ,PYTHONPATH='tools',ONEAPI_DEVICE_SELECTOR='level_zero:gpu');env.pop('STRATA_SYCL_VERIFY_NATIVE_CAPTURE',None);env.pop('STRATA_SYCL_ADAPT_SYNC',None);env.pop('STRATA_SYCL_VERIFY_LOGITS',None);env.pop('STRATA_TRACE',None)
p=subprocess.Popen(['/home/yayoi/.local/share/strata-sycl/venv/bin/python','serve/server.py','--engine','strata','--config','/home/yayoi/.local/share/strata-sycl/serve-config.json','--host','127.0.0.1','--port','18085'],env=env,stdout=log,stderr=subprocess.STDOUT)
try:
 deadline=time.monotonic()+180
 while True:
  if p.poll() is not None:raise RuntimeError('HTTP server exited before ready')
  try:
   with urllib.request.urlopen('http://127.0.0.1:18085/v1/models',timeout=2) as r:
    if r.status==200:break
  except OSError:pass
  if time.monotonic()>deadline:raise TimeoutError('HTTP server readiness')
  time.sleep(1)
 subprocess.run(['/home/yayoi/.local/share/strata-sycl/venv/bin/python','/tmp/strata-sycl-mtp-floor-http-check.py'],env=env,check=True,timeout=240)
finally:
 if p.poll() is None:p.send_signal(signal.SIGINT);p.wait(timeout=60)
 log.close()
