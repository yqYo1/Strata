"""Bounded VTune subprocesses; keep process identities and check descendants."""
from pathlib import Path
import datetime
import json
import os
import signal
import subprocess
import time


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0],
                    ppid=int(fields[1]), pgrp=int(fields[2]), session=int(fields[3]))
    except (FileNotFoundError, ProcessLookupError):
        return None


class OwnedSession:
    def __init__(self, argv, out, env, stdin=subprocess.DEVNULL):
        self.out = Path(out)
        self.out.mkdir(mode=0o700)
        self.started = time.monotonic()
        self.stdout = (self.out/'stdout').open('wb')
        self.stderr = (self.out/'stderr').open('wb')
        self.child = subprocess.Popen(argv, stdin=stdin, stdout=self.stdout,
                                      stderr=self.stderr, start_new_session=True, env=env)
        self.root = identity(self.child.pid)
        assert self.root and self.root['session'] == self.child.pid
        self.owned = {self.child.pid: self.root}
        self.signals = []
        self.record = dict(argv=argv, root=self.root, started_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(), active=True, environment={
                k:v for k,v in env.items() if k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_',
                'ZEL_', 'ONEAPI_', 'VTUNE_', 'NEO')) or k in ['LD_LIBRARY_PATH',
                'LD_PRELOAD', 'EnableDirectSubmission']})
        self.scan()

    def scan(self):
        rows = {}
        for p in Path('/proc').iterdir():
            if p.name.isdigit():
                item = identity(int(p.name))
                if item: rows[item['pid']] = item
        roots = {pid for pid, old in self.owned.items() if pid in rows
                 and rows[pid]['start_ticks'] == old['start_ticks']}
        while True:
            new = {pid for pid, item in rows.items() if item['ppid'] in roots
                   or item['session'] == self.root['session']}
            if new <= roots: break
            roots |= new
        for pid in roots:
            self.owned[pid] = rows[pid]
        self.record.update(owned=list(self.owned.values()), elapsed_seconds=time.monotonic()-self.started)
        (self.out/'record.json').write_text(json.dumps(self.record, indent=2)+'\n')

    def adopt_target(self, path):
        if not Path(path).exists(): return
        item = json.loads(Path(path).read_text())
        now = identity(item['pid'])
        assert now and now['start_ticks'] == item['start_ticks']
        assert now['session'] == self.root['session'] or now['ppid'] in self.owned
        self.owned[item['pid']] = now
        self.record['target'] = item

    def live(self):
        result=[]
        for pid, old in self.owned.items():
            now=identity(pid)
            if now and now['start_ticks']==old['start_ticks'] and now['state']!='Z':
                result.append(now)
        return result

    def close(self, force=False):
        self.scan()
        if force or self.live():
            for sig, grace in [(signal.SIGTERM, 2), (signal.SIGKILL, 2)]:
                for item in self.live():
                    # Identity check immediately before signaling; no device reset.
                    now=identity(item['pid'])
                    if now and now['start_ticks']==item['start_ticks']:
                        try: os.kill(item['pid'],sig); self.signals.append([item['pid'],sig])
                        except ProcessLookupError: pass
                until=time.monotonic()+grace
                while time.monotonic()<until:
                    self.child.poll();self.scan()
                    if not self.live(): break
                    time.sleep(.05)
                if not self.live(): break
        self.child.poll()
        self.stdout.close();self.stderr.close()
        self.record.update(active=False, exit_code=self.child.returncode, signals=self.signals,
                           survivors=self.live(), finished_utc=datetime.datetime.now(
                           datetime.timezone.utc).isoformat())
        self.scan()
        return self.record


def run(argv, out, env, seconds=90):
    job=OwnedSession(argv,out,env)
    try:
        while job.child.poll() is None:
            if time.monotonic()-job.started>seconds: raise TimeoutError('profiler deadline')
            job.scan();time.sleep(.1)
    except BaseException as error:
        job.record['error']=repr(error)
        raise
    finally:
        receipt=job.close(force=job.child.poll() is None)
    assert not receipt['survivors']
    return receipt
