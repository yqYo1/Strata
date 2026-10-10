"""Private direct owner for long compiler builds; derived from qualified direct_owner v4.
Retires only dead non-leader pidfds; finite live and cumulative identity budgets are distinct.

Leader stays unreaped until final session scan. Signals only pinned pidfds;
no numeric killpg, GDB, unknown-owner adoption or automatic retry.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import select
import selectors
import signal
import subprocess
import time
import resource


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def identity(pid):
    try:
        v = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=v[0], ppid=int(v[1]), pgid=int(v[2]), sid=int(v[3]),
                    start_ticks=int(v[19]), rss_bytes=int(v[21])*os.sysconf('SC_PAGE_SIZE'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def closed(entry):
    return (entry.get('observation_complete') and entry.get('direct_child_reaped')
            and entry.get('session_empty') and not entry.get('error')
            and not entry.get('errors') and not entry.get('cleanup'))


class Interactive:
    def __init__(self, proc, entry, stdout, poll, exit_watch):
        self.proc, self.entry, self.stdout = proc, entry, stdout
        self.poll, self.exit_watch = poll, exit_watch
        self.pending = bytearray()
        self.eof = False
        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stdin.fileno(), False)

    def read(self):
        try:
            data = os.read(self.proc.stdout.fileno(), 65536)
        except BlockingIOError:
            return
        if data:
            self.stdout.write(data); self.stdout.flush(); self.pending.extend(data)
            require(len(self.pending) < 1 << 20, 'protocol pending/line cap')
        else:
            self.eof = True

    def line(self, seconds=700):
        deadline = time.monotonic() + seconds
        with selectors.DefaultSelector() as ready:
            ready.register(self.proc.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                if b'\n' in self.pending:
                    line, _, tail = self.pending.partition(b'\n')
                    self.pending[:] = tail
                    require(len(line) < 1 << 20, 'protocol line cap')
                    return line.decode('utf-8').strip()
                self.poll()
                require(not self.eof, 'protocol EOF before expected line')
                if ready.select(.02):
                    self.read()
        raise TimeoutError('protocol line deadline')

    def send(self, data, seconds=20):
        require(len(data) < 1 << 20, 'input size cap')
        deadline = time.monotonic() + seconds
        view = memoryview(data)
        with selectors.DefaultSelector() as ready:
            ready.register(self.proc.stdin, selectors.EVENT_WRITE)
            while view:
                self.poll()
                require(time.monotonic() < deadline, 'protocol send deadline')
                require(not self.exit_watch.poll(0), 'child exited during input')
                if not ready.select(.02):
                    continue
                try:
                    count = os.write(self.proc.stdin.fileno(), view[:8192])
                except BlockingIOError:
                    continue
                require(count > 0, 'input pipe closed')
                view = view[count:]

    def exit(self, seconds):
        deadline = time.monotonic() + seconds
        with selectors.DefaultSelector() as ready:
            ready.register(self.proc.stdout, selectors.EVENT_READ)
            while not self.exit_watch.poll(0):
                self.poll()
                require(time.monotonic() < deadline, 'child exit deadline')
                if ready.select(.02):
                    self.read()
            # Drain all pipe bytes after death, without reaping the session anchor.
            while not self.eof:
                require(time.monotonic() < deadline, 'stdout held by surviving writer')
                self.poll()
                if ready.select(.02):
                    self.read()
            require(not self.pending.strip(), 'unexpected trailing protocol')


class Owner:
    def __init__(self, output, persist=lambda: None):
        self.output = output
        self.output.mkdir(exist_ok=True)
        self.commands = []
        self.persist = persist
        self.active = None

    def run(self, label, argv, env, cwd, interaction=None, wall=1000, text_cap=64 << 20,
            rss_cap=104 << 30, total_cap=2 << 30, cpu=1000, file_cap=64 << 20):
        require(all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), 'explicit string environment')
        entry = dict(label=label, argv=list(map(str, argv)), environment=dict(env), started_utc=utc(),
                     wall_seconds=wall, errors=[], owners=[], cleanup=[], observation_complete=False,
                     direct_child_reaped=False, session_empty=False, session_anchored=False)
        self.commands.append(entry); self.persist()
        so, se = self.output / (label+'.stdout'), self.output / (label+'.stderr')
        require(not so.exists() and not se.exists(), 'no overwrite of owned logs')
        proc = None; owned = {}; leader = None; watch = select.poll()
        start = time.monotonic(); next_scan = start

        def anchor():
            require(leader is not None, 'leader identity unavailable')
            now = identity(leader['pid'])
            require(now and now['start_ticks'] == leader['start_ticks'] and now['sid'] == leader['pid'], 'unreaped session anchor lost')

        def register(v):
            key = (v['pid'], v['start_ticks'])
            if key in owned: return
            anchor()
            try: fd = os.pidfd_open(v['pid'])
            except ProcessLookupError: return  # child exited before pinning; do not adopt a recycled PID
            try:
                now = identity(v['pid'])
                if not now or now['start_ticks'] != v['start_ticks']:
                    os.close(fd); return
                require(now['sid'] == leader['pid'], 'new pidfd not in anchored session')
                require(len(owned)<256, 'live owned process population cap')
                require(len(entry['owners'])<200000, 'cumulative owned identity budget')
                owned[key] = (fd, dict(now)); entry['owners'].append({k: now[k] for k in ('pid','start_ticks','sid','pgid')})
            except BaseException:
                os.close(fd); raise

        def scan():
            anchor(); members=[]
            # A build creates thousands of short-lived compiler children. Retain identity evidence,
            # but close descriptors for confirmed dead non-leaders before admitting new members.
            # The unreaped direct leader anchors the session and is never retired here.
            for key, (fd, _) in list(owned.items()):
                if key == (leader['pid'], leader['start_ticks']): continue
                v = identity(key[0])
                if v is None or v['start_ticks'] != key[1] or v['state'] == 'Z':
                    os.close(fd); del owned[key]
            for path in Path('/proc').iterdir():
                if not path.name.isdecimal(): continue
                try: v=identity(int(path.name))
                except PermissionError:
                    try:
                        if path.stat().st_uid != os.getuid(): continue
                    except (FileNotFoundError, ProcessLookupError): continue
                    raise
                if v and v['sid'] == leader['pid'] and v['state'] != 'Z':
                    register(v); members.append(v)
            # Previously pinned members that leave the session remain ours, not
            # inferred from a recycled numeric group. Track them explicitly.
            keys={(v['pid'],v['start_ticks']) for v in members}
            for key in owned:
                v=identity(key[0])
                if v and v['start_ticks']==key[1] and v['state']!='Z' and key not in keys:
                    members.append(v)
            require(len(owned)<=256, 'live owned process population cap')
            entry['peak_live_owned_identities']=max(entry.get('peak_live_owned_identities',0), len(owned))
            return members

        def poll():
            nonlocal next_scan
            require(time.monotonic()-start < wall, 'owned whole-process wall deadline')
            require(so.stat().st_size+se.stat().st_size<=text_cap, 'combined log cap')
            require(sum(p.stat().st_size for p in self.output.iterdir() if p.suffix in ('.stdout','.stderr'))<=64<<20,'aggregate owner log cap')
            if time.monotonic()>=next_scan:
                members=scan(); rss=sum(v['rss_bytes'] for v in members)
                entry['peak_session_rss_bytes']=max(entry.get('peak_session_rss_bytes',0),rss)
                require(rss<=rss_cap,'owned session RSS cap')
                require(sum(p.stat().st_size for p in self.output.parent.rglob('*') if p.is_file())<=total_cap,'artifact cap')
                self.persist(); next_scan=time.monotonic()+.25

        def signals(sig):
            try: scan()
            except BaseException as e: entry['errors'].append('cleanup observation: '+repr(e))
            for (pid,ticks),(fd,_) in list(owned.items()):
                try:
                    now=identity(pid)
                    if not now or now['start_ticks']!=ticks or now['state']=='Z': continue
                    signal.pidfd_send_signal(fd,sig)
                    entry['cleanup'].append(dict(pid=pid,start_ticks=ticks,signal=sig.name,result='sent'))
                except ProcessLookupError: pass
                except BaseException as e: entry['errors'].append('pidfd signal: '+repr(e))
            try: self.persist()
            except BaseException as e: entry['errors'].append('cleanup persistence: '+repr(e))

        def limits():
            for key,value in ((resource.RLIMIT_CPU,(cpu,cpu+1)), (resource.RLIMIT_FSIZE,(file_cap,file_cap)),
                              (resource.RLIMIT_NOFILE,(256,256)), (resource.RLIMIT_CORE,(0,0))):
                resource.setrlimit(key,value)

        try:
            with so.open('xb') as stdout, se.open('xb') as stderr:
                entry['spawn_begin_monotonic']=time.monotonic()
                proc=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
                                      start_new_session=True,preexec_fn=limits,bufsize=0)
                self.active=proc; entry['pid']=proc.pid; entry['spawn_return_monotonic']=time.monotonic()
                leader=identity(proc.pid)
                require(leader and leader['sid']==proc.pid and leader['pgid']==proc.pid,'direct leader/session identity')
                entry['direct_identity']=dict(leader); entry['session_anchored']=True
                register(leader); watch.register(owned[(leader['pid'],leader['start_ticks'])][0],select.POLLIN)
                self.persist(); io=Interactive(proc,entry,stdout,poll,watch)
                if interaction: interaction(io,entry)
                else:
                    # Utility commands have no stdin/protocol. Collect arbitrary
                    # bounded stdout, including non-newline output.
                    proc.stdin.close(); deadline=time.monotonic()+wall
                    with selectors.DefaultSelector() as ready:
                        ready.register(proc.stdout,selectors.EVENT_READ)
                        while not watch.poll(0) or not io.eof:
                            poll(); require(time.monotonic()<deadline,'utility deadline')
                            if ready.select(.02):
                                io.read(); io.pending.clear()
                require(watch.poll(0),'interaction must complete child exit')
                require(not scan(),'session member survives direct leader exit')
                require(so.stat().st_size+se.stat().st_size<=text_cap,'exit log cap')
                require(sum(p.stat().st_size for p in self.output.iterdir() if p.suffix in ('.stdout','.stderr'))<=64<<20,'aggregate exit log cap')
                entry.update(session_empty=True,observation_complete=True)
        except BaseException as e:
            entry['error']=repr(e)
        finally:
            if proc is not None:
                # Never poll()/wait()/communicate() before this final owned scan.
                try: stop=not watch.poll(0) or bool(scan())
                except BaseException as e: stop=True; entry['errors'].append('closure observation: '+repr(e))
                if stop:
                    for sig in (signal.SIGTERM,signal.SIGKILL):
                        signals(sig); deadline=time.monotonic()+3
                        while time.monotonic()<deadline:
                            try:
                                if watch.poll(0) and not scan(): break
                            except BaseException as e: entry['errors'].append('cleanup scan: '+repr(e)); break
                            time.sleep(.05)
                try:
                    entry['survivors']=scan(); entry['session_empty']=not entry['survivors']; entry['ownership_status']='observed'
                except BaseException as e:
                    entry['session_empty']=False; entry['ownership_status']='unknown'; entry['errors'].append('final scan: '+repr(e))
                # Final scan precedes reap; no numeric SID scan after this point.
                try:
                    entry['exit_code']=proc.wait(timeout=.1); entry['direct_child_reaped']=True
                except BaseException as e: entry['errors'].append('reap: '+repr(e))
                for stream in (proc.stdin,proc.stdout):
                    if stream:
                        try: stream.close()
                        except OSError as e: entry['errors'].append('pipe close: '+repr(e))
            for fd,_ in owned.values():
                try: os.close(fd)
                except OSError as e: entry['errors'].append('pidfd close: '+repr(e))
            entry.update(finished_utc=utc(),elapsed_seconds=time.monotonic()-start,
                         logs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (so,se) if p.exists()})
            self.active=proc if proc and (not entry.get('direct_child_reaped') or not entry.get('session_empty')) else None
            self.persist()
        return entry,so,se
