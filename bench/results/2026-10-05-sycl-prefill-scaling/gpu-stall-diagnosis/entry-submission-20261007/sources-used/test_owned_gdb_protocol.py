"""Actual CPU-only PTY/MI protocol separation and owned cleanup checks."""
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import tty

from owned_gdb import OwnedGdb, process_identity

out = Path(sys.argv[1]); out.mkdir(mode=0o700)
source = out / 'protocol.c'
source.write_text(r'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
    fprintf(stderr,"DIAGNOSTIC-START\n"); fflush(stderr);
    puts("READY"); fflush(stdout);
    char *line=NULL; size_t capacity=0;
    while (1) {
        if (!strcmp(argv[1],"blocked")) { pause(); continue; }
        ssize_t n=getline(&line,&capacity,stdin);
        if (n<0) { free(line); return 91; }
        if (!strcmp(line,"QUIT\n")) { free(line); return 0; }
        unsigned long long sum=0;
        for (ssize_t i=0;i<n;++i) sum=(sum*131+(unsigned char)line[i])&0xffffffffULL;
        fprintf(stderr,"DIAGNOSTIC-REQUEST %zd\n",n); fflush(stderr);
        printf("ACK %zd %llu\n",n,sum); fflush(stdout);
    }
}
''')
env = dict(os.environ, PATH='/usr/bin:/bin', LD_LIBRARY_PATH='/usr/lib/x86_64-linux-gnu')
env.pop('LD_PRELOAD', None)
binary = out / 'protocol-probe'
compiled = subprocess.run(['/usr/bin/gcc', '-g', '-O0', str(source), '-o', str(binary)],
                          env=env, capture_output=True, timeout=10)
(out/'compile.stdout').write_bytes(compiled.stdout)
(out/'compile.stderr').write_bytes(compiled.stderr)
record = {'scope': 'CPU-only real GDB/PTY stdin/stdout/stderr isolation; no GPU runtime or system changes',
          'passed': False, 'compile_exit_code': compiled.returncode, 'cases': []}

def save():
    (out/'record.json').write_text(json.dumps(record, indent=2)+'\n')

def gone(identity):
    current = process_identity(identity['pid']) if identity else None
    return not current or current['start_ticks'] != identity['start_ticks'] or current['state'] == 'Z'

save()
assert compiled.returncode == 0
try:
    for mode in ['protocol', 'blocked']:
        case = {'mode': mode, 'passed': False, 'requests': []}; record['cases'].append(case)
        master, slave = os.openpty(); tty.setraw(slave); os.set_blocking(master, False)
        g = None; pending = bytearray()
        raw = (out/(mode+'.protocol.raw')).open('wb')
        try:
            g = OwnedGdb([str(binary), mode], out/mode, env, inferior_tty_fd=slave)
            g.run()
            def line(seconds=6):
                end = time.monotonic()+seconds
                while time.monotonic()<end:
                    if b'\n' in pending:
                        value, _, tail = pending.partition(b'\n'); pending[:]=tail
                        return value.decode()
                    g.poll(.01)
                    try: data=os.read(master,65536)
                    except BlockingIOError: continue
                    if data: pending.extend(data); raw.write(data); raw.flush()
                raise TimeoutError('PTY reply deadline exceeded')
            def send(data, seconds=6):
                end = time.monotonic()+seconds; view=memoryview(data)
                with selectors.DefaultSelector() as writable:
                    writable.register(master, selectors.EVENT_WRITE)
                    while view:
                        g.poll(.01)
                        if time.monotonic()>=end: raise TimeoutError('PTY input deadline exceeded')
                        if not writable.select(.01): continue
                        try: n=os.write(master,view[:8192])
                        except BlockingIOError: continue
                        view=view[n:]
            assert line() == 'READY'
            os.close(slave); slave=None
            snap=g.snapshot('waiting-protocol', resume=True)
            assert snap and snap['resumed']
            if mode == 'protocol':
                for data in [b'GEN 2 literal $x "quoted" \\ tail\n', b'GEN 2 '+b'123456,'*180000+b'9\n']:
                    checksum=0
                    for byte in data: checksum=(checksum*131+byte)&0xffffffff
                    send(data); response=line()
                    assert response == f'ACK {len(data)} {checksum}', response
                    case['requests'].append({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'reply':response})
                send(b'QUIT\n')
                end=time.monotonic()+6
                while g.exit_code is None and time.monotonic()<end: g.poll(.01)
                assert g.exit_code == 0
            stderr=(out/mode/'inferior.stderr').read_text()
            assert 'DIAGNOSTIC-START' in stderr
            assert 'DIAGNOSTIC' not in (out/(mode+'.protocol.raw')).read_text()
            assert 'READY' not in (out/mode/'gdb-mi.stdout').read_text()
            case.update(inferior=g.inferior, debugger=g.debugger_identity, terminal=g.terminal,
                        exit_code=g.exit_code, snapshots=g.snapshots, stderr=stderr)
        finally:
            if g:
                case['cleanup']=g.close()
                assert not case['cleanup']['inferior_survived'] and not case['cleanup']['gdb_survived']
                assert gone(g.inferior) and gone(g.debugger_identity)
            raw.close(); os.close(master)
            if slave is not None: os.close(slave)
            save()
        case['passed']=True; save()
    record['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__), Path(__file__).with_name('owned_gdb.py'), source]}
    record['passed']=True
except BaseException as e:
    record['error']=repr(e); raise
finally:
    save()
print(json.dumps(record, indent=2))
