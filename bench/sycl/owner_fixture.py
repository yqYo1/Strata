#!/usr/bin/env python3
"""Main-authored bounded CPU-only supervision fixture; repository LICENSE."""
import os
import select
import signal
import subprocess
import sys
import time

signal.alarm(20)  # Orphan backstop; no fixture process can wait indefinitely.


def eof():
    if os.read(0, 1): raise RuntimeError('only private stdin EOF may stop fixture')


mode = sys.argv[1]
if mode == 'wait':
    print('ready', flush=True)
    for tick in range(1, 11):
        if select.select([0], [], [], 1)[0]: raise RuntimeError('early fixture EOF')
        print('tick=' + str(tick), flush=True)
    eof(); print('done', flush=True)
elif mode == 'normal':
    print('done', flush=True)
elif mode in ('family', 'cleanup'):
    child = subprocess.Popen([sys.executable, '-I', '-S', '-B', __file__,
        'escape-child' if mode == 'family' else 'hold-child'], stdin=subprocess.PIPE)
    print('child=' + str(child.pid), flush=True)
    if sys.stdin.readline() != 'go\n': raise RuntimeError('parent gate')
    child.stdin.write(b'go\n'); child.stdin.flush()
    if mode == 'family':
        if sys.stdin.readline() != 'stop\n': raise RuntimeError('family stop gate')
        child.stdin.close()
        if child.wait(timeout=3) != 0: raise RuntimeError('child failed')
        print('done', flush=True)
    else:
        eof(); raise RuntimeError('cleanup fixture should be stopped by owner')
elif mode in ('escape-child', 'hold-child'):
    if sys.stdin.readline() != 'go\n': raise RuntimeError('child gate')
    if mode == 'escape-child':
        os.setsid()
        payload = bytearray(32 << 20)
        payload[-1] = 1
        print('escaped=' + str(os.getpid()), flush=True)
    else: print('held=' + str(os.getpid()), flush=True)
    eof()
else: raise RuntimeError('unknown fixture')
