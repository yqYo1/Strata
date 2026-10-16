"""Bounded passive samples of one pinned native child; no attach or signals."""
import datetime
import json
import os
import select
import threading
import time
from pathlib import Path


class Sampler:
    def __init__(self, native, root):
        self.native, self.root = native, Path(root)
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.summary = dict(passed=False, samples=0, bytes=0, pid=native['pid'],
                            start_ticks=native['start_ticks'], byte_cap=256 << 10,
                            max_seconds=900, interval_seconds=1, max_threads=256,
                            attach_or_signal=False, stack_capture=False)

    def start(self):
        self.thread.start()

    def close(self):
        self.cancel.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('passive sampler did not stop')

    def run(self):
        fd = None
        started = time.monotonic()
        last = {}
        try:
            fd = os.pidfd_open(self.native['pid'])
            watch = select.poll()
            watch.register(fd, select.POLLIN)
            proc = Path('/proc', str(self.native['pid']))
            with (self.root / 'owned/thread-samples.jsonl').open('wb') as out:
                while not self.cancel.is_set() and not watch.poll(0):
                    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                    if int(fields[19]) != self.native['start_ticks']:
                        raise RuntimeError('native identity changed')
                    if time.monotonic() - started >= self.summary['max_seconds']:
                        self.summary['stopped_for_time_cap'] = True
                        break
                    tasks = sorted((proc / 'task').iterdir(), key=lambda x: int(x.name))
                    rows = []
                    for task in tasks[:self.summary['max_threads']]:
                        try:
                            f = (task / 'stat').read_text().rsplit(')', 1)[1].split()
                            tid, ticks = int(task.name), int(f[11]) + int(f[12])
                            row = dict(tid=tid, state=f[0], ticks=ticks, delta=ticks-last.get(tid, ticks),
                                       wchan=(task / 'wchan').read_text().strip()[:80])
                            last[tid] = ticks
                            rows.append(row)
                        except (FileNotFoundError, ProcessLookupError):
                            continue
                    rows.sort(key=lambda x: (-x['delta'], x['tid']))
                    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    data = (json.dumps(dict(utc=stamp, elapsed_s=round(time.monotonic()-started, 3),
                        observed_threads=len(tasks), scanned_threads=len(rows), most_active=rows[:8])) + '\n').encode()
                    if self.summary['bytes'] + len(data) > self.summary['byte_cap']:
                        self.summary['stopped_for_byte_cap'] = True
                        break
                    out.write(data); out.flush()
                    self.summary['samples'] += 1; self.summary['bytes'] += len(data)
                    if self.summary['samples'] % 5 == 1:
                        with (proc / 'maps').open('rb') as stream:
                            maps = stream.read(96 << 10)
                            self.summary['maps_truncated'] = bool(stream.read(1))
                        (self.root / 'owned/native-maps.txt').write_bytes(maps)
                        self.summary['maps_snapshot_utc'] = stamp
                    if self.cancel.wait(self.summary['interval_seconds']):
                        break
            self.summary['passed'] = True
        except (FileNotFoundError, ProcessLookupError) as error:
            self.summary.update(passed=True, stopped_on_process_exit=type(error).__name__)
        except BaseException as error:
            self.summary['error'] = repr(error)
        finally:
            if fd is not None:
                os.close(fd)
            self.summary['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            (self.root / 'sampler-summary.json').write_text(json.dumps(self.summary, indent=2) + '\n')
