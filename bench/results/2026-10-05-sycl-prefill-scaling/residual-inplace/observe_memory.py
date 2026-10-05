"""Read process memory and DRM fdinfo during a run; no GPU API calls."""
import json
from pathlib import Path
import re
import threading
import time


def read_memory(pid):
    proc = Path(f'/proc/{pid}')
    values = {}
    for line in (proc / 'status').read_text().splitlines():
        match = re.match(r'(VmRSS|VmSwap|VmPeak|VmSize):\s+(\d+) kB', line)
        if match:
            values[match[1] + '_kib'] = int(match[2])
    for line in Path('/proc/meminfo').read_text().splitlines():
        match = re.match(r'(MemAvailable|SwapFree|SwapTotal):\s+(\d+) kB', line)
        if match:
            values['host_' + match[1] + '_kib'] = int(match[2])
    clients = {}
    for fd in (proc / 'fdinfo').iterdir():
        try:
            fields = dict(line.split(':', 1) for line in fd.read_text().splitlines() if ':' in line)
        except FileNotFoundError:
            continue
        if 'drm-client-id' not in fields:
            continue
        # Duplicate handles for the same DRM client describe the same allocations.
        client = fields.get('drm-pdev', '').strip() + ':' + fields['drm-client-id'].strip()
        clients[client] = {key: int(value.split()[0]) for key, value in fields.items()
                           if re.fullmatch(r'drm-(total|resident)-vram\d+', key)}
    values['drm_clients'] = clients
    for kind in ('total', 'resident'):
        values[f'vram_{kind}_kib'] = sum(value for fields in clients.values() for key, value in fields.items()
                                        if key.startswith(f'drm-{kind}-vram'))
    return values


class MemoryObserver:
    def __init__(self, pid, path, interval=1.0):
        self.pid, self.path, self.interval = pid, Path(path), interval
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.peak = {}
        self.samples = 0
        self.errors = []
        self.thread = threading.Thread(target=self.observe, name='memory-observer', daemon=True)
        self.thread.start()

    def observe(self):
        with self.path.open('w') as output:
            while True:
                row = {'pid': self.pid, 'seconds': time.monotonic() - self.started}
                try:
                    row.update(read_memory(self.pid))
                    self.samples += 1
                    for key in ('VmRSS_kib', 'VmSwap_kib', 'VmPeak_kib', 'vram_total_kib', 'vram_resident_kib'):
                        if key in row:
                            self.peak[key] = max(self.peak.get(key, 0), row[key])
                except FileNotFoundError:
                    row['process_exited'] = True
                except Exception as exc:
                    row['error'] = repr(exc)
                    self.errors.append(repr(exc))
                output.write(json.dumps(row) + '\n')
                output.flush()
                if row.get('process_exited') or self.stop_event.wait(self.interval):
                    break

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        return {'path': str(self.path), 'samples': self.samples, 'peak': self.peak,
                'sample_interval_seconds': self.interval, 'errors': self.errors}
