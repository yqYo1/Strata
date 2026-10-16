"""Bounded read-only host-side sensor sampling, separate from inference threads."""
import json
import time
from pathlib import Path


class Sensors:
    def __init__(self, device, native_log):
        self.device = Path(device)
        self.native_log = Path(native_log)
        self.next_sample = 0.0
        self.paths = {
            'gt0_act_freq': self.device/'tile0/gt0/freq0/act_freq',
            'gt0_cur_freq': self.device/'tile0/gt0/freq0/cur_freq',
            'gt0_throttle_reasons': self.device/'tile0/gt0/freq0/throttle/reasons',
            'gt0_throttle_status': self.device/'tile0/gt0/freq0/throttle/status',
            'pcie_current_speed': self.device/'current_link_speed',
            'pcie_current_width': self.device/'current_link_width',
        }
        hwmon = list((self.device/'hwmon').glob('hwmon*'))
        if len(hwmon) == 1:
            h = hwmon[0]
            for key in ('energy1_input', 'energy1_label', 'temp2_input', 'temp2_label',
                        'temp3_input', 'temp3_label', 'power1_cap'):
                self.paths[key] = h/key
        self.data = dict(schema='strata-read-only-sysfs-observation-v1',
            period_seconds=1.0, sample_cap=2000, byte_cap=4<<20,
            paths={k: str(v) for k,v in self.paths.items()}, samples=[],
            stopped_at_cap=False,
            consumer='Main paired-profile frequency/thermal/power comparison; retire samples after verified summary',
            limitations='Host 1 Hz non-atomic reads. No GPU utilization, XMX/VRAM/PCIe throughput, per-kernel attribution or exclusive GPU ownership is established. Zero actual frequency can occur when idle. Energy differences measure the card, not this process alone.',
            references=['https://docs.kernel.org/gpu/xe/xe_gt_freq.html',
                        'https://docs.kernel.org/hwmon/sysfs-interface.html'])

    def sample(self):
        now = time.monotonic()
        if now < self.next_sample or self.data['stopped_at_cap']:
            return
        self.next_sample = now + self.data['period_seconds']
        if len(self.data['samples']) >= self.data['sample_cap']:
            self.data['stopped_at_cap'] = True
            return
        row = dict(monotonic_start=now, realtime_ns=time.time_ns(), values={}, errors={})
        for key,path in self.paths.items():
            try:
                with path.open('rb') as f:
                    value = f.read(256)
                if len(value) == 256:
                    raise ValueError('sensor text exceeds 255-byte limit')
                row['values'][key] = value.decode('ascii').strip()
            except (OSError, UnicodeError, ValueError) as error:
                row['errors'][key] = repr(error)
        # A bounded progress excerpt gives approximate layer alignment. It does
        # not establish exact prompt boundaries or simultaneous sensor values.
        try:
            with self.native_log.open('rb') as f:
                f.seek(0,2)
                f.seek(max(0,f.tell()-4096))
                lines = f.read(4096).decode('utf-8', errors='replace').splitlines()
            progress = [line for line in lines if 'layer-major layer ' in line]
            if progress:
                row['last_layer_progress'] = progress[-1][:240]
        except OSError as error:
            row['progress_error'] = repr(error)
        row['monotonic_end'] = time.monotonic()
        self.data['samples'].append(row)
        if len(json.dumps(self.data, separators=(',',':')).encode()) > self.data['byte_cap']:
            self.data['samples'].pop()
            self.data['stopped_at_cap'] = True
