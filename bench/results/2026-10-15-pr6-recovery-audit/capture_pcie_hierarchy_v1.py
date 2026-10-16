#!/usr/bin/env python3
"""Read the complete current PCI ancestor path; never configure or execute GPU work."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess

A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def main():
    target = A / 'own-pcie-hierarchy-v1.json'
    assert not target.exists()
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        endpoint = Path('/sys/bus/pci/devices/0000:05:00.0').resolve(strict=True)
        nodes = []
        for path in [endpoint, *endpoint.parents]:
            if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', path.name):
                continue
            fields = {}
            for name in ['vendor', 'device', 'class', 'current_link_speed', 'current_link_width',
                         'max_link_speed', 'max_link_width', 'power/runtime_status', 'power/control']:
                try:
                    fields[name] = (path / name).read_text().strip()
                except OSError as exc:
                    fields[name] = dict(errno=exc.errno, error=exc.strerror)
            fields['path'] = str(path)
            fields['BDF'] = path.name
            fields['driver'] = str((path / 'driver').resolve()) if (path / 'driver').exists() else None
            nodes.append(fields)
        commands = []
        for argv in [['/usr/bin/lspci', '-t'], ['/usr/bin/lspci', '-s', '05:00.0', '-vv']]:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=10)
            commands.append(dict(argv=argv, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr))
            assert p.returncode == 0
        record = dict(schema='strata-pcie-full-hierarchy-v1',
                      created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      endpoint_resolved=str(endpoint), nodes_endpoint_to_root=nodes, commands=commands,
                      GPU_work_executed=False, configuration_changes=False,
                      scope='Idle read-only metadata after transfer cohorts, not transfer-time utilization.',
                      interpretation=dict(
                          observed_upstream_link='03:00.0 and 02:02.0 both report16GT/s x4; 01:00.0 and 00:01.2 agree',
                          observed_endpoint_subtree='05:00.0 and 04:01.0 report2.5GT/s x1; preserve this observation',
                          inferred_link_estimate_only=dict(speed_GT_s=16, lanes=4, encoding='128/130',
                              one_direction_GB_s_before_transaction_overhead=16e9 * (128/130) * 4 / 8 / 1e9),
                          limits=['Hierarchy fields do not measure attained PCIe bandwidth, traffic, utilization or all shared devices.',
                                  'No source-backed explanation of endpoint/internal-bridge link reporting is claimed.',
                                  'This later snapshot does not prove upstream state during older cohorts.',
                                  'Fresh320MiB end-to-end payload4.35/3.89GB/s remains distinct from physical wire traffic.',
                                  'Lower new transfer rate versus older6.02/5.54GB/s has no proven cause.']))
        target.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(path=str(target), nodes=[dict(BDF=n['BDF'], speed=n['current_link_speed'], width=n['current_link_width']) for n in nodes])))


if __name__ == '__main__':
    main()
