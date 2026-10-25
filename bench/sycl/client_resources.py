#!/usr/bin/env python3
"""Read-only xe client/CPU diagnostics, separate from clean speed measurements.

Main-authored from the Linux fdinfo ABI, not another engine implementation:
https://www.kernel.org/doc/html/latest/gpu/drm-usage-stats.html
https://www.kernel.org/doc/html/latest/gpu/xe/xe-drm-usage-stats.html
Busy cycles include engine execution stalls; they are not FLOPs or bus bytes.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time


class TargetExited(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unsigned(value):
    require(re.fullmatch(r'[0-9]+', value) is not None, 'invalid unsigned counter')
    return int(value)


def canonical_pci(value):
    require(isinstance(value, str) and
        re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', value) is not None,
        'invalid PCI identity')
    require(int(value[8:10], 16) < 32, 'invalid PCI device number')
    return value.lower()


def receipt_target(receipt, boot):
    require(re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', boot) is not None,
        'invalid current boot identity')
    require(receipt.get('boot_id') == boot, 'native receipt boot identity missing or changed')
    target = receipt['native']
    require(all(type(target.get(k)) is int and target[k] > 0 for k in ('pid', 'start_ticks', 'sid')),
        'invalid native identity')
    return {k: target[k] for k in ('pid', 'start_ticks', 'sid')}


def bounded_entries(path, label, limit=256):
    # Path.iterdir/listdir may materialize the entire directory before returning.
    paths = []
    with os.scandir(path) as entries:
        for entry in entries:
            require(len(paths) < limit, 'target ' + label + ' budget exceeded')
            unsigned(entry.name)
            paths.append(Path(entry.path))
    return sorted(paths, key=lambda p: int(p.name))


def parse_fdinfo(text):
    values = {}
    for line in text.splitlines():
        key, sep, value = line.partition(':')
        if not sep or not key.startswith('drm-'):
            continue
        require(key not in values, 'duplicate fdinfo key: ' + key)
        values[key] = value.strip()
    if values.get('drm-driver') != 'xe':
        return None
    client = unsigned(values['drm-client-id'])
    pci = canonical_pci(values['drm-pdev'])
    engines, memory = {}, {}
    for key, value in values.items():
        if key.startswith('drm-cycles-'):
            name = key.removeprefix('drm-cycles-')
            total = 'drm-total-cycles-' + name
            require(total in values and 'drm-maxfreq-' + name not in values, 'unsupported engine cycle denominator')
            capacity = unsigned(values.get('drm-engine-capacity-' + name, '1'))
            require(capacity > 0, 'zero engine capacity')
            engines[name] = dict(busy=unsigned(value), total=unsigned(values[total]), capacity=capacity)
        elif key.startswith(('drm-resident-', 'drm-total-', 'drm-shared-')) and not key.startswith('drm-total-cycles-'):
            match = re.fullmatch(r'([0-9]+)(?:\s+(KiB|MiB))?', value)
            require(match is not None, 'invalid memory counter')
            memory[key] = int(match[1]) * {None: 1, 'KiB': 1024, 'MiB': 1024**2}[match[2]]
    require(bool(engines), 'xe client exports no supported busy/total cycle pair')
    return dict(driver='xe', pci=pci, client_id=client, engines=engines, memory_bytes=memory)


def deduplicate_clients(fdinfos, pci):
    """First descriptor per DRM client; repeated reads are not simultaneous."""
    pci = canonical_pci(pci)
    clients = {}
    for fd, text in fdinfos:
        row = parse_fdinfo(text)
        if row is None or row['pci'] != pci:
            continue
        key = (row['driver'], row['pci'], row['client_id'])
        if key in clients:
            require({k: v['capacity'] for k, v in row['engines'].items()} ==
                {k: v['capacity'] for k, v in clients[key]['engines'].items()}, 'duplicate client capacity changed')
            clients[key]['descriptors'].append(fd)
        else:
            clients[key] = dict(row, descriptors=[fd])
    return list(clients.values())


def process_stat(path):
    # comm may itself contain parentheses; remaining numeric fields do not.
    fields = path.read_text().rsplit(')', 1)[1].split()
    require(len(fields) >= 37, 'incomplete process stat')
    return dict(state=fields[0], sid=int(fields[3]), start_ticks=int(fields[19]),
                cpu_ticks=int(fields[11]) + int(fields[12]), processor=int(fields[36]))


def snapshot(target, pci, proc=Path('/proc')):
    """Validate native PID/start/SID before AND after the bounded proc read."""
    root = proc / str(target['pid'])

    def checked_stat():
        row = process_stat(root / 'stat')
        require(all(row[k] == target[k] for k in ('start_ticks', 'sid')), 'target process identity changed')
        if row['state'] == 'Z':
            raise TargetExited()
        return row

    begin = time.monotonic_ns()
    checked_stat()
    paths = bounded_entries(root / 'fdinfo', 'descriptor')
    fdinfos = []
    for path in paths:
        try:
            with path.open() as stream:
                text = stream.read(65537)
            require(len(text) <= 65536, 'fdinfo text budget exceeded')
            fdinfos.append((int(path.name), text))
        except FileNotFoundError:
            continue  # A closed descriptor does not prove that the target exited.
    threads, lost = [], 0
    paths = bounded_entries(root / 'task', 'thread')
    for path in paths:
        try:
            threads.append(dict(tid=int(path.name), **process_stat(path / 'stat')))
        except FileNotFoundError:
            lost += 1
    final = checked_stat()
    return dict(begin_monotonic_ns=begin, end_monotonic_ns=time.monotonic_ns(), target=dict(target),
        process=final, threads=threads, disappeared_threads=lost,
        clients=deduplicate_clients(fdinfos, pci))


class CycleIntervals:
    """Keep the last good high-water mark; report gaps rather than clamp values."""
    def __init__(self):
        self.previous = {}
        self.target = None

    @staticmethod
    def baseline(counter, stamp):
        return dict(counter=dict(counter), stamp=stamp, recoveries=0,
                    high_busy=counter['busy'], high_total=counter['total'])

    def update(self, sample):
        require(type(sample['begin_monotonic_ns']) is int and type(sample['end_monotonic_ns']) is int and
            0 <= sample['begin_monotonic_ns'] <= sample['end_monotonic_ns'], 'invalid observation timestamps')
        if self.target is None:
            self.target = dict(sample['target'])
        require(self.target == sample['target'], 'interval target identity changed')
        seen, rows = set(), []
        for client in sample['clients']:
            for engine, current in client['engines'].items():
                key = (client['pci'], client['client_id'], engine)
                require(key not in seen, 'duplicate client engine in snapshot')
                seen.add(key)
                row = dict(pci=key[0], client_id=key[1], engine=engine, eligible=False)
                previous = self.previous.get(key)
                stamp = dict(begin_monotonic_ns=sample['begin_monotonic_ns'],
                             end_monotonic_ns=sample['end_monotonic_ns'])
                if previous is not None:
                    require(stamp['begin_monotonic_ns'] >= previous['stamp']['end_monotonic_ns'],
                            'overlapping or backwards observation span')
                if previous is None or current['capacity'] != previous['counter']['capacity']:
                    row['reason'] = 'baseline' if previous is None else 'capacity_changed'
                    self.previous[key] = self.baseline(current, stamp)
                elif (current['busy'] < previous['high_busy'] or current['total'] < previous['high_total'] or
                      current['total'] <= previous['counter']['total']):
                    row['reason'] = 'counter_below_high_water_or_nonadvancing_total'
                    previous['recoveries'] += 1
                    previous['high_busy'] = max(previous['high_busy'], current['busy'])
                    previous['high_total'] = max(previous['high_total'], current['total'])
                else:
                    busy = current['busy'] - previous['counter']['busy']
                    total = current['total'] - previous['counter']['total']
                    row.update(busy_cycles=busy, total_cycles=total, capacity=current['capacity'],
                        span_begin=previous['stamp'], span_end=stamp, recovered_samples=previous['recoveries'])
                    if busy > total * current['capacity']:
                        row['reason'] = 'busy_delta_exceeds_engine_capacity'
                        previous['recoveries'] += 1
                        previous['high_busy'] = max(previous['high_busy'], current['busy'])
                        previous['high_total'] = max(previous['high_total'], current['total'])
                    else:
                        row.update(eligible=True, busy_fraction=busy / (total * current['capacity']))
                        self.previous[key] = self.baseline(current, stamp)
                rows.append(row)
        self.previous = {k: v for k, v in self.previous.items() if k in seen}
        return rows


class CollectionEvidence:
    """Collector completion requires data; model/phase coverage is a separate gate."""
    def __init__(self):
        self.samples = self.client_samples = self.eligible_samples = self.eligible_intervals = 0

    def observe(self, row):
        self.samples += 1
        self.client_samples += bool(row['clients'])
        eligible = sum(item['eligible'] and item['engine'] == 'ccs' for item in row['engine_intervals'])
        self.eligible_samples += bool(eligible)
        self.eligible_intervals += eligible

    def terminal(self, reason, error=None):
        complete = reason in ('deadline', 'target_exited') and self.eligible_samples >= 2
        return dict(kind='terminal', reason=reason, error=error, samples=self.samples,
            client_samples=self.client_samples, expected_engine='ccs',
            eligible_samples=self.eligible_samples, eligible_intervals=self.eligible_intervals,
            minimum_eligible_samples=2, collection_complete=complete, model_phase_qualified=False,
            limits='Process-bounded collection with a maximum deadline. Completion requires two samples with eligible CCS intervals; planned phase coverage must be checked separately.')


def summarize_window(samples, begin, end, ticks_per_second):
    """Use only wholly contained observation spans; never extrapolate gaps."""
    engines, threads, cpu_ticks, wall, missing = {}, {}, 0, 0., 0
    for sample in samples:
        for interval in sample['engine_intervals']:
            if not interval['eligible'] or not (begin <= interval['span_begin']['begin_monotonic_ns'] and
                    interval['span_end']['end_monotonic_ns'] <= end):
                continue
            key=(interval['pci'],interval['client_id'],interval['engine'])
            value=engines.setdefault(key,dict(pci=key[0],client_id=key[1],engine=key[2],
                eligible_intervals=0,busy_cycles=0,capacity_total_cycles=0))
            value['eligible_intervals']+=1
            value['busy_cycles']+=interval['busy_cycles']
            value['capacity_total_cycles']+=interval['total_cycles']*interval['capacity']
    for first,last in zip(samples,samples[1:]):
        if not (begin <= first['begin_monotonic_ns'] and last['end_monotonic_ns'] <= end):continue
        elapsed=((last['begin_monotonic_ns']+last['end_monotonic_ns'])-
                 (first['begin_monotonic_ns']+first['end_monotonic_ns']))/2e9
        require(elapsed>0,'nonadvancing CPU observation interval')
        delta=last['process']['cpu_ticks']-first['process']['cpu_ticks']
        require(delta>=0,'process CPU counter decreased')
        cpu_ticks+=delta;wall+=elapsed
        previous={(v['tid'],v['start_ticks']):v for v in first['threads']}
        for row in last['threads']:
            key=(row['tid'],row['start_ticks'])
            if key not in previous:
                missing+=1;continue
            delta=row['cpu_ticks']-previous[key]['cpu_ticks']
            require(delta>=0,'thread CPU counter decreased')
            value=threads.setdefault(key,dict(tid=key[0],start_ticks=key[1],cpu_ticks=0,observed_last_cpus=set()))
            value['cpu_ticks']+=delta;value['observed_last_cpus'].add(row['processor'])
    require(wall>0 and any(v['engine']=='ccs' and v['eligible_intervals']>=2 for v in engines.values()),
        'insufficient contained CPU/CCS observation spans for the client interval')
    for value in engines.values():
        value['busy_fraction']=value['busy_cycles']/value['capacity_total_cycles']
    top=[]
    for value in sorted(threads.values(),key=lambda v:(-v['cpu_ticks'],v['tid']))[:8]:
        top.append(dict(value,observed_last_cpus=sorted(value['observed_last_cpus']),
            cpu_seconds=value['cpu_ticks']/ticks_per_second))
    return dict(window_begin_monotonic_ns=begin,window_end_monotonic_ns=end,
        contained_CPU_wall_seconds=wall,process_cpu_seconds=cpu_ticks/ticks_per_second,
        process_cpu_core_equivalents=cpu_ticks/ticks_per_second/wall,
        top_scheduled_threads=top,unmatched_thread_intervals=missing,
        engines=sorted(engines.values(),key=lambda v:(v['pci'],v['client_id'],v['engine'])))


def summarize_capture(records, target, boot, client_interval, *, before_only=False):
    require(type(before_only) is bool, 'explicit boolean observation scope required')
    require(len(records)>=5 and records[0].get('kind')=='header' and records[-1].get('kind')=='terminal',
        'missing resource header or explicit terminal')
    header,terminal=records[0],records[-1]
    require(header['schema']==2 and header['target']==target and header['boot_id']==boot and
        header['pci']==canonical_pci(header['pci']) and type(header['ticks_per_second']) is int and
        header['ticks_per_second']>0,'resource header identity or clock differs')
    samples=records[1:-1];intervals=CycleIntervals();evidence=CollectionEvidence()
    previous_cpu = -1
    for i,row in enumerate(samples):
        require(row.get('kind')=='sample' and row['ordinal']==i and row['target']==target and
            row['ticks_per_second']==header['ticks_per_second'] and
            row['process']['start_ticks']==target['start_ticks'] and row['process']['sid']==target['sid'],
            'resource sample identity, order or clock differs')
        require(intervals.update(row)==row['engine_intervals'],'resource cycle interval recomputation differs')
        require(row['process']['cpu_ticks'] >= previous_cpu, 'process CPU counter decreased')
        previous_cpu = row['process']['cpu_ticks']
        evidence.observe(row)
    require({k:v for k,v in terminal.items() if k!='end_monotonic_ns'}==
        evidence.terminal(terminal['reason'],terminal.get('error')) and
        terminal['collection_complete'],'resource collection is incomplete')
    keys=('request_begin_monotonic_ns','first_output_monotonic_ns','request_end_monotonic_ns')
    require(isinstance(client_interval,dict) and all(type(client_interval.get(k)) is int for k in keys),
        'missing exact client interval clocks')
    begin,first,end=(client_interval[k] for k in keys)
    require(0<=begin<first<end,'invalid client output interval')
    require(type(terminal.get('end_monotonic_ns')) is int and terminal['end_monotonic_ns']>=end and
        samples[0]['begin_monotonic_ns']<=begin and
        end-samples[-1]['end_monotonic_ns'] <= 2*header['interval']*1e9,
        'collector does not cover the completed client request')
    return dict(collector_complete=True,request_coverage_verified=True,samples=len(samples),
        final_sample_to_response_end_seconds=max(0,end-samples[-1]['end_monotonic_ns'])/1e9,
        before_first_output=summarize_window(samples,begin,first,header['ticks_per_second']),
        after_first_output=None if before_only else summarize_window(samples,first,end,header['ticks_per_second']),
        limits='Client observation windows, not native kernel-stage boundaries. Scheduled CPU includes polling/stalls. Associated DRM clients may be shared; cycle busy fraction is not arithmetic efficiency, VRAM bandwidth, physical PCIe traffic or hardware saturation.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_receipt', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--interval', type=float, default=1.)
    parser.add_argument('--pci', default='0000:05:00.0')
    args = parser.parse_args(argv)
    require(0 < args.seconds <= 3600 and .2 <= args.interval <= 10, 'finite sampling budget required')
    pci = canonical_pci(args.pci)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    target = receipt_target(json.loads(args.native_receipt.read_text()), boot)
    deadline, count = time.monotonic() + args.seconds, 0
    intervals = CycleIntervals()
    evidence = CollectionEvidence()
    reason, error = 'deadline', None
    with args.out.open('x') as stream:
        stream.write(json.dumps(dict(kind='header', schema=2, target=target, pci=pci, boot_id=boot,
            kernel=Path('/proc/sys/kernel/osrelease').read_text().strip(),
            ticks_per_second=os.sysconf('SC_CLK_TCK'), interval=args.interval, seconds=args.seconds,
            limits='Associated DRM-client busy cycles and scheduled CPU ticks; no stage alignment, FLOPs, VRAM bandwidth, PCIe bytes or throughput qualification.')) + '\n')
        try:
            while time.monotonic() < deadline:
                require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot, 'boot changed')
                try:
                    row = snapshot(target, pci)
                except (FileNotFoundError, TargetExited):
                    # Distinguish confirmed target loss from a transient missing proc entry.
                    try:
                        final = process_stat(Path('/proc', str(target['pid']), 'stat'))
                    except FileNotFoundError:
                        final = None
                    require(final is None or (final['state'] == 'Z' and
                        all(final[k] == target[k] for k in ('sid', 'start_ticks'))), 'incomplete proc snapshot or reused PID')
                    reason = 'target_exited'
                    break
                row.update(kind='sample', ordinal=count, ticks_per_second=os.sysconf('SC_CLK_TCK'),
                           engine_intervals=intervals.update(row))
                encoded = json.dumps(row) + '\n'
                require(stream.tell() + len(encoded.encode()) + 2048 <= 32 << 20, 'raw sample budget exceeded')
                stream.write(encoded); stream.flush()
                evidence.observe(row)
                count += 1
                time.sleep(min(args.interval, max(0, deadline-time.monotonic())))
        except (OSError, ValueError, KeyError) as exc:
            reason, error = 'error', (type(exc).__name__ + ': ' + str(exc))[:256]
        terminal = evidence.terminal(reason, error)
        terminal['end_monotonic_ns'] = time.monotonic_ns()
        stream.write(json.dumps(terminal) + '\n')
    return 0 if terminal['collection_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
