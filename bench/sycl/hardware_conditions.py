"""Read-only host/xe conditions for an owned model diagnostic (repository LICENSE).

Main-authored from Linux ABIs, not another engine's implementation:
https://docs.kernel.org/gpu/xe/xe_gt_freq.html
https://docs.kernel.org/hwmon/sysfs-interface.html
https://docs.kernel.org/admin-guide/pm/cpufreq.html
https://docs.kernel.org/accounting/psi.html
These samples do not measure physical bus bytes, FLOPs or CPU APERF/MPERF.
"""
import os
from pathlib import Path
import re
import statistics
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    with path.open() as stream:
        value = stream.read(16385)
    require(len(value) <= 16384, 'condition source text budget')
    return value.strip()


def natural(value):
    require(re.fullmatch(r'[0-9]+',value), 'malformed unsigned condition')
    return int(value)


class Conditions:
    def __init__(self, pci, sysfs=Path('/sys'), proc=Path('/proc')):
        require(re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-1][0-9a-f]\.[0-7]', pci), 'PCI selector')
        self.proc = proc
        self.device = sysfs / 'bus/pci/devices' / pci
        self.identity = {k: read(self.device / k) for k in ('vendor', 'device')}
        require(self.identity == dict(vendor='0x8086', device='0xe20c'), 'expected B570 identity')
        self.sources = {}
        for root in sorted(self.device.glob('tile*/gt*/freq0')):
            tag = str(root.relative_to(self.device)).removesuffix('/freq0')
            for name in ('act_freq', 'cur_freq', 'min_freq', 'max_freq', 'rp0_freq', 'power_profile',
                         'throttle/status', 'throttle/reasons'):
                self.add(tag + '/' + name, root / name, 'text' if name in ('power_profile', 'throttle/reasons') else 'uint')
        for root in sorted(self.device.glob('hwmon/hwmon*')):
            require(read(root / 'name') == 'xe', 'unexpected GPU hwmon provider')
            for path in sorted(root.iterdir()):
                if re.fullmatch(r'(?:temp\d+_input|energy\d+_input|power\d+_cap|fan\d+_input)', path.name):
                    stem, kind = path.name.rsplit('_', 1)
                    label = root / (stem + '_label')
                    name = read(label) if label.exists() else stem
                    self.add('gpu/' + stem + '/' + name + '/' + kind, path, 'uint')
        for root in sorted((sysfs / 'devices/system/cpu/cpufreq').glob('policy*')):
            for name in ('scaling_cur_freq', 'scaling_min_freq', 'scaling_max_freq',
                         'scaling_governor', 'scaling_driver', 'energy_performance_preference'):
                self.add('cpu/' + root.name + '/' + name, root / name,
                         'uint' if name.endswith('_freq') else 'text')
        # Enumerate only this bounded PCI hierarchy, not all host devices.
        root = self.device.resolve()
        for _ in range(8):
            if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', root.name):
                break
            for name in ('current_link_speed', 'current_link_width', 'max_link_speed', 'max_link_width'):
                self.add('pci/' + root.name + '/' + name, root / name, 'text')
            root = root.parent
        require('tile0/gt0/act_freq' in self.sources, 'no primary GT clock capability')
        require(len(self.sources) <= 192, 'condition source count budget')

    def add(self, key, path, kind):
        if path.exists():
            require(key not in self.sources, 'duplicate condition key')
            self.sources[key] = dict(path=str(path), kind=kind)

    def describe(self):
        return dict(schema=1, device=dict(path=str(self.device), **self.identity), sources=self.sources,
            page_size_bytes=os.sysconf('SC_PAGE_SIZE'),
            scope='Global device/host signals sampled serially,not atomic or exclusive to this model.',
            units='GPU frequency MHz;cpufreq-reported frequency kHz;temp input millidegreesC;energy input microjoules;power cap microwatts;proc counters raw.',
            limits='CPU scaling_cur_freq is a reported signal,not an APERF/MPERF effective-clock measurement. PCI link fields do not prove wiring/bus traffic. Counts of sampled throttle flags are not time durations. No saturation or clean-speed qualification.')

    def capture(self):
        begin = time.monotonic_ns()
        require({k: read(self.device / k) for k in self.identity} == self.identity, 'GPU identity changed')
        values, errors = {}, {}
        for key, source in self.sources.items():
            try:
                value = read(Path(source['path']))
                if source['kind'] == 'uint':
                    value = natural(value)
                values[key] = value
            except (OSError, ValueError) as error:
                errors[key] = type(error).__name__
        host = {}
        for name in ('stat', 'vmstat', 'meminfo', 'pressure/cpu', 'pressure/io', 'pressure/memory'):
            try:
                text = read(self.proc / name)
                if name == 'stat':
                    line = next(v for v in text.splitlines() if v.startswith('cpu '))
                    fields = line.split()[1:9]
                    require(len(fields) == 8 and all(re.fullmatch(r'[0-9]+', v) for v in fields), 'host CPU counters')
                    host[name] = [int(v) for v in fields]
                elif name == 'vmstat':
                    wanted = {'pswpin','pswpout','pgmajfault','pgfault','pgpgin','pgpgout'}
                    host[name] = {k: natural(v) for k,v in (line.split() for line in text.splitlines()) if k in wanted}
                elif name == 'meminfo':
                    host[name] = {k: natural(v) for k,v in re.findall(r'^(MemAvailable|SwapFree|SwapTotal):\s+(\d+) kB$', text, re.M)}
                else:
                    host[name] = {line.split()[0]: natural(line.rsplit('total=',1)[1]) for line in text.splitlines()}
            except (OSError, ValueError, StopIteration, IndexError) as error:
                errors['host/' + name] = type(error).__name__
        return dict(begin_monotonic_ns=begin,end_monotonic_ns=time.monotonic_ns(),values=values,host=host,errors=errors)


def interval_delta(first, last):
    """Unknown/reset counters cannot masquerade as zero activity."""
    if first is None or last is None or type(first) is not int or type(last) is not int or last < first:
        return None
    return last - first


def summarize(records, begin, end):
    require(records[0]['kind'] == 'header' and records[0].get('conditions',{}).get('schema') == 1,
            'hardware-enabled capture header required')
    require(records[-1]['kind'] == 'terminal' and records[-1]['collection_complete'], 'closed capture required')
    selected=[]
    previous=-1
    for row in records[1:-1]:
        require(row.get('kind')=='sample' and isinstance(row.get('conditions'),dict), 'missing condition sample')
        c=row['conditions'];lo,hi=c['begin_monotonic_ns'],c['end_monotonic_ns']
        require(type(lo) is int and type(hi) is int and previous<lo<=hi and
                row['end_monotonic_ns']<=lo, 'condition read clock differs')
        previous=hi
        if begin<=row['begin_monotonic_ns']<=lo<=hi<=end:selected.append(row)
    samples=[v['conditions'] for v in selected]
    require(len(samples) >= 10 and end > begin, 'insufficient contained hardware samples')
    sources = records[0]['conditions']['sources']
    gauges, text, counters = {}, {}, {}
    for key, source in sources.items():
        values = [s['values'][key] for s in samples if key in s['values']]
        if source['kind'] == 'text':
            text[key] = dict(observed=len(values),counts={v:values.count(v) for v in sorted(set(values))})
        elif '/energy' in key:
            entries=[];invalid=0
            for first,last in zip(samples,samples[1:]):
                delta=interval_delta(first['values'].get(key),last['values'].get(key))
                seconds=(last['begin_monotonic_ns']-first['begin_monotonic_ns'])/1e9
                if delta is None or seconds <= 0:invalid+=1
                else:entries.append((delta,seconds))
            counters[key]=dict(valid_intervals=len(entries),invalid_intervals=invalid,
                average_watts=sum(v for v,t in entries)/1e6/sum(t for v,t in entries) if entries and not invalid else None)
        else:
            require(all(type(v) is int and v >= 0 for v in values), 'invalid gauge')
            gauges[key]=dict(observed=len(values),missing=len(samples)-len(values),
                minimum=min(values) if values else None,median=statistics.median(values) if values else None,
                maximum=max(values) if values else None,zero_samples=values.count(0))
            if key.endswith('/act_freq'):
                active=[v for v in values if v>0]
                gauges[key]['nonzero_median']=statistics.median(active) if active else None
    host={}
    for name in ('vmstat','pressure/cpu','pressure/io','pressure/memory'):
        keys=set().union(*(s['host'].get(name,{}) for s in samples))
        host[name]={}
        for key in sorted(keys):
            deltas=[interval_delta(a['host'].get(name,{}).get(key),b['host'].get(name,{}).get(key)) for a,b in zip(samples,samples[1:])]
            host[name][key]=dict(sum=sum(x for x in deltas if x is not None),
                valid_intervals=sum(x is not None for x in deltas),invalid_intervals=deltas.count(None))
    cpu=[]
    for a,b in zip(samples,samples[1:]):
        x,y=a['host'].get('stat'),b['host'].get('stat')
        if x is None or y is None:continue
        deltas=[interval_delta(u,v) for u,v in zip(x,y)]
        if None in deltas or sum(deltas)<=0:continue
        cpu.append((sum(deltas)-deltas[3]-deltas[4],sum(deltas),deltas[4],deltas[7]))
    host['CPU']=dict(valid_intervals=len(cpu),busy_fraction=sum(x[0] for x in cpu)/sum(x[1] for x in cpu) if cpu else None,
        iowait_fraction=sum(x[2] for x in cpu)/sum(x[1] for x in cpu) if cpu else None,
        steal_fraction=sum(x[3] for x in cpu)/sum(x[1] for x in cpu) if cpu else None)
    host['native_faults']={}
    for name in ('minor_faults','major_faults'):
        deltas=[interval_delta(a['process'].get(name),b['process'].get(name)) for a,b in zip(selected,selected[1:])]
        host['native_faults'][name]=dict(sum=sum(x for x in deltas if x is not None),
            valid_intervals=sum(x is not None for x in deltas),invalid_intervals=deltas.count(None))
    errors={}
    for sample in samples:
        for key,value in sample['errors'].items():errors.setdefault(key,{});errors[key][value]=errors[key].get(value,0)+1
    primary=gauges.get('tile0/gt0/act_freq',{})
    require(primary.get('observed',0) >= .95*len(samples), 'primary GT clock coverage failed')
    return dict(contained_samples=len(samples),gauges=gauges,text=text,energy=counters,host_counters=host,errors=errors,
        maximum_reader_ms=max(s['end_monotonic_ns']-s['begin_monotonic_ns'] for s in samples)/1e6,
        reader_wall_sum_ms=sum(s['end_monotonic_ns']-s['begin_monotonic_ns'] for s in samples)/1e6,
        clean_speed_qualified=False,cpu_effective_frequency_qualified=False,hardware_saturation_qualified=False,
        limits='Wholly contained read spans;serial global samples are not device-stage clocks or exclusive model demand. Zero GPU frequency retained separately. Read span cost is not a causal observer-overhead bound;CPU reported frequency is not APERF/MPERF.')
