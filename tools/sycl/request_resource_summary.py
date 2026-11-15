#!/usr/bin/env python3
"""Conserved resource categories from the completed current request cohort.

Main-authored, repository LICENSE. This offline analysis does not execute the
engine. Copy sizes are requested API bytes, not measured PCIe traffic. Intel
PTI's M denotes ZE_MEMORY_TYPE_UNKNOWN, not proven host memory.
"""
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import request_profile_reader as reader

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
EVIDENCE = 'a1f90cdff2be49e2aaa64b70e21824bdca29434f'
RESULT = 'bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json'
KEY = 'unitrace_current_request_profile_v4'
COPY = re.compile(r'^zeCommandListAppendMemoryCopy\(([HDMS]2[HDMS])\)\[(\d+)\]$')
PLAN = dict(question='All copy directions and kernel families across each complete request',
    processes=[1, 2], repetition='all three requests separately; first-use and warmed never pooled',
    gate='Exact trace SHA and reproduced original request summary; closed normal cohort; '
         'conserved counts/service; crossing intervals reported separately',
    wall_seconds=180, CPU_seconds=120, RSS_bytes=12 << 30,
    text_bytes=128 << 10, file_bytes=2 << 20, output_bytes=4 << 20,
    GPU_executed=False, retention='Top twelve normalized kernel names plus conserved tail, '
    'all copy directions, interval unions and overlap; retire full traces after verification')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def blob(revision, path):
    return subprocess.check_output(['git', '-C', str(P), 'show', revision + ':' + path], timeout=30)


def summarize(trace, receipt, original):
    check = reader.summarize(trace, receipt)
    reader.require(all(check[k] == original[k] for k in check), 'original request summary mismatch')
    output = []
    for request, prior in zip(receipt['requests'], original['requests']):
        low, high = request['begin_raw_ns'], request['end_raw_ns']
        groups = defaultdict(lambda: dict(calls=0, service_ns=0, requested_API_bytes=0))
        kernels = defaultdict(lambda: dict(calls=0, service_ns=0))
        intervals = dict(copy=[], kernel=[], other_command=[])
        crossing = 0
        for event in trace['traceEvents']:
            if event.get('cat') != 'gpu_op': continue
            start = reader.nanoseconds(event['ts']); stop = start + reader.nanoseconds(event['dur'])
            if stop <= low or start >= high: continue
            name = event['name']; match = COPY.fullmatch(name)
            kind = 'copy' if match else ('other_command' if name.startswith('zeCommandList') else 'kernel')
            intervals[kind].append((max(start, low), min(stop, high)))
            if start < low or stop > high:
                crossing += 1
                continue
            if match:
                value = groups[match[1]]; value['requested_API_bytes'] += int(match[2])
            elif kind == 'other_command':
                value = groups[name]
            else:
                # Keep template parameters; combine launch geometries only.
                value = kernels[re.sub(r'\[SIMD[^\n]*$', '', name).rstrip()]
            value['calls'] += 1; value['service_ns'] += stop-start
        ranked = sorted(kernels.items(), key=lambda item: (-item[1]['service_ns'], item[0]))
        top, tail = ranked[:12], ranked[12:]
        counts = sum(x['calls'] for x in groups.values()) + sum(x['calls'] for x in kernels.values())
        service = sum(x['service_ns'] for x in groups.values()) + sum(x['service_ns'] for x in kernels.values())
        reader.require(counts == prior['contained_GPU_events'] and service == prior['contained_GPU_service_ns']
                       and crossing == prior['boundary_crossing_GPU_events'], 'conservation failure')
        unions = {kind: reader.union_ns(value) for kind, value in intervals.items()}
        all_union = reader.union_ns([v for values in intervals.values() for v in values])
        reader.require(all_union == prior['observed_GPU_interval_union_ns'], 'union mismatch')
        output.append(dict(request_id=request['request_id'], sample_scope=request['sample_scope'],
            request_sha256=request['request_sha256'], actual_read_tokens=request['actual_read_tokens'],
            generated_tokens=request['generated_tokens'], window_ns=high-low,
            contained_GPU_events=counts, contained_GPU_service_ns=service,
            boundary_crossing_GPU_events=crossing, copy_and_other_commands=dict(sorted(groups.items())),
            top_kernel_families=[dict(name=name, **value) for name,value in top],
            kernel_tail=dict(families=len(tail), calls=sum(v['calls'] for _,v in tail),
                             service_ns=sum(v['service_ns'] for _,v in tail)),
            interval_union_ns=unions, all_interval_union_ns=all_union,
            copy_kernel_overlap_ns=unions['copy']+unions['kernel']-
                reader.union_ns(intervals['copy']+intervals['kernel'])))
    return dict(requests=output, totals_match_original=True,
        physical_bus_bytes_qualified=False, hardware_saturation_qualified=False,
        phase_attribution_qualified=False, clean_speed_qualified=False)


def main(index, out):
    for name in ('tools/sycl/request_resource_summary.py', 'tools/test_request_resource_summary.py',
                 'tools/sycl/request_profile_reader.py', 'tools/sycl/measurement_owner.py'):
        reader.require((P/name).read_bytes() == blob('HEAD',name), 'commit exact analysis inputs: '+name)
    cohort = json.loads(blob(EVIDENCE, RESULT))[KEY]
    reader.require(index in PLAN['processes'] and cohort['complete'] and cohort['passed']
        and cohort['ownership_closed'] and not cohort['active']
        and cohort['decision']['observer_overhead_qualified'], 'closed qualified observer cohort required')
    row = cohort['processes'][index]
    reader.require(row['native_exit_code'] == 0 and row['ownership_closed'] and row['fresh_faults'] == []
        and row['boot_unchanged'] and not row['clean_speed'], 'closed traced target required')
    paths = list((B/'current-request-profile-v4'/str(index)/'trace').glob('*.json'))
    reader.require(len(paths) == 1 and digest(paths[0]) == row['trace_sha256'], 'exact complete raw trace required')
    bounds = row['enclosing_owned_run_raw_ns']
    receipt = dict(schema=reader.SCHEMA, clock_domain=reader.CLOCK, pti_revision=reader.PTI_REVISION,
        unitrace_system_time='1', level_zero_only=True, native_exit_code=0, ownership_closed=True,
        fresh_faults=[], boot_unchanged=True, source=row['request_profile']['source'],
        native_lifetime=dict(begin_raw_ns=bounds['begin'], end_raw_ns=bounds['end']),
        requests=[dict(request_id=str(i), request_sha256=s['request_sha256'], actual_read_tokens=s['actual_read_tokens'],
            expected_read_tokens=32769, actual_reused_tokens=s['actual_reused_tokens'],
            generated_tokens=s['generated_tokens'], thinking=True, sample_scope='first_long' if i==0 else 'warmed',
            begin_raw_ns=w['begin'], end_raw_ns=w['end']) for i,(s,w) in
            enumerate(zip(row['samples'],row['external_request_raw_windows']))])
    result = summarize(reader.load_json(paths[0]),receipt,row['request_profile'])
    result.update(evidence_commit=EVIDENCE, trace_sha256=row['trace_sha256'], process_index=index,
        analysis_commit=subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip(),
        plan=PLAN, source=receipt['source'],
        copy_type_reference=dict(repository='intel/pti-gpu', revision=reader.PTI_REVISION,
            path='tools/unitrace/src/levelzero/ze_collector.h', meaning='H host allocation,D device,M unknown,S shared'),
        limits='Requested bytes are not physical PCIe/VRAM traffic. Whole request includes24decode tokens. '
               'Kernel service, interval union and critical-path wall time differ; no saturation claim.')
    with out.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(process=index,requests=len(result['requests']),totals_match_original=True)))


def run_owned():
    import measurement_owner as owner
    root=B/'current-request-resource-v1'
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Acquire the existing measurement lock and refuse a duplicate campaign.
        root.mkdir(exist_ok=False)
        source=subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()
        inputs={name:digest(P/name) for name in ('tools/sycl/request_resource_summary.py',
            'tools/test_request_resource_summary.py','tools/test_request_profile_reader.py',
            'tools/sycl/request_profile_reader.py','tools/sycl/measurement_owner.py')}
        reader.require(all((P/name).read_bytes()==blob(source,name) for name in inputs),
                       'commit all analysis/host/owner inputs before launch')
        recipe=dict(source_commit=source,inputs=inputs,plan=PLAN,commands=[])
        (root/'record.json').write_text(json.dumps(recipe,indent=2)+'\n')
        commands=[('host',[sys.executable,'-B','-m','unittest',
            'tools.test_request_resource_summary','tools.test_request_profile_reader'])]
        commands += [(str(i),[sys.executable,'-B','tools/sycl/request_resource_summary.py',str(i),
            str(root/(str(i)+'.json'))]) for i in PLAN['processes']]
        for label,command in commands:
            host=label=='host'
            entry,out,err=owner.Owner(root/label).run(label,command,
                dict(PATH='/usr/bin:/bin',LANG='C',PYTHONDONTWRITEBYTECODE='1'),P,
                wall=90 if host else PLAN['wall_seconds'],cpu=60 if host else PLAN['CPU_seconds'],
                rss_cap=1<<30 if host else PLAN['RSS_bytes'],text_cap=PLAN['text_bytes'],
                file_cap=PLAN['file_bytes'],total_cap=PLAN['output_bytes'])
            good=owner.closed(entry) and entry['exit_code']==0
            if host:good=good and 'Ran 11 tests' in err.read_text() and err.read_text().rstrip().endswith('OK')
            recipe['commands'].append(dict(label=label,command=command,passed=bool(good),
                exit_code=entry['exit_code'],ownership_closed=bool(owner.closed(entry))))
            (root/'record.json').write_text(json.dumps(recipe,indent=2)+'\n')
            reader.require(good,'owned resource analysis failed: '+label+' '+err.read_text()[-2000:])
            print('Closed resource analysis stage '+label,flush=True)


if __name__ == '__main__':
    if sys.argv[1:]==['run']:run_owned()
    else:
        reader.require(len(sys.argv)==3,'run | process_index output.json')
        main(int(sys.argv[1]),Path(sys.argv[2]))
