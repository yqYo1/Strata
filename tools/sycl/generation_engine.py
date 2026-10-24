#!/usr/bin/env python3
"""Shared source and request-evidence gates for official long-generation diagnostics.

Closed comparison/native-runtime launchers are preserved in original Git history.
The maintained resource_engine.py owns current passive model execution.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import bench_engine as common
import measurement_guards as guards
from build_engine import blobs

P, B, A = common.P, common.B, common.A
RESULT = A / 'decode-long-generation-current.json'
SETTINGS = common.SETTINGS
WORKLOAD_ID = 'native16_decode_long_20261019'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P)


def committed(path):
    data = path.read_bytes()
    guards.require(data == git('show', 'HEAD:' + str(path.relative_to(P))),
                   'commit the exact input first: ' + str(path))
    return json.loads(data)


def benchmark_pins(revision, match_worktree=True):
    names = git('ls-tree', '-r', '--name-only', revision, '--', 'serve', 'tools', 'bench.sh').decode().splitlines()
    names = [n for n in names if n == 'bench.sh' or
             (n.endswith('.py') and not n.startswith('tools/hip/'))]
    pins = {}
    for name, data in blobs(P, revision, names):
        if match_worktree:
            guards.require((P / name).read_bytes() == data, 'uncommitted benchmark input: ' + name)
        pins[name] = hashlib.sha256(data).hexdigest()
    return pins


def native_end(root, record, plan, boot, leader):
    """Require the exact owned native exit, with only a declared passive reader."""
    start=json.loads((root/'engine-start.json').read_text())
    end=json.loads((root/'engine-exit.json').read_text())
    allowed={'binary','args','boot_id','native','wrapper','native_validation_probe'}
    guards.require(set(start)<=allowed and start['boot_id']==boot and
        start['binary']==record['binary'] and start['args']==['--serve',*plan['config']['args']] and
        start.get('native_validation_probe')==record['native_validation_probe'], 'native start identity differs')
    identities=[]
    for key in ('native','wrapper'):
        identity=start[key]
        guards.require(set(identity)=={'pid','start_ticks','sid'} and
            all(type(v) is int and v>0 for v in identity.values()) and identity['sid']==leader,
            'native ownership identity differs')
        identities.append(identity['pid'])
    expected=dict(start,exit_code=0)
    collection=record.get('resource_collection')
    if collection is not None:
        native_plan=json.loads((root/'plan.json').read_text())
        guards.require(native_plan.get('resource_collection')==collection and
            set(collection)=={'source','seconds','interval'} and
            set(collection['source'])=={'path','sha256'} and
            all(type(collection[k]) in (int,float) and math.isfinite(collection[k]) for k in ('seconds','interval')) and
            0<collection['seconds']<=900 and .2<=collection['interval']<=10 and
            common.digest(collection['source']['path'])==collection['source']['sha256'],
            'passive collector differs from the frozen native plan')
        identity=end['resource_collector']
        guards.require(set(identity)=={'pid','start_ticks','sid'} and
            all(type(v) is int and v>0 for v in identity.values()) and identity['sid']==leader and
            identity['start_ticks']>=start['native']['start_ticks'] and
            type(end.get('resource_collector_exit_code')) is int and end['resource_collector_exit_code']==0,
            'passive collector ownership/exit differs')
        identities.append(identity['pid'])
        expected.update(resource_collector=identity,resource_collector_exit_code=0)
    guards.require(len(set(identities))==len(identities) and type(end.get('exit_code')) is int and end==expected,
        'native exit/observer identity differs')
    return end


def draft_rows(text):
    guards.require(not re.search(r'^strata (?:draft kv|serve: SNAPSHOT_VERIFY)', text, re.M),
                   'unexpected timer or snapshot-restore observer')
    pattern = r'strata serve: DRAFT_PREFILL path=(batched|token) mode=(-?\d+) cells=(\d+)'
    parsed = []
    for line in text.splitlines():
        if not line.startswith('strata serve: DRAFT_PREFILL'): continue
        match = re.fullmatch(pattern, line)
        guards.require(match is not None, 'malformed draft-KV marker')
        parsed.append(match.groups())
    return [dict(path=path, mode=int(mode), cells=int(cells)) for path, mode, cells in parsed]


def request_evidence(text, samples):
    """Bind draft work to every native request, including all four warmups."""
    headers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$', text, re.M))
    guards.require(len(headers) == len(samples), 'missing/extra native request footer')
    previous = 0; evidence = []
    for header, sample in zip(headers, samples):
        guards.require(tuple(map(int, header.groups())) == (sample['prompt_tokens'], sample['reused_tokens'],
            sample['read_tokens'], sample['generated_tokens']) and sample['reused_tokens'] == 0,
            'native/journal request binding differs')
        rows = draft_rows(text[previous:header.start()]); cells = sample['read_tokens'] - 1
        expected = [8192] * (cells // 8192) + ([cells % 8192] if cells % 8192 else [])
        guards.require(rows and all(r['path'] == 'batched' and r['mode'] == 2 for r in rows)
            and [r['cells'] for r in rows] == expected, 'draft-KV marker route/chunks do not conserve request')
        evidence.append(dict(read_tokens=sample['read_tokens'], draft_cells=sum(r['cells'] for r in rows),
            batch_count=len(rows), path='batched', mode=2))
        previous = header.end()
    guards.require(not draft_rows(text[previous:]), 'unbound trailing draft batch')
    return evidence


def iq2_pair_evidence(text, samples, arm, expected_cpus=None):
    """Bind drained host counters to requests; observers remain charged to timing.

    Registration identifies threads only. The request boundary is owned by the
    synchronous driver, not inferred from TLS. This proves dispatch/work, not
    clean speed or CPU/DRAM saturation.
    """
    guards.require(arm in (0,1), 'unknown IQ2 pair arm')
    headers=list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, '
        r'(\d+) generated[^\n]*$',text,re.M))
    guards.require(len(headers)==len(samples),'IQ2 audit/native request extent')
    header=re.compile(r'strata IQ2_GU_AUDIT stage=(begin|end) request=(\d+) enabled=(\d+) overflow=(\d+) threads=(\d+)')
    thread=re.compile(r'strata IQ2_GU_THREAD stage=(begin|end) request=(\d+) slot=(\d+) tid=(\d+) '
        r'calls=(\d+) row_ops=(\d+) nt1_rows=(\d+) pair_rows=(\d+) cpu=(-?\d+) pinned=(\d+)')
    fields=('calls','row_ops','nt1_rows','pair_rows');limit=(1<<64)-1
    result=[];previous=0
    for request,(native,sample) in enumerate(zip(headers,samples),1):
        guards.require(tuple(map(int,native.groups()))==(sample['prompt_tokens'],sample['reused_tokens'],
            sample['read_tokens'],sample['generated_tokens']),'IQ2 audit/native sample binding')
        lines=[s for s in text[previous:native.start()].splitlines() if s.startswith('strata IQ2_GU_')]
        offset=0;snapshots=[]
        for stage in ('begin','end'):
            m=header.fullmatch(lines[offset]) if offset<len(lines) else None;offset+=1
            guards.require(m is not None,'missing/malformed IQ2 audit snapshot')
            phase,req,enabled,overflow,size=m.groups();size=int(size)
            guards.require(phase==stage and int(req)==request and enabled=='1' and overflow=='0' and
                0<=size<=64,'IQ2 audit phase/request/overflow')
            snapshot=[]
            for slot in range(size):
                m=thread.fullmatch(lines[offset]) if offset<len(lines) else None;offset+=1
                guards.require(m is not None,'missing/malformed IQ2 audit thread')
                phase,req,index,tid,*counts,cpu,pinned=m.groups();values=dict(zip(fields,map(int,counts)))
                guards.require(phase==stage and int(req)==request and int(index)==slot and
                    0<int(tid)<=limit and int(cpu)>=0 and pinned in ('0','1') and
                    all(0<=v<=limit for v in values.values()) and
                    values['pair_rows']<=values['nt1_rows']<=values['row_ops'], 'IQ2 audit thread identity/bounds')
                snapshot.append(dict(tid=int(tid),cpu=int(cpu),pinned=int(pinned),**values))
            guards.require(len({t['tid']for t in snapshot})==len(snapshot),'duplicate IQ2 audit thread')
            snapshots.append(snapshot)
        before,after=snapshots
        guards.require(offset==len(lines) and len(before)<=len(after),'extra IQ2 audit rows/lost thread')
        totals=dict.fromkeys(fields,0);active=[]
        for slot,end in enumerate(after):
            begin=before[slot] if slot<len(before) else dict.fromkeys(fields,0)
            if slot<len(before):
                guards.require(all(begin[k]==end[k]for k in ('tid','cpu','pinned')),'IQ2 audit thread changed')
            delta={k:end[k]-begin[k]for k in fields}
            guards.require(all(v>=0 for v in delta.values()) and
                delta['pair_rows']<=delta['nt1_rows']<=delta['row_ops'],'IQ2 audit counter regression/delta')
            for k,v in delta.items():
                totals[k]+=v;guards.require(totals[k]<=limit,'IQ2 audit aggregate overflow')
            if delta['row_ops']>0:
                guards.require(delta['calls']>0,'IQ2 audit work without callback');active.append(end)
        guards.require(totals['pair_rows']==(totals['nt1_rows']if arm else 0),'IQ2 actual selected dispatch differs')
        cpus=sorted(t['cpu']for t in active)
        if expected_cpus is not None and sample['read_tokens']>=32768:
            guards.require(totals['nt1_rows']>0 and cpus==sorted(expected_cpus) and
                all(t['pinned']==1 for t in active),'IQ2 long request missing actual pinned work')
        result.append(dict(request=request,**totals,active_threads=len(active),
            pinned_threads=sum(t['pinned']for t in active),cpus=cpus))
        previous=native.end()
    guards.require(not any(s.startswith('strata IQ2_GU_')for s in text[previous:].splitlines()),
        'unbound trailing IQ2 audit')
    return result


def decode_work_evidence(text, samples):
    """Existing trace/footer counts: logical work, never physical RAM/PCIe traffic."""
    headers=list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, '
        r'(\d+) generated[^\n]*, drafts accepted (\d+) of (\d+),[^\n]*$',text,re.M))
    guards.require(len(headers)==len(samples),'missing/extra decode work headers')
    keys=('submitted_blobs','submitted_bytes','completed_blobs','completed_bytes','host_view_groups',
          'gpu_miss_groups','cpu_unique_misses','cpu_entries')
    pattern=r'^strata serve: SYCL_EXPERT_COPY '+r' '.join(k+r'=(\d+)' for k in keys)+r'$'
    guards.require(len(re.findall(pattern,text,re.M))==len(samples),'missing/extra copy/work footer')
    result=[];previous=0
    for i,(header,row) in enumerate(zip(headers,samples)):
        guards.require(tuple(map(int,header.groups()))==(row['prompt_tokens'],row['reused_tokens'],row['read_tokens'],
            row['generated_tokens'],row['drafts_accepted'],row['drafts_offered']) and row['reused_tokens']==0,
            'native/client decode work identity differs')
        segment=text[previous:header.start()]
        windows=[tuple(map(int,m)) for m in re.findall(r'^strata trace: window (\d+) (\d+)$',segment,re.M)]
        guards.require(windows and windows[0]==(row['read_tokens']-1,1) and
            all(t==4 for _,t in windows[1:]) and
            all(1<=b[0]-a[0]<=a[1] for a,b in zip(windows,windows[1:])) and
            sum(t-1 for _,t in windows)==row['drafts_offered'],'actual fixed-spec4 target windows differ')
        tail=text[header.end():headers[i+1].start() if i+1<len(headers) else len(text)]
        parsed=re.findall(pattern,tail,re.M)
        guards.require(len(parsed)==1,'work footer escaped its completed request')
        counters=dict(zip(keys,map(int,parsed[0])))
        guards.require(all(counters[k]==0 for k in keys[:6]),'expert-copy route differs from admitted pcie0')
        jobs,entries=counters['cpu_unique_misses'],counters['cpu_entries']
        target_rows=sum(t for _,t in windows)
        guards.require(0<jobs<=entries<=4*jobs and entries<=target_rows*48*10,'CPU job/entry conservation differs')
        n=row['generated_tokens']
        result.append(dict(target_windows=len(windows),target_rows=target_rows,emitted_tokens=n,
            target_rows_per_emitted=target_rows/n,cpu_unique_jobs=jobs,cpu_entries=entries,
            cpu_unique_jobs_per_emitted=jobs/n,minimum_NT1_jobs=max(0,2*jobs-entries),
            minimum_NT1_job_fraction=max(0,2*jobs-entries)/jobs,
            expert_copy_requested_bytes=counters['submitted_bytes'],
            mathematical_dot_equivalent_ops=entries*9830400,
            limits='Spec4,48 layers,top10,pcie0,no peer. NT1 bound is max(0,2*jobs-entries). Dot equivalents exclude decode/index/quant/nonlinear work and are not issued FP/integer instructions. Blob visits and expert-copy counters are not physical DRAM/PCIe bytes; no saturation or critical-path qualification.'))
        previous=header.end()
    return result


def decode_timing_evidence(text, samples):
    """Legacy spec4 work contract through the shared T1/T4 footer parser."""
    import decode_footer
    work = decode_work_evidence(text, samples)
    try:
        parsed = decode_footer.analyze(text, samples, work)
    except ValueError as error:
        guards.require(False, str(error))
    return [dict(windows=v['target_windows'], ms_per_window=v['ms_per_window'],
        outer_residual_ms_per_window=v['outer_residual_ms_per_window'],
        host_nested_residual_ms_per_window=v['host_nested_residual_ms_per_window'], work=w,
        clock_binding=dict(v['clock_binding'], average_quantum_ms=.01, DONE_total_quantum_ms=.1),
        limits='Existing rounded nested host wall clocks,not additive hardware costs. CPU_pool_run includes scheduling/waits;GPU_reach_wait is not device arithmetic service. No clean-speed/saturation qualification.')
        for v, w in zip(parsed, work)]


def sample_eligible(row, maximum_output=16384):
    """Independently reject mislabeled short, incomplete or unclocked samples."""
    def integer(key, minimum):
        value = row.get(key)
        return type(value) is int and value >= minimum

    offered, accepted = row.get('drafts_offered'), row.get('drafts_accepted')
    return bool(type(maximum_output) is int and maximum_output in (16384,32768,65536) and
        row.get('eligible') is True and row.get('speed_eligible') is True and
        row.get('ok') is True and row.get('stream_complete') is True and
        row.get('finish_reason') in ('stop', 'length') and row.get('reused_tokens') == 0 and
        integer('prompt_tokens', 32768) and integer('read_tokens', 32768) and
        row['prompt_tokens'] == row['read_tokens'] and integer('usage_generated_tokens', 1024) and
        row['usage_generated_tokens'] <= maximum_output and
        row.get('generated_tokens') == row['usage_generated_tokens'] and
        row.get('generated_tokens_source') == 'usage' and integer('output_chars', 512) and
        integer('reasoning_chars', 1) and row.get('completion_includes_thinking') is True and
        type(offered) is int and offered >= 256 and type(accepted) is int and 0 <= accepted <= offered and
        isinstance(row.get('acceptance_rate'), (int, float)) and
        math.isclose(row['acceptance_rate'], accepted / offered, rel_tol=0, abs_tol=1e-12) and
        row.get('decode_tps_source') == 'engine' and all(
            isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool) and
            math.isfinite(row[key]) and row[key] > 0 for key in ('decode_ms', 'decode_tps', 'total_s')))


def collect_samples(generation, maximum_output=16384):
    # Preserve every completed/failed client row before assessing eligibility.
    rows = []
    for workload, case in generation.get('workloads', {}).items():
        for request in case.get('requests', []):
            row = dict(request, workload=workload)
            row['controller_eligible'] = sample_eligible(row, maximum_output)
            rows.append(row)
    return sorted(rows, key=lambda r: r.get('order', -1))


def require_complete_generation(generation, rows, workloads=('regular-code', 'varied-prose')):
    guards.require(generation.get('workload_version') == 4 and generation.get('thinking') is True and
        generation.get('reasoning_budget_tokens') == 0 and generation.get('max_tokens') == 16384,
        'unexpected long-generation workload or thinking/output budget')
    guards.require(len(rows) == len(workloads) and
        [(r.get('workload'), r.get('run'), r.get('order')) for r in rows] ==
        [(name, 0, i) for i, name in enumerate(workloads)] and
        all(r['controller_eligible'] for r in rows), 'ineligible or incomplete long-generation pilot')
    if len(workloads) == 1:
        guards.require(generation.get('diagnostic_subset') is True and
            generation.get('selected_workloads') == list(workloads), 'single-workload diagnostic selection differs')


def journal_evidence(root):
    sys.path.insert(0, str(P / 'tools'))
    from strata_bench import verify_request_journal
    paths = list((root / 'bench').glob('*/requests.jsonl'))
    guards.require(len(paths) == 1, 'missing or ambiguous official request journal')
    journal = verify_request_journal(paths[0].parent)
    rows = []
    for line in paths[0].read_text().splitlines()[:journal['complete_prefix_rows']]:
        row = json.loads(line)
        if row['kind'] == 'result' and row['phase'].startswith('long-generation/'):
            rows.append(dict(row['sample'], workload=row['phase'].split('/', 1)[1],
                run=row['run'], order=len(rows)))
    return journal, rows


def warmup_reconciliation(entries, workloads=('regular-code',), maximum_output=16384):
    """Require the actual official warmups, not just a successful final request."""
    expected = [('warm-up/chat', None, 24, True), ('calibration', 200, 1, False),
                ('calibration', 800, 1, False), ('warm-up/prompt', None, 8, True)]
    guards.require(type(maximum_output) is int and maximum_output in (16384,32768,65536),
        'unsupported long-generation output budget')
    expected += [('long-generation/' + name, 0, maximum_output, True) for name in workloads]
    events = [row for row in entries if row['kind'] != 'identity']
    guards.require(len(events) == 2 * len(expected), 'missing or extra official request events')
    hashes = []
    for i, (phase, run, maximum, thinking) in enumerate(expected):
        start, result = events[2*i:2*i+2]
        guards.require(start['kind'] == 'start' and result['kind'] == 'result' and
            all(row.get('phase') == phase and row.get('run') == run and row.get('turn') is None
                for row in (start, result)) and start.get('max_tokens') == maximum and
            start.get('thinking') is thinking and result.get('request_index') == start['index'] and
            result.get('sample', {}).get('ok') is True and
            result['sample'].get('request_sha256') == start.get('request_sha256') and
            re.fullmatch('[0-9a-f]{64}', start.get('request_sha256', '')) is not None,
            'official warmup/request identity, order or success differs')
        hashes.append(start['request_sha256'])
    return dict(successful_warmups=4, successful_long_requests=len(workloads),
                phase_order=[row[0] for row in expected], request_sha256=hashes)


def warmup_evidence(root, expected_journal):
    journal, _ = journal_evidence(root)
    guards.require(journal['valid'] and journal['complete'] and journal == expected_journal,
        'warmup journal is invalid, incomplete or differs from the closed receipt')
    path, = (root / 'bench').glob('*/requests.jsonl')
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    return dict(warmup_reconciliation(entries), journal_sha256=common.digest(path),
                journal_state_sha256=common.digest(path.with_name('journal-state.json')))


def runtime_assets_unchanged(record):
    delta = record.get('runtime_asset_delta', {})
    guards.require(not delta or record['mode'] in ('cpu-check', 'cpu-overhead', 'engine-compare', 'client-resources', 'pool-capacity', 'arena-copy-check', 'arena-generation-check', 'arena-generation-screen', 'early-pcie-check', 'early-generation-check', 'early-generation-screen', 'cache-owned-check', 'cache-capacity-clean'), 'proposed runtime is diagnostic-only')
    settings = json.loads(SETTINGS.read_text())
    guards.require(set(delta) <= set(settings['runtime_elfs']), 'unknown proposed runtime asset')
    for name, item in settings['runtime_elfs'].items():
        proposed = delta.get(name)
        if proposed:
            guards.require(proposed['qualified_sha256'] == item['sha256'], 'original runtime baseline changed')
        expected = proposed['candidate_sha256'] if proposed else item['sha256']
        guards.require(common.digest(name) == expected, 'runtime asset changed: ' + name)
