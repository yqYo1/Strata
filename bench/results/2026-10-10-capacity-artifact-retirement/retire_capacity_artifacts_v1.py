import collections, csv, datetime, fcntl, hashlib, json, re, shutil, subprocess
from pathlib import Path

B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
Q=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-qualification-20261010')
A=W/'bench/results/2026-10-10-capacity-artifact-retirement'
CP=W/'bench/results/2026-10-10-cpu-capacity'
P=W/'bench/results/2026-10-10-hardware-concurrency/profile'
def ident(p):
    p=Path(p);return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def closed(r):
    assert r['passed'] and r['complete'] and not r['active']
    for c in r['commands']:
        assert c['normal_exit'] and c['exit_code']==0 and c['session_empty'] and c['observation_complete']
        assert c['direct_child_reaped'] and not c['errors'] and not c['survivors']
        for o in c['owners']:
            p=Path('/proc')/str(o['pid'])/'stat'
            if p.exists():assert int(p.read_text().rsplit(')',1)[1].split()[19])!=o['start_ticks']
def tracked(p,cwd=W):
    subprocess.run(['git','ls-files','--error-unmatch',str(Path(p).relative_to(cwd))],cwd=cwd,check=True,stdout=subprocess.DEVNULL)
def available():
    s=__import__('os').statvfs(B);return s.f_bavail*s.f_frsize

with open(B/'owned-v0141-measurement.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()=='7ea2650ba3ae1b15807a32a67b62d56131046569'
    status=subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
    assert not status or status=='?? bench/results/2026-10-10-capacity-artifact-retirement/'
    assert not A.exists() or {p.name for p in A.iterdir()}<={'native-pool-diagnostic-summary.json'};A.mkdir(exist_ok=True)
    entries=[];pool_summaries=[]
    def admit(path,reason,replacements):
        p=Path(path);assert p.is_file() and not p.is_symlink()
        assert B in p.parents
        for q in replacements:tracked(q,Q if Q in Path(q).parents else W)
        entries.append(dict(path=str(p),**ident(p),allocated_bytes=p.stat().st_blocks*512,reason=reason,replacements=[dict(path=str(q),**ident(q)) for q in replacements],status='verified-ready-to-retire'))
    plan=json.loads((B/'native-capacity-v2-matrix-plan-v1.json').read_text());assert plan['passed'] and plan['complete'] and not plan['active'] and len(plan['closed'])==24
    canonical=json.loads((CP/'selected-payload-identity-rows.json').read_text())
    keep={'ROUND','WARMUP','ROUND_COUNTS','REFERENCE_COUNTS','REFERENCE_WORST','GATE','ROUTE','WORKING_SET','ACTIVE_LOGICAL_WORKING_SET','HOT_ID','PLACEMENT','TASK_PLAN','RESULT','META','ENV'}
    for cell in plan['closed']:
        p=Path(cell['path']);assert ident(p)['sha256']==cell['sha256'];r=json.loads(p.read_text());closed(r)
        q=CP/'native'/(p.parent.name+'.json');a=json.loads(q.read_text());tracked(q)
        assert a['original_receipt']==dict(path=str(p),**ident(p))
        for k,v in r.items():
            if k!='output_rows':assert a[k]==v
        rows=r['output_rows'];assert a['output_rows']==[x for x in rows if x[0] in keep]
        assert [x for x in rows if x[0] in ('ID','EXTENT')]==canonical
        refs=[x for x in rows if x[0]=='REFERENCE_TOKEN']
        assert hashlib.sha256(json.dumps(refs,separators=(',',':')).encode()).hexdigest()==a['all_reference_vectors_original_sha256']
        assert len(refs)==384*cell['NT']*3
        for stage,info in a['verified_reference_summary'].items():
            ss=[x for x in refs if x[6]==stage]
            assert info==dict(vectors=len(ss),rows=sum(int(x[13]) for x in ss),maxabs=max(float(x[7]) for x in ss),max_nrms=max(float(x[8]) for x in ss))
        assert set(x[0] for x in rows)<=keep|{'ID','EXTENT','REFERENCE_TOKEN','REFERENCE'}
        aggregate=[x for x in rows if x[0]=='REFERENCE'];assert len(aggregate)==384
        stdout=p.parent/'native-cell.stdout';stderr=p.parent/'native-cell.stderr'
        for log in (stdout,stderr):assert ident(log)==r['commands'][0]['logs'][log.name]
        assert list(csv.reader(stdout.read_text().splitlines()))==rows
        # Record each already-completed pool state without repeated English messages.
        states=[];diags=[];threads=[];thread_states=[]
        for line in stderr.read_text().splitlines():
            m=re.fullmatch(r'  expert pool: epoch (\d+), batch epoch (\d+): (\d+) of (\d+) jobs claimed, (\d+) done; (\d+) of (\d+) workers parked, (\d+) sleeping; mode (\d+)',line)
            if m:
                v=list(map(int,m.groups()));assert v[0]==v[1] and v[2]==v[3]==v[4]==18 and v[5]==v[6]==5 and 0<=v[7]<=5 and v[8]==0;states.append(v);continue
            m=re.fullmatch(r'  expert pool threads: w0=(parked|sleeping) w1=(parked|sleeping) w2=(parked|sleeping) w3=(parked|sleeping) w4=(parked|sleeping); host idle for (\d+) ms',line)
            if m:thread_states.append(list(m.groups()[:5]));threads.append(int(m.group(6)));continue
            m=re.fullmatch(r'DIAG,(\d+),(-?\d+)',line)
            assert m,('unexpected stderr',line);diags.append(list(map(int,m.groups())))
        assert len(states)==len(threads)==57 and len(diags)==56
        assert diags==[[cohort,round_] for cohort in (0,1) for round_ in range(-3,25)]
        pool_summaries.append(dict(cell={k:cell[k] for k in ('NT','mode','repeat')},stderr_identity=ident(stderr),state_columns=['epoch','batch_epoch','claimed','jobs','done','parked','workers','sleeping','mode'],states=states,thread_states=thread_states,host_idle_ms=threads,DIAG_markers=diags,aggregate_REFERENCE_count=len(aggregate),aggregate_REFERENCE_maxabs_by_GU_Down_full=[max(float(x[i]) for x in aggregate) for i in (5,6,7)],aggregate_REFERENCE_sha256=hashlib.sha256(json.dumps(aggregate,separators=(',',':')).encode()).hexdigest()))
        reason='Closed successful timing process; all 250 measured and 30 warmup rows, source/environment/ownership/status and reference count/worst/hash already committed; canonical payload IDs once. Four qualifier receipts retained as live reproduction inputs.'
        for path in (p,stdout,stderr):admit(path,reason,[q,CP/'selected-payload-identity-rows.json',CP/'native-capacity-v2-matrix-plan-v1.json'])
    write(A/'native-pool-diagnostic-summary.json',dict(scope='All original 57 completed pool states and 56 DIAG markers per process, with idle milliseconds; no worker CPU-utilization or active service inference. Aggregate references are redundant with independently emitted per-token reference checks.',processes=pool_summaries))
    profile=json.loads((B/'hardware-concurrency-v1-profile/record.json').read_text());closed(profile)
    analysis=json.loads((P/'profile-analysis.json').read_text());trace=Path(analysis['trace']['path'])
    assert ident(trace)=={k:analysis['trace'][k] for k in ('bytes','sha256')}
    assert ident(B/'hardware-concurrency-v1-profile/profile-analysis.json')==ident(P/'profile-analysis.json')
    assert ident(B/'hardware-concurrency-v1-profile/compact-operation-intervals.json')==ident(P/'compact-operation-intervals.json')
    compact=json.loads((P/'compact-operation-intervals.json').read_text());assert len(compact['rows'])==analysis['GPU_operations']==3330
    assert analysis['device_copy_GEMM_intersection_ns']==0
    admit(trace,'Raw-consumer independent R225 review completed and committed in round60. Root complete-operation analysis plus all 3330 normalized GPU intervals preserve the resolved overlap question; no active complete-history consumer remains.',[P/'profile-analysis.json',P/'compact-operation-intervals.json',P/'record.json',W/'bench/results/2026-10-10-parallel-round60/research/round225-hardware-concurrency-profile-evidence-audit.txt'])
    qr=json.loads((B/'prefill-service-small-device-qualification-v1/record.json').read_text());closed(qr)
    qp=Q/'bench/results/2026-10-10-prefill-service-device-qualification'
    qs=json.loads((qp/'diagnostic-trace-summary.json').read_text());trace=Path(qs['original_path'])
    assert ident(trace)=={k:qs[k] for k in ('bytes','sha256')}
    lines=trace.read_text().splitlines();assert len(lines)==qs['lines']==1446
    assert len(re.findall(r'-> UR_RESULT_SUCCESS;',trace.read_text()))==qs['UR_results']['UR_RESULT_SUCCESS']==210
    verified_windows=[]
    for window in qs['relevant_windows']:
        starts=[i for i in range(len(lines)) if lines[i:i+len(window['lines'])]==window['lines']]
        assert len(starts)==1
        verified_windows.append(dict(original_label=window['first_line'],window_first_line_one_based=starts[0]+1,exact_window_match=True))
    write(A/'qualification-window-verification.json',dict(original_summary=ident(qp/'diagnostic-trace-summary.json'),windows=verified_windows,clarification='Original first_line labels 838/886 do not identify the beginning of the stored window; exact verified window beginnings are one-based 837/885. Original summary remains unchanged; text windows, successful result counts and qualification outcome agree.'))
    admit(trace,'Healthy small qualification closed and expected graph-capture query-false statuses resolved; complete API chatter is represented by result counts, relevant exact windows, receipt, faults and source/binary identities.',[qp/'diagnostic-trace-summary.json',qp/'device-record.json',qp/'README.md'])
    for e in entries:
        if '/native-capacity-v2-measure-' in e['path']:e['additional_replacement']=dict(path=str(A/'native-pool-diagnostic-summary.json'),**ident(A/'native-pool-diagnostic-summary.json'))
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Only resolved successful duplicate native timing logs and two raw diagnostic traces. No source/model/pack/binary, original failed status, canonical tensor/session, or four active native qualifier fixtures removed.',entries=entries,files=len(entries),logical_bytes=sum(e['bytes'] for e in entries),allocated_bytes=sum(e['allocated_bytes'] for e in entries),available_bytes_before=available(),status='validated-before-retirement')
    write(A/'retirement-manifest.json',manifest);shutil.copyfile(__file__,A/Path(__file__).name)
    (A/'README.md').write_text('Closed native timing receipts are normalized in ../2026-10-10-cpu-capacity with every individual timing, exact original hashes/statuses, full command/ownership/policy, independent numerical summaries and canonical payload identities. This retirement verifies all retained rows and reference summaries against originals before deletion. Additional compact evidence preserves each completed pool diagnostic state. Four original qualifiers remain controller inputs; original failed qualification remains failed. Complete PTI and successful small API trace consumer questions are resolved; committed summaries retain their relevant evidence. The manifest preserves retired path, size/hash, replacement and actual allocation, separately from filesystem free-space change.\n')
    for args in [('diff','--check'),('add',str(A.relative_to(W))),('diff','--cached','--check'),('commit','-m','bench: verify compact evidence before retiring resolved capacity logs')]:subprocess.run(['git',*args],cwd=W,check=True)
    for e in entries:
        path=Path(e['path']);assert ident(path)=={k:e[k] for k in ('bytes','sha256')};path.unlink();e['status']='retired'
    assert all(not Path(e['path']).exists() for e in entries)
    for nt in (1,2,3,4):assert (B/f'native-capacity-v2-qualify-nt{nt}-streaming-r1-controller2/record.json').exists()
    assert not json.loads((B/'native-capacity-v2-qualify-nt1-streaming-r1/record.json').read_text())['passed']
    manifest.update(status='retirement-complete',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),available_bytes_after=available())
    manifest['available_bytes_change']=manifest['available_bytes_after']-manifest['available_bytes_before'];write(A/'retirement-manifest.json',manifest)
    for args in [('add',str((A/'retirement-manifest.json').relative_to(W))),('diff','--cached','--check'),('commit','-m','bench: record completed retirement of duplicate capacity evidence'),('push',)]:subprocess.run(['git',*args],cwd=W,check=True)
    print(json.dumps({k:manifest[k] for k in ('status','files','logical_bytes','allocated_bytes','available_bytes_before','available_bytes_after','available_bytes_change')}))
