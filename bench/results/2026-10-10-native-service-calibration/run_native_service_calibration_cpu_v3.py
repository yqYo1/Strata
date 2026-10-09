"""Root-owned exact-cell streaming CPU calibration; never an engine benchmark."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-role-plan-v0141-20261010')
GGML = Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
PACK = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
PRIMARY = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
BUILD = B / 'native-service-calibration-cpu-build-v2/record.json'
ROUTE = B / 'owned-cache-route-pairs-v0141-code32k-tasks6-diagnostic-r6/record.json'
PARSER = B / 'cache_route_pairs_parser_v1.py'
OWNER = B / 'native-service-calibration-cohort-22-20-nt1-v2'
SELECTION_SEED = 'strata-native-service-20261010-v1'
ROUND_SEED = 2026101001
PINS = {
    BUILD: '17751d280ac856f301f1b87c5ab72d6585c7fbb346d5dfbca305afd11633a07d',
    ROUTE: 'f606f5a1f2d00dba240213a7feeb3cc84c9da4f1487075d1e97c336a558e4c89',
    PARSER: '2ff44099f7385054dc5663f04a8e226d364d4d349a84b2493bc2bb1d6f3585cd',
    PACK / 'native_experts.txt': 'd9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d',
    PACK / 'conversions.json': '51df3cd6ffd0d38c2da8e2d3a95a604b4a96c20b06fc639a9bf477282a42b86a',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def identity(path):
    path = Path(path)
    s = path.stat()
    return dict(path=str(path), bytes=s.st_size, dev=s.st_dev, ino=s.st_ino,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def members(pgid):
    result = []
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():
            continue
        try:
            words = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(words[2]) == pgid and words[0] != 'Z':
                result.append(dict(pid=int(p.name), start_ticks=int(words[19]),
                                   rss_bytes=int(words[21]) * os.sysconf('SC_PAGE_SIZE')))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def observation():
    cpu = Path('/proc/cpuinfo').read_text()
    governor = {}
    for p in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor')):
        governor[str(p)] = p.read_text().strip()
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                loadavg=Path('/proc/loadavg').read_text().strip(),
                cpu_model=next(line.partition(':')[2].strip() for line in cpu.splitlines() if line.startswith('model name')),
                flags=next(line.partition(':')[2].strip() for line in cpu.splitlines() if line.startswith('flags')),
                allowed_cpu_list=sorted(os.sched_getaffinity(0)),
                governors=governor)


def check_build(build):
    assert build['passed'] and not build['active'] and not build['cleanup'] and not build['survivors']
    assert not build['gpu_work_submitted'] and not build['inference_run']
    for path, pin in PINS.items():
        assert sha(path) == pin, str(path)
    for path, pin in build['pins'].items():
        assert sha(path) == pin, path
    for rel, pin in build['production_source_pins'].items():
        assert sha(W / rel) == pin, rel
    for rel, pin in build['ggml_source_pins'].items():
        assert sha(GGML / rel) == pin, rel
    assert sha(build['binary']['path']) == build['binary']['sha256']


def select_cohort():
    route = json.loads(ROUTE.read_text())
    assert route['completed'] and route['healthy'] and route['math_gate_passed'] and route['pair_shape_gate_passed']
    assert not route['active'] and route['exit_code'] == 0
    assert route['cleanup'] == dict(forced=False, inferior_survived=False, gdb_survived=False)
    spec = importlib.util.spec_from_file_location('pinned_route_parser', PARSER)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    profile = Path(route['profile']['path'])
    assert sha(profile) == route['profile']['sha256']
    residents, fingerprint = parser.static_profile(profile.read_bytes())
    parsed = []
    for ordinal, request in enumerate(route['requests'], 1):
        replay = request['cache_route_pairs']['replay_input']
        data = Path(replay['stderr_path']).read_bytes()[replay['begin']:replay['end']]
        assert hashlib.sha256(data).hexdigest() == replay['sha256']
        done = [x for x in request['protocol'] if x.startswith('DONE ')]
        assert len(done) == 1
        frame = parser.parse_request(data, ordinal, done[0], residents, fingerprint)
        assert frame['passed'] and frame['end'] == request['cache_route_pairs']['end']
        parsed.append({(p['layer'], p['expert']): p for p in frame['pairs']})
    assert len(parsed) == 4 and parsed[0] == parsed[2] and parsed[1] == parsed[3]
    layers = {}
    for line in (PACK / 'native_experts.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        layers[int(fields[0])] = dict(gu=int(fields[1]), down=int(fields[2]), blob_bytes=int(fields[4]))
    assert set(layers) == set(range(48))
    eligible = {}
    for key in sorted(set(parsed[0]) | set(parsed[1])):
        l, e = key
        if (layers[l]['gu'], layers[l]['down']) != (22, 20):
            continue
        counts = []
        for frame in parsed[:2]:
            p = frame.get(key)
            if p is None:
                counts.append(0)
            else:
                assert 0 <= p['entries'] - p['callback_pairs'] <= p['callback_pairs']
                counts.append(2 * p['callback_pairs'] - p['entries'] if p['refused'] else 0)
        if sum(counts) > 0:
            eligible.setdefault(l, []).append(dict(layer=l, expert=e, nt1_canonical_AB=counts,
                                                   frequency=sum(counts)))
    selected = {'train': [], 'holdout': []}
    strata = []
    layer_ids = sorted(eligible)
    assert len(layer_ids) == 15
    # Fixed layer-balanced counts, equal disjoint train/holdout per stratum.
    # Frequency tertiles are ranked by canonical A+B counts, not timing or repeats.
    for rank, l in enumerate(layer_ids):
        quota = 192 // len(layer_ids) + (rank < 192 % len(layer_ids))
        ids = sorted(eligible[l], key=lambda x: (x['frequency'], x['expert']))
        for band in range(3):
            rows = ids[len(ids) * band // 3:len(ids) * (band + 1) // 3]
            count = quota // 3 + (band < quota % 3)
            assert len(rows) >= 2 * count
            rows.sort(key=lambda x: hashlib.sha256(f"{SELECTION_SEED}:{l}:{band}:{x['expert']}".encode()).digest())
            for i, x in enumerate(rows[:2 * count]):
                selected['train' if i % 2 == 0 else 'holdout'].append({**x, 'frequency_tertile': band})
            strata.append(dict(layer=l, frequency_tertile=band, eligible=len(rows),
                               each_cohort=count, minimum_frequency=min(x['frequency'] for x in rows),
                               maximum_frequency=max(x['frequency'] for x in rows)))
    assert all(len(rows) == 192 for rows in selected.values())
    ids = [(x['layer'], x['expert']) for rows in selected.values() for x in rows]
    assert len(set(ids)) == 384
    total = sum(layers[l]['blob_bytes'] for l, e in ids)
    assert total <= 1 << 30
    return dict(gu=22, down=20, nt=1, selection_seed=SELECTION_SEED, round_seed=ROUND_SEED,
                provenance='Only actual nonresident NT1 IDs in canonical A/B of closed r6; A3/B4 duplicates excluded from frequency. The prompts are dependent, not independent workload holdout.',
                selection='Equal train/holdout per layer and rank-frequency tertile; frozen SHA256 ordering, never timing-selected. Layer-balanced format-kernel service; not routing-frequency-weighted latency.',
                full_blob_bytes=total, strata=strata, cohorts=selected,
                route_receipt_sha256=PINS[ROUTE], parser_sha256=PINS[PARSER])


def run_child(record, out, label, argv, env, expect_error=None, wall=600):
    dest = out / label
    dest.mkdir(mode=0o700)
    stdout, stderr = dest / 'stdout.csv', dest / 'stderr.txt'
    entry = dict(label=label, argv=argv, environment=env, cwd=str(W),
                 start_observation=observation(), deadline_seconds=wall)
    record['commands'].append(entry)
    record['phase'] = label
    save(record, out)
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = None
    try:
        with stdout.open('wb') as so, stderr.open('wb') as se:
            def limits():
                for kind, pair in ((resource.RLIMIT_AS, (96 << 30, 96 << 30)),
                                   (resource.RLIMIT_CPU, (900, 901)),
                                   (resource.RLIMIT_FSIZE, (4 << 20, 4 << 20)),
                                   (resource.RLIMIT_NOFILE, (256, 256)),
                                   (resource.RLIMIT_CORE, (0, 0))):
                    resource.setrlimit(kind, pair)
            proc = subprocess.Popen(argv, cwd=W, env=env, stdout=so, stderr=se,
                                    preexec_fn=limits, start_new_session=True)
            stat = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()
            entry['owner'] = dict(pid=proc.pid, pgid=proc.pid, start_ticks=int(stat[19]))
            save(record, out)
            while proc.poll() is None:
                assert time.monotonic() - start < wall, 'child wall deadline'
                files = [p for p in out.rglob('*') if p.is_file()]
                assert sum(p.stat().st_size for p in files) <= 8 << 20, 'aggregate text budget'
                rss = sum(x['rss_bytes'] for x in members(proc.pid))
                entry['peak_group_rss_bytes'] = max(entry.get('peak_group_rss_bytes', 0), rss)
                assert rss <= 1536 << 20, 'owned RSS limit'
                time.sleep(.1)
        entry['exit_code'] = proc.returncode
        assert not members(proc.pid), 'owned descendants survived'
        if expect_error is None:
            assert proc.returncode == 0, 'child failure'
        else:
            assert proc.returncode == 1 and expect_error in stderr.read_text(), 'negative admission did not reject as expected'
        return list(csv.reader(stdout.read_text().splitlines())), stderr.read_text()
    finally:
        if proc is not None:
            current = members(proc.pid)
            if proc.poll() is None or current:
                record['cleanup'].append(dict(label=label, signal='TERM', pgid=proc.pid, members=current))
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                if proc.poll() is None or members(proc.pid):
                    record['cleanup'].append(dict(label=label, signal='KILL', pgid=proc.pid))
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=5)
            record['survivors'].extend(members(proc.pid))
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        entry.update(elapsed_seconds=time.monotonic()-start, end_observation=observation(),
                     whole_child_rusage_delta=dict(user_seconds=after.ru_utime-before.ru_utime,
                                                  system_seconds=after.ru_stime-before.ru_stime,
                                                  minor_faults=after.ru_minflt-before.ru_minflt,
                                                  major_faults=after.ru_majflt-before.ru_majflt),
                     files={str(p):dict(bytes=p.stat().st_size, sha256=sha(p)) for p in dest.rglob('*') if p.is_file()})
        save(record, out)


def save(record, out):
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')


def expected_orders(seed):
    mask = (1 << 64) - 1
    state = seed
    result = {}
    def next64():
        nonlocal state
        state = (state + 0x9e3779b97f4a7c15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
        return z ^ (z >> 31)
    for cohort in (0, 1):
        for r in range(28):
            order = list(range(cohort*192, (cohort+1)*192))
            for i in range(191, 0, -1):
                j = next64() % (i+1)
                order[i], order[j] = order[j], order[i]
            h = 14695981039346656037
            for v in order:
                for byte in v.to_bytes(4, 'little'):
                    h = ((h ^ byte) * 1099511628211) & mask
            result[(cohort, r-3)] = str(h)
    return result


def validate_rows(rows, manifest, task, batch, correctness):
    end = ['RESULT', 'correctness_only_pass', 'no_calibration_rounds'] if correctness else ['RESULT', 'correctness_pass', 'statistical_holdout_adoption_and_full_lifecycle_not_qualified']
    assert rows[-1] == end
    ids = [x for x in rows if x[:1] == ['ID']]
    assert len(ids) == 384
    expected = [(cohort, i + cohort*192, item['layer'], item['expert'])
                for cohort, label in enumerate(('train', 'holdout')) for i, item in enumerate(manifest['cohorts'][label])]
    assert [tuple(map(int, x[1:5])) for x in ids] == expected
    reference = [x for x in rows if x[:1] == ['REFERENCE']]
    assert len(reference) == 384 and all(len(x) == 7 for x in reference)
    assert [tuple(map(int, x[1:5])) for x in reference] == [(i, l, e, 1) for co, i, l, e in expected]
    assert all(float(x[5]) == float(x[6]) == 0 for x in reference), 'NT1 independent GGML must be exact'
    extents = [x for x in rows if x[:1] == ['EXTENT']]
    assert len(extents) == 1152 and all(len(x) == 9 for x in extents)
    extent_keys = [tuple(map(int, x[1:5])) for x in extents]
    assert len(set(extent_keys)) == 1152
    assert set(extent_keys) == {(co, l, e, role) for co, i, l, e in expected for role in range(3)}
    assert [x for x in rows if x[:1] == ['TASK_PLAN']] == [['TASK_PLAN', str(task or 18), str(task or 18)]]
    env = {x[1]:x[2] for x in rows if x[:1] == ['ENV']}
    assert 'STRATA_IQ_PREFETCH' in env and all(v == '<unset>' for v in env.values())
    workers = [x for x in rows if x[:2] == ['PLACEMENT', 'planned_worker']]
    assert len(workers) == 5
    host = [x for x in rows if x[:2] == ['PLACEMENT', 'planned_host']]
    assert len(host) == 1 and host[0][2] == host[0][3]
    rounds = [x for x in rows if x[:1] == ['ROUND'] and x[1] != 'cohort']
    warmups = [x for x in rows if x[:1] == ['WARMUP']]
    orders = expected_orders(ROUND_SEED)
    assert len(rounds) == (0 if correctness else 250) and len(warmups) == (0 if correctness else 30)
    seen = set()
    for x in rounds + warmups:
        assert len(x) == 18
        cohort, arm, r = int(x[1]), x[2], int(x[3])
        assert cohort in (0, 1) and arm in ('GU','FFquant','Down','direct_complete','pool')
        assert r in (range(-3, 0) if x[0] == 'WARMUP' else range(25))
        key = (cohort, arm, r)
        assert key not in seen
        seen.add(key)
        assert tuple(map(int, x[4:8])) == (22,20,1,task)
        assert tuple(map(int, x[8:11])) == ((batch,192//batch,192) if arm == 'pool' else (1,192,192))
        assert all(math.isfinite(float(v)) and float(v) >= 0 for v in x[12:16])
        assert float(x[12]) > 0 and x[17] == host[0][2]
        assert x[11] == orders[(cohort, r)], 'independent frozen schedule hash'
    # Bind each arm to exactly the same frozen order per cohort/round.
    for cohort in (0, 1):
        for r in range(25):
            same = [x for x in rounds if int(x[1]) == cohort and int(x[3]) == r]
            if not correctness:
                assert len(same) == 5 and len({x[11] for x in same}) == 1
    return extents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('correctness', 'timing'))
    ap.add_argument('--tasks', type=int, choices=(0,6), default=6)
    ap.add_argument('--batch', type=int, choices=(1,6), default=6)
    ap.add_argument('--repeat', type=int, choices=range(1,5), default=1)
    args = ap.parse_args()
    assert __debug__
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        build=json.loads(BUILD.read_text()); check_build(build)
        binary=build['binary']['path']; env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        name=f'native-service-calibration-22-20-nt1-{args.mode}-tasks{args.tasks}-batch{args.batch}-r{args.repeat}'
        out=B/name;out.mkdir(mode=0o700)
        record=dict(active=True,complete=False,passed=False,scope='Actual-weight homogeneous-cell standalone CPU calibration, not model/GPU inference',
                    controller_sha256=sha(__file__),build_receipt_sha256=PINS[BUILD],binary=build['binary'],
                    input_pins={str(p):h for p,h in PINS.items()},commands=[],cleanup=[],survivors=[],
                    gpu_work_submitted=False,inference_run=False,performance_eligible=False,statistical_qualified=False,
                    adopted=False,full_lifecycle_passed=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    limits=dict(AS_each_bytes=96<<30,RSS_group_poll_bytes=1536<<20,CPU_each_soft_seconds=900,
                                CPU_each_hard_seconds=901,FSIZE_each_bytes=4<<20,NOFILE=256,CORE_bytes=0),
                    text_budget_bytes=8<<20,configuration=dict(tasks=args.tasks,batch=args.batch,nt=1,gu=22,down=20,round_seed=ROUND_SEED),
                    timing_scope='Resident owned weight cohort beyond L3; no DRAM/engine-latency claim. Fixed synthetic layer/token inputs. Direct complete outer wall time; pool phase completion is separate, never divided into per-job service.')
        save(record,out);start=time.monotonic()
        try:
            if args.mode=='correctness':
                assert not OWNER.exists()
                OWNER.mkdir(mode=0o700)
                manifest=select_cohort()
                cohort=OWNER/'cohort.tsv'
                cohort.write_text('cohort\tlayer\texpert\n'+''.join(f"{label}\t{x['layer']}\t{x['expert']}\n" for label in ('train','holdout') for x in manifest['cohorts'][label]))
                manifest['cohort_sha256']=sha(cohort)
                (OWNER/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            else:
                admission=B/'native-service-calibration-22-20-nt1-correctness-tasks6-batch6-r2/record.json'
                prior=json.loads(admission.read_text());assert prior['complete'] and prior['passed'] and not prior['active'] and not prior['cleanup'] and not prior['survivors']
                assert sha(OWNER/'manifest.json')==prior['cohort']['manifest_sha256'] and sha(OWNER/'cohort.tsv')==prior['cohort']['tsv_sha256']
                record['correctness_receipt_sha256']=sha(admission)
                manifest=json.loads((OWNER/'manifest.json').read_text());cohort=OWNER/'cohort.tsv'
            record['cohort']=dict(manifest_sha256=sha(OWNER/'manifest.json'),tsv_sha256=sha(cohort),owner=str(OWNER),full_blob_bytes=manifest['full_blob_bytes'])
            conversions=json.loads((PACK/'conversions.json').read_text())
            model=[identity(PRIMARY.parent/x['name']) for x in conversions['source_shards']]
            assert all(x['bytes']==y['size'] for x,y in zip(model,conversions['source_shards']))
            record['model_identity']=model;record['whole_model_payload_hashed']=False
            def argv(tsv=cohort,gu=22):
                return [binary,str(PACK),str(PRIMARY),str(tsv),str(gu),'20','1',str(args.tasks),str(args.batch),str(ROUND_SEED)]
            if args.mode=='correctness':
                negatives=[('wrong-header','cohort\tlayer\tid\n','cohort header'),
                           ('wrong-count','cohort\tlayer\texpert\ntrain\t2\t0\n','exact192 train and holdout required'),
                           ('overlap','cohort\tlayer\texpert\ntrain\t2\t0\nholdout\t2\t0\n','duplicate/global train-holdout overlap'),
                           ('out-of-range','cohort\tlayer\texpert\ntrain\t48\t0\n','cohort ID range')]
                for label,text,error in negatives:
                    p=out/(label+'.tsv');p.write_text(text)
                    run_child(record,out,label,argv(p),env,expect_error=error,wall=20)
                run_child(record,out,'wrong-format',argv(gu=18),env,expect_error='nonhomogeneous format/dimensions',wall=30)
                trace=out/'correctness-syscalls.txt'
                launch=['/usr/bin/strace','-f','-yy','-s','512','-e','trace=open,openat,openat2,close,close_range,mmap,ioctl,execve','-o',str(trace)]+argv()+['--correctness-only']
                rows,stderr=run_child(record,out,'correctness-observed',launch,{**env,'LD_DEBUG':'libs'})
                extents=validate_rows(rows,manifest,args.tasks,args.batch,True)
                syscalls=trace.read_text()
                assert not re.search(r'/dev/(dri|nvidia|kfd)|lib(?:sycl|ur_|ze_loader|mkl|igc|intelocl)',syscalls,re.I),'device/runtime path observed'
                objects=sorted(set(re.findall(r'calling init: (.+)',stderr)))
                assert objects and ('transferring control: '+binary) in stderr
                assert not any(re.search(r'lib(?:sycl|ur_|ze_loader|mkl|igc|intelocl)',p,re.I) for p in objects)
                record['runtime_audit']=dict(scope='Pre-exec to exit strace-f-yy open/ioctl/loader paths and LD_DEBUG=libs; no device/runtime observed; not universal device-API interposition.',observed_loader_initialization_objects=objects,trace_sha256=sha(trace),trace_bytes=trace.stat().st_size)
                assert trace.stat().st_size <= 512<<10
                record['retained_raw_trace']=dict(owner='/root',byte_budget=512<<10,consumer='Current standalone binary scoped runtime observation',review_point='After compact runtime evidence and first untraced timing close; then retire redundant earlier host-v1 trace')
                extent_pins=[]; phase=[dict(gu=0,down=0),dict(gu=0,down=0)]
                for x in extents:
                    co,l,e,role=map(int,x[1:5]);path=Path(x[5]);offset,length=map(int,x[7:9])
                    assert co in (0,1) and role in (0,1,2) and path.parent==PRIMARY.parent and offset>=0 and 0<length<4<<20
                    with path.open('rb') as f:
                        f.seek(offset);data=f.read(length)
                    assert len(data)==length
                    extent_pins.append(dict(cohort=co,layer=l,expert=e,role=role,path=str(path),tensor=x[6],offset=offset,bytes=length,sha256=hashlib.sha256(data).hexdigest()))
                    phase[co]['gu' if role<2 else 'down']+=length
                assert sum(x['gu']+x['down'] for x in phase)==manifest['full_blob_bytes']
                assert all(x['gu']>64<<20 and x['down']>64<<20 for x in phase)
                record['selected_extent_sha256']=extent_pins;record['phase_working_set_bytes']=phase
                record['negative_admission_cases']=len(negatives)+1
            else:
                rows,_=run_child(record,out,'uninstrumented-rounds',argv(),env)
                extents=validate_rows(rows,manifest,args.tasks,args.batch,False)
                prior_extents=[{k:x[k] for k in ('cohort','layer','expert','role','path','tensor','offset','bytes')} for x in prior['selected_extent_sha256']]
                actual=[dict(cohort=int(x[1]),layer=int(x[2]),expert=int(x[3]),role=int(x[4]),path=x[5],tensor=x[6],offset=int(x[7]),bytes=int(x[8])) for x in extents]
                assert actual==prior_extents,'selected role extents changed'
                record['recorded_rounds']=250;record['warmup_rounds']=30
                record['timing_samples_complete']=True
            assert all(identity(x['path'])==x for x in model),'model shard metadata changed'
            check_build(build)
            assert sha(cohort)==record['cohort']['tsv_sha256'] and sha(OWNER/'manifest.json')==record['cohort']['manifest_sha256']
            record.update(complete=True,passed=True)
        except BaseException as e:
            record['error']=type(e).__name__+': '+str(e)
        finally:
            # Exit/source guards are not waived after failure.
            try:
                check_build(build)
                assert sha(__file__) == record['controller_sha256'], 'controller changed'
                if 'model_identity' in record:
                    assert all(identity(x['path']) == x for x in record['model_identity']), 'model shard metadata changed at exit'
                record['exit_source_pin_gate_passed']=True
            except BaseException as e:
                record['passed']=False;record['exit_source_error']=type(e).__name__+': '+str(e)
            record.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save(record,out)
        print(json.dumps(dict(record=str(out/'record.json'),sha256=sha(out/'record.json'),passed=record['passed'],error=record.get('error'),elapsed_seconds=record['elapsed_seconds'])))
        return 0 if record['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
