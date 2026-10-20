#!/usr/bin/env python3
"""Main-owned CPU-only production expert/RAM calibration; never launches a GPU.

prepare -> commit plan -> build -> check -> measure. Each stage is single-use.
Pin source before building and preserve under-floor/failed samples unchanged.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_owner as owner

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'cpu-roofline-v2'
A = P / 'bench/results/2026-10-15-pr6-recovery-audit'
PLAN = A / 'cpu-roofline-plan.json'
RESULT = A / 'cpu-roofline-current.json'
METADATA = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/native_experts.txt')
NATIVE_DIR = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S')
GGML = Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp/.worktree/dep-ggml-3cf-xe-e32-20261011')
GGML_REV = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
LLVM = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
PREFIXES = ('bench/sycl/cpu_roofline', 'tools/sycl/cpu_roofline.py',
            'tools/sycl/measurement_owner.py', 'tests/test_sycl_cpu_roofline.py',
            'src/kernels/cpu', 'include/strata/kernels/cpu',
            'include/strata/core/progress.hpp', 'third_party/ggml')
PAIRS = {(18,20):14,(22,42):5,(22,20):15,(18,42):3,(21,20):9,(21,42):1,(23,20):1}
SIZES = {18:980,22:820,21:1100,23:1360}  # bytes per H2560 GU row, checked by native_fmt


def need(ok, message):
    if not ok: raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git','-C',str(root),*args])


def source_pin(root, revision, prefixes):
    names = git(root,'ls-tree','-r','--name-only',revision,'--',*prefixes).decode().splitlines()
    need(names, 'empty source scope')
    # Recover original Git bytes, not a nearby snapshot or only an artifact hash.
    hashes = {}
    for at in range(0,len(names),64):
        batch=names[at:at+64]
        request=''.join(revision+':'+n+'\n' for n in batch).encode()
        data=subprocess.check_output(['git','-C',str(root),'cat-file','--batch'],input=request)
        pos=0
        for name in batch:
            end=data.index(b'\n',pos); header=data[pos:end].split()
            need(len(header)==3 and header[1]==b'blob','unrecoverable source '+name)
            size=int(header[2]); start=end+1; end=start+size
            need(data[end:end+1]==b'\n','Git extent')
            content=data[start:end]; pos=end+1
            need((root/name).read_bytes()==content, 'uncommitted/changed input '+name)
            hashes[name]=hashlib.sha256(content).hexdigest()
        need(pos==len(data),'Git trailing data')
    untracked=git(root,'ls-files','--others','--exclude-standard','--',*prefixes).decode().splitlines()
    need(not untracked,'uncommitted input files '+str(untracked[:3]))
    return dict(revision=revision,count=len(names),aggregate_sha256=fingerprint(hashes))


def identity(path):
    s=Path(path).stat()
    return dict(path=str(Path(path).resolve()),dev=s.st_dev,ino=s.st_ino,bytes=s.st_size,
                mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def formats(metadata=METADATA, native_dir=NATIVE_DIR):
    lines=metadata.read_text().splitlines()
    first=lines[0].split('absolute offsets in ')[1].split(',')[0]
    rows={}; counts={}
    for line in lines:
        if not line or line.startswith('#'): continue
        f=line.split(); layer,gu,dn,offset,n,go,uo,do=map(int,f[:8]); key=(gu,dn)
        counts[key]=counts.get(key,0)+1
        if key in rows: continue
        need(key in PAIRS,'unexpected model type pair')
        shard=native_dir/(f[8] if len(f)==9 else first)
        gbytes=SIZES[gu]*640; dbytes=(180 if dn==42 else 360)*2560
        need(2*gbytes+dbytes==n,'metadata geometry mismatch')
        count=6*math.ceil(math.ceil((96<<20)/n)/6)
        need(count<=96 and count<=512,'fixture count')
        rows[key]=dict(layer=layer,gu=gu,down=dn,blob_bytes=n,gate=go,up=uo,down_offset=do,
                       shard=str(shard),experts=count,gu_bytes=gbytes,down_bytes=dbytes)
    need(counts==PAIRS,'layer format frequencies changed')
    return [rows[p] for p in sorted(rows)]


def fixture_hash(row):
    path=Path(row['shard']); before=identity(path); h=hashlib.sha256()
    with path.open('rb',buffering=0) as f:
        for e in range(row['experts']):
            for start,n in ((row['gate'],row['gu_bytes']),(row['up'],row['gu_bytes']),
                            (row['down_offset'],row['down_bytes'])):
                off=start+e*n; need(0<=off and off+n<=before['bytes'],'fixture slice bounds')
                f.seek(off); data=f.read(n); need(len(data)==n,'fixture short read'); h.update(data)
    need(identity(path)==before,'model changed during fixture digest')
    return h.hexdigest()


def machine():
    cpus=sorted(os.sched_getaffinity(0)); topology=[]
    for c in cpus:
        t=Path(f'/sys/devices/system/cpu/cpu{c}/topology')
        topology.append([c,int((t/'physical_package_id').read_text()),int((t/'core_id').read_text())])
    primaries=[]; seen=set()
    for c,p,k in topology:
        if (p,k) not in seen: seen.add((p,k)); primaries.append(c)
    need(len(primaries)==6 and len(cpus)==12,'requires this six-core/12-thread CPU')
    return dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                kernel=os.uname().release,affinity=cpus,topology=topology,physical_cpus=primaries,
                cpuinfo_sha256=hashlib.sha256('\n'.join(s for s in Path('/proc/cpuinfo').read_text().splitlines()
                    if s.startswith(('model name','microcode','flags'))).encode()).hexdigest())


def environment():
    # Installed official toolchain only; no XeStrata implementation input.
    return dict(PATH=str(LLVM/'bin')+':/usr/bin:/bin',LD_LIBRARY_PATH=str(LLVM/'lib'),HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',
                OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')


def command(row, workers, mode):
    return [str(ROOT/'build/cpu_roofline'),row['shard'],str(row['gu']),str(row['down']),
            str(row['gate']),str(row['up']),str(row['down_offset']),str(row['blob_bytes']),
            str(workers),mode,'20261020']


def load_plan():
    raw=git(P,'show','HEAD:'+str(PLAN.relative_to(P)))
    need(PLAN.read_bytes()==raw,'commit exact plan first')
    plan=json.loads(raw)
    need(source_pin(P,plan['source']['revision'],PREFIXES)==plan['source'],'project input drift')
    need(source_pin(GGML,GGML_REV,('ggml',))==plan['ggml'],'ggml input drift')
    need(machine()==plan['machine'],'machine/affinity/boot drift')
    for path,h in plan['tools'].items(): need(digest(path)==h,'compiler/tool drift')
    need(digest(METADATA)==plan['metadata_sha256'],'model metadata drift')
    for row in plan['formats']:
        need(identity(row['shard'])==plan['model_identity'][row['shard']],'model file identity drift')
        need(fixture_hash(row)==row['fixture_sha256'],'selected expert bytes drift')
    return plan


def prepare():
    need(not PLAN.exists() and not RESULT.exists() and not ROOT.exists(),'single-use root/plan already exists')
    revision=git(P,'rev-parse','HEAD').decode().strip()
    rows=formats()
    for r in rows:r['fixture_sha256']=fixture_hash(r)
    order=[(r['gu'],r['down'],w) for r in rows for w in (1,5)]
    random.Random(20261020).shuffle(order)
    plan=dict(schema='strata-cpu-roofline-v1',source=source_pin(P,revision,PREFIXES),
        ggml=source_pin(GGML,GGML_REV,('ggml',)),tools={str(Path(p).resolve()):digest(p)
            for p in (LLVM/'bin/clang++',LLVM/'bin/clang','/usr/bin/cmake','/usr/bin/ninja')},
        environment=environment(),machine=machine(),metadata_sha256=digest(METADATA),formats=rows,
        model_identity={r['shard']:identity(r['shard']) for r in rows},order=order,
        configure=['/usr/bin/cmake','-S',str(P/'bench/sycl/cpu_roofline'),'-B',str(ROOT/'build'),
            '-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER='+str(LLVM/'bin/clang'),
            '-DCMAKE_CXX_COMPILER='+str(LLVM/'bin/clang++'),'-DSTRATA_GGML_DIR='+str(GGML)],
        build=['/usr/bin/cmake','--build',str(ROOT/'build'),'--target','cpu_roofline','--parallel','6'],
        protocol=dict(nt=[1,2,3,4],group=6,reps=5,minimum_sample_s=1,
            pilot_target_s=1.3,under_floor='Keep all attempts; double iterations and repeat the whole five-sample cell once, then fail.',
            hot='Six distinct real expert blobs, 8.6–15.3 MiB including weights only.',
            stream='At least96 MiB of distinct real blobs; shuffled six-expert groups, no revisit per pass.',
            scan='256 MiB read-only checksum; persistent pinned1/6/12 threads.',
            workers='1 worker + noncomputing waiting host; or production5 workers + computing host. Host wait CPU is included.',
            validation='Before/after each NT cell: same-input GU and Down rel-L1<=1e-5 vs ggml; pool exact vs serial same-route. Finite outputs and external buffer canaries; private pool scratch is not instrumented.',
            gate='Require all14 checks, all112 five-sample expert cells and3 scan cells; keep failures/under-floor attempts. No automatic run retry.',
            interpretation='Hot/stream and1/6-compute-thread scaling identify sensitivity. Requested byte rates and scan are not physical DRAM traffic, saturation proof, model speed or70tok/s evidence.'),
        budgets=dict(build_wall_s=600,check_child_s=120,measure_child_s=180,scan_child_s=40,
                     raw_text_bytes=64<<20,root_bytes=512<<20,rss_bytes=3<<30),
        retention='Auxiliary CPU fixture, no new inference generation. Keep during calibration/next identified regression; retire successful raw/build cache after compact evidence/source recovery. Existing two engine generations retain their117K comparison consumer.')
    PLAN.write_text(json.dumps(plan,indent=2)+'\n')
    print('Prepared committed-source CPU plan; commit it before build.')


def summarize(rows):
    cells=[]
    for entry in rows:
        need(entry['closed'] and entry['exit_code']==0,'incomplete/failed calibration child')
        samples=[r for r in entry['rows'] if r['kind']=='sample']
        grouped={}
        for row in samples:
            key=(row['mode'],row.get('nt',0));grouped.setdefault(key,[]).append(row)
        for (mode,nt),attempts in grouped.items():
            last=max(r['attempt'] for r in attempts); selected=[r for r in attempts if r['attempt']==last]
            need(len(selected)==5 and [r['rep'] for r in selected]==list(range(5)) and
                 all(r['wall_s']>=1 for r in selected),'ineligible timed cell')
            rates=[r['requested_bytes']/r['wall_s']/1e9 for r in selected]
            cells.append(dict(pair=entry.get('pair'),workers=entry.get('workers'),mode=mode,nt=nt,
                scan_threads=entry.get('scan_threads'),samples=attempts,
                requested_GB_s_median=statistics.median(rates),min=min(rates),max=max(rates),
                requested_GB_s_sd=statistics.stdev(rates)))
    return cells


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','build','check','measure'])
    args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # No model/native run is permitted concurrently, even though this uses no GPU.
        live=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:exe=(proc/'exe').resolve(strict=True)
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            if exe.name in ('strata','cpu_roofline'): live.append(int(proc.name))
        need(not live,'active measurement executable(s): '+str(live))
        if args.stage=='prepare':return prepare()
        plan=load_plan();ROOT.mkdir(exist_ok=True)
        stagepath=ROOT/(args.stage+'.json');need(not stagepath.exists(),'closed/active stage cannot retry')
        state=dict(stage=args.stage,source=plan['source'],plan_sha256=digest(PLAN),active=True,passed=False,commands=[])
        def persist():stagepath.write_text(json.dumps(state,indent=2)+'\n')
        # Owner measures artifact_bytes(output.parent); keep that parent private.
        own=owner.Owner(ROOT/'owned',persist);state['commands']=own.commands;persist()
        def run(label, argv, wall):
            entry,_,_=own.run(args.stage+'-'+label,argv,plan['environment'],P,wall=wall,text_cap=64<<20,
                rss_cap=3<<30,total_cap=512<<20,cpu=600,file_cap=64<<20)
            need(owner.closed(entry),'measurement ownership did not close')
            need(entry['exit_code']==0,'calibration child failed: '+label)
            return entry
        try:
            if args.stage=='build':
                run('configure',plan['configure'],120);run('compile',plan['build'],600)
                state['binary']=dict(path=str(ROOT/'build/cpu_roofline'),sha256=digest(ROOT/'build/cpu_roofline'))
                run('linked-libraries',['/usr/bin/ldd',str(ROOT/'build/cpu_roofline')],20)
                links=(ROOT/'owned/build-linked-libraries.stdout').read_text()
                need(not any(x in links.lower() for x in ('sycl','level_zero','cuda','amdhip','libze')),'GPU runtime linked')
            else:
                build=json.loads((ROOT/'build.json').read_text());need(build['passed'] and not build['active'],'unqualified build')
                need(build['plan_sha256']==digest(PLAN),'build plan mismatch')
                need(digest(ROOT/'build/cpu_roofline')==build['binary']['sha256'],'binary drift')
                state['binary']=build['binary'];state['results']=[]
                if args.stage=='measure':
                    check=json.loads((ROOT/'check.json').read_text())
                    need(check['passed'] and not check['active'] and check['binary']==build['binary'] and check['plan_sha256']==digest(PLAN),'complete numerical gate required')
                lookup={(r['gu'],r['down']):r for r in plan['formats']}
                jobs=[(list((gu,dn)),w,None) for gu,dn,w in plan['order']]
                if args.stage=='measure': jobs += [(None,None,n) for n in (1,6,12)]
                for i,(pair,w,n) in enumerate(jobs):
                    argv=command(lookup[tuple(pair)],w,args.stage) if pair else [str(ROOT/'build/cpu_roofline'),'scan',','.join(map(str,plan['machine']['physical_cpus'] if n==6 else plan['machine']['affinity'][:n]))]
                    label=f'cell-{i}';e=run(label,argv,180 if pair and args.stage=='measure' else 120)
                    lines=[json.loads(line) for line in (ROOT/'owned'/(args.stage+'-'+label+'.stdout')).read_text().splitlines()]
                    need(lines and lines[-1]==dict(kind='closed',passed=True),'missing native completion')
                    if pair:
                        need(len([x for x in lines if x['kind']=='validation']) >=4,'four NT checks required')
                        placement=[x for x in lines if x['kind']=='placement']
                        need(len(placement)==1 and placement[0]['workers']==w and placement[0]['host_works']==(w==5),'production placement mismatch')
                    state['results'].append(dict(pair=pair,workers=w,scan_threads=n,rows=lines,closed=True,exit_code=e['exit_code']))
                    persist();print('Closed '+args.stage+' '+str(i+1)+'/'+str(len(jobs)),flush=True)
                if args.stage=='measure':
                    cells=summarize(state['results']);need(len(cells)==115,'complete112+3 matrix required')
                    compact=dict(schema='strata-cpu-roofline-result-v1',plan_sha256=digest(PLAN),plan_source_commit=plan['source']['revision'],
                        binary=state['binary'],GPU_executed=False,cells=cells,
                        validations=[dict(pair=r['pair'],workers=r['workers'],values=[x for x in r['rows'] if x['kind'] in ('validation','placement')]) for r in state['results'] if r['pair']],
                        hardware_saturation_qualified=False,model_quality_qualified=False,
                        physical_DRAM_traffic='unknown',limits=plan['protocol']['interpretation'])
                    RESULT.write_text(json.dumps(compact,indent=2)+'\n')
            need(machine()==plan['machine'],'machine changed during stage')
            for path,v in plan['model_identity'].items():need(identity(path)==v,'model file changed during stage')
            state['passed']=True
        except BaseException as e:
            state['error']=str(e);raise
        finally:state['active']=False;persist()
        print('CPU-only '+args.stage+' passed; source='+plan['source']['revision'],flush=True)


if __name__=='__main__':main()
