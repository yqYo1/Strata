#!/usr/bin/env python3
"""Compare unchanged expert arithmetic with different CPU work distribution."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import select

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--gguf', type=Path, required=True)
parser.add_argument('--pack', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.pairs, args.rounds = 9, 3
counts = [1, 2, 12, 48, 96]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
model, pack = args.gguf.resolve(), args.pack.resolve()
binaries = {'reference': args.reference.resolve(), 'candidate': args.candidate.resolve()}
env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare(left, right):
    assert left.stat().st_size == right.stat().st_size
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            x, y = a.read(1024 * 1024), b.read(1024 * 1024)
            assert x == y, 'Expert output bytes differ'
            if not x:
                return


def status(pid):
    path = Path(f'/proc/{pid}')
    try:
        stat = (path / 'stat').read_text().split(')', 1)[1].split()
        row = dict(pid=pid, state=stat[0], cpu_ticks=int(stat[11])+int(stat[12]), last_cpu=int(stat[36]))
        for line in (path / 'status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmSwap:')):
                key, value = line.split(':', 1)
                row[key] = value.strip()
        row['dri_fds'] = []
        for fd in (path / 'fd').iterdir():
            try:
                target = os.readlink(fd)
                if target.startswith('/dev/dri/'):
                    row['dri_fds'].append(target)
            except OSError:
                pass
        assert not row['dri_fds'], 'CPU harness opened a GPU device'
        return row
    except FileNotFoundError:
        return dict(pid=pid, exited=True)


manifest = dict(scope='Actual production constant-9 CPU ExpertPool archive versus unchanged constant-3; no GPU or model inference', completed=False,
                args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, model=str(model), pack=str(pack), validation={}, datasets=[],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                binary_sha256={arm:digest(binaries[arm]) for arm in ['reference','candidate']},
                runner_sha256=digest(Path(__file__)),
                environment={k:v for k,v in env.items() if k.startswith(('ONEAPI_','SYCL_','UR_','ZE_','OMP_','KMP_')) or k=='LD_LIBRARY_PATH'})


def save():
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


manifest['timing_conditions'] = 'Run with no other agent-owned build or CPU benchmark; external host load is not controlled'
save()
# The actual constant-9 production object is checked over
# every layer, all token group sizes, mixed groups and the 96-entry boundary.
for arm in ['reference', 'candidate']:
    prefix = out/f'validation-{arm}'
    runenv = dict(env)
    with prefix.with_suffix('.jsonl').open('w') as log, prefix.with_suffix('.stderr').open('w') as err:
        run = subprocess.run([str(binaries[arm]),str(model),str(pack),'validate',str(prefix.with_suffix('.bin'))],env=runenv,stdout=log,stderr=err,timeout=120)
    assert run.returncode == 0, (arm, run.returncode)
    records = [json.loads(line) for line in prefix.with_suffix('.jsonl').read_text().splitlines()]
    assert records[-1]['kind']=='completed' and records[-1]['cases']==396
    manifest['validation'][arm]=dict(summary=records[-1],bytes=prefix.with_suffix('.bin').stat().st_size,sha256=digest(prefix.with_suffix('.bin')))
    save()
compare(out/'validation-reference.bin',out/'validation-candidate.bin')
manifest['validation']['all_bytes_identical']=True
save()
print(json.dumps({'kind':'full_validation_passed',**manifest['validation']}),flush=True)


class Peer:
    def __init__(self, arm, name, round_no):
        self.arm=arm
        self.log=(out/f'{name}-{round_no}-{arm}.jsonl').open('w')
        self.err=(out/f'{name}-{round_no}-{arm}.stderr').open('w')
        runenv=dict(env)
        binary=binaries['reference'] if name=='null' else binaries[arm]
        self.child=subprocess.Popen([str(binary),str(model),str(pack),'bench'],env=runenv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,bufsize=1)
        self.ready=self.receive()
        assert self.ready['kind']=='ready'

    def receive(self):
        assert select.select([self.child.stdout], [], [], 30)[0], (self.arm, 'CPU reply timeout', self.child.poll())
        line=self.child.stdout.readline()
        assert line,(self.arm,self.child.poll())
        self.log.write(line);self.log.flush()
        return json.loads(line)

    def command(self,text):
        self.child.stdin.write(text+'\n');self.child.stdin.flush()
        return self.receive()

    def close(self):
        if self.child.poll() is None:
            self.child.stdin.write('QUIT\n');self.child.stdin.flush()
        assert self.child.wait(timeout=30)==0
        self.log.close();self.err.close()


configs = [('selected', 9, 0), ('null', 3, 0)]
pairs=[]
with (out/'pairs.jsonl').open('w') as raw:
    for round_no in range(args.rounds):
        for name,factor,quant in (configs if round_no%2==0 else list(reversed(configs))):
            peers={arm:Peer(arm,name,round_no) for arm in ['reference','candidate']}
            try:
                assert peers['reference'].ready==peers['candidate'].ready
                for layer in ([1,2,17,21] if round_no%2==0 else [21,17,2,1]):
                    for count in (counts if round_no%2==0 else list(reversed(counts))):
                        for pattern in ([1,2,4,-4] if round_no%2==0 else [-4,4,2,1]):
                            meta=None
                            for arm,peer in peers.items():
                                ready=peer.command(f'CASE {layer} {count} {pattern}')
                                assert ready['kind']=='case_ready'
                                if meta is None:meta=ready
                                else:assert ready==meta
                                assert peer.command(f'DUMP {out/(arm+"-case.bin")}')['kind']=='dumped'
                            compare(out/'reference-case.bin',out/'candidate-case.bin')
                            manifest['datasets'].append(dict(config=name,round=round_no,**meta,output_sha256=digest(out/'reference-case.bin'),all_bytes_identical=True,placement=peers['reference'].ready))
                            save()
                            repeats=max(1,96//count)
                            for pair_no in range(args.pairs):
                                first='reference' if (round_no+pair_no+layer+pattern)%2==0 else 'candidate'
                                order=[first,'candidate' if first=='reference' else 'reference']
                                timings={};observed={arm:status(peer.child.pid) for arm,peer in peers.items()}
                                for arm in order:
                                    time.sleep(.06)
                                    timings[arm]=peers[arm].command(f'BENCH {repeats}')
                                    assert timings[arm]['kind']=='timing'
                                row=dict(config=name,factor=factor,parallel_quant=quant,round=round_no,pair=pair_no,first=first,**meta,**timings,status=observed)
                                row['speed_ratio']=timings['reference']['wall_ms']/timings['candidate']['wall_ms']
                                raw.write(json.dumps(row)+'\n');raw.flush();pairs.append(row)
                            print(json.dumps(dict(kind='case_completed',config=name,round=round_no,layer=layer,experts=count,pattern=pattern,median_speed_ratio=statistics.median(row['speed_ratio'] for row in pairs[-args.pairs:]))),flush=True)
            finally:
                for peer in peers.values():peer.close()

summary=[]
for name,factor,quant in configs:
    rows=[row for row in pairs if row['config']==name]
    cases=[]
    for layer in [1,2,17,21]:
        for count in counts:
            for pattern in [1,2,4,-4]:
                subset=[row for row in rows if (row['layer'],row['experts'],row['pattern'])==(layer,count,pattern)]
                case=dict(layer=layer,experts=count,pattern=pattern,pairs=len(subset),speed_ratio=statistics.median(row['speed_ratio'] for row in subset),process_speed_ratios=[statistics.median(row['speed_ratio'] for row in subset if row['round']==round_no) for round_no in range(args.rounds)])
                for phase in ['wall_ms','gu_ms','quant_ms','down_ms']:
                    case[phase]={arm:statistics.median(row[arm][phase] for row in subset) for arm in ['reference','candidate']}
                cases.append(case)
    summary.append(dict(config=name,factor=factor,parallel_quant=quant,pairs=len(rows),cases=cases,case_ratio_geomean=math.exp(statistics.mean(math.log(case['speed_ratio']) for case in cases)),minimum_case_ratio=min(case['speed_ratio'] for case in cases),maximum_case_ratio=max(case['speed_ratio'] for case in cases)))
manifest['completed']=True;save()
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'kind':'completed','configs':[{k:v for k,v in row.items() if k!='cases'} for row in summary]}),flush=True)
