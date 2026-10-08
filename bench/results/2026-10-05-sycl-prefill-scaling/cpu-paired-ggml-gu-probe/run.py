"""Validate and time paired one-token gate/up; no GPU queue is created."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

root=Path(__file__).resolve().parents[4]
out=Path.home()/'.local/state/strata-sycl/cpu-paired-ggml-gu-probe'
model=Path.home()/'.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
files=[out/x for x in ('paired.c','paired-icx-original.o','paired-gcc-original.o','paired-icx.o','paired-gcc.o','rows.cpp','rows.o','probe.cpp','probe.o','probe')]
files += [root/'build-sycl-upstream-jit'/x for x in ('libstrata_kernels_cpu.a','ggml/src/libggml-cpu.a','ggml/src/libggml-base.a','strata')]
manifest={'scope':'CPU-only paired one-token native gate/up with original ggml arithmetic',
          'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'environment':{k:v for k,v in env.items() if k.startswith(('STRATA_','ONEAPI','SYCL','LD_LIBRARY','OMP'))},
          'gcc':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
          'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
timings={};validation=[]
for i in range(1,4):
    with (out/f'round{i}.jsonl').open('w') as stdout,(out/f'round{i}.stderr').open('w') as stderr:
        run=subprocess.run([str(out/'probe'),str(model)],env=env,stdout=stdout,stderr=stderr)
    assert run.returncode==0,(i,run.returncode)
    rows=[json.loads(x) for x in (out/f'round{i}.jsonl').read_text().splitlines()]
    assert rows[-1]['kind']=='completed' and rows[-1]['all_bits_equal']
    for r in rows:
        if r['kind']=='validation':validation.append(dict(r,run=i))
        if r['kind']=='timing':timings.setdefault((r['type'],r['arm'],r['experts']),[]).append(dict(r,run=i))
summary={'scope':manifest['scope'],'validation':validation,'timings':[]}
for (type_,arm,experts),rows in sorted(timings.items()):
    summary['timings'].append({'type':type_,'arm':arm,'experts':experts,'pairs':len(rows),
                              'median_speed_ratio':statistics.median(r['speed_ratio'] for r in rows),
                              'process_medians':[statistics.median(r['speed_ratio'] for r in rows if r['run']==i) for i in range(1,4)],
                              'min_speed_ratio':min(r['speed_ratio'] for r in rows),'max_speed_ratio':max(r['speed_ratio'] for r in rows)})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
