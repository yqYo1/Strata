import hashlib,json,os,subprocess,time
from pathlib import Path

root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exe=Path('/tmp/strata-upstream-arc-dequant-workgroup')
source=exe.with_suffix('.cpp')
out=Path('/tmp/strata-upstream-arc-dequant-workgroup-results')
out.mkdir(exist_ok=True)
env=dict(os.environ,**json.loads(Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json').read_text())['env'])
env.update(ONEAPI_DEVICE_SELECTOR='level_zero:gpu',SYCL_CACHE_PERSISTENT='1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def peer():
    proc=Path('/proc/6151')
    if not proc.exists():return None
    stat=(proc/'stat').read_text().split(') ')[-1].split()
    return dict(pid=6151,comm=(proc/'comm').read_text().strip(),user_ticks=int(stat[11]),system_ticks=int(stat[12]),cgroup=(proc/'cgroup').read_text().strip())
report=dict(binary_sha256=sha(exe),source_sha256=sha(source),
    included_source_sha256={s:sha(root/s) for s in ('sycl/src/kernels/cuda/iq_kernels.dp.cpp','sycl/include/strata/sycl_math.hpp','sycl/include/strata/sycl_queue.hpp','third_party/ggml/ggml-common.h')},
    build=json.loads(Path('/tmp/strata-upstream-arc-dequant-workgroup-build.json').read_text()),
    env={k:v for k,v in env.items() if k.startswith(('ONEAPI_','SYCL_','STRATA_','NEO_')) or k=='LD_LIBRARY_PATH'},
    command=[str(exe)],peer_before=peer(),rounds=[],checks=[],
    note='Microbenchmark using unchanged engine dequantization formulas on finite random quantized tensors, seed 7. Nine types, flat and interleaved GU, row counts 1/640/641 including tail groups; all half output bytes checked against original functions. Timing uses 640-row shapes, ten warmup calls each and five alternating original/candidate pairs of 128 calls. Wall time includes all queue submission and completion; allocation, input preparation and full byte checks are outside timers. Separate embedding service remains resident on GPU; its CPU counters are recorded, without claiming an exclusive GPU condition. No engine integration or model-level correctness/performance gain is claimed.')
def save():(out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
save();start=time.monotonic()
with (out/'stdout.jsonl').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
    child=subprocess.Popen([str(exe)],env=env,stdout=stdout,stderr=stderr,cwd=root)
    report['pid']=child.pid;save()
    rc=child.wait()
report.update(exit_code=rc,wall_seconds=time.monotonic()-start,peer_after=peer())
for line in (out/'stdout.jsonl').read_text().splitlines():
    item=json.loads(line)
    (report['rounds'] if 'round' in item else report['checks']).append(item)
save()
assert rc==0, 'Dequant probe failed; partial evidence preserved'
assert len(report['rounds'])==720 and len(report['checks'])==432
assert all(c['all_half_bytes_identical'] for c in report['checks'])
for ty in sorted({c['type'] for c in report['checks']}):
    for gu in (False,True):
        candidates=[c for c in report['checks'] if c['type']==ty and c['gu']==gu and c['n_ff']==640]
        best=min(candidates,key=lambda c:c['candidate_ms'])
        print(json.dumps(dict(type=ty,gu=gu,best=best,speed_ratio=best['original_ms']/best['candidate_ms'])),flush=True)
print('PASS all 432 complete FP16-byte checks and 720 timing rounds',flush=True)
