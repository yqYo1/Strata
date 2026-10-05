import hashlib,json,math,os,subprocess,time
from pathlib import Path

exe=Path('/tmp/strata-upstream-arc-pcie-bandwidth')
source=exe.with_suffix('.cpp')
out=Path('/tmp/strata-upstream-arc-pcie-bandwidth-results')
out.mkdir(exist_ok=True)
ref=json.loads(Path('/tmp/strata-upstream-arc-attn-paired-wall/base-p1/run.json').read_text())
env=dict(os.environ,**ref['env'])
env['ONEAPI_DEVICE_SELECTOR']='level_zero:gpu'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def links():
    result=[]
    for card in Path('/sys/class/drm').glob('card[0-9]*'):
        if not card.name[4:].isdigit():continue
        d=(card/'device').resolve()
        if (d/'vendor').read_text().strip()!='0x8086':continue
        for p in (d,*list(d.parents)[:4]):
            r=dict(path=str(p))
            for key in ('vendor','device','current_link_speed','current_link_width','max_link_speed','max_link_width'):
                if (p/key).exists():r[key]=(p/key).read_text().strip()
            result.append(r)
    return result
# This controller is deliberately not an automatic recovery operation. Run it
# after the xe kernel fault has been cleared and the normal-wall controller is idle.
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        status=(proc/'status').read_text()
        if 'strata-upstream' in status:
            for task in (proc/'task').iterdir():
                state=(task/'stat').read_text().split(') ')[-1].split()[0]
                if state=='D':raise RuntimeError(f'xe recovery still pending: process {proc.name}, thread {task.name}')
        cmd=(proc/'cmdline').read_bytes().split(b'\0')
        if cmd and cmd[0] in (b'/tmp/strata-upstream-arc-layer-major-gpu-jit',b'/tmp/strata-upstream-arc-layer-major-aot'):
            raise RuntimeError(f'An inference measurement is active: process {proc.name}')
    except OSError:pass
report=dict(binary_sha256=sha(exe),source_sha256=sha(source),command=[str(exe)],
    env={k:v for k,v in env.items() if k.startswith(('ONEAPI_','SYCL_','STRATA_','NEO_')) or k=='LD_LIBRARY_PATH'},
    link_before=links(),rounds=[],summaries=[],
    note='No inference or disk workload. One in-order profiling-enabled GPU queue. Ordinary host memory versus the same USM host allocation API used by engine staging. H2D sources rotate within 1 GiB; staged mode adds serial CPU copies to an eight-slot host-USM ring, with DMA completion checked before slot reuse. It has one issuer, unlike the engine worker pool. Warmup, allocation, source initialization and complete final-copy checks are outside timers. Decimal GB/s from bytes/wall time and the summed memcpy event durations; raw event timing is not a separate electrical link measurement.')
def save(): (out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
save()
start=time.monotonic()
with (out/'stdout.jsonl').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
    child=subprocess.Popen([str(exe)],env=env,stdout=stdout,stderr=stderr)
    report['pid']=child.pid;save()
    # Do not terminate a successful long copy merely because an observer fails.
    # This run is finite and prints progress after each measured round.
    rc=child.wait()
report.update(exit_code=rc,wall_seconds=time.monotonic()-start,link_after=links())
for line in (out/'stdout.jsonl').read_text().splitlines():
    item=json.loads(line)
    (report['summaries'] if 'rounds' in item else report['rounds']).append(item)
save()
assert rc==0, 'Bandwidth process failed; partial raw evidence preserved'
assert len(report['rounds'])==175 and len(report['summaries'])==35
assert all(r['final_copy_all_bytes_identical'] for r in report['rounds']+report['summaries'])
assert all(math.isfinite(r[k]) and r[k]>0 for r in report['summaries']
           for k in ('median_wall_ms','median_dma_ms','wall_GB_s','dma_GB_s'))
for r in report['summaries']:print(json.dumps(r),flush=True)
print('PASS all 35 bandwidth cases, five rounds each, with complete final-copy byte checks',flush=True)
