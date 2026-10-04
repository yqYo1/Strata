from pathlib import Path
import json,re,statistics,hashlib,shutil
out=Path('bench/results/2026-10-04-sycl-xmx-signspread');out.mkdir(exist_ok=True,parents=True)
base=Path('/tmp')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected=[11855,248046,198,248044,248045,846,198,248046]
def short(layout,tag):
 p=base/f'strata-sycl-perf-goal-signspread-{layout}-short-{tag}-profile.log';s=p.read_text()
 m=re.search(r'^prefill\s+(\d+) tokens in ([\d.]+) ms',s,re.M);ids=list(map(int,re.search(r'^output  : (.*)',s,re.M)[1].split()));assert ids==expected
 return dict(prefetched_tokens=int(m[1]),prefill_ms=float(m[2]),output_ids=ids,log=p.name,log_sha256=sha(p))
def long(layout,tag):
 p=base/f'strata-sycl-goal-signspread-{layout}-4k-{tag}.json';d=json.loads(p.read_text());assert d['exit_code']==0 and d['output_ids']==expected
 log=d.pop('log');d['log_file']=p.with_suffix('.log').name;d['log_sha256']=hashlib.sha256(log.encode()).hexdigest();return d
result=dict(status='paired prefill validation complete; persistent requests and full suites pending',source_parent_commit='3bd22ad',hardware=dict(gpu='Intel Arc B57010GiB',cpu='Ryzen5 5600X',ram_gib=125),change='Spread four sign bits with a constant multiply, then fill byte masks; the old negative-value and blend operations remain unchanged.',proof='All256 sign-byte values matched the original mask; high sign bits are ignored by both formulas.',arithmetic='No change to integer products, quantized data, FP32 accumulation order or CPU/GPU placement.',binary_sha256=dict(old=sha(base/'strata-sycl-goal-signspread-baseline-aot'),new=sha('build-sycl-aot/strata')),raw_pair_order=['old1','new1','new2','old2','old3','new3'],raw_pairs={},packed_single_pair={},medians={})
result['method']=dict(short='826prefetched tokens,ctx2048,cache1649,prefill1024,workers5,adapt0',long='4007prefetched tokens,ctx8192,cache512,prefill4096,workers5,adapt0',common='IQ3_S,exactXMXtile8,STRATA_IQ_SINGLE_EXACT1,allCPUobjects unchanged,profiling disabled',concurrency='No owned compilation or other GPU tests overlapped timing; existing workstation services active',selection='All three raw pairs retained, including the slower first new short run. The separate packing1 pair is only a single screen.')
for tag in result['raw_pair_order']:result['raw_pairs'][tag]=dict(short=short('raw',tag),long=long('raw',tag))
for tag in ['old1','new1']:result['packed_single_pair'][tag]=dict(short=short('packed',tag),long=long('packed',tag))
for shape in ['short','long']:
 values={tag:[result['raw_pairs'][tag+str(i)][shape]['prefill_ms'] for i in [1,2,3]] for tag in ['old','new']};med={tag:statistics.median(v) for tag,v in values.items()};tokens=826 if shape=='short' else 4007
 result['medians'][shape]=dict(samples_ms=values,median_ms=med,median_tok_s={tag:tokens*1000/v for tag,v in med.items()},rate_gain_percent=100*(med['old']/med['new']-1))
result['device_metadata']=json.loads((base/'strata-sycl-goal-signspread-aot.registers.json').read_text())
result['device_text_comparison']=json.loads((base/'strata-sycl-goal-signspread-text-compare.json').read_text())
result['code_bytes_exact_raw_tile8']={'18':{'old':126400,'new':116480},'21':{'old':134720,'new':118720},'22':{'old':135808,'new':121856}}
result['validation']=dict(JIT_MMQ='3tests50geometries passed,35.97s',AOT_MMQ='3tests50geometries passed,2.48s',raw_outputs='all12short/long runs matched eight expected ids',packed_outputs='bothshort/long pairs matched expected ids',full_suites='pending',persistent='pending')
(out/'run.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ['signspread-profile.py','signspread-text-compare.py','gpu-metadata.py','extract-zebin.py','signspread-record.py']:
 shutil.copy2(base/('strata-sycl-goal-'+name),out/name)
archive=Path.home()/'.local/state/strata-sycl/measurement-archive/2026-10-04-speed-goal'
for pattern in ['strata-sycl-goal-signspread-*','strata-sycl-perf-goal-signspread-*']:
 for p in base.glob(pattern):
  if p.is_file() and 'serve' not in p.name:shutil.copy2(p,archive/p.name)
print(json.dumps(result['medians'],indent=2))
