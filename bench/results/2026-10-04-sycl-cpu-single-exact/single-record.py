import csv, hashlib, json, shutil, statistics
from pathlib import Path
base=Path('/tmp')
out=Path('bench/results/2026-10-04-sycl-cpu-single-exact')
out.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def compact(p,flag):
 d=json.loads(p.read_text());assert d['exit_code']==0 and d['comparison_ids_equal'] and d['cancel_recovery_first_eight_ids']
 d['environment']['STRATA_IQ_SINGLE_EXACT']=str(flag)
 for r in d['runs']:
  ids=r.pop('output_ids');r['output_count']=len(ids);r['output_ids_sha256']=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
 d['source_file']=p.name;d['source_sha256']=sha(p)
 return d
result=dict(status='paired engine measurement complete; boundary validation pending',hardware=dict(cpu='AMD Ryzen 5 5600X',gpu='Intel Arc B570 10 GiB',ram_gib=125),
 change='Clang znver3 single-token IQ3_XXS inner unroll2 and IQ2_S unroll4. GGML integer lanes, FMA accumulation and hsum_float_8 order retained.',
 selection='STRATA_IQ_SINGLE_EXACT=1 is opt-in; only NT1, no AVX512, STRATA_NO_IQ256 absent, after existing multi-token dispatch. Other types/compiler targets fall back.',
 oracle='Linked production ggml_get_type_traits_cpu(type)->vec_dot; pinned GGML source unchanged.',
 scope='Same AOT executable with flag0/1, packing0, exact XMX tile8, ctx512, auto cache2137, workers5, prefill1024, adapt64, speculation4 p0.9; 128-token writing/coding twice per process.',
 conditions='No owned compilation, GPU tests or microbenchmarks overlapped timing. Existing workstation services remained active.',
 pair_order=['off1','on1','on2','off2','off3','on3'],screens={},pairs={},medians={},production_component={})
for label in ['off','on']:
 result['screens'][label]=compact(base/f'strata-sycl-goal-single-serve-{label}.json',label=='on')
 # Screen harness predates recording the new env flag; shell script is the provenance.
 result['screens'][label]['environment']['STRATA_IQ_SINGLE_EXACT']='1' if label=='on' else '0'
for label in result['pair_order']:
 result['pairs'][label]=compact(base/f'strata-sycl-goal-single-pairs-{label}.json',int(label.startswith('on')))
for i,name in enumerate(['writing_first','writing_repeat','coding_first','coding_repeat']):
 vals={k:[result['pairs'][k+str(t)]['runs'][i]['decode_ms'] for t in [1,2,3]] for k in ['off','on']}
 med={k:statistics.median(v) for k,v in vals.items()}
 result['medians'][name]=dict(samples_ms=vals,median_ms=med,median_tok_s={k:128000/v for k,v in med.items()},rate_gain_percent=100*(med['off']/med['on']-1))
for e in [8,64]:
 data=list(csv.DictReader((base/f'strata-sycl-goal-single-production-{e}.csv').open()))
 result['production_component'][str(e)]={}
 for ty in [18,22]:
  d=[r for r in data if int(r['type'])==ty];assert all(int(r['unequal'])==0 for r in d)
  vals={str(c):[float(r['microseconds']) for r in d if int(r['candidate'])==c] for c in [0,1]}
  med={k:statistics.median(v) for k,v in vals.items()}
  result['production_component'][str(e)][str(ty)]=dict(samples_us=vals,median_us=med,speedup=med['0']/med['1'])
result['validation']=dict(screen_output_ids_equal=True,paired_output_ids_equal=True,cancellation_recovery_all_passed=True,production_component_all_bitwise_equal=True,boundary_test='pending')
for name in ['iq_single_avx2','iq_avx2']:
 result[name+'_object_sha256']=sha(f'build-sycl-aot/CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/{name}.cpp.o')
(out/'run.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ['single-production-bench.cpp','single-production-build.sh','single-production-run.sh','single-pairs.sh','single-record.py']:
 p=base/('strata-sycl-goal-'+name)
 if p.exists():shutil.copy2(p,out/name)
archive=Path.home()/'.local/state/strata-sycl/measurement-archive/2026-10-04-speed-goal'
for p in base.glob('strata-sycl-goal-single-*'):
 if p.is_file():shutil.copy2(p,archive/p.name)
print(json.dumps(result['medians'],indent=2))
