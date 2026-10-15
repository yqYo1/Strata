from pathlib import Path
import subprocess,hashlib,json,re
out=Path(__file__).resolve().parent;base='5942f1853f006b0df45e15fd4f1c60bc37b7b3b3'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
def read(p):return subprocess.check_output(['git','show',base+':'+p]).decode()
def body(s,marker):
 start=s.index(marker);at=s.index('{',start);depth=0
 for m in re.finditer(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/|[{}]',s[at:]):
  if m.group()=='{':depth+=1
  if m.group()=='}':
   depth-=1
   if depth==0:return s[start:at+m.end()]
 raise ValueError(marker)
p='src/program/generate.cpp';a=read(p);b=(out/'main-review-consumer'/p).read_text();checks={}
for marker in ['struct GpuStage {','auto observe=','bool run_prompt_with_mtp_experts(']:
 x,y=body(a,marker),body(b,marker);assert x==y;checks[marker]={'base_sha256':sha(x),'proposed_sha256':sha(y),'byte_identical':True}
unchanged=['src/core/expert_source.cpp','include/strata/core/expert_source.hpp','src/kernels/cpu/pool.cpp','include/strata/kernels/cpu/pool.hpp','src/core/mtp.cpp','include/strata/core/mtp.hpp','include/strata/core/mtp_expert_source.hpp','src/core/session.cpp','include/strata/core/session.hpp','src/kernels/sycl/native_mmvq.cpp','src/kernels/sycl/native_mmq_xmx.cpp','include/strata/sycl/native_mmq.hpp','include/strata/core/native_dense_bank_policy.hpp']
(out/'retained-invariants.json').write_text(json.dumps({'scope':'source preservation only; no runtime qualification','immutable_base':base,'generator_blocks':checks,'untouched_files':{p:sha(read(p)) for p in unchanged},'requirements':['single NativeDensePhase reader lease','default-off and immutable child environment','exact converted host upload capture; no D2H; 3GiB canonical payload cap','3GiB native snapshot cap separately; aggregate max 6GiB payload','HC96/lookahead and all canonical PLE projections persistent','reserve both destinations before reader retirement and withdrawal','confirmed OOM-null rollback; unknown/contradictory status fail-stop','all SessionGraphs arrays retained and retired; canonical norm commit graph destroyed under exclusive lease','Full owner publications and all PLE rebindings before persistent-pool free and recapture','MTP/NativeEmbed/private resolver/short-input full layout preserved','current native-call exception fail-stop untouched']},indent=2)+'\n')
