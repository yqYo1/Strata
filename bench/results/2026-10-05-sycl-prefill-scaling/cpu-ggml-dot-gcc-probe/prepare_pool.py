"""Prepare a production-pool check selecting only the faster IQ3_XXS dot."""
import hashlib
import json
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[4]
prior=root/'bench/results/2026-10-05-sycl-prefill-scaling/cpu-gcc-pool-probe'
out=Path.home()/'.local/state/strata-sycl/cpu-ggml-dot-gcc-probe/pool-iq3xxs'
out.mkdir(exist_ok=True)
harness=(prior/'probe.cpp').read_text()
needle='''        uint64_t checked = 0;
        int cases = 0;'''
replacement='''        uint64_t checked = 0;
        int cases = 0;
        std::vector<int> target_layers;
        for (int layer=0;layer<48;++layer) {
            Dataset meta(file,layer,1,1,1071+layer);
            std::printf("{\\"kind\\":\\"layer_metadata\\",\\"layer\\":%d,\\"gu_type\\":%d,\\"down_type\\":%d}\\n",layer,meta.fmt.gu_type,meta.fmt.d_type);
            if(meta.fmt.gu_type==18)target_layers.push_back(layer);
        }
        require(!target_layers.empty(),"no IQ3_XXS gate/up layer");'''
assert needle in harness
harness=harness.replace(needle,replacement)
harness=harness.replace('const int layer = round % 2 ? 1 : 2;', 'const int layer = round % 2 ? target_layers.front() : 2;')
(out/'probe.cpp').write_text(harness)
wrapper='''#include "ggml-cpu.h"
extern "C" const ggml_type_traits_cpu* __real_ggml_get_type_traits_cpu(ggml_type);
extern "C" void alternate_ggml_vec_dot_iq3_xxs_q8_K(int,float*,size_t,const void*,size_t,const void*,size_t,int);
extern "C" const ggml_type_traits_cpu* __wrap_ggml_get_type_traits_cpu(ggml_type type) {
    const auto* original=__real_ggml_get_type_traits_cpu(type);
#if defined(SELECT_IQ3XXS)
    if(type==GGML_TYPE_IQ3_XXS) {
        static const ggml_type_traits_cpu selected=[&] {
            auto copy=*original;
            copy.vec_dot=alternate_ggml_vec_dot_iq3_xxs_q8_K;
            return copy;
        }();
        return &selected;
    }
#endif
    return original;
}
'''
(out/'wrap.cpp').write_text(wrapper)
runner=(prior/'run.py').read_text().replace('default','baseline').replace('gcc','iq3xxs')
needle='''for arm in ("baseline", "iq3xxs"):
    prefix = out / f"validation-{arm}"'''
assert needle in runner
runner=runner.replace(needle,'''layers = None
for arm in ("baseline", "iq3xxs"):
    prefix = out / f"validation-{arm}"''')
needle='''    assert events[-1]["kind"] == "completed"
    manifest["validation"][arm]'''
assert needle in runner
runner=runner.replace(needle,'''    assert events[-1]["kind"] == "completed"
    metadata = [r for r in events if r["kind"] == "layer_metadata"]
    assert len(metadata)==48
    targets = [r["layer"] for r in metadata if r["gu_type"]==18]
    assert len(targets)>=2
    chosen = [targets[0], targets[-1]]
    chosen += [next(r["layer"] for r in metadata if r["gu_type"]==t) for t in (22,21)]
    assert len(set(chosen))==4
    if layers is None:
        layers=chosen
        manifest["timed_layers"]=[metadata[x] for x in chosen]
    else: assert layers==chosen
    manifest["validation"][arm]''')
assert runner.count('for layer in (1, 2, 17, 21):')==2
runner=runner.replace('for layer in (1, 2, 17, 21):','for layer in layers:')
(out/'run.py').write_text(runner)
archive=root/'build-sycl-upstream-jit/libstrata_kernels_cpu.a'
for arm in ('baseline','iq3xxs'):shutil.copy2(archive,out/f'production-{arm}.a')
candidate=out.parent/'gcc-contract/alternate_quants.o'
shutil.copy2(candidate,out/'alternate_quants.o')
tracked=(archive,candidate,out/'probe.cpp',out/'wrap.cpp',out/'run.py')
manifest={'scope':'Actual unchanged production pool/quantizers; linker wrapper selects GCC IQ3_XXS vec_dot only',
          'production_iq2s_gcc':True,'selected_type':18,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}}
(out/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Prepared selective IQ3_XXS full-pool check:',out)
