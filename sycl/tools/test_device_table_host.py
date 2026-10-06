"""Check production codebook initialization and byte lookup without a GPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from test_shutdown_spin_host import function
from lazy_device_tables import lazy_device_tables


LIFETIME = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
int initializations=0;
namespace sycl {
template<int D=1> struct range { range(int) {} };
range(int)->range<1>;
}
namespace dpct {
template<class T,int D> struct global_memory {
    global_memory(sycl::range<1>,std::initializer_list<T>) { ++initializations; }
};
template<class T,int D> struct constant_memory {
    constant_memory(int,int) { ++initializations; }
};
}
#include "tables.inc"
int main() {
#if CONTROL
    assert(initializations==2);
    delete &iq4nl_values; delete &c_codes;
    std::puts("PASS old table constructors ran before main");
#else
    assert(initializations==0);
    auto* a=&iq4nl_storage(); auto* b=&s2_codes_storage();
    assert(initializations==2 && a==&iq4nl_storage() && b==&s2_codes_storage());
    delete a; delete b; // CPU stand-ins only: no captured users remain.
    std::puts("PASS table constructors delayed until use and retained across calls");
#endif
}
'''

LOOKUP = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#define __dpct_inline__ inline
namespace sycl {
struct int2 { uint32_t x,y; int2(uint32_t a,uint32_t b):x(a),y(b) {} };
}
namespace dpct {
#include "permute.inc"
}
#include "lookup.inc"
int main(int argc,char** argv) {
    const bool misaligned=argc==2 && std::string(argv[1])=="misaligned";
    alignas(16) int8_t storage[32]{};
    const int8_t table[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    const unsigned int first=misaligned ? 1 : 0;
    const unsigned int offsets=CONTROL ? 1 : 4;
    uint64_t compared=0;
    for (unsigned int offset=first;offset<first+offsets;++offset) {
        std::memcpy(storage+offset,table,sizeof table);
        for (uint32_t word=0;word<65536;++word) {
            const uint32_t codes=word | ((word ^ 0xffffu)<<16);
            const auto actual=iq4_table_lookup((int)codes,storage+offset);
            uint32_t even=0,odd=0;
            for (unsigned int lane=0;lane<8;++lane) {
                uint32_t value=static_cast<uint8_t>(table[(codes>>(lane*4))&15]);
                ((lane&1) ? odd : even) |= value << ((lane/2)*8);
                ++compared;
            }
            assert(actual.x==even && actual.y==odd);
        }
    }
    std::printf("PASS %llu exact codebook byte comparisons\n",(unsigned long long)compared);
}
'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    control='60fc342df60ec970b60c54a3d07a3f7a53273bdd'
    paths=['sycl/src/kernels/cuda/native_mmvq.dp.cpp','sycl/src/kernels/cuda/s2_gemv_fast.dp.cpp']
    sources={'control':[subprocess.check_output(['git','show',control+':'+p],cwd=root,text=True) for p in paths],
             'candidate':[(root/p).read_text() for p in paths]}
    permute=function((root/'sycl/include/dpct/util.hpp').read_text(),'inline unsigned int byte_level_permute(')
    env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1:abort_on_error=1',
             UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),control_commit=control,
                scope='Production table constructors, lookup and byte permutation; CPU memory/device stand-ins, no GPU',
                sources={},runs=[],passed=False)
    for variant,(mmvq,s2) in sources.items():
        record['sources'][variant]=dict(zip(paths,[hashlib.sha256(s.encode()).hexdigest() for s in (mmvq,s2)]))
        if variant=='control':
            a=mmvq.index('inline dpct::global_memory<int8_t, 1>& iq4nl_values')
            b=s2.index('inline dpct::constant_memory<float, 2>& c_codes')
            tables=mmvq[a:mmvq.index(';',a)+1]+'\n'+s2[b:s2.index(';',b)+1]
            for old in (mmvq,s2):
                regenerated=lazy_device_tables(old)
                assert regenerated==lazy_device_tables(regenerated)
                assert '= *new dpct::' not in regenerated
                assert 'reinterpret_cast<const uint32_t*>(iq4nl_values)' not in regenerated
        else:
            tables=function(mmvq,'inline dpct::global_memory<int8_t, 1>& iq4nl_storage()')+'\n'+function(s2,'inline dpct::constant_memory<float, 2>& s2_codes_storage()')
        for group,program,fragments in [('lifetime',LIFETIME,{'tables.inc':tables}),
                                         ('lookup',LOOKUP,{'lookup.inc':function(mmvq,'__dpct_inline__ sycl::int2 iq4_table_lookup('),'permute.inc':permute})]:
            out=args.output/variant/group;out.mkdir(parents=True)
            (out/'probe.cpp').write_text(program)
            for name,text in fragments.items():(out/name).write_text(text+'\n')
            p=subprocess.run(['g++','-std=c++20','-O2','-g','-fno-omit-frame-pointer','-fno-pie','-no-pie',
                              '-fsanitize=address,undefined','-DCONTROL='+str(int(variant=='control')),
                              '-I'+str(out),str(out/'probe.cpp'),'-o',str(out/'probe')],capture_output=True,text=True,timeout=30)
            (out/'build.stderr').write_text(p.stderr);assert p.returncode==0,p.stderr
            for mode in (['aligned','misaligned'] if variant=='control' and group=='lookup' else ['normal']):
                run=subprocess.run([str(out/'probe'),mode],capture_output=True,text=True,env=env,timeout=10)
                (out/(mode+'.stdout')).write_text(run.stdout);(out/(mode+'.stderr')).write_text(run.stderr)
                passed=(run.returncode!=0 and 'load of misaligned address' in run.stderr
                        if mode=='misaligned' else run.returncode==0 and 'PASS' in run.stdout)
                record['runs'].append(dict(variant=variant,group=group,mode=mode,exit_code=run.returncode,expected_result=passed))
                (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
                assert passed,(variant,group,mode,run.stdout,run.stderr)
    record.update(passed=True,regeneration_idempotent=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
