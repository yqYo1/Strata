"""Main-authored Page4 specialization; no narrowing or inference dispatch edit."""
from pathlib import Path

NATIVE = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
          'src/kernels/sycl/prompt_attention.cpp', 'src/kernels/sycl/qsa_bf16_prefix.cpp')
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX = NATIVE[-1]
BENCH = 'tests/sycl/qsa_bf16_prefix_layout_bench.cpp'
EXPORT = 'qsa_bf16_prefix_ordered_exp_batch'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('source edit is not unique: '+before[:90])
    return text.replace(before, after, 1)


def generated(project):
    source=(project/PREFIX).read_text()
    wrapper=source[source.index('bool '+EXPORT+'('):source.index('bool qsa_bf16_prefix_state_probe_batch(')]
    prototype=wrapper[:wrapper.index(' {')]+';\n'
    original=source[:source.index('bool qsa_bf16_prefix_batch(')]+wrapper+'} // namespace strata::kernels\n'
    if 'int64_t{4}' in original:raise ValueError('specialization reversal token already present')
    result={}
    guard='''  if (shape.page_size != 4) {
    ++generic_routes;
    return ::strata::kernels::qsa_bf16_prefix_ordered_exp_batch(query,pools,ids,steps,
        cap,shape,output,queries,bounds,live_cells,scratch,stream);
  }
  ++specialized_routes;
'''
    counters='''namespace {
// Component-only host route proof. One calling thread; no device reads.
uint64_t specialized_routes=0,generic_routes=0;
}
std::array<uint64_t,2> route_counts() { return {specialized_routes,generic_routes}; }
'''
    for variant in ('control','page4'):
        namespace=f'namespace strata::kernels::{variant} {{'
        body=replace_once(original,'namespace strata::kernels {',namespace)
        if variant=='page4':
            body=body.replace('shape.page_size','int64_t{4}')
            body=replace_once(body,namespace,namespace+'\n'+counters)
            # Keep the caller's actual shape for admission. Non-4 shapes go
            # through the original generic wrapper before any specialized work.
            mark='void *stream) {\n  return prefix_batch_impl<Cells + 1, false, false, true, true, true, true>'
            body=replace_once(body,mark,'void *stream) {\n'+guard+mark.split('\n',1)[1])
            body='#include <array>\n'+body
            restored=body.removeprefix('#include <array>\n').replace(guard,'',1).replace(counters,'',1)
            restored=restored.replace(namespace+'\n',namespace,1).replace('int64_t{4}','shape.page_size')
        else:restored=body
        if restored.replace(namespace,'namespace strata::kernels {',1)!=original:
            raise ValueError('Page4 overlay changed math,guards,lifetimes or another body byte')
        result[variant+'.cpp']=body
    header='// Main-authored component declarations; repository LICENSE applies.\n#pragma once\n'
    header+='#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include <array>\n#include <algorithm>\n#include <limits>\n'
    for variant in ('control','page4'):
        header+=f'namespace strata::kernels::{variant} {{\n'+prototype
        if variant=='page4':header+='std::array<uint64_t,2> route_counts();\n'
        header+='}\n'
    header+='inline auto qsa_page4_functions() {\n  using namespace strata::kernels;\n'
    header+=f'  return std::array<decltype(&{EXPORT}),3>{{{EXPORT},control::{EXPORT},page4::{EXPORT}}};\n}}\n'
    result['qsa-page4-variants.hpp']=header

    numerical=(project/'tests/sycl/qsa_bf16_prefix.cpp').read_text()
    numerical=replace_once(numerical,'constexpr int Heads=24,KV=2,Dim=256,Page=16;',
        'constexpr int Heads=24,KV=2,Dim=256; const int Page=qsa_page_fixture_page;')
    numerical=replace_once(numerical,
        '        const size_t row=(size_t(table[0])*KV+head)*Page+invalid_cell;\n        scales[row*4]=0x7c00;',
        '''        const int logical=invalid_cell/Page;
        require(logical>=0 && size_t(logical)<table.size(),"diagnostic logical page bounds");
        const int physical=table[size_t(logical)];
        require(physical>=0 && physical<pages,"diagnostic physical page bounds");
        const size_t row=(size_t(physical)*KV+head)*Page+invalid_cell%Page;
        require(row<size_t(rows),"diagnostic row bounds");
        require(scales.size()>=4 && row<scales.size()/4,"diagnostic scale element bounds");
        scales[row*4]=0x7c00;''')
    numerical=replace_once(numerical,'    if (cap>5) ids[b*cap+5]=-1;', '''    if (cap>5) ids[b*cap+5]=-1;
    if (cap>11) {
      ids[b*cap+2]=std::numeric_limits<int32_t>::min();
      ids[b*cap+3]=std::numeric_limits<int32_t>::max();
      ids[b*cap+4]=cells; ids[b*cap+6]=live_cells;
      ids[b*cap+7]=live_cells-1; ids[b*cap+8]=cells-1;
      ids[b*cap+9]=Page-1; ids[b*cap+10]=Page;
      ids[b*cap+11]=0;
    }''')
    numerical=replace_once(numerical,'<<" live_cells="<<live_cells','<<" page_size="<<Page<<" live_cells="<<live_cells')
    result['numerical-reference.inc']=numerical

    timing=(project/BENCH).read_text()
    start=timing.index('#if defined(STRATA_QSA_BENCH_ORDERED_EXP)')
    end=timing.index('#endif',start)+len('#endif')
    timing=timing[:start]+'''const auto attention_variants=qsa_page4_functions();
const std::array<const char*,3> variant_names{"default","control","page4"};
long long timer_floor_ns=0;'''+timing[end:]
    timing='#include "qsa-page4-variants.hpp"\n'+timing
    edits=[
        ('constexpr int Dim=256,Heads=24,KV=2,Page=16;','constexpr int Dim=256,Heads=24,KV=2,Page=4;'),
        ('(variant ? candidate_attention : baseline_attention)','attention_variants[size_t(variant)]'),
        ('const std::array<int,2> variants{0,1};','const std::array<int,3> variants{0,1,2};'),
        ('index<2','index<3'),('(process_id+round+index)%2','(process_id+round+index)%3'),
        ('(variant ? candidate_name : baseline_name)','variant_names[size_t(variant)]'),
        ('<<" samples=12 full_bits=PASS','<<" samples=18 full_bits=PASS'),
        ('rt->wait();const auto start=std::chrono::steady_clock::now();',
         'rt->wait();const auto routes_before=strata::kernels::page4::route_counts();\n    const auto start=std::chrono::steady_clock::now();'),
        ('rt->wait();return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();',
         '''rt->wait();const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    const auto routes_after=strata::kernels::page4::route_counts();
    require(routes_after[0]-routes_before[0]==uint64_t(variant==2) && routes_after[1]==routes_before[1],"actual Page4 timing route");
    return elapsed;'''),
        ('const double ms=invoke(variant);','''const double ms=invoke(variant);
    if (queries>=8192) require(ms>=.1 && ms*1e6>=1000.*timer_floor_ns,"timing batch below declared floor");'''),
        ('<<" milliseconds="<<std::setprecision(12)<<ms','<<" page_size=4 route=PASS milliseconds="<<std::setprecision(12)<<ms'),
        ('std::cout.setf(std::ios::unitbuf);','''std::cout.setf(std::ios::unitbuf);
    timer_floor_ns=std::numeric_limits<long long>::max();
    for (int i=0;i<128;++i) {
      const auto a=std::chrono::steady_clock::now(),b=std::chrono::steady_clock::now();
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
      if (ns>0) timer_floor_ns=std::min(timer_floor_ns,static_cast<long long>(ns));
    }
    require(timer_floor_ns<std::numeric_limits<long long>::max(),"clock calibration");
    std::cout<<"timer_floor_ns="<<timer_floor_ns<<'\\n';'''),
        ('    rt->wait();rt.reset();gate(gated,"after-cleanup");',
         '''    const auto routes=strata::kernels::page4::route_counts();
    std::cout<<"page4_timing_routes fast="<<routes[0]<<" generic="<<routes[1]<<'\\n';
    rt->wait();rt.reset();gate(gated,"after-cleanup");'''),
        ('else { run(78453,8192,33,false);run(78453,8192,2051,false);run(78453,8192,2051,true); }',
         'else { run(32768,8192,2051,false);run(117488,8192,2051,false);run(117488,8192,2051,true); }')]
    for before,after in edits:timing=replace_once(timing,before,after)
    result['timing.cpp']=timing
    return {name:body.encode() for name,body in result.items()}
