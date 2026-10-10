
#include "iq4nl_caller_census.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <climits>
using Ledger=strata::prefill::detail::Iq4nlCallerCensus;
void check(bool v){if(!v)throw std::runtime_error("host_ledger_assertion");}
void base(Ledger& q){q.begin(8192,1,1,false,false,false);}
void filled(Ledger& q,long long n,bool major){
 q.begin(n,48,512,major,true,false);
 for(int l=0;l<48;++l) for(long long pos=0;pos<n;pos+=8192){
  const auto t=std::min(8192LL,n-pos);bool priv=(pos/8192)%2;
  q.chunk(l,pos,t);q.routed(l,t,1,t);q.decision(l,0,t);
  q.selected(l,0,42,20,2560,640,priv);q.returned(l,0,priv);
 }
 for(int l=0;l<48;++l){
  auto& e=q.expert[l*512];auto& c=q.layer[l];
  check(c.positions==(unsigned long long)n && c.chunks==(unsigned long long)n/8192);
  check(e.decisions==(unsigned long long)n/8192 && e.rows==(unsigned long long)n);
  check(e.compressed==(unsigned long long)n/8192*921600ULL);
  check(e.output==(unsigned long long)n/8192*3276800ULL);
 }
 check(!q.invalid&&!q.overflow&&!q.unsupported);
}
void label(const char* name){std::fprintf(stderr,"HOST_CASE,%s\n",name);}
int main(){
 try{
  auto holder=std::make_unique<Ledger>();auto& q=*holder;
  static_assert(sizeof(Ledger)<4*1024*1024);
  label("valid32k");filled(q,32768,false);q.report(true,true);check(!q.invalid&&!q.overflow&&!q.active);
  label("full_count_layermajor_HOST_SIMULATION");filled(q,262144,true);q.report(true,true);check(!q.invalid&&!q.overflow&&!q.active);
  label("bounds");base(q);check(!q.bounds(-1)&&q.invalid);q.report(false,false);
  label("chunk_order");base(q);q.chunk(0,1,8192);check(q.invalid);q.report(false,false);
  label("private_wrong_type");base(q);q.selected(0,0,42,42,2560,640,true);check(q.invalid);q.report(false,false);
  label("type20_extent");base(q);q.selected(0,0,42,20,1,33,false);check(q.invalid);q.report(false,false);
  label("shape_change");base(q);q.selected(0,0,42,20,2560,640,false);q.selected(0,0,42,20,1280,640,false);check(q.invalid);q.report(false,false);
  label("add_overflow");base(q);unsigned long long x=ULLONG_MAX;uint64_t v=x;q.add(v,1);check(q.overflow&&v==ULLONG_MAX);q.report(false,false);
  label("mul_overflow");base(q);check(q.mul(ULLONG_MAX,2)==0&&q.overflow);q.report(false,false);
  label("returned_without_selected");base(q);q.chunk(0,0,8192);q.routed(0,8192,1,8192);q.decision(0,0,8192);q.returned(0,0,false);q.report(true,true);check(q.invalid);
  label("missing_return");base(q);q.chunk(0,0,8192);q.routed(0,8192,1,8192);q.decision(0,0,8192);q.selected(0,0,42,20,2560,640,false);q.report(true,true);check(!q.invalid);
  label("rows_mismatch");base(q);q.chunk(0,0,8192);q.routed(0,8192,1,8191);q.decision(0,0,8191);q.selected(0,0,42,20,2560,640,false);q.returned(0,0,false);q.report(true,true);check(q.invalid);
  label("unsupported");q.begin(8192,1,1,false,false,true);q.report(false,false);check(q.unsupported);
  label("pipeline_incomplete");filled(q,32768,false);q.report(true,false);check(!q.invalid);
  std::printf("HOST_CONTRACT,cases=14,ledger_bytes=%zu,expert_bytes=%zu,layer_bytes=%zu,no_GPU=1,full_count_is_host_simulation=1\n",sizeof(Ledger),sizeof(Ledger::Expert),sizeof(Ledger::Layer));
  return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"HOST_FAIL,%s\n",e.what());return 1;}
}
