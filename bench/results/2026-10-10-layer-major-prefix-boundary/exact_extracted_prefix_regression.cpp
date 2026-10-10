#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
struct Cache { int64_t count; int64_t slots() const { return count; } };
struct Geometry { int64_t n_layers=48, n_expert=512; };
struct Impl { Cache* cache; Geometry* g; int64_t T; void* transfer_context=nullptr; int64_t residual_gpu_tokens=0; };
int64_t outer_old(Impl& m, const Cache& layer_cache, int64_t n) {
auto* impl_=&m; const auto& g=*m.g; (void)impl_; (void)g; (void)layer_cache;
    const char* gpu_env = std::getenv("STRATA_PREFILL_LAYER_MAJOR_R_GPU");
    int64_t gpu_tokens = gpu_env ? std::clamp<int64_t>(std::atoll(gpu_env), 0, n) : 0;
    const char* first_env = std::getenv("STRATA_PREFILL_FIRST");
    const bool all_cached = impl_->cache && impl_->g &&
        impl_->cache->slots() >= impl_->g->n_layers * impl_->g->n_expert;
    const int64_t first = first_env ? std::atoll(first_env) : (all_cached ? 256 : 0);
    const int64_t first_len = first > 0 && first < m.T && n > 2 * first ? first : std::min(m.T, n);
    if (gpu_tokens < n) {
        // Keep a complete prefix of chunks on-device. A tail never straddles the two stores.
        gpu_tokens = gpu_tokens < first_len ? 0 : first_len + (gpu_tokens - first_len) / m.T * m.T;
    }

return gpu_tokens;
}
int64_t outer_fixed(Impl& m, const Cache& layer_cache, int64_t n) {
auto* impl_=&m; const auto& g=*m.g; (void)impl_; (void)g; (void)layer_cache;
    const char* gpu_env = std::getenv("STRATA_PREFILL_LAYER_MAJOR_R_GPU");
    int64_t gpu_tokens = gpu_env ? std::clamp<int64_t>(std::atoll(gpu_env), 0, n) : 0;
    const char* first_env = std::getenv("STRATA_PREFILL_FIRST");
    // run_impl uses this temporary cache, so resolve its first chunk before
    // rounding the GPU prefix. The original cache may have held every layer.
    const bool all_cached = layer_cache.slots() >= g.n_layers * g.n_expert;
    const int64_t first = first_env ? std::atoll(first_env) : (all_cached ? 256 : 0);
    const int64_t first_len = first > 0 && first < m.T && n > 2 * first ? first : std::min(m.T, n);
    if (gpu_tokens < n) {
        // Keep a complete prefix of chunks on-device. A tail never straddles the two stores.
        gpu_tokens = gpu_tokens < first_len ? 0 : first_len + (gpu_tokens - first_len) / m.T * m.T;
    }

return gpu_tokens;
}
std::vector<int64_t> chunk_lengths(Impl& m, int64_t n) {
    const bool all_cached = m.cache != nullptr && m.g != nullptr && m.cache->slots() >= m.g->n_layers * m.g->n_expert;
    static const char* const first_env = std::getenv("STRATA_PREFILL_FIRST");
    const int64_t first_chunk = first_env ? std::atoll(first_env) : (all_cached ? 256 : 0);
    auto chunk_len = [&](int64_t c0) {
        if (c0 == 0 && first_chunk > 0 && first_chunk < m.T && n > 2 * first_chunk) return first_chunk;
        return std::min(m.T, n - c0);
    };

std::vector<int64_t> r; for(int64_t c0=0;c0<n;c0+=chunk_len(c0)) {
const auto length=chunk_len(c0); assert(length>0 && length<=m.T && length<=n-c0); r.push_back(length);
} return r;
}
bool actual_guard(Impl& m, int64_t c0, int64_t T, std::string& err) {
        if (m.transfer_context && c0 < m.residual_gpu_tokens && c0 + T > m.residual_gpu_tokens) {
            err = "prefill layer-major: GPU residual prefix splits a chunk";
            return false;
        }

return true;
}

bool spans_valid(int64_t n,int64_t G,const std::vector<int64_t>& chunks) {
 if(G<0||G>n) return false;
 int64_t row=0;
 for(auto count:chunks) {
  if(row+count<=G) { if(row<0 || count>G-row) return false; }
  else { const auto offset=row-G; if(offset<0 || offset>n-G || count>n-G-offset) return false; }
  row+=count;
 }
 return row==n;
}
int main() {
 Geometry g; Cache layer{512}; uint64_t cases=0,old_bad=0,guard_rejections=0;
 const int64_t ns[]={1,255,256,257,511,512,513,8191,8192,8193,16385,32768,262143,262144};
 const int64_t ts[]={128,256,257,512,8192,32768};
 const int64_t slots[]={0,511,512,24575,24576,24577};
 for(auto n:ns) for(auto T:ts) for(auto sl:slots) {
  Cache original{sl}; Impl m{&original,&g,T};
  const int64_t requests[]={-1,0,1,255,256,257,8191,8192,8193,8448,16384,16640,32768,n-1,n,n+1};
  for(auto want:requests) {
   const auto value=std::to_string(want); assert(!setenv("STRATA_PREFILL_LAYER_MAJOR_R_GPU",value.c_str(),1));
   m.cache=&original; auto oldG=outer_old(m,layer,n); auto G=outer_fixed(m,layer,n);
   m.cache=&layer; const auto chunks=chunk_lengths(m,n);
   assert(spans_valid(n,G,chunks));
   m.transfer_context=&m; m.residual_gpu_tokens=G; int64_t c0=0;
   for(auto length:chunks) { std::string error; assert(actual_guard(m,c0,length,error)); assert(error.empty()); c0+=length; }
   if(!spans_valid(n,oldG,chunks)) {
    ++old_bad; m.residual_gpu_tokens=oldG; c0=0; bool rejected=false;
    for(auto length:chunks) { std::string error; if(!actual_guard(m,c0,length,error)) { assert(error=="prefill layer-major: GPU residual prefix splits a chunk"); rejected=true; } c0+=length; }
    assert(rejected); ++guard_rejections;
   }
   ++cases;
  }
 }
 Cache full{24576}; Impl m{&full,&g,8192};
 assert(!setenv("STRATA_PREFILL_LAYER_MAJOR_R_GPU","8192",1));
 const auto oldG=outer_old(m,layer,32768), newG=outer_fixed(m,layer,32768);
 m.cache=&layer; const auto chunks=chunk_lengths(m,32768);
 const bool unset=std::getenv("STRATA_PREFILL_FIRST")==nullptr;
 if(unset) { assert(oldG==256 && newG==8192); assert(!spans_valid(32768,oldG,chunks)); assert(spans_valid(32768,newG,chunks)); }
 m.transfer_context=&m; m.residual_gpu_tokens=1; std::string err; assert(!actual_guard(m,0,8192,err));
 m.transfer_context=nullptr; err.clear(); assert(actual_guard(m,0,8192,err) && err.empty());
 std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"original_invalid_spans\":"<<old_bad
 <<",\"guard_rejected_original_invalid_spans\":"<<guard_rejections<<",\"fixed_invalid_spans\":0,\"named_unset_original_G\":"<<oldG
 <<",\"named_unset_fixed_G\":"<<newG<<",\"named_unset_original_first_host_offset\":"<<-oldG<<"}\n";
}
