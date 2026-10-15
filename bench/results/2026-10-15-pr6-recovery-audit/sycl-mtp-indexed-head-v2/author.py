from pathlib import Path
import re,difflib,hashlib,json
root=Path.cwd(); a=root/'bench/results/2026-10-15-pr6-recovery-audit'; v1=a/'sycl-mtp-indexed-head-v1'; out=a/'sycl-mtp-indexed-head-v2'
patch=(v1/'source.diff').read_text().splitlines(keepends=True); sources={}; bases={}; i=0
while i<len(patch):
    if not patch[i].startswith('--- '): i+=1;continue
    old=patch[i][4:].strip(); name=patch[i+1][6:].strip(); i+=2
    base='' if old=='/dev/null' else (root/name).read_text(); bases[name]=base
    original=base.splitlines(keepends=True); result=[]; pos=0
    while i<len(patch) and not patch[i].startswith('--- '):
        m=re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@',patch[i])
        if not m: raise ValueError(patch[i])
        start=max(0,int(m[1])-1);result+=original[pos:start];pos=start;i+=1
        while i<len(patch) and not patch[i].startswith(('@@ ','--- ')):
            line=patch[i];i+=1
            if line[0] in ' -':
                if original[pos]!=line[1:]: raise ValueError('root base mismatch '+name)
                pos+=1
            if line[0] in ' +':result.append(line[1:])
    result+=original[pos:];sources[name]=''.join(result)
# Validate pinned base/output while reconstructing, not running product/tests.
rec=json.loads((v1/'reconstruction.json').read_text())['files']
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
for n,s in sources.items():
    assert sha(s)==rec[n]['output_sha256'],n
    assert (sha(bases[n]) if rec[n]['base_sha256'] else None)==rec[n]['base_sha256'],n
n='include/strata/core/indexed_head_contract.hpp'
s=sources[n].replace('#include <stdexcept>','#include <stdexcept>\n#include <vector>\n#include <cstdio>\n#include <cerrno>\n#include <cstring>\n#include <string>')
s=s.replace('} // namespace strata::core::detail',r'''
// Production MTP logits are max_t * full, unlike the generic indexed kernel.
inline void validate_mtp_head_capacity(size_t count,int64_t full) {
    if(full<=0 || full>INT32_MAX || count>uint64_t(full))
        throw std::invalid_argument("MTP subset exceeds full-vocabulary logits capacity");
}
struct IndexedHeadMap {
    bool present=false;
    std::vector<int32_t> ids;
};
// Shared bounded reader: EOF, full payload alignment, capacity and every ID are
// validated before any caller can claim indexed storage savings or publish IDs.
// A FILE* overload permits a host fixture without filesystem-path assumptions.
inline IndexedHeadMap read_mtp_head_map(FILE* f,int64_t full) {
    IndexedHeadMap map;
    if(!f)return map;
    validate_mtp_head_capacity(0,full);
    map.present=true;
    const size_t cap=indexed_head_product(uint64_t(full),sizeof(int32_t));
    std::vector<uint8_t> raw;
    uint8_t chunk[4096];
    for(;;) {
        const size_t got=std::fread(chunk,1,sizeof(chunk),f);
        if(got>cap-raw.size())throw std::invalid_argument("MTP subset exceeds full-vocabulary logits capacity");
        raw.insert(raw.end(),chunk,chunk+got);
        if(got<sizeof(chunk)) {
            if(std::ferror(f) || !std::feof(f))throw std::invalid_argument("unreadable MTP vocabulary map");
            break;
        }
    }
    if(raw.empty() || raw.size()%sizeof(int32_t))throw std::invalid_argument("invalid MTP vocabulary map length");
    map.ids.resize(raw.size()/sizeof(int32_t));
    std::memcpy(map.ids.data(),raw.data(),raw.size());
    validate_indexed_head_ids(map.ids.data(),map.ids.size(),full);
    return map;
}
inline IndexedHeadMap read_mtp_head_map(const std::string& path,int64_t full) {
    errno=0;
    FILE* f=std::fopen(path.c_str(),"rb");
    if(!f) {
        if(errno==ENOENT)return {};
        throw std::invalid_argument("cannot open MTP vocabulary map");
    }
    struct Close {FILE* f;~Close(){std::fclose(f);}} close{f};
    return read_mtp_head_map(f,full);
}
inline bool mtp_indexed_map_ready(bool eligible,const IndexedHeadMap& map) noexcept {
    return eligible && map.present && !map.ids.empty();
}
} // namespace strata::core::detail
''')
sources[n]=s
n='src/core/mtp.cpp';s=sources[n]
start=s.index('                bool indexed=false;',s.index('int head_type) const'))
end=s.index('                if (!indexed)',start)
s=s[:start]+'''                bool indexed=false;
                if (g_ && detail::indexed_head_eligible(indexed_head_requested(),head_type,g_->n_embd,n_vocab,head_row_bytes)) {
                    try {
                        const auto map=detail::read_mtp_head_map(rt_dir_ + "/draft_vocab.bin",n_vocab);
                        indexed=detail::mtp_indexed_map_ready(true,map);
                    } catch (const std::exception&) {
                        // Invalid/unreadable input never advertises saved copied-row bytes.
                        // bind reports the validation error; reservation remains conservative.
                    }
                }
'''+s[end:]
start=s.index('        std::vector<uint8_t> raw;',s.index('// The indexed opt-in'))
end=s.index('            const auto allocation_status=',start)
s=s[:start]+'''        detail::IndexedHeadMap map;
        try { map=detail::read_mtp_head_map(rt_dir_ + "/draft_vocab.bin",n_vocab_); }
        catch (const std::exception& error) {err=std::string("mtp: ")+error.what();return false;}
        if (detail::mtp_indexed_map_ready(true,map)) {
            const size_t map_bytes=map.ids.size()*sizeof(int32_t);
            try {
                const detail::IndexedHeadShape shape(g_->n_embd,n_vocab_,int64_t(map.ids.size()),max_t_);
                if (shape.weights!=head->weight_bytes()) {
                    err="mtp: indexed native head full span mismatch";return false;
                }
            } catch (const std::exception& error) {err=std::string("mtp: ")+error.what();return false;}
'''+s[end:]
start=s.index('            const auto allocation_status=',s.index('// The indexed opt-in'));end=s.index('\n    // the draft head',start)
part=s[start:end].replace('raw.size()','map_bytes').replace('std::move(ids)','std::move(map.ids)');s=s[:start]+part+s[end:]
needle='''            n_dvocab_ = (int64_t) (raw.size() / 4);'''
s=s.replace(needle,'''#ifdef STRATA_ENABLE_SYCL
            try { detail::validate_mtp_head_capacity(raw.size()/4,n_vocab_); }
            catch (const std::exception& error) {err=std::string("mtp: ")+error.what();return false;}
#endif
'''+needle)
s=s.replace('if (g_ && detail::indexed_head_eligible(indexed_head_requested(),head->type(),g_->n_embd,n_vocab_,head->row_bytes())) {\n        indexed_head_weights_=head->weights();indexed_bind_complete_=true;', 'if (indexed_head_) {\n        indexed_bind_complete_=true;')
sources[n]=s
# Full root-based patch is the only product source copy; no duplicate large TUs.
output=[]; manifest={}
for n,s in sources.items():
    output.extend(difflib.unified_diff(bases[n].splitlines(True),s.splitlines(True),fromfile='a/'+n if rec[n]['base_sha256'] else '/dev/null',tofile='b/'+n))
    manifest[n]={'base_sha256':rec[n]['base_sha256'],'output_sha256':sha(s)}
(out/'source.diff').write_text(''.join(output));(out/'reconstruction.json').write_text(json.dumps({'schema':'sycl-mtp-indexed-head-reconstruction-v2','files':manifest},indent=2)+'\n')
(out/'pins.json').write_bytes((v1/'pins.json').read_bytes())
# Compact always-on host fixture, still the actual production helper.
s=(v1/'cpu_contract_fixture.cpp').read_text().replace('#include <cassert>','#include <stdexcept>\n#include <cstring>\n#include <cstdio>')
s=s.replace('assert(', 'need(')
s=s.replace('static void rejects(', 'static void need(bool ok){if(!ok)throw std::runtime_error("host contract failed");}\nstatic void rejects(')
s=s.replace('int main(){',r'''static IndexedHeadMap from_bytes(const std::vector<uint8_t>& raw,int64_t full){
    FILE* f=std::tmpfile();need(f!=nullptr);
    struct Close{FILE* f;~Close(){std::fclose(f);}} close{f};
    need(std::fwrite(raw.data(),1,raw.size(),f)==raw.size());need(std::fseek(f,0,SEEK_SET)==0);
    return read_mtp_head_map(f,full);
}
static std::vector<uint8_t> bytes(std::initializer_list<int32_t> ids){std::vector<uint8_t> b(ids.size()*4);std::memcpy(b.data(),ids.begin(),b.size());return b;}
int main(){
    auto absent=read_mtp_head_map(static_cast<FILE*>(nullptr),37);
    need(!mtp_indexed_map_ready(true,absent)); // absent subset: full-head fallback
    auto valid=from_bytes(bytes({36,0,17,17,1}),37);
    need(mtp_indexed_map_ready(true,valid));
    need(!mtp_indexed_map_ready(indexed_head_eligible(true,13,2560,37,2100),valid)); // noneligible legacy
    need(valid.ids==std::vector<int32_t>({36,0,17,17,1}));
    for(auto raw:std::vector<std::vector<uint8_t>>{{},{1},{1,2,3},bytes({-1}),bytes({37})})
        rejects([&]{(void)from_bytes(raw,37);});
    rejects([&]{(void)from_bytes(bytes({0,0,0}),2);}); // bounded map, before publication
    rejects([]{validate_mtp_head_capacity(38,37);});
    validate_mtp_head_capacity(37,37);
    IndexedHeadShape boundary(256,37,51,1);need(boundary.row==210); // generic duplicate count allowed
''')
(out/'cpu_contract_fixture.cpp').write_text(s)
s=(v1/'gpu_parity_fixture.cpp').read_text()
s=s.replace('static void run(int cols,const std::vector<int32_t>& ids,cudaStream_t stream,bool exact){\n    constexpr int ni=2560,full=37;const int subset=int(ids.size());', 'static void run(int ni,int full,int cols,const std::vector<int32_t>& ids,cudaStream_t stream,bool exact,bool negatives=true){\n    const int subset=int(ids.size());')
start=s.index('    checked(cudaGraphLaunch(exec,stream)');end=s.index('    checked(cudaGraphExecDestroy(exec)',start)
s=s[:start]+'''    const auto reference=old.read();std::vector<uint8_t> actual;
    for(int replay=0;replay<3;++replay) {
        indexed.upload(); // poison all output and guards again before every replay
        checked(cudaGraphLaunch(exec,stream),"launch indexed graph");checked(cudaStreamSynchronize(stream),"indexed graph drain");
        actual=indexed.read();need(reference==actual,"complete indexed/gather MMVQ bits differ");old.guards(reference);indexed.guards(actual);
        for(size_t i=0;i<shape.output/4;++i){float value;std::memcpy(&value,actual.data()+64+4*i,4);need(std::isfinite(value),"nonfinite output");}
        weights.immutable();mapping.immutable();input.immutable();need(x.read()==quantized,"Q8 activation payload changed");x.guards(quantized);gathered.guards(gathered.read());
    }
'''+s[end:]
start=s.index('    auto invoke=');end=s.index('    std::printf(',start)
s=s[:start]+'    if(negatives) {\n'+s[start:end]+'    }\n'+s[end:]
# log no invalid-map result for production case unless actually checked
s=s.replace('\\"invalid_map_zero\\":true','\\"graph_replays\\":3')
s=s.replace('run(cols,ids,stream,exact);++cases;', 'for(int ni:{256,2560}){run(ni,37,cols,ids,stream,exact);++cases;}')
s=s.replace('    checked(cudaStreamSynchronize(stream),"all cases drain");', '''    // Finite production T=1 case: about 750 MiB device payload plus compact guards.
    // Exact mode is process-wide and this case runs in BOTH separate invocations.
    std::vector<int32_t> production;production.reserve(106299);
    for(int i=0;i<106299;++i)production.push_back((248319-int64_t(i)*7%248320+248320)%248320);
    run(2560,248320,1,production,stream,exact,false);++cases;
    checked(cudaStreamSynchronize(stream),"all cases drain");''')
# invalid zeros variable scope removed from log? log only actual hash.
(out/'gpu_parity_fixture.cpp').write_text(s)
