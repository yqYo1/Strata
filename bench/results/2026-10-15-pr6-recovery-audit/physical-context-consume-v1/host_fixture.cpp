// Main-owned, unexecuted host fixture. No runtime/GPU and no environment changes.
// Uses actual proposed CLI/admission/coverage/terminal helper code.
#include "strata/program/consume_prompt.hpp"
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <vector>
using namespace strata::program::consume_prompt;
static void need(bool b,const char* m){if(!b){std::fprintf(stderr,"FAIL %s\n",m);std::abort();}}
static void cli_admission() {
    need(cli_flag("--consume-prompt") && !cli_flag("--consume-prompt=1") && !cli_flag("--max-new") && !cli_flag(nullptr),"exact CLI flag");
    Admission yes{true,true,true,true,true,true,true,true,true,true,true};
    need(supported(yes),"valid diagnostic admission");
    bool Admission::* const fields[]={&Admission::sycl,&Admission::standalone,&Admission::native_cli,&Admission::single_gpu,
        &Admission::mtp4,&Admission::prefill,&Admission::full_kv,&Admission::ple,&Admission::no_token_graph,
        &Admission::suffix_disabled,&Admission::no_competing_diagnostics};
    for(auto field:fields){auto bad=yes;bad.*field=false;need(!supported(bad),"each unsupported mode rejects");}
    need(!context_admitted(false,0,512,511),"ordinary maxnew0 stays invalid");
    need(context_admitted(false,1,512,511),"ordinary legal output reserve");
    need(!context_admitted(false,1,512,512),"ordinary full prompt still reserves output");
    for(int n:{512,262144}){
        need(context_admitted(true,0,n,n),"exact consume zero output");
        need(context_admitted(true,16,n,n),"diagnostic ignores ordinary default output budget");
        need(!context_admitted(true,-1,n,n),"negative maxnew rejects");
        need(!context_admitted(true,0,n,size_t(n)+1),"input overflow rejects");
    }
    need(!context_admitted(true,0,262145,262144),"no physical limit extension");
    need(!context_admitted(true,0,0,0) && !context_admitted(true,0,512,1),"empty and unbatched mode reject");
    need(context_admitted(true,0,1024,512),"partial-capacity consume is distinctly receipted");
}
static void coverage_and_terminal(int64_t input,int64_t chunk) {
    Coverage target(input-1),mtp(input-1);
    for(int64_t at=0;at<input-1;){const auto count=std::min(chunk,input-1-at);
        need(target.accept(at,count) && mtp.accept(at,count),"contiguous final-output callbacks");at+=count;}
    need(target.complete() && mtp.complete(),"complete N-1 coverage");
    const auto before=target;
    need(!target.accept(0,1) && !target.accept(input-1,1) && !target.accept(input-1,0) &&
         !target.accept(-1,1) && !target.accept(input-1,std::numeric_limits<int64_t>::max()),"negative coverage");
    need(target.next==before.next && target.callbacks==before.callbacks,"reject leaves coverage unchanged");
    Coverage gap(input-1);need(!gap.accept(1,1) && gap.next==0,"gap rejects before publication");
    const int64_t last=42;
    Terminal t;
    need(!t.ran() && !t.committed(1) && !t.waited(),"out of order terminal stages reject");
    need(!t.begin(false,1,input-1,last,input,last,0,target) &&
         !t.begin(true,2,input-1,last,input,last,0,target) &&
         !t.begin(true,1,input,last,input,last,0,target) &&
         !t.begin(true,1,input-1,last+1,input,last,0,target) &&
         !t.begin(true,1,input-1,last,input,last,1,target),"pre-mutation terminal negatives");
    need(t.windows==0,"invalid begin unchanged");
    need(t.begin(true,1,input-1,last,input,last,0,target),"exact terminal begin");
    need(!t.waited() && t.ran() && !t.ran(),"run stage once");
    need(!t.committed(0) && !t.committed(2) && t.committed(1) && !t.committed(1),"commit exactly one");
    need(!t.complete(0) && t.waited() && !t.waited() && t.complete(0) && !t.complete(1),"wait and no emitted output");
    need(!t.begin(true,1,input-1,last,input,last,0,target) && t.windows==1,"no second target window");
    need(target.next+1==input && t.position==input-1,"last prompt input is distinct from prediction");
    need(target.last_count==chunk-1,"exact-context partial final chunk");
}
int main(){cli_admission();coverage_and_terminal(512,128);coverage_and_terminal(262144,8192);
    const int64_t a[]={1,2},b[]={2,1};need(token_fingerprint(a,2)!=token_fingerprint(b,2),"ordered input fingerprint");
    std::puts("HOST_ONLY policy/coverage/terminal fixture complete; no GPU/full-context result");}
