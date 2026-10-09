#include "repeat_capture.hpp"
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <signal.h>
using strata::prefill::detail::RepeatCapture;
static void check(bool ok,const char* reason) { if(!ok) throw std::runtime_error(reason); }
static std::string try_record(RepeatCapture& c,sycl::queue& q,const void* src,int64_t rows=2,int64_t width=4) {
    try { c.record(q,3,"test",98304,8192,98304,rows,width,1,src); }
    catch(const std::exception& e) { return e.what(); }
    return "";
}
int main(int argc,char** argv) {
    try {
        check(argc==2,"one case required"); const std::string name=argv[1]; sycl::queue q;
        float data[8]={1.25f,-2.5f,3,4,5,6,7,8}; std::string error;
        if(name=="disabled") { check(!RepeatCapture::enabled(),"disabled flag"); }
        else {
            check(RepeatCapture::enabled(),"enabled flag");
            if(name=="normal") {
                RepeatCapture c;
                c.record(q,3,"float_rows",98304,8192,98304,2,4,1,data);
                const int32_t ids[4]={-1,0,17,262143};
                c.record(q,7,"selected_ids",98304,8192,98328,1,4,2,ids);
                check(q.copies==2 && q.completions==2 && q.drains==0,"normal queue work");
            } else if(name=="bad-shape") {
                RepeatCapture c;
                check(try_record(c,q,data,0,4).find("invalid shape")!=std::string::npos,"zero shape accepted");
                check(try_record(c,q,data,1,33554433).find("invalid shape")!=std::string::npos,"oversize accepted");
                check(try_record(c,q,data,INT64_MAX,INT64_MAX).find("invalid shape")!=std::string::npos,"overflow accepted");
                check(q.copies==0 && q.initial_waits==0,"invalid shape queue work");
            } else if(name=="existing" || name=="symlink") {
                RepeatCapture c; error=try_record(c,q,data);
                check(error.find("fresh owned private regular file")!=std::string::npos,"unsafe file accepted");
                check(q.copies==0 && q.initial_waits==0,"unsafe file queue work");
            } else if(name=="initial-error") {
                q.initial_error=true; RepeatCapture c; error=try_record(c,q,data);
                check(error=="initial-error" && q.copies==0 && q.drains==0,"initial error contract");
            } else if(name=="event-drained" || name=="submit-drained" || name=="event-unknown" || name=="submit-unknown") {
                const bool submit=name.find("submit")==0, unknown=name.find("unknown")!=std::string::npos;
                q.event_error=!submit; q.submit_error=submit; q.drain_error=unknown;
                {
                    RepeatCapture c; error=try_record(c,q,data);
                    check(error==(submit?"submit-error":"event-error"),"primary error not preserved");
                    check(q.copies==1 && q.drains==1,"one copy and one fallback required");
                    if(unknown) {
                        const auto later=try_record(c,q,data);
                        check(later.find("prior readback completion unknown")!=std::string::npos,"poison missing");
                        check(q.copies==1 && q.initial_waits==1 && q.pending,"poison submitted extra copy");
                    } else check(!q.pending && q.completions==1,"fallback did not complete");
                }
                if(unknown) { q.complete(); check(q.completions==1 && !q.pending,"late copy failed"); }
            } else if(name=="partial-write") {
                struct rlimit before{}, bounded{}; check(getrlimit(RLIMIT_FSIZE,&before)==0,"getrlimit");
                bounded=before; bounded.rlim_cur=150; check(setrlimit(RLIMIT_FSIZE,&bounded)==0,"setrlimit");
                const auto prior=signal(SIGXFSZ,SIG_IGN);
                { RepeatCapture c; error=try_record(c,q,data); }
                check(setrlimit(RLIMIT_FSIZE,&before)==0,"restore limit"); signal(SIGXFSZ,prior);
                check(error.find("file write failed")!=std::string::npos,"partial write accepted");
                check(q.completions==1 && !q.pending && q.drains==0,"write failure before copy retirement");
            } else if(name=="two-full-budget") {
                std::vector<uint32_t> payload(32*10240,0x3f800000); RepeatCapture c;
                for(int full=0;full<2;++full) for(int layer=0;layer<=12;++layer) {
                    for(const char* phase:{"R_input","R_post_attention_gdn","R_post_moe"})
                        c.record(q,layer,phase,98304,8192,98304,32,10240,1,payload.data());
                    if(layer==3 || layer==7 || layer==11) {
                        const char* phases[]={"K_quant_input","V_quant_input","indexer_raw","query","q_indexer","attention_output"};
                        const int widths[]={512,512,128,6144,512,6144};
                        for(int i=0;i<6;++i) c.record(q,layer,phases[i],98304,8192,98304,32,widths[i],1,payload.data());
                        for(int row=0;row<32;++row) c.record(q,layer,"scores",98304,8192,98304+row,1,24577+row/4,1,payload.data());
                        for(int row=0;row<32;++row) {
                            c.record(q,layer,"steps",98304,8192,98304+row,1,4,2,payload.data());
                            c.record(q,layer,"selected_ids",98304,8192,98304+row,1,2051,2,payload.data());
                        }
                    }
                }
                check(q.copies==690 && q.completions==690,"two full frame count");
                error=try_record(c,q,payload.data(),32,10240);
                check(error.find("128 MiB budget exceeded")!=std::string::npos,"budget excess accepted");
                check(q.copies==690 && q.initial_waits==690,"excess submitted queue work");
            } else throw std::runtime_error("unknown test case");
        }
        std::cout << "{\"case\":\""<<name<<"\",\"copies\":"<<q.copies<<",\"initial_waits\":"<<q.initial_waits
                  <<",\"event_waits\":"<<q.event_waits<<",\"drains\":"<<q.drains<<",\"completions\":"<<q.completions
                  <<",\"pending\":"<<(q.pending?"true":"false")<<",\"primary_error\":\""<<error<<"\"}\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"test failure: "<<e.what()<<"\n";return 1; }
}
