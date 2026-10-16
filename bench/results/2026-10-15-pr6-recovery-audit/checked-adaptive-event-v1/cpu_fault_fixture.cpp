// SOURCE ONLY: Main owns host compilation/execution; no runtime/GPU API used.
#include "strata/sycl/checked_adaptive_event.hpp"
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

static void mark(int fd,char value){ssize_t n;do{n=::write(fd,&value,1);}while(n<0 && errno==EINTR);if(n!=1)std::_Exit(90);}
struct Owner{int fd;~Owner(){mark(fd,'D');}};
static void scenario(int fd,int mode){
    Owner dma_source{fd},cache_slot{fd},pending_commit{fd};
    strata::sycl_backend::checked_adaptive_event_record(
        [&]{mark(fd,'R');if(mode==2)throw std::runtime_error("injected record exception");return mode==1?77:0;},
        [&](int code){assert(code==77);mark(fd,'E');return "injected original event detail";},
        "host fault fixture");
    mark(fd,'P'); // models subsequent pending->host_res publication
    mark(fd,'C'); // models the next consumer; neither can run on record failure
}
static std::string read_all(int fd){std::string text;char bytes[512];for(;;){ssize_t n=::read(fd,bytes,sizeof(bytes));if(n<0 && errno==EINTR)continue;assert(n>=0);if(!n)break;text.append(bytes,size_t(n));}return text;}
int main(){
    for(int mode:{0,1,2}){
        int steps[2],errors[2];assert(::pipe(steps)==0 && ::pipe(errors)==0);
        const pid_t child=::fork();assert(child>=0);
        if(child==0){::close(steps[0]);::close(errors[0]);if(::dup2(errors[1],STDERR_FILENO)<0)std::_Exit(91);::close(errors[1]);scenario(steps[1],mode);std::_Exit(0);}
        ::close(steps[1]);::close(errors[1]);
        int status=0;pid_t waited;do{waited=::waitpid(child,&status,0);}while(waited<0 && errno==EINTR);assert(waited==child && WIFEXITED(status));
        const auto sequence=read_all(steps[0]);const auto diagnostic=read_all(errors[0]);::close(steps[0]);::close(errors[0]);
        if(mode==0){assert(WEXITSTATUS(status)==0 && sequence=="RPCDDD" && diagnostic.empty());}
        else{
            assert(WEXITSTATUS(status)==EXIT_FAILURE);
            assert(sequence==(mode==1?"RE":"R")); // no publication, consumer or owner destructor
            assert(diagnostic.find("retaining owners until process exit")!=std::string::npos);
            if(mode==1)assert(diagnostic.find("status=77: injected original event detail")!=std::string::npos);
            else assert(diagnostic.find("injected record exception")!=std::string::npos);
        }
    }
}
