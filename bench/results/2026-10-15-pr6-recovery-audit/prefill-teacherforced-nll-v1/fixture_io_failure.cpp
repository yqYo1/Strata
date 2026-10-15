// SOURCE ONLY. GNU/LLVM ELF linker fixture:
// -Wl,--wrap=fwrite,--wrap=fflush,--wrap=fclose,--wrap=fprintf
// No GPU. Private NEW output directory required. Asserts must remain enabled.
#include "strata/program/prefill_nll.hpp"
#include <cassert>
#include <cstdarg>
static int write_at=0,flush_at=0,close_at=0,print_at=0;
static bool trip(int& n) { return n>0 && --n==0; }
extern "C" {
size_t __real_fwrite(const void*,size_t,size_t,FILE*);
int __real_fflush(FILE*);
int __real_fclose(FILE*);
size_t __wrap_fwrite(const void* p,size_t s,size_t n,FILE* f) {
  if(trip(write_at)) return 0; return __real_fwrite(p,s,n,f);
}
int __wrap_fflush(FILE* f) { if(trip(flush_at)) return EOF; return __real_fflush(f); }
int __wrap_fclose(FILE* f) {
  const bool fail=trip(close_at); const int r=__real_fclose(f); return fail?EOF:r;
}
int __wrap_fprintf(FILE* f,const char* fmt,...) {
  if(trip(print_at)) return -1;
  va_list ap; va_start(ap,fmt); const int r=std::vfprintf(f,fmt,ap); va_end(ap); return r;
}
}
int main(int argc,char** argv) {
  assert(argc==2); const std::filesystem::path dir=argv[1]; assert(std::filesystem::create_directory(dir));
  using strata::program::prefill_nll::Writer;
  for(int mode=0;mode<9;++mode) {
    const auto p=(dir/std::to_string(mode)).string(); std::string e;
    { Writer w;
      if(mode==0) { write_at=1; assert(!w.begin(p,{0,1},4,1,e)); }
      else if(mode==1) { print_at=1; assert(!w.begin(p,{0,1},4,1,e)); }
      else {
        assert(w.begin(p,{0,1},4,1,e)); assert(w.chunk(0,1,e)); w.host()[0]=1;
        if(mode==2) { write_at=1; assert(!w.append(0,1,e)); }
        else {
          assert(w.append(0,1,e));
          if(mode==3) write_at=1; // committed row count
          if(mode==4) flush_at=1; // raw flush
          if(mode==5) close_at=1; // raw close
          if(mode==6) print_at=1; // completion footer
          if(mode==7) flush_at=2; // metadata flush
          if(mode==8) close_at=2; // metadata close
          assert(!w.finish(e));
        }
      }
      write_at=flush_at=close_at=print_at=0;
    }
    assert(!std::filesystem::exists(p+".meta"));
  }
}
