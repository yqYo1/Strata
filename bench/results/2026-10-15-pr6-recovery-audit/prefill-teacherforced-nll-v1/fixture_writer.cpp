// SOURCE ONLY. Run in a private directory owned by this fixture; no GPU.
#include "strata/program/prefill_nll.hpp"
#include <cassert>
#include <fstream>
#include <iterator>
#include <cstring>
using strata::program::prefill_nll::Writer;
namespace fs=std::filesystem;
int main(int argc,char** argv) {
    assert(argc==2); const fs::path dir=argv[1];
    assert(fs::create_directory(dir)); // fixture never cleans/overwrites existing output
    int serial=0; auto path=[&] { return (dir/std::to_string(serial++)).string(); };
    std::string e;
    { Writer w; assert(!w.begin(path(),{0},4,1,e)); }
    { Writer w; assert(!w.begin(path(),{0,-1},4,1,e)); }
    { Writer w; assert(!w.begin(path(),{0,4},4,1,e)); }
    { Writer w; assert(!w.begin(path(),{0,INT64_MAX},4,1,e)); }
    { Writer w; assert(!w.begin(path(),{0,1},INT64_MAX,1,e)); }
    { Writer w; std::vector<int64_t> ids(262145,0); assert(!w.begin(path(),ids,4,1,e)); }
    { Writer w; assert(!w.begin((dir/"missing"/"file").string(),{0,1},4,1,e)); }
    { Writer w; const auto p=path(); assert(w.begin(p,{0,1,2,3},4,1,e));
      assert(w.chunk(0,2,e)); w.host()[0]=0; w.host()[1]=1000;
      assert(w.append(0,2,e)); assert(w.chunk(2,1,e)); w.host()[2]=1.5;
      assert(w.append(2,1,e)); assert(w.finish(e));
      assert(fs::file_size(p)==16+16+24); assert(fs::file_size(p+".meta")<4096);
      std::ifstream f(p,std::ios::binary); std::vector<char> b((std::istreambuf_iterator<char>(f)),{});
      int32_t h[4]; std::memcpy(h,b.data(),16); assert(h[0]==0x314c4c4e && h[1]==4 && h[2]==4 && h[3]==3);
      for (int i=0;i<3;++i) { int32_t gold; float nll; std::memcpy(&gold,b.data()+32+8*i,4);
        std::memcpy(&nll,b.data()+36+8*i,4); assert(gold==i+1); assert(nll==float(i==0?0:i==1?1000:1.5)); }
      Writer same; assert(!same.begin(p,{0,1},4,1,e)); }
    // Missing, overlapping, repeated, reordered and impossible callback coverage.
    for (int mode=0;mode<6;++mode) { Writer w; auto p=path(); assert(w.begin(p,{0,1,2},4,1,e));
      if(mode==0) assert(!w.chunk(1,1,e));
      if(mode==1) assert(!w.chunk(0,0,e));
      if(mode==2) assert(!w.chunk(0,3,e));
      if(mode==3) { assert(w.chunk(0,1,e)); assert(!w.chunk(1,1,e)); }
      if(mode==4) { assert(w.chunk(0,1,e)); w.host()[0]=1; assert(w.append(0,1,e)); assert(!w.chunk(0,1,e)); }
      if(mode==5) { assert(w.chunk(0,2,e)); assert(!w.append(1,2,e)); }
      assert(!w.finish(e)); assert(!fs::exists(p+".meta")); }
    for(float x:{NAN,INFINITY,-INFINITY,-1.f}) { Writer w; auto p=path();
      assert(w.begin(p,{0,1},4,1,e)); assert(w.chunk(0,1,e)); w.host()[0]=x;
      assert(!w.append(0,1,e)); assert(!w.finish(e)); assert(!fs::exists(p+".meta")); }
    // Publication race: no replacing a competing final metadata file.
    { Writer w; auto p=path(); assert(w.begin(p,{0,1},4,1,e)); assert(w.chunk(0,1,e));
      w.host()[0]=1; assert(w.append(0,1,e)); { std::ofstream f(p+".meta"); f<<"competitor"; }
      assert(!w.finish(e)); std::ifstream f(p+".meta"); std::string s; f>>s; assert(s=="competitor"); }
    // Full 262143 prediction budget boundary, not a model/physical-context test.
    { Writer w; auto p=path(); std::vector<int64_t> ids(262144,1); ids[0]=0;
      assert(w.begin(p,ids,4,1,e)); assert(w.chunk(0,262143,e));
      for(int i=0;i<262143;++i) w.host()[i]=1;
      assert(w.append(0,262143,e)); assert(w.finish(e));
      assert(fs::file_size(p)+fs::file_size(p+".meta") < (4u<<20)); }
}
