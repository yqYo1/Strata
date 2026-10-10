// Offline row oracle: link the frozen original ngram.cpp, never a copied hash.
// CPU only. Creates a NEW raw uint32 little-endian row file; refuses overwrite.
#include "strata/kernels/ngram.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>
namespace {
constexpr size_t limit=262144, chunk=8192, input_bytes=8*1024*1024;
int64_t number(std::string_view s, int64_t lo, int64_t hi) {
    const size_t begin=(!s.empty() && s.front()=='-') ? 1 : 0;
    if(s.empty() || s.size()>11 || begin==s.size() ||
       (s[begin]=='0' && s.size()-begin>1)) throw std::runtime_error("invalid integer");
    int64_t v=0; auto r=std::from_chars(s.data(),s.data()+s.size(),v);
    if(s.empty() || r.ec!=std::errc{} || r.ptr!=s.data()+s.size() || v<lo || v>hi)
        throw std::runtime_error("invalid integer");
    return v;
}
struct Fd { int fd=-1; ~Fd(){if(fd>=0) ::close(fd);} };
std::string json_string(const char* s) {
    std::string out="\"";
    const char* hex="0123456789abcdef";
    for(const unsigned char* p=reinterpret_cast<const unsigned char*>(s);*p;++p) {
        if(*p=='\"' || *p=='\\') { out+='\\'; out+=static_cast<char>(*p); }
        else if(*p<32) { out+="\\u00"; out+=hex[*p>>4]; out+=hex[*p&15]; }
        else out+=static_cast<char>(*p);
    }
    return out+'\"';
}
void write_all(int fd,const unsigned char* p,size_t n) {
    while(n) { ssize_t k=::write(fd,p,n); if(k<0 && errno==EINTR) continue;
        if(k<=0) throw std::runtime_error("row output write failed");
        p+=k; n-=static_cast<size_t>(k); }
}
std::vector<int32_t> load(const char* path) {
    if(std::strlen(path)>4096) throw std::runtime_error("input path too long");
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("input open failed");
    std::vector<int32_t> t; t.reserve(limit);
    std::string word; size_t bytes=0; char ch;
    auto finish=[&] { if(!word.empty()) {
        if(t.size()==limit) throw std::runtime_error("too many input tokens");
        t.push_back(static_cast<int32_t>(number(word,INT32_MIN,INT32_MAX))); word.clear(); }};
    while(f.get(ch)) {
        if(++bytes>input_bytes) throw std::runtime_error("input byte bound exceeded");
        if(std::isspace(static_cast<unsigned char>(ch))) finish();
        else { if(word.size()==11) throw std::runtime_error("token text too long"); word+=ch; }
    }
    if(!f.eof()) throw std::runtime_error("input read failed");
    finish(); if(t.empty()) throw std::runtime_error("empty input"); return t;
}
}
int main(int argc,char** argv) {
    try {
        if(argc<4) throw std::runtime_error("usage: oracle INPUT PP_COUNT NEW_ROWS [--insert198] [--prev OLDEST NEWEST]");
        const size_t count=static_cast<size_t>(number(argv[2],1,limit));
        if(std::strlen(argv[3])>4096) throw std::runtime_error("output path too long");
        bool insert=false; std::array<int32_t,2> initial{-1,-1}; bool prev_seen=false;
        for(int i=4;i<argc;++i) {
            const std::string_view opt=argv[i];
            if(opt=="--insert198" && !insert) insert=true;
            else if(opt=="--prev" && !prev_seen && i+2<argc) {
                prev_seen=true;
                initial[0]=static_cast<int32_t>(number(argv[++i],INT32_MIN,INT32_MAX));
                initial[1]=static_cast<int32_t>(number(argv[++i],INT32_MIN,INT32_MAX));
            } else throw std::runtime_error("unknown or duplicate option");
        }
        auto tokens=load(argv[1]); const size_t original_count=tokens.size();
        if(insert) {
            if(tokens.size()!=32768) throw std::runtime_error("insert198 requires exactly 32768 input tokens");
            tokens.insert(tokens.begin()+32759,198);
        }
        if(count>tokens.size()) throw std::runtime_error("prefix exceeds input");
        using namespace strata::kernels;
        static_assert(NGRAM_SIZE==3 && PLE_N_HEADS==16 && PLE_TABLE_ROWS==320001536ull && PLE_ROW_BYTES==90);
        const auto constants=ple_artifact_consts();
        for(int h=0;h<PLE_N_HEADS;++h)
            if(constants.vocab[h]==0 || constants.offset[h]>PLE_TABLE_ROWS ||
               constants.vocab[h]>PLE_TABLE_ROWS-constants.offset[h])
                throw std::runtime_error("artifact constants outside table");
        std::vector<int32_t> prev(chunk*2);
        std::vector<uint32_t> rows(chunk*PLE_N_HEADS);
        std::vector<unsigned char> encoded(chunk*PLE_N_HEADS*4);
        Fd output; output.fd=::open(argv[3],O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if(output.fd<0) throw std::runtime_error("new output open failed (existing files refused)");
        uint64_t fnv=14695981039346656037ull, written=0;
        for(size_t c0=0;c0<count;c0+=chunk) {
            const size_t n=std::min(chunk,count-c0);
            // Same at(c0) history contract as Prefill::ple_gather; oldest first.
            for(size_t i=0;i<n;++i) for(size_t j=0;j<2;++j) {
                const size_t at=c0+i+j;
                prev[2*i+j]=at<2 ? initial[at] : tokens[at-2];
            }
            ngram_rows(tokens.data()+c0,prev.data(),static_cast<int>(n),constants,rows.data());
            for(size_t i=0;i<n*PLE_N_HEADS;++i) {
                if(rows[i]>=PLE_TABLE_ROWS) throw std::runtime_error("row outside table");
                for(int b=0;b<4;++b) {
                    const auto v=static_cast<unsigned char>(rows[i]>>(8*b));
                    encoded[4*i+b]=v; fnv=(fnv^v)*1099511628211ull;
                }
            }
            write_all(output.fd,encoded.data(),n*PLE_N_HEADS*4);
            written+=n*PLE_N_HEADS*4;
        }
        const int fd=output.fd; output.fd=-1;
        if(::close(fd)!=0) throw std::runtime_error("row output close failed");
        std::cout<<"{\"success\":true,\"scope\":\"CPU offline original ngram_rows only\","
                 <<"\"input_tokens\":"<<original_count<<",\"transformed_tokens\":"<<tokens.size()
                 <<",\"pp_count\":"<<count<<",\"chunk_tokens\":8192,\"chunks\":"<<(count+chunk-1)/chunk
                 <<",\"heads\":16,\"initial_prev\":["<<initial[0]<<","<<initial[1]<<"]"
                 <<",\"insert198_before_index32759\":"<<(insert?"true":"false")
                 <<",\"row_count\":"<<count*16<<",\"row_bytes\":"<<written
                 <<",\"rows_le_fnv1a64\":"<<fnv<<"}\n";
        if(!std::cout) throw std::runtime_error("receipt write failed");
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"{\"success\":false,\"error\":"<<json_string(e.what())
                 <<",\"partial_output_must_not_be_used\":true}\n"; return 1;
    } catch(...) { std::cerr<<"{\"success\":false,\"error\":\"unknown error\"}\n"; return 1; }
}
