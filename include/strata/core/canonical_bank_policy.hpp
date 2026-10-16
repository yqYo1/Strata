#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>
namespace strata::core::detail {
inline uint64_t canonical_stride(uint64_t n) {
    if(n>UINT64_MAX-255) throw std::length_error("canonical alignment overflow");
    return (n+255)&~uint64_t(255);
}
// Original pack alignment is distinct from the 256-byte layer-bank stride.
inline bool canonical_capture_alignment(uint64_t n) noexcept { return n==64 || n==256; }
inline int canonical_layer(const std::string& name) {
    if(name.rfind("blk.",0)) return -1;
    size_t end=name.find('.',4);
    if(end==std::string::npos || end==4) throw std::invalid_argument("canonical layer name");
    int layer=0;for(size_t i=4;i<end;++i) {
        if(name[i]<'0' || name[i]>'9' || layer>47) throw std::invalid_argument("canonical layer index");
        layer=layer*10+name[i]-'0';
    }
    if(layer>=48) throw std::invalid_argument("canonical layer range");return layer;
}
inline bool canonical_persistent(const std::string& n) {
    return canonical_layer(n)<0 || n.ends_with(".hc_attn_norm.weight") ||
        n.ends_with(".hc_ffn_norm.weight") || n.rfind("blk.1.ple_",0)==0;
}
inline uint64_t canonical_fingerprint(const uint8_t* bytes,size_t n) {
    uint64_t h=14695981039346656037ULL;for(size_t i=0;i<n;++i){h^=bytes[i];h*=1099511628211ULL;}return h;
}
}
