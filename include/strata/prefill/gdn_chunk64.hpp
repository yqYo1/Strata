// Main-authored private Strata candidate; repository LICENSE applies.
#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/event.hpp>
namespace strata::gdn_chunk64 {
using BF = sycl::ext::oneapi::bfloat16;
// Real typed BF16 storage. Every pointer is a distinct registered Device span.
// Token padding is local to one <=8192-token prefill call, not context capacity.
struct Workspace {
  BF *q, *k, *kb, *vb, *inverse, *w, *u, *history, *delta, *raw;
  float *g, *lower;
};
struct Sizes { size_t qk, values, inverse, history, gates, lower; };
Sizes sizes(int tokens);
struct Arena {
  uint16_t *qkv=nullptr,*z=nullptr,*normalized=nullptr,*output=nullptr;
  float *ab=nullptr,*gate=nullptr,*beta=nullptr;
  Workspace workspace{};
};
// Shared by Prefill sizing, actual carving and the device fixture. QKV storage
// is reused by lower KKT only after conv/history completion on the same queue.
size_t arena_bytes(int tokens);
Arena bind_arena(void *base,size_t bytes,int tokens);
// Immutable opt-in; the normal/short-input path remains the existing route.
bool requested();
// Admission only: reject an unsupported queue/device before model mutation.
void validate_device(void *stream);
// Span/domain/capability validation only; call before changing conv history.
void validate_run(float *state,const uint16_t *h,const float *gate,const float *beta,
                  const uint16_t *z,const float *gamma,float eps,uint16_t *output,
                  int tokens,Workspace workspace,void *stream);
struct Events {sycl::event gates,pack,kkt,inverse,wu,carry,readout,norm;};
struct ConvEvents {sycl::event convolution,history;};
// BF16 projection -> FP32 conv/SiLU -> BF16 -> FP32 Q/K L2 -> BF16.
// The existing FP32 history is consumed unchanged. Final history keeps
// decoded projection values in FP32, in the existing [10240,3] decode layout.
// All old history reads complete before a separate ordered update kernel.
void conv_bf16(float *history,const uint16_t *qkv,const float *weights,
               uint16_t *normalized,int tokens,float eps,void *stream,
               ConvEvents *events=nullptr);
// h is existing tiled [T,10240] Q16/K16/V48 order, z [T,48,128].
// BF16 activation producer contract; FP32 gate/beta/gamma and persistent carry.
// State remains [128,48,128]; output is genuine FP16 for the existing consumer.
// No allocations, environment mutation, host tensor math, or implicit waits.
void run(float *state, const uint16_t *h, const float *gate, const float *beta,
         const uint16_t *z, const float *gamma, float eps, uint16_t *output,
         int tokens, Workspace workspace, void *stream, Events *events=nullptr,
         bool packed_inverse=true, bool narrow_carry=true);
}
