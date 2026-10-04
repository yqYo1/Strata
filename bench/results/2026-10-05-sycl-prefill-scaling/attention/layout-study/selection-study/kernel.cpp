// Experimental smaller work-groups; original integer radix selection and tie order.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/qsa_select.hpp"
namespace strata::kernels {
namespace {
constexpr int R = 4;
__dpct_inline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = sycl::bit_cast<unsigned int>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

template <int TK_T>
__dpct_inline__ int block_excl_scan(int v, int *s_warp, int &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {

        const int y = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;

    item_ct1.barrier();
    if (warp == 0) {
        int w = lane < TK_T / 32 ? s_warp[lane] : 0;
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {

            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;               // exclusive per warp
        if (lane == 31) s_warp[32] = z;     // total
    }

    item_ct1.barrier();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];

    item_ct1.barrier();
    return r;
}

template <int TK_T, int PER, bool PARALLEL>

__dpct_inline__ void block_topk_reg_kernel(const float *__restrict__ scores,
                                           const int32_t *__restrict__ steps,
                                           int64_t max_blocks, int64_t cap,
                                           int32_t *__restrict__ ids) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &hist =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<int[TK_T / 32][256]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_warp =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[33]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_digit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_above = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t qi = item_ct1.get_group(2);
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = item_ct1.get_local_id(2), lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
#pragma unroll
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t)j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
#pragma unroll
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask))
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &hist[warp][(key[j] >> shift) & 255], w);
        }


        item_ct1.barrier();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int s = 0;
#pragma unroll
            for (int w2 = 0; w2 < TK_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
        }


        item_ct1.barrier();
        if constexpr (PARALLEL) {
            const int count=t<256 ? hist[0][255-t] : 0;
            const int before=sycl::exclusive_scan_over_group(item_ct1.get_group(),count,0,sycl::plus<int>());
            const int candidate=t<256 && above+before+count>=width ? t : 256;
            const int chosen=sycl::reduce_over_group(item_ct1.get_group(),candidate,sycl::minimum<int>());
            if(t==chosen) {s_digit=255-t;s_above=above+before;}
        } else {
        if (t == 0) {
            int cum = above, d = 255;
#pragma unroll
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }


        }
        item_ct1.barrier();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;


        item_ct1.barrier();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan<TK_T>(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan<TK_T>(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
#pragma unroll
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t)(b * R + c);
        } else if (key[j] == thr) {
#pragma unroll
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}



template<int TK_T, int PER, bool PARALLEL> class TopkTuned;
template<int TK_T, int PER, bool PARALLEL>
void tuned_launch(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
                  int64_t cap, int32_t* ids, void* stream) {
    strata::q_of(stream)->parallel_for<TopkTuned<TK_T,PER,PARALLEL>>(sycl::nd_range<3>(sycl::range<3>(1,1,nq*TK_T), sycl::range<3>(1,1,TK_T)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            block_topk_reg_kernel<TK_T,PER,PARALLEL>(scores,steps,max_blocks,cap,ids);
        });
    static bool reported=false;
    if(!reported) {
        reported=true;
        try {
            auto* queue=strata::q_of(stream);
            auto id=sycl::get_kernel_id<TopkTuned<TK_T,PER,PARALLEL>>();
            auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(queue->get_context(),{queue->get_device()},{id});
            auto kernel=bundle.get_kernel(id);
            auto device=queue->get_device();
            std::fprintf(stderr,"topk resources threads=%d per=%d parallel=%d declared_histogram=%zu private=%zu spill=%zu bytes\n",
                TK_T,PER,(int)PARALLEL,
                size_t(TK_T/32)*256*sizeof(int),
                kernel.template get_info<sycl::info::kernel_device_specific::private_mem_size>(device),
                kernel.template get_info<sycl::ext::intel::info::kernel_device_specific::spill_memory_size>(device));
        } catch(const sycl::exception& e) {
            std::fprintf(stderr,"topk resource query unavailable: %s\n",e.what());
        }
    }
}
}
bool qsa_block_topk_tuned(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
    int64_t cap, const QsaShapes& s, int32_t* ids, void* stream, int variant, int64_t active_blocks) {
    if (nq <= 0 || s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells,s)) return false;
    const int64_t reach=active_blocks>0 && active_blocks<=max_blocks ? active_blocks : max_blocks;
    const int threads=variant%10000;
    const bool parallel=variant>=10000;
    if (threads==256) {
        if(reach<=256*9) {if(parallel) tuned_launch<256,9,true>(scores,steps,nq,max_blocks,cap,ids,stream); else tuned_launch<256,9,false>(scores,steps,nq,max_blocks,cap,ids,stream);}
        else if(reach<=256*33) {if(parallel) tuned_launch<256,33,true>(scores,steps,nq,max_blocks,cap,ids,stream); else tuned_launch<256,33,false>(scores,steps,nq,max_blocks,cap,ids,stream);}
        else return false;
    } else if (threads==512) {
        if(reach<=512*5) {if(parallel) tuned_launch<512,5,true>(scores,steps,nq,max_blocks,cap,ids,stream); else tuned_launch<512,5,false>(scores,steps,nq,max_blocks,cap,ids,stream);}
        else if(reach<=512*17) {if(parallel) tuned_launch<512,17,true>(scores,steps,nq,max_blocks,cap,ids,stream); else tuned_launch<512,17,false>(scores,steps,nq,max_blocks,cap,ids,stream);}
        else return false;
    } else if (threads==1024) {
        if(reach<=1024*3) {if(parallel) tuned_launch<1024,3,true>(scores,steps,nq,max_blocks,cap,ids,stream); else tuned_launch<1024,3,false>(scores,steps,nq,max_blocks,cap,ids,stream);}
        else if(reach<=1024*9) {if(parallel) tuned_launch<1024,9,true>(scores,steps,nq,max_blocks,cap,ids,stream); else tuned_launch<1024,9,false>(scores,steps,nq,max_blocks,cap,ids,stream);}
        else return false;
    } else return false;
    return true;
}
}
