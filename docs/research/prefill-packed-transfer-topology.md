# Packed expert transfer topology: reusable source reference

Use this note when Main decides whether the layer-major packed expert path needs a source-preparation or upload-submission change. Source collection: `prefill-weight-transfer-topology-20261110/report.md`, report SHA256 `65b5f1348c36ec844fe888e0dbf6f6aefc1bdcf5fd8e8cf25fecc0bf6bb87c9d`; manifest SHA256 `4bf2c6446d26cd8c77426a3cc8e350bab0d09e9aa4a5bca631141369384758b4`; source commit recorded as `8b57207dc5ccfe1101156a294224d27e374ca64c`.

## Route and accounting

This applies to the opt-in packed expert bank on the layer-major long-prefill route with mode-10 MoE accumulator, BF16 residuals, chunk 8192, all residual rows resident on GPU, and the required single in-order queue (`copy == cs`). It does not describe the ordinary decode expert cache or the unchanged 512-expert fallback.

For layer `l`, let `B_l` be the actual packed gate/up/down blob bytes from the native expert layout, `E=512`, `S=ceil(max_l(B_l)/256)*256`, and `cache_bytes` be the packed-bank allocation. The owned source slab reserves `E*S` bytes but copies `E*B_l` payload bytes for that layer, with source offsets separated by `S`. Across the model, synchronous `copy_blob` preparation touches `E*sum_l(B_l)` payload bytes per request; stride padding is reserved space, not payload copied to the device.

The packed bank copies one `B_l` payload for each upload enqueue. Effective per-layer capacity is `C_l=min(512,floor(cache_bytes/B_l))`. Except when all 512 experts fit, the first `C_l-1` distinct experts become permanent residents and one slot is scratch. The upload count is the number of expert transitions in the routed job order that are not permanent hits: an expert in scratch can be uploaded again after another expert replaces it. Thus `H_l=upload_enqueues_l*B_l`; distinct routed expert count alone does not determine `H_l` unless jobs are grouped so each expert appears in one run. The full-cache fallback instead uploads all `512*B_l` bytes per layer.

After an expert change, the source calls two device dequantizers, writing 26,214,400 bytes for gate/up FP16 weights and 13,107,200 bytes for down FP16 weights at the reported `N=10240`, `n_ff=640`. Those device writes are separate from host-to-device expert payload and from the two route-map copies per MoE window.

An exact immutable resident host view can remove slab allocation and the `copy_blob` preparation pass. It does not by itself reduce selected expert payload bytes, device dequantization, or copy enqueue count. The checked provider/lifetime contract and actual effective-view marker must hold.

## Profile interpretation and next gate

The observed 32,769-input warmed window reports 46.505 GB M2D requested / 10.1518 s service, about 35.386 GB packed-bank source accounting, 0.9584 GB H2D, 4.5490 s outside-wait upload submission, a 45.5320 s client window, 40.6341 s GPU interval union, and 40.6447 s summed services. These values do not measure physical PCIe traffic or prove a saturated bus. The source-byte model cannot reconcile the M2D excess or identify the limiting resource; the near-equal GPU union and summed services only indicate little aggregate overlap in that sample.

Before another source-view A/B, crosswalk the exact source, route, configuration, and workload to the already indexed source-view comparison, whose prior speed gate passed 4.5512%. The index leaves the 1000 tok/s target and broader task/full-context/session/generation gates open. If source preparation is already removed and submission remains material, a batching study needs actual upload-transition counts and a bounded batch/ring lifetime design that retains each destination until its last reader; a second queue requires explicit copy, consumer, reuse, and host-source dependencies. No source audit establishes safety or speed.

## Provenance limit

All 12 manifest-listed source files matched their recorded hashes in the reviewed checkout. The report also relies on `src/prefill/moe_window.inc`, `src/kernels/cpu/native_expert.cpp`, and `include/strata/prefill/expert_upload_staging.hpp`, which are absent from that manifest. Treat their formulas and staging details as unpinned until a future source-bound task records immutable pins. This reference is for the named packed-bank transfer decision; review it at the source-view applicability crosswalk and retire it when that decision is recorded or superseded.
