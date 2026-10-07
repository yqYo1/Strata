# B70 offload reference inspected on 2026-10-07

User-supplied [qwen38-flash-next-b70-offload](https://github.com/0xSero/qwen38-flash-next-b70-offload/)
was read at commit 557f80fe0f4235a76d4cf835f89e108d599b5e94. [File hashes](reference.json)
pin the source examined; none of its code or scripts was executed.

Its platform is B70 32 GB, PCIe Gen4 x16, 32 GB host RAM and an NVMe RAID.
It uses EXL3 weights and SGLang, whereas the current B570 run uses IQ3_S,
10 GiB VRAM and measured explicit H2D about 6.16 GB/s. Its speeds and
kernel choices cannot serve as measurements for this host.

The most relevant implementation is plugin/exl3xpu/nvtier.py:529-623.
It allocates two full-layer VRAM buffers and a configurable ring of RAM
buffers, reads experts ahead with a worker pool, coalesces consecutive
missing expert IDs into one DMA call, and orders reuse with H2D and compute
events. A RAM buffer waits for its previous copy before being rewritten;
a VRAM buffer waits for its previous compute before receiving new data;
compute waits for the new copy. This event ordering is useful for a bounded
pipeline on B570. Our current layer-major loop waits for compute, loads all
512 compressed experts and waits for copy before running the layer.

Full-layer double buffering requires another approximately 1.36 GB on our
native pack, which the observed full-256K free budget cannot provide.
Keep the single-buffer fallback, or examine RAM-only read-ahead and partial
GPU staging. Record peak VRAM and actual transfer/compute overlap rather
than inferring overlap from the presence of two queues. The reference's
research/18-b70-ingest-ceiling.md reports serialization within one process
and overlap across processes; that report is a hypothesis to test on this
runtime, not proof of a driver fault on this host.

Two other useful directions are post-GEMM hyper-connection fusion, with
explicit rounding boundaries, and event-ordered decode bookkeeping instead
of a device-wide wait per step. The reference already adapts Strata attention
and GDN code, so compare the actual source and existing local profiles to
avoid porting an equivalent implementation back as an assumed improvement.

Do not treat its no-victim-write-back figures as exact inference: README
documents masked expert picks. The experimental victim ring has standalone
checks but no server matrix; when full it can drop victims and increase
masking. The reported direct-write-back race (a pending host page punch
versus GPU residency publication) reinforces the invariant that residency
must be published only after data lands and storage is no longer being
reclaimed. Our optimization must retain every selected expert and have a
lossless fallback under pressure.

Next measurements: unchanged-result dequant reuse, bounded RAM read-ahead/
coalesced copies, and an isolated logged copy-compute overlap probe. Keep
original phase waits until the timing-dependent parity issue is resolved;
follow short full-head comparisons with repeated 262144-cell capacity gates.
