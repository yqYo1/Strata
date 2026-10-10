# Host packed-copy operation capacity

Ryzen5 5600X, physical CPU mask0–5, unchanged powersave/EPPpower, glibc2.39, GCC13.3 -O3 -march=znver3. CPU-only synthetic ordinary aligned RAM,1971200B/blob,192sourceblobs,16ring slots. The fixture represents one selected packed format size; it is not production topology or a universal native type. Every copy's entire payload is compared and hashed before its destination is overwritten. Source/destination canaries, exact job/generation identity and final immutable source checks pass.15parser rejection checks,4fresh qualifiers and12fresh timingprocesses close normally with ownership/survivor checks. All19,968copied payloads are verified, including warmups/qualifiers.84timedsample rows plus12warmup rows, perbatch/job timing, worker distribution, source hashes and correctness are retained in the exact owned receipt.

| Workers | Median payload GB/s | Range of process medians | Logical read+write GB/s |
| ---: | ---: | --- | ---: |
|1|11.637|11.395–11.980|23.275|
|3|13.743|12.943–13.946|27.487|
|4|13.667|13.345–14.080|27.334|
|6|13.711|13.476–14.281|27.422|

These are medians of three process medians per arm, each process with one warmup and seven samples. Predeclared order is1/3/4/6, reverse, then3/1/6/4; this is not an exactly balanced Latin square. Three workers are the source RAM-default expression ifhardware_concurrency12; explicit four/sixworker arms show no clear added capacity here. Actual runtime environment and source-transient32thread arm still need exact attestation.

Each sample sums12 independently timed16-slot copy bursts. It includes issue/mutex/wakeup/claims/memcpy/worker timestamps/completion barrier/mainwait. Full memcmp/hash/canary/acknowledgment, initialization/source first-touch, thread startup/join, final source hashing and output are outside those windows. Verification perturbs caches between bursts. The fixture is a host acknowledgment surrogate, not live Stager's DMA-dependent scheduling or uninterrupted streaming. Ordinary aligned RAM is not pinned SYCLhostUSM. No GGUF/source-I/O/device dequant/GPU work or wholemodel speed is measured.

The3worker payload capacity13.74GB/s exceeds the separate host-USM H2Dcontrol6.447GB/s by about2.13times. This shows operation-level headroom for plain-host copying in this control; it does not prove production's stager is never critical, or that pinned allocation/source assembly/cache/DMA handshakes attain the same rate. Do not add summed worker spans to whole wall or substitute this rate for actual PCIe service. Avoid raising thread count just from an assumed copy bottleneck; next correlate real stager readiness/H2D/dequant service and actual expert row histogram. Nooptimization adopted or full262144-position inference claim.
