# MTP decode weights during layer-major prefill

This experiment is off by default. Its first real short request succeeds,
but the repeat crashes inside the CPU Level Zero runtime before a BCS fault.
The [failure](../gpu-stall-diagnosis/host-boundaries/draft-lease-runtime-crash/README.md)
is recorded with exact matching debug symbols. After an external reboot,
the updated graph-retirement candidate completes four processes and 16 requests
with ordinary exits and no new xe faults, with direct submission disabled.
Both retained-weight controls pass full-head equality, while releasing weights
changes repeat results even with prompt/conversation reuse disabled. See the
[post-reboot comparison](../gpu-stall-diagnosis/host-boundaries/post-reboot-retirement/README.md).
The experiment
does not establish stable model output, a speed improvement or successful
full-context MTP operation.

On Arc B570 10 GiB, Ryzen 5 5600X and 128 GiB RAM, an earlier frozen candidate's
normal-MTP serve run at exactly 262,144 context failed before prefill. Its
1.27 GiB layer expert cache did not fit the 0.43 GiB free after releasing the
target's decode expert cache. The capacity is retained. See the
[failed real run](../residual-inplace/post-reboot-proof/serve-262144-capacity-failure.json).
The separate [full CLI run](../residual-inplace/post-reboot-proof/cli-262144.json)
completed at that capacity; it does not validate normal-MTP serve.

`STRATA_PREFILL_RELEASE_DRAFT=1` retains the immutable draft expert file bytes
in RAM and uses virtual memory for the draft experts and subset output head.
The subset head is gathered normally at startup and copied once into RAM.
Layer-major prefill waits for outstanding device work, unmaps those decode
weights, and keeps the dense projections and draft K/V used during prefill.
After temporary prefill VRAM is freed, it maps the original virtual addresses
and uploads the RAM images. The candidate discards existing decode graphs
before physical release and captures fresh ones after restoration. Unchanged
virtual addresses alone do not validate cached backend allocation metadata.
The restoration callback also runs on cancellation and error returns.

Both arenas request 8 MiB physical segments. With this model's 675 MiB draft
expert payload and 106,299 subset rows of 2,100 bytes each, rounding gives
896 MiB of physical segments. The failed repeat test measures three paired
releases/restores at this size and verifies all restored payload bytes. The
931,016,700-byte RAM uploads take 175.987-176.243 ms, excluding mapping and the
optional verification. Automatic cache sizing reserves the head padding, and
the MTP VRAM counter includes physical padding. These transfer observations
do not turn the failed model run into a passing lease test.

`STRATA_PREFILL_DRAFT_VERIFY=1` compares every payload byte with its immutable
RAM image before unmapping and after restoration. This diagnostic performs
device-to-host copies and should be disabled for performance comparisons.
The engine logs total release time, queue wait, unmap and diagnostic compare
time, then restoration time split into mapping, RAM-to-VRAM copy and compare.
The ordinary restore path does not download weights at each prompt boundary.

`cache-lease-probe.cpp` links the real `ExpertCache` implementation. It checks
a partial segment and the two draft weight sizes across three releases each,
compares every restored word, and executes a previously captured kernel that
reads every payload word. Its first actual GPU attempt fails with
`DEVICE_LOST` before completing any case. Its cleanup wait throws inside a
`unique_ptr` destructor, so the abort stack does not identify the initial
failing probe operation. This is not proof of unsupported 8 MiB segments.
A later current-library run passes all nine storage/replay rounds with no new
xe fault: [receipt](../gpu-stall-diagnosis/host-boundaries/cli-and-storage/record.json).
The subsequently failing real model demonstrates that this storage control
does not validate model-level cached graphs or repeated request state.

The full-context controller accepts `--release-draft --verify-draft` with an
alternate frozen executable. Its reports use a distinct directory for these
flags, record runtime adapter overrides and the controller digest, and require
paired release/restore records for the first multi-chunk MTP request. At 256
and 262,144, the controller still requires the last allocated KV cell, clipped
speculative tails, exact token counts, finite heads/log probabilities, refused
overflow requests and a valid request afterward. Its CPU protocol stub passes
all eleven existing cases; the stub never runs model or GPU code and cannot
validate weight restoration.

The installed xe module was checked to retain the first dump without
overwriting it during later hangs. After that check, the storage probe and
an independent existing llama-bench control both fail with `DEVICE_LOST`;
llama-bench fails while clearing its initial KV buffer, before inference.
Both processes have exited. The retained dump is subsequently deleted by
the kernel at 14:48 without a saved copy. See
[the investigation](../gpu-stall-diagnosis/README.md) for bounded stacks,
kernel records and the limits of these observations.
Next checks are identifying the invalid residency object, validating graph
retirement, normal-MTP output equality, cancellation/checkpoint reuse,
full 262,144 normal-MTP execution, and repeated warm PP/TG measurements.
The optimization goal remains PP 1,000 and TG 70 token/s.
