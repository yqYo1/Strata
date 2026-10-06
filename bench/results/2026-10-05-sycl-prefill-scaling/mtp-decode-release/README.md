# MTP decode weights during layer-major prefill

This experiment is off by default. Its build succeeds; GPU execution and
model-output equality have not yet been verified. It does not establish a
speed improvement or a successful full-context MTP run.

On Arc B570 10 GiB, Ryzen 5 5600X and 128 GiB RAM, the frozen candidate's
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
and uploads the RAM images. Captured decode graphs retain their addresses.
The restoration callback also runs on cancellation and error returns.

Both arenas request 8 MiB physical segments. With this model's 675 MiB draft
expert payload and 106,299 subset rows of 2,100 bytes each, rounding gives
896 MiB of physical segments. These numbers are calculated from the payload
geometry, not measurements of a successful release. Automatic cache sizing
reserves the head padding, and the MTP VRAM counter includes physical padding.
Actual remaining capacity and driver support still need GPU checks.

`STRATA_PREFILL_DRAFT_VERIFY=1` compares every payload byte with its immutable
RAM image before unmapping and after restoration. This diagnostic performs
device-to-host copies and should be disabled for performance comparisons.
The engine logs total release time, queue wait, unmap and diagnostic compare
time, then restoration time split into mapping, RAM-to-VRAM copy and compare.
The ordinary restore path does not download weights at each prompt boundary.

`cache-lease-probe.cpp` links the real `ExpertCache` implementation. It checks
a partial segment and the two draft weight sizes across three releases each,
compares every restored word, and executes a previously captured kernel that
reads every payload word. It is compiled but has not run on the GPU. Even a
successful probe would only verify storage and captured graph replay.

The full-context controller accepts `--release-draft --verify-draft` with an
alternate frozen executable. Its reports use a distinct directory for these
flags, record runtime adapter overrides and the controller digest, and require
paired release/restore records for the first multi-chunk MTP request. At 256
and 262,144, the controller still requires the last allocated KV cell, clipped
speculative tails, exact token counts, finite heads/log probabilities, refused
overflow requests and a valid request afterward. Its CPU protocol stub passes
all eleven existing cases; the stub never runs model or GPU code and cannot
validate weight restoration.

The new xe dump from the failed full serve run is awaiting privileged
preservation outside Git. No further GPU test has been launched since the
request to preserve it. Next checks are the actual storage probe, normal-MTP
output equality with and without this option, cancellation/checkpoint reuse,
full 262,144 normal-MTP execution, and repeated warm PP/TG measurements.
The optimization goal remains PP 1,000 and TG 70 token/s.
