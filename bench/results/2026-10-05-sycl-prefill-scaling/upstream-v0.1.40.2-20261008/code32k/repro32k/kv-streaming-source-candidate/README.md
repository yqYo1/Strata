# Private KV-streaming source candidate: compiled, not GPU-tested

Existing upstream KV streaming (`--kv-resident 32768`) keeps authoritative
context K/V in host memory and a bounded subset in VRAM. It is an avenue to
investigate full262,144-cell capacity, not a new measured result. The existing
SYCL path is audited before activation; earlier accepted32K controls use
KV mode0 and do not validate this path.

The [source candidate](source/kv_stream.candidate.dp.cpp) changes only one
KV translation unit. A legal pair of sorted cell selections `[16,32]` and
`[4,16]` hits page4 from query0/lane0 and query1/lane1. The initial hit loop
contains no inter-query barrier and writes the same slot's stamp/reference
bits non-atomically. The candidate uses relaxed device-scope global atomic
stores for these shared hit writes. Its later work-group barrier and unique
victim writes remain unchanged. This is a source-level possible conflicting
access; it is not a claim this caused earlier non-streaming hangs.

Packed64-bit host counters are read from the int32_t copy through `memcpy`,
avoiding the previous type-punning/alignment assumptions. Four unused
root-sync properties are removed from reset/resolve/copy/ring launches.
Kernel arithmetic, ranges and required SG32/WG1024 resolve geometry remain
unchanged. This is not device capability or runtime correctness proof.

The first preparer expects three root properties, but its guard finds four
and stops before writing a candidate. Its CPU negative and immutable
controller are retained. The corrected preparer accounts for reset too and
produces the isolated source; no original source/binary changes.

The [private build](compile-link-receipt/record.json) passes compilation/linking with the
accepted e82 compiler flags, changing only the KV object in a copied kernel
archive and reusing the accepted no-root prefill archive. Every other archive
member/link input and both baseline/production binaries remain unchanged.
Compilation takes5.974 s and linking19.698 s on Ryzen5600X; these are build
durations, not inference measurements. The candidate executable hash is
`4af18d831559e68215090d55ddba878986a07af61d834485351af81d793d6e77`.

No GPU run, performance result or production adoption is claimed. Host-USM
ownership, failure cleanup, ring restoration, staging/clipping and snapshot
identity reconstruction remain source gates before a first bounded logged
normal-MTP32K request at max-context262144/kv-resident32768. Its state/head,
64IDs/logprobs and MTP counts must match the accepted control. Any clean
comparison uses at least32,768 tokens with matching settings and separates
initial load/capture from later full rereads. Full262,144-cell occupancy,
repeat, restore, clipped-tail, refusal and later-valid gates remain mandatory.
No reset, rebind, reboot, service, package, runtime or global change occurs.
