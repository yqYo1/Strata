# Completion stress and cooperative audit on 2026-10-07

Four owned diagnostic processes exit normally on Arc B570 10 GiB / Ryzen
5600X / 128 GiB RAM, kernel 7.0.0-38-generic, NEO 26.31.39395.14 and
oneAPI 2026.1.1. Each performs 20,000 GU/down pairs through the actual updated
IQ wrappers with regular launches, full Level Zero/UR logs and validation.
Each checks all guarded FP16 output bytes 40 times. The same real
layer17/expert0 IQ3_S/IQ4_NL weights are replicated across 512 aligned slots
in a 1,363,148,800-byte device arena. This is a real-weight component stress
test, not all experts or a complete model request.

The two extended cases also use unchanged production Gemm::f16 and
swiglu_interleaved wrappers, eight synthetic FP16 activation rows, and
the original producer/consumer wait boundaries. All GU FP32, SwiGLU FP16
and down FP32 outputs are finite, guarded and byte-identical to each
process's initial reference. Their reference outputs also match each other
between modes. The actual SwiGLU cooperative limit is 144 groups; this
fixture uses 20. No arithmetic or production launch geometry was rewritten.

One mode per fixture consumes a 64 MiB virtual-memory mapping, waits,
unmaps the full range, releases physical memory and frees the virtual range
before the stress. The extended fixture also performs real GEMM consumers
before retiring the mapping. This follows the ownership/order requirements
of the [SYCL virtual-memory extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_virtual_mem.asciidoc).
It excludes recorded decode graphs, other model state, routing and staging
threads. A single mapping does not cover the model's whole restoration path.

Diagnostic process durations are 9.100/9.269 seconds for the basic cases and
29.352/29.199 seconds for the extended cases. These include logs, validation,
GDB and initialization; none is accepted as clean throughput. All owners
are gone and no new xe fault is recorded. No reset, rebind, reboot, service
or package change was used. These results fail to reproduce the model stall;
they do not establish general hang prevention or model output parity.

The separate resource query submits no kernels. Native handles and complete
group/local shapes are recovered from the failed unchanged-control trace,
then 25 original prefill/IQ kernel/local-size combinations are queried using
unchanged production objects. Twenty-two observed cooperative launches
exceed the measured maximum. Examples are GDN output normalization
49,152 groups versus 288, broadcast 40,960 versus 144, RMS rows 24,576 versus 144,
and the original large SwiGLU 1,988 versus 144. Removing only the two IQ
dequant flags leaves these invalid settings. See the complete
[query receipt](prefill-cooperative-query/query-record.json) and
[observed launch metadata](observed-cooperative-launches.json).

The query uses zero additional dynamic local bytes, so its limit is an
optimistic bound for any launch with extra dynamic local storage. Counts
above it violate the [Level Zero cooperative launch contract](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#cooperative-kernels);
counts below it do not validate every other launch requirement. Resource
limits alone do not identify which operation caused the completion stall.
All 45 observed kernel/local-size combinations map uniquely to eight source
files; [source hashes and class names](observed-cooperative-source-targets.json)
record that mapping. This is not a complete call-graph or synchronization
proof. The observed gather_rows kernel in verify_kernels.dp.cpp is a row
copier, but other kernels in that file include persistent/global coordination.
Review properties per function; do not remove them from the entire file.
Production source and binary remain unchanged.
Full-model and all 262144-cell gates and PP1000/TG70 remain incomplete.

The first extended CPU link failed because the original archive preceded
the explicit regular IQ object and introduced duplicate definitions. Its
failed receipt is retained. The corrected v2 link places the explicit object
before archives and is the only extended fixture used on GPU. Controllers,
sources, input-region hashes, bounded tails and full private-log hashes are
included. Executables, whole output arrays and large logs stay private.
The manifest covers every archived file except itself.
