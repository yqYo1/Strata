# Native expert role reader: host fixture

This standalone project compiles the shared GGUF reader and `NativeRolePlan`
without the SYCL engine or ggml CPU kernels. The plan owns the mappings it
validates and assembles gate, up and down with independent expert strides.
`Extent.offset` is an absolute file byte offset; `tensor_bytes` covers the
whole role tensor, and `bytes_per_expert` is that role's expert stride.
Files must stay unchanged while mapped and the destination must be a separate
caller-owned buffer. Paths retain the production reader's lexical resolution.

Root built with `/usr/bin/g++`, C++20, ASan and UBSan on 2026-10-10. The 22
independent synthetic fixture groups passed in a 5.057-second configure/build/
execution sequence, with normal exits and no forced cleanup or survivor.
They cover primary, normalized per-layer and per-role shard names, distinct
physical role ordering/strides, mapped-file rename, exact guarded output,
closed/invalid coordinates and capacity, malformed layout geometry/overflow,
missing shards, wrong tensor type/name/shape and truncated payload.
This does not test the native layout parser or model weights.

The exact owner receipt is
`native-role-plan-host-cpu-validation-v1/record.json`, SHA-256
`aa75a19110984c01b77fdaa19768949b5a05bedf425f812ec0268e0f78a3a201`,
under the shared `post-reboot-tuning-20261007` state directory. Its controller
records sources, commands, flags, limits and logs. Direct dynamic imports have
no SYCL/UR/Level Zero/oneMKL library; this is not a runtime device-API audit.

The production `check_experts_gguf` and `FileExpertSource` are unchanged.
No actual-weight service measurement, model correctness, full-context result,
performance result or adoption follows from this fixture. The next consumer
is the standalone actual-weight CPU service harness and subsequent shared
production validator integration, each with root-owned qualification.
