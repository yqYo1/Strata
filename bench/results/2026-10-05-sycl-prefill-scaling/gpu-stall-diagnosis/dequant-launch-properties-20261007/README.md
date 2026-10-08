# Dequant launch-property preparation on 2026-10-07

These are CPU-only audit/build receipts on the existing B570 host with
oneAPI 2026.1.1. Neither new executable has run on a GPU. Production
sources, objects, archives and the executable remain unchanged.

The [2K profile](../profiling-20261007/README.md) recorded 100,060 flat/GU
expert dequant launches taking 4.628 seconds of device execution. Source
review finds `use_root_sync` on both wrappers and no root-group access in
this source file. The earlier short model's [kernel-specific API excerpts](dequant-no-root-control-api-snippets.json)
show both actual wrappers launching with `UR_KERNEL_LAUNCH_FLAG_COOPERATIVE`.
The [extension specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_root_group.asciidoc)
imposes a root-group-compatible launch and a work-group limit. The actual
device limit was not queried; this is not a demonstrated invalid range,
undefined behavior or a stall cause.

The [previous work-group probe](../../dequant-workgroup-probe/README.md)
already passed 432 full FP16 comparisons, but changed kernel names and
wrappers and skipped the original capability check. Its exploratory times
do not isolate this property.

The new private engine replaces only these two declarations with empty
property lists. Kernel names, formulas, ND-ranges, 32-item work-groups,
queue choice, capability checks, other properties and every other archive
member remain as before. The [source diff](dequant-no-root-build/source.diff)
and [build receipt](dequant-no-root-build/record.json) record the change;
offline compile/link passed in 28.277 seconds. This is not an engine runtime
measurement or an adopted optimization.

A [second build](dequant-no-root-probe-build/record.json) links one common
host fixture object separately against the original and changed kernel
objects. It passed offline compile/link in 9.134 seconds. The prepared
fixture will compare 144 full guarded FP16 outputs per executable across
nine types, two seeds, small/odd/640-row shapes, flat/GU layout and
zero/positive/negative scales. No comparisons have run yet.

The [deferred controller](sources-used/run_dequant_actual_wrapper_probe.py)
requires the separate full-context job's successful terminal receipt,
recorded absence of live jobs and disappearance of its pinned owners
before loading a GPU runtime. It captures API logs/environment through
owned GDB and bounds each binary to 180 seconds. Full-model head parity,
clean timings and the complete 256K gates remain separate requirements.
The [audit](dequant-no-root-audit.json) records provenance and limits;
the manifest hashes every archived file except itself. Executables,
objects, archives and unbounded logs are not included.
