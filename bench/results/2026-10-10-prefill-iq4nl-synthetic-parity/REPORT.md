# Private IQ4_NL synthetic qualification

On Arc B57010GiB at PCI0000:05:00.0, source2ab09413117044d7499ac6acc6c12db404024f9d builds both prefill_iq4nl_parity and the production strata_prefill archive. Release precise-math SPIR64 compilation used oneAPI2026.1, subgroup32, per-kernel device split, native experts and the private option enabled. The full119-step CPU build completed in321.627seconds.

The root-owned diagnostic run passed84 paired cases, with45,896,704 active half words checked per arm, full unused output/guard comparisons and immutable input. Both paths matched independently computed RNE reference bits exactly. The unchanged device half constructor also matched20 explicit boundary inputs on this target. Advertised fp16 flags are2,4,8,16,32,64; this is target observation, not a portable rounding/denormal contract.

The host-only admission gate passed before queue construction. Actual Level Zero logs show one constructor kernel,84 generic launches and84 private launches, with28 launches per arm at1,2 and6400groups of32. All observed UR and native result receipts were successful. All341 progress stages and the terminal record after USM/queue teardown were present; normal exit0, fully observed/reaped empty sessions, no cleanup or survivors. Peak sampled GPU-process RSS was149,897,216bytes. The complete new kernel-journal interval contained no GPU entries and no devcoredump appeared.

This is synthetic numerical/lifetime qualification. It does not establish production route counts, model correctness, speed, default-OFF device-image absence, strict physical262144-token lifecycle or adoption. The production async error boundary remains unqualified; subsequent model and timing work must address it. No candidate is adopted.

Successful verbose compiler/API traces are not copied wholesale. Original byte/hash identities remain in receipts; trace-summary preserves actual kernel identities/counts/shapes and successful results. Small exact stdout, controllers, compile/link commands and receipts are retained.
