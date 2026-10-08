# Prefill host waits: stop propagation and failure bounds

The former Stager `finish()` disabled new job claims, then waited for active
workers. A worker already waiting for DMA acknowledgement could never leave
that wait if completion was lost. Separately, IssuerJoin set its stop flag
and joined the issuer, but an issuer blocked in Stager's host-buffer `wait()`
did not test that flag. These are CPU control-flow defects; they do not
identify the trigger of the original GPU fault.

Stager workers now stop waiting when `finish()` cancels outstanding claims,
and skip the buffer copy. They never overwrite a slot whose previous DMA has
not finished. The issuer passes its stop flag into buffer readiness waits
and checks it before issuing each next transfer. Ordinary successful DMA
acknowledgements, row bytes and transfer order remain unchanged.

Host-only polls have a five-minute deadline **per wait**, independent of the
whole prompt length. The polling predicate makes no driver call. A deadline
failure prints the wait stage and exits with status 1 through `_Exit`, without
unwinding GPU resources or permitting unsafe buffer reuse. A producer-side
exception likewise reports the error and exits. This does not guarantee that
kernel-side device close can finish after device loss, and it does not make
blocked SYCL calls or file I/O cancellable.

## Actual validation

The CPU test extracts the actual Stager class from `prefill.cpp`. Device
selection and allocation use CPU stubs; a deferred CPU queue models DMA reads
and their acknowledgement callbacks. It does not create a SYCL queue or use
GPU hardware. The old production control at `f1dde98` reaches the first two
queued DMAs, then remains blocked in `finish()` until the runner kills it
after two seconds. The control output and source digests are preserved.

With strict ASan/UBSan settings, four candidate cases pass:

- Finish with both DMA reads pending returns without overwriting their source
  slots; the deferred reads still match every source byte afterward.
- A stop flag releases an issuer waiting for an unpublished third job.
- 256 generations / 2,048 actual ring copies retain every byte, including
  final reads deliberately left pending into the next generation.
- A missing completion exits with status 1 and the expected timeout message;
  the test uses a 10 ms deadline, not the production five-minute budget.

The normal CPU processes take approximately 10–18 ms on this host; these
are regression-run durations, not engine performance measurements. The
binary links no SYCL/Level Zero/UR runtime; see `ldd.txt` and `record.json`.
The test checks the host protocol with a fake queue, not actual GPU buffer
lifetime or scheduler correctness. It can be reproduced with
`python3 sycl/tools/test_stager_host.py --output NEW_DIRECTORY`.

The actual engine compiles and links with oneAPI. The factor-0 binary has
SHA-256 `cd3542afeb65c0ff39fa53e83c6f9e63978fc327a1d55fb57cc3bc2cb763fec4`.
The factor-9 candidate has SHA-256 `b69fcfb5b6e403d2aa93a2caac7a8564eb9fe6e45f12b221b58bfc1d890b407c`.
Both CPU archives and comparison programs are identical to the already
[measured and fully compared CPU artifacts](../../cpu-pool-scheduling/README.md).
Factor 0 is restored after the candidate build. No latest engine was run on
the GPU; arithmetic/state/full-262,144-context, cancellation/checkpoint and
prefill/TG integration gates remain pending healthy hardware. Frozen binaries
and compiler logs are retained in persistent host state outside Git.

## Remaining failure paths

Blocking driver calls, async file reads and resource/graph restoration still
need review. This deadline does not bound those calls or recover a wedged GPU.
The original small stock probe failed before its integer kernel on the
already lost device. Xorg currently owns B570 in an active local X11 session;
recovery was refused before reset. No display service was stopped.
