# Equivalent GPU memory and warm-table long measurements

The user temporarily stops the embedding service. All GPU jobs run
sequentially, and the controller restores that service after the finite
measurement sequence. Its health check returns HTTP 200 with status ok.
The source and binary remain the validated AOT build. All four AOT CLI
observations preserve every finite first-head value and generated ID
against the original 32K/8K-chunk K8/V8 control. All use the same fixed
65,538-position context, cache 128 and original model arithmetic.

| AOT observation | Prefill seconds | Tokens/s | PLE startup seconds |
| --- | ---: | ---: | ---: |
| Direct PLE, layer-major, GPU residual 32K, normal wall | 50.654 | 646.90 | — |
| RAM PLE, layer-major, GPU residual 32K, transfer markers | 50.174 | 653.09 | 1.5 |
| RAM PLE, original traversal, normal wall | 72.158 | 454.11 | 1.5 |
| RAM PLE, layer-major, GPU residual 32K, normal wall | 50.245 | 652.16 | 1.5 |

These are single AOT observations, not repeated medians. All RAM table
loads are warm and report loaded, not locked. The first full-table load
in the earlier RAM proof takes 42.3 seconds. The Q2_0 and native IQ3_S
PLE shards are hard links to the same file. The profile helper reports
startup separately from prefill and records whole process time, memory
and each locking outcome.

The direct-mode AOT log also reports much faster logical file reads
than the earlier pre-restart observations: read p50 597 µs and total
reported blocking of 772 ms, versus the earlier approximately 20 ms
median read latency. The table has been fully touched before this
sequence. ZFS properties remain `primarycache=metadata`, compression
off and 128K records; they are read back and unchanged. Physical SSD
traffic and the cause of the shorter read waits are not isolated here.
These warm observations must not be compared with the earlier results
to attribute the whole difference to AOT or to an engine change.

The RAM versus direct layer-major difference is about 0.8% in these
single observations; it does not establish a repeatable RAM-mode gain.
The [three JIT pairs](../paired-normal-wall/summary.json), measured in this
same warm-table sequence, provide the actual original-versus-layer-major
comparison. Their medians are 72.289 and 50.736 seconds, respectively.
All raw commands, hashes, memory samples and full-head checks are saved.
