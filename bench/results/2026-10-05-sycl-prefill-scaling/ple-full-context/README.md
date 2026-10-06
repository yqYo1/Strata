# Full-context PLE gather, CPU only

On Ryzen 5 5600X / 128 GiB RAM, the original PLE hash, table reader and
decoder complete all sixteen comparisons for the actual full-CLI prefill
fixture's first 262,141 tokens. Each case compares every one of its
2,684,323,840 output bytes against a 1,024-token direct-reader reference.
All reference floats are finite. No GPU or complete inference runs here.

The table is the native IQ4_NL second GGUF shard, 28,800,138,432 bytes.
`ple_full_context_probe.cpp` changes only the gather batch size and the
existing `direct` / RAM-table policy. It does not replace the hash or decoder.
Batch sizes are 1,024, 4,096, 8,192 and the entire 262,141-token prefill.
Both batch and policy order reverse in the second round, with 16 I/O workers.

Gather seconds, excluding table open and byte comparison:

| Batch tokens | Direct, round 1 before RAM touch | Direct, round 2 after RAM touch | RAM, round 1 | RAM, round 2 |
| --- | ---: | ---: | ---: | ---: |
| 1,024 | 134.148 | 4.689 | 0.873 | 0.857 |
| 4,096 | 125.800 | 4.746 | 0.882 | 0.876 |
| 8,192 | 123.175 | 4.619 | 0.885 | 0.887 |
| 262,141 | 147.418 | 4.837 | 0.843 | 0.872 |

The file cache was not flushed. The first direct cases also follow the
134.063-second reference read. Each direct reader opens a fresh row cache,
while the later RAM-table touch warms the underlying file cache. Therefore
the two direct rounds must not be combined as a single speed estimate or
described as matched cold/warm trials. Whole-context gathering is slower
than the 1K batch in both observed direct rounds; these results do not
support adopting whole-context PLE batching.

The first RAM-table open/touch takes 42.043 seconds. Subsequent opens take
1.478–1.516 seconds. Every `mlock` attempt fails under the existing host
limit, so the production fallback touches the table pages without locking
them; every RAM case reports `locked=false`. The RAM gather results are
promising for the next model-level comparison with existing `--ple-io ram`.
They do not establish a complete prefill or TG improvement.

The process exits zero after 762.345 seconds. Its 762 memory samples peak at
33,625,592 KiB RSS and show zero process swap, zero GPU allocation and no
sampler errors. The original engine sources are unchanged and their hashes,
the frozen executable hash, fixture hash, commands and individual times are
in [summary.json](summary.json). Raw gather records and I/O statistics are
in [stdout.jsonl](stdout.jsonl) and [stderr.log](stderr.log). The private
memory record and executable digests are in [evidence-manifest.json](evidence-manifest.json).

Build from the repository root:

```sh
g++ -O3 -std=c++17 -march=native -Iinclude \
  sycl/tools/ple_full_context_probe.cpp src/kernels/ngram.cpp \
  src/ngram/ple_reader.cpp src/platform/direct_file.cpp -pthread \
  -o /path/to/ple-full-context-probe
STRATA_IO_THREADS=16 /path/to/ple-full-context-probe \
  /path/to/native-shard-2.gguf /path/to/fills-context.tokens.txt \
  262141 1024,4096,8192,0 2
```

The full-context controller now accepts `--ple-io ram`, writes to a separate
`-ple-ram` report directory, and requires the actual table-load time and lock
status in the engine's startup log. It retains every full-context and
speculative-tail check. The [CPU protocol check](controller-smoke-ple-ram.json)
passes four valid direct/RAM cases at 64 and 262,144 and rejects ten invalid
replies, including a missing RAM startup record. This checks the controller,
not model arithmetic, GPU recovery or a real full-context RAM-table run.
