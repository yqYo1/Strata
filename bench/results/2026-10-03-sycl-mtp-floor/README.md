# B570 MTP probability-floor measurements, 2026-10-03

`run.json` contains all four screen settings, six repeated-pair runs, decoded
screen continuations, full flags, the fixed-cache numerical comparison and HTTP
responses. Individual trial JSON and stdout/stderr logs retain raw samples.
The screen's 0.5 and 0.9 trials also form pair one; eight fresh engine processes
were measured in total. No screen or paired sample was discarded.

The engine body is commit `63e5604`; checkout `af21a91` adds only IQ4 screening
records. The only adopted change here is the local server's `--spec-min-p 0.9`.
The preceding and updated server configs are included. Their asset paths point
to the measured workstation, with model/MTP assets outside Git.

The Python scripts are archived measurement drivers, retaining their original
`/tmp` filenames. They run from the repository root after sourcing oneAPI.
Their captured JSON inputs and prompt token files are included; restore these
to their recorded paths to rerun on this workstation. Use the local venv Python
for the numerical check (NumPy is required). Full-vocabulary binary dumps are
excluded from Git; their SHA-256 hashes are in `run.json`. The 0.5 reference is
the fixed2048-slot diagnostic from the verifier-overlap record. Its reference
file must be regenerated before rerunning the comparison; the corresponding
flags and comparison tool are recorded there and in the supplied verifier JSON.

Throughput excludes startup and prompt processing. Requests within each process
retain prompt and expert caches; the first request includes graph preparation.
Background CPU workloads were not isolated. The fixed-cache check covers 64
committed rows sharing the same history; it does not compare different rejected
drafts. HTTP checks cover functionality and clean shutdown, not speed. See
[the implementation notes](../../../docs/SYCL_IMPLEMENTATION.md) for the current
settings, measured limits and server start command.
