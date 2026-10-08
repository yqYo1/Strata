# Observed regular-launch model checks on 2026-10-07

A private executable changes only 44 use_root_sync declarations across eight
source files for 41 manually reviewed kernel classes. Original arithmetic,
kernel names, ranges, caller waits and every other archive member remain.
The offline receipt checks all source/object/archive/binary input hashes;
production inputs are unchanged. The binary SHA is
33bef89feb49a97c152c114c48dd002612215e9b44000f334891b24a3804237b.
Selected kernels own separate rows/elements or use only subgroup/work-group
coordination. Other verifier persistent/global functions are retained.

On Arc B570 10 GiB / Ryzen 5600X / 128 GiB RAM, kernel 7.0.0-38-generic,
NEO 26.31.39395.14 and oneAPI 2026.1.1, four context-128 normal-MTP requests
match frozen updated-control IDs, printed logprobs and every first-head byte.
Six MTP release/restore pairs complete; the owned process exits normally with
no new xe fault. These full Level Zero/UR logging and validation checks are
not clean throughput or full-capacity results.

The subsequent three-repeat 2048-input test stops on its first request,
at layer 40 of the chunk starting at token 256. The last completed phase is
dequant, mark 170571. GU/down APIs return success; the following
zeCommandListHostSynchronize has no successful return. After 60 seconds of
no progress the engine watchdog raises SIGABRT. GDB captures the stop and
main-thread sample; owned cleanup removes both inferior and debugger.
The ordinary-launch candidate is not sufficient to prevent this wait.

The complete traced interval has 258,136 successful native/UR kernel enqueue
associations and zero cooperative UR enqueue flags, with complete native
group/local shapes. No selected class retains a cooperative flag. Successful
API returns are submission evidence, not proof that those kernels complete.
This accounting covers the traced enqueue path, not a general assertion
about every recorded command-buffer operation or all possible configurations.
It narrows the diagnosis beyond the cooperative flag; it does not prove a
particular cache, recorded-graph or runtime cause.

No new xe fault appears. A fresh logged H2D/kernel/D2H health probe then passes
three rounds of 16,384 exact words on the same boot, without reset/rebind,
reboot, service or package changes. The main sample is in the vDSO clock path;
its truncated unwind is not used as proof of a particular CSR object.

The successful short trace still contains seven other cooperative kernel/
local-size combinations. A separate unchanged-object query submits no kernels
and finds four above their optimistic zero-extra-local limits: original
attention chunk 1980 versus 144 groups, add-streams broadcast and to_f16
1200 versus 144, and original attention merge 720 versus 144. Recurrence
192 versus 288, serial convolution 80 versus 288 and top-k 30 versus 36
do not exceed that queried bound. Counts within it do not validate additional
dynamic local storage or all synchronization requirements. Those paths still
need per-function correction, independently of the zero-flag long stall.

The retained main/MTP-backing controller is prepared but has not run. It uses
the same private binary, long-path settings and strict whole-head/ID/logprob
checks, disables only both releases, and requires two repeated requests with
zero MTP release/restore pairs. This is a diagnostic comparison, not an adopted
memory policy. Full 262144-cell CLI/normal-MTP serving, clipped-tail, refusal,
later-valid requests and PP1000/TG70 remain incomplete.

Receipts, exact source copies/diffs, both executed parser versions, bounded
log tails, full private-log hashes and GDB state are included. Binaries, whole
output arrays and large logs stay private. The manifest covers every file
except itself.
