# Registered copy: default counter conversion, three exact32K checks

On the same Arc B57010GiB / Ryzen5 5600X /128GiB boot, kernel7.0.0-38,
NEO26.31.39395.14 and oneAPI2026.1.1, the private scheduling candidate
e82fc5de5480b72255b601759497d5810e86bf691350417ed0dc11ed6c429323
is unchanged from the [completed no-CNR controls and clean comparison](../registered-copy-no-cnr-and-prefill-scheduling/README.md).
Only EnableImplicitConvertionToCounterBasedEvents=0 is removed.
MKL_CBWR remains absent; queue properties, arithmetic, subgroups, ranges,
barriers, drains and graphs remain unchanged. Other common process-local
settings, including disabled direct submission/persistent cache/copy offload,
remain recorded. This tests default counter conversion, not every environment
variable at its default.

The first logged/validated32K request and two fresh unlogged state/head
repetitions all match all66prefill state parts, all248,320first-head float
bytes, all64generated IDs and every protocol logprob with the prior counter-off
control and one another. Every process reads32,768 input tokens, with context
33024,8192-token chunks, int8 KV, normal MTP4, cache128, five CPU workers,
pcie0, FIRST0/RING8 and no prompt/conversation caching. All three exit normally,
complete owned cleanup and record no new xe fault. The [state sequence](sequences/event-ack-no-root-prefill-default-v01402-state-sequence.json)
passes both fresh comparisons.

These executions show that forced-off implicit conversion and MKL CNR are
unnecessary for the tested32K candidate controls. They do not prove general
hang prevention, full262,144-cell serving/repeats/restore or the PP1000/TG70
target. State/head dumping is enabled in all three; the first also has UR/
Level Zero API logs and parameter validation. No timing from these jobs is
used for a speed claim or substituted into the earlier clean comparison.
The candidate remains private and unadopted while full-context gates are open.

Exact environment/argv, fixture hashes, raw protocol, project messages, journal
and process ownership are public. Large API/state/head payloads remain private
with byte counts and SHA256 hashes in private-artifacts.json. No reset, rebind,
reboot, package, service or global environment setting is changed.
