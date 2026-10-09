# Whole-document prompt preparer v2: bounded fake fixtures

The separate source-only implementation addresses v1's reviewed admission gaps:
nonblocking regular-file reads, explicit xhigh/template config, 1MiB pre-encode
rendered text, and a candidate that always has completed=false. Owner acceptance
requires normal child exit and independently verified whole-bundle outputs.
A parseable candidate alone never establishes completion.

Root ran17 independent fake-tokenizer/renderer fixture methods under the shared
measurement lock. They passed in0.254s with normal exit0, no forced cleanup or
survivors. The child had wall60s, CPU30/31s, AS1GiB, monitored RSS768MiB,
FSize32MiB, NOFILE64 and core0 budgets. Polling is not a hard RSS limit; AS is.
No actual model-pack tokenizer or payload, inference or GPU work ran.

Fixtures include R63-style32719-common-prefix near-copy rejection, exact80%
unique16-shingle boundary, FIFO rejection, unused-tail hashes, UTF8 byte limits,
final-record write/fsync/close errors and literal special IDs/types. The R67
read-only source review found no new blocker in the supported one-user,
marker-free thinking path. Actual pack input preparation remains a separate
root-owned step with asset/document/source admission and process supervision.

- Helper SHA: `829f6335ff7b47d587832c40b9f30b9aafddce112e7b980095aa17225b0940cd`
- Fixture SHA: `a6e36b7da9bc8d7928bc7417a6eb4d4277f87f72ae435dda95afe3191c68a729`
- Original root receipt SHA: `a1bffdfbd9b104278042d632037bd08c7d0633aae2e97c3756314b92ceea5c5a`

V1 and its original fixture pass are unchanged. No live policy/performance
adoption or full physical262144-position lifecycle claim follows.
