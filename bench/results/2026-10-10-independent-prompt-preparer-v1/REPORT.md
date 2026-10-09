# Whole-document prompt preparer v1: fake fixtures only

A separate gpt-6.1-sol prepared the source without execution. Root ran the eight
independent fake-tokenizer/renderer fixtures under the shared measurement lock.
They passed in 0.152 seconds, normal exit 0, without forced cleanup or survivors.
No production tokenizer, model payload, inference or GPU work ran.

Read-only gpt-6-luna R66 reviews then identified FIFO pre-open blocking, missing
process resource supervision, and a completion claim that can survive a final
record fsync/close error. This source is held for actual input preparation. The
fixture pass remains valid for its tested scope; it does not waive those gaps.
The separate v2 implementation addresses them before root-owned qualification.

Source pins:

- `sycl/tools/prepare_independent_prompt_tokens.py`: `365d3a32b55252cafcb1947e7743dfc9d0758589d91f9242ca730fe71a9925a3`
- `sycl/tools/test_prepare_independent_prompt_tokens.py`: `016bb3a42b15f2fd21211cc30f6c687c7e87c9a96de4785f56a4de41a1a4b2f7`
- Original root receipt: `1ef223ccbc2e51a4e69e6d2f5ace5d28aa14bca8285adbec950d5200a4326a09`

The original receipt and source are unchanged. Documents remain whole; no
truncation, padding, repetition or validation-score-driven selection is used.
The structural shingle guard and declared source IDs/hashes do not establish
statistical independence. Actual pack inputs and full physical 262144-position
model lifecycle remain unqualified by this CPU fixture run.
