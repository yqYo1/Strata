# IQ2_S NT1 index-only source handoff

This commit preserves a source-only implementation produced by a separate gpt-6.1-sol agent and reviewed by root plus two gpt-6-luna investigators. No build, test, timing, model inference or adoption has occurred. The experiment is standalone and does not alter engine dispatch.

The controlled comparison is direct_control against index_candidate. Trait baseline against direct_control is a composite comparison: ABI/call path, width checks and the shared sign memcpy differ. No call-only benefit is inferred. Root owns all serialized validation.

The explicit exhaustive fixture uses uniform low bytes. Mixed-lane/unaligned coverage, actual binary codegen, sanitizer, real weights and live activation evidence remain open. Source and review identities are in root-source-review.json; the original implementation report is preserved unchanged.
