# Main serial validation plan — all NOT_RUN

Do not apply/build/test/clean or inspect active owners while the clean full OFF/ON suite is running. Main selects the later serial review point. No author process interaction occurred.

1. Verify source-pins against actual sources and configured resolver CMakeLists -> cmake/sycl_backend.cmake -> src/kernels/sycl/iq.cpp. Apply this patch directly to the selective product in Main's owned build. Do not apply old six-type proposal or v1 patch first. Record original and proposed source, actual compile flags/commands, linked archives and native/fixture hashes. Existing try_parallel_iq is unchanged; review its five switch cases and both type18-only insertion sites. Public validation precedes all candidate dispatch. Check device IR for SG16/SG32, uniform collectives, source0 broadcast, multiply order, no unexpected spills; CPU formula checks do not validate GPU exchange.
2. Main builds/reviews a new versioned fixture/controller with existing fail-stop Owner and shared exclusive measurement lock. Use separate native children, fresh capture dirs and complete pre-exec env dictionaries. No setenv/unsetenv, runtime arm switching, external process attach or retry after submission failure. Main owns fresh journal boundaries, boot/PID/start identities, deadlines, byte budgets, normal/abnormal native exit and reaping. A compiler/required-width failure is a failed gate, not unsupported-device fallback.

| fixture ARM | parallel selector | SG selector |
| --- | --- | --- |
| off | 0 | absent or empty |
| on | 1 | absent or empty |
| sg16 | 0 | sg16 |
| sg32 | 0 | sg32 |
| on-sg16 | 1 | sg16 |
| on-sg32 | 1 | sg32 |
| invalid | 0 | one invalid value |
| on-invalid | 1 | one invalid value |

Invalid values are SG16, sg16 followed by a space, 16, 1, sg64; each is a fresh process. Invoke gpu_fixture ARM quick NEW_CAPTURE_DIR, require69 cases plus six F32/20 invalid, guards/input/CPU checks and fresh faults0, then full with75 cases. compare_captures.py OFF_DIR CANDIDATE_DIR NEW_REPORT.json must pass strict full bytes for every candidate. Never reuse directories or rewrite failed receipts. The actual-SG probe is separate from product launches. Unsupported width must produce DECLINED_BEFORE_DISPATCH plus exact scalar arrays; do not force unsupported device kernel dispatch. No unsupported-device run means that portability gate remains NOT_RUN.
3. Verify actual product route/launch count separately using Main's reviewed diagnostic mechanism and exact compiled source/IR. In parallel0+SGoff every half path is scalar. In parallel1+SGoff only GU21/22/23 and generic20/42 use per-element; type18 remains scalar. SG16/SG32 affects only type18 generic/GU; combined arms add those SG routes while retaining the same five per-element roles. GU20/42, generic21/22/23, every other format and all F32 stay scalar. Invalid SG never widens the five-role list. Supported type18 replaces one scalar launch with one SG launch; failed submission propagates without trying another width/scalar. Probe and fixture markers are not product launches. Array equality cannot establish dispatch and is not accepted as that proof.
4. Only after source/build/device/full-array/fault gates, Main may run separate clean OFF/SG16/SG32 and combined-selector measurements with observers/profilers disabled. Include small/padded shapes, real GU640x2560, ne1/8/64 and materializer-plus-oneMKL complete-route work. Record every sample, process/order, output bytes and actual selected width; distinguish host-submit, completion and route wall. Logical byte counts do not establish bandwidth. Exclude unsupported fallback from SG speed conclusions. Retain the five-role default behavior and faster scalar choice unless repeated route measurements support a type18 decision.
5. Actual-model same-work prefill/decode, tensor/logit/ID parity, restore and cancel/recovery/context gates remain required before adoption. Do not claim fixture success is performance, task quality, RESTORE or physical262144 qualification. New SG selector remains defaultOFF. Preserve original failures and later per-component decisions.

Retention: Main uses64MiB text/512MiB total job/128MiB per-arm capture budget unless a finite replacement is documented before launch. Keep compact commands/env/source/binary identities, individual samples, full correctness/fault statuses. Raw arrays have one current consumer: strict comparison at this candidate boundary. Retire only after recorded hashes/comparison, closed owners and Main's serialized retention manifest; keep one representative per distinct unresolved mechanism. No cleanup was performed here. Next review: Main source/build/fixture result or candidate rejection.
