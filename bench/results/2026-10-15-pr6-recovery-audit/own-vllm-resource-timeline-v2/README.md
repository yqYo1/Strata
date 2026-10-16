Main v2 parser correction, SOURCE ONLY UNRUN: the official verbose Kernel Properties table retains the complete shape-suffixed name. Require exact full-name equality and coverage against Device rows, preserve every original raw name and numerical/oracle/fault gate. v1 tiny remains FAIL with normal native/oracle completion; see source-provenance.json. The tiny fixture now reflects the official schema. Main retained the stdout/stderr 64MiB cap correction. Use fresh run-id v2 for both tiny and model.

This is a source-only proposal for Main. Preparation, Python syntax checking, offline fixtures, tiny oracle, model run and collection are **UNRUN**. No speed, hardware scarcity, idle capacity or adoption claim follows from these files.

The new controller preserves the qualified v6 ownership/lifecycle scaffolding. `controller-v6.delta.patch` is the complete controller delta against v6 SHA256 `a358eda324f751c7935e8c3938162cf82705e0e2fd846dd829f8450a9fe4e967`; `run_server_job.py` is byte-identical to v6 SHA256 `3d6df33f033ad675a8b497da2f62b521c9f8c29ea05221cf505c333f691566c6`. Read `implementation-report.md`, all source, `source-provenance.json`, and `proposed_elf_inventory.json` before preparing or running. Newly authored code follows repository LICENSE. Intel code was inspected as schema/design evidence and was not copied into these Python files.

Main must finish the current serialized cohort, hold the existing measurement lock for any execution and keep all other GPU/build/test/cleanup work stopped. Do not run this proposal concurrently with another worker or the restore child. A selected health receipt must be Main's latest eligible closed sameboot softmax0 long case; the fixed diagnostic0 receipt is a known example, not automatic discovery of the latest gate. Source/products/runtime/ELF drift is an error requiring review, never an invitation to refresh pins.

After full review, Main can run these host-only preparation/fixture commands from the repository root. These commands have not been executed by the worker:

```sh
proposal=bench/results/2026-10-15-pr6-recovery-audit/own-vllm-resource-timeline-v2
python3 -B "$proposal/prepare_manifest.py" \
  --reviewed-package-sha256 "$(sha256sum "$proposal/SHA256SUMS" | cut -d' ' -f1)" \
  --reviewed-inventory-sha256 "$(sha256sum "$proposal/proposed_elf_inventory.json" | cut -d' ' -f1)" \
  --reviewed-source-sha256 "$(sha256sum "$proposal/source-provenance.json" | cut -d' ' -f1)"
python3 -B -m unittest discover -s "$proposal" -p test_timeline.py -v
```

The preparation script checks original receipt identities, the 432 current source pins, original products, all inherited file pins and reviewed ELF/runtime candidates. It creates only `manifest.json` in this proposal, exclusively; it performs no native/tool/version/device query. Inherited historical build inputs stay pinned, and every new product identity comes from the original `own-prompt-softmax-build-v3.json` receipt. Do not edit generated manifest pins. Preparation is not a tiny/health/collection qualification.

For execution, Main selects a concrete latest health receipt and its externally verified digest. For the known diagnostic0 example its SHA256 is recorded in `source-provenance.json`; replace it with the latest eligible softmax0 receipt when necessary. Pick a UTC retention review within 48 hours of launch. Do not reuse a run ID or an output directory, including after a failed run. The controller never retries automatically.

```sh
proposal=bench/results/2026-10-15-pr6-recovery-audit/own-vllm-resource-timeline-v2
# Set these explicitly after review: health_receipt, health_sha256, review_utc.
manifest_sha256=$(sha256sum "$proposal/manifest.json" | cut -d' ' -f1)
python3 -B "$proposal/controller.py" --mode tiny --run-id v1 \
  --manifest-sha256 "$manifest_sha256" \
  --health-receipt "$health_receipt" --health-sha256 "$health_sha256" \
  --raw-budget-mib 64 --retention-review-utc "$review_utc"
```

Stop and review the tiny receipt, complete schema, requested-byte evidence, exact summary reconciliation, normal exit0/oracle/ownership/loaded-map/fault gates before scheduling a model job. A tiny failure is a failure; no model launch is authorized by a surviving partial timeline. Tiny uses the current qualified BF16 fixture's existing `--smoke`/FP64 oracle through backend `own`, in its own pre-exec environment. Long model uses backend `vllm`. Both use the same official collector and the same newly enabled timeline/verbose flags.

After the tiny PASS has been reviewed, Main may schedule the single bounded model diagnostic:

```sh
tiny_receipt=bench/results/2026-10-15-pr6-recovery-audit/own-vllm-resource-timeline-tiny-v1.json
tiny_sha256=$(sha256sum "$tiny_receipt" | cut -d' ' -f1)
python3 -B "$proposal/controller.py" --mode model --run-id v1 \
  --manifest-sha256 "$manifest_sha256" \
  --health-receipt "$health_receipt" --health-sha256 "$health_sha256" \
  --tiny-receipt "$tiny_receipt" --tiny-sha256 "$tiny_sha256" \
  --raw-budget-mib 128 --retention-review-utc "$review_utc"
```

The model command predeclares a **128 MiB total raw exception** solely to answer the all-command/QSA-envelope copy-compute overlap question with all individual events and copy payload sizes. The default remains 64 MiB; Main may choose it instead before launch. The only alternatives are 64/128/256 MiB, always private total <=1 GiB. The measured historical v6 model has tens of thousands of long demangled kernel names/copies, so text timeline rows may exceed 64 MiB; this is an estimate, not a measured size for the proposed capture. No API call logging, Chrome, metrics, KMD, XPTI, per-tile, conditional/filter controls or new builds are enabled.

All collector sides are included in each live Owner budget scan, including unexpected/partial files. A budget stop makes the result incomplete/FAIL. Whole/native deadlines are 1200/900 seconds for model and 300/90 seconds for tiny; 150 seconds of parent time is reserved for closure/fault/pin checks. The existing runner also reserves up to 120 seconds after the sole completed benchmark for owned-native QUIT and both PIDfd exits before server shutdown. These are diagnostic captures, not clean speed samples.

Successful model extraction creates a compact structured receipt beside this directory and `timeline-samples.json` in its private state root. The sample table interns kernel names and preserves every line's thread/device/append/submit/start/end nanoseconds, with exact raw/source hashes and representative schema rows in the receipt. All gaps and shape windows can be reconstructed from these rows. Retain only compact verified decision evidence in Git; keep the full sample table outside Git if its size is unsuitable. Main must verify extraction and the needed individual QSA/timing evidence, then retire successful raw timeline/protocol under the exclusive lock with a path/size/SHA256/replacement/reason manifest. This controller performs no cleanup. No trace must be deleted while its owned process is live or if it is the last unresolved-failure fixture.
