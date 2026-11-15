# Arc Control setup port map

## Provenance and boundary

This is a source-only map of the required Control/App/MCP Arc setup fix. The hotfix is at Niko commit `1130d824aa1791f763d0083008bded14fc09dcb9`; the observed endpoint is `504b143342a9fe1aa8320a7b31e11f5c0ac265c3`. The seven manifest-listed endpoint files have matching SHA-256 values at both revisions, including the hotfix, tests, and root license; this says nothing about unlisted intervening changes. The root [Niko LICENSE](https://github.com/Niko1221/Strata/blob/1130d824aa1791f763d0083008bded14fc09dcb9/LICENSE) is MIT, 1,093 bytes, SHA-256 `949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0`, matching the fork license at source `93d908410c15d526b855b3de0c54f9d703ea7b48).

## Required function slice

- **Hardware and engine choice:** Niko `strata_control/intel_app.py` provides Linux support, Arc discovery, Control's selected SYCL engine check, family/local-file helpers, and a safe card-list gate. Add Intel cards only on Linux, when no usable NVIDIA/AMD card has precedence and Control selects SYCL; probe failure should not break the hardware page. The fork's MCP source at `93d90841:tools/strata_mcp.py` needs the connected `_hardware`, `_engine_vendor`, `recommend`, schema, and `install_plan` adaptations. Explicit `sycl` must be validated and represented in the generated setup command; `auto` must follow the selected engine/vendor rules.
- **Local model plan:** Niko `install_plan` treats Arc as local-model-only: validate an existing absolute GGUF folder or use the existing data/model path, require an eligible family and complete files, set the explicit backend, and explain when files are absent. Keep the project exclusions for DeepSeek and 35B execution. Do not add downloads or widen model selection through shared code.
- **Control setup:** Niko `setup.py:intel_control_setup` handles CONTROL-mode `--backend sycl`: check Linux and model files, prepare a local pack through `prepare_pack_only` (which may fetch the small GGUF helper, not model weights), call `adopt.plan`, then write a `component:strata-engine` config. No native engine build or model download is part of this path. The upstream `adopt.py` uses driver-dependent `--ple-io ram` / `--stream-experts` flags and upstream defaults; these are assumptions until the fork's actual engine contract accepts them.
- **App dispatch:** The Niko test at `504b1433:apps/host/tests/arcfix-updated-control.test.mjs` checks selecting updated Control and launching its bundled Python/MCP with `--src`, cwd, and `STRATA_CONTROL_DIR`. The desktop host implementation is not in this fork's pinned port set; Python source or mock tests cannot qualify packaged dispatch.

## Fork call-graph correction

At fork source `93d90841:setup.py:8248..8271`, `sycl_setup()` targets `ROOT / "sycl" / "setup_intel.py"`; `main()` calls it at `setup.py:10120`. Main independently confirmed that target is absent. `tools/setup_intel.py` is a separate adapter and is not the target of this route. Treat the CONTROL path as unwired until Main establishes its actual entrypoint and adaptation.

The existing CLI adoption path in `strata_cli/launch/runner.py:1388..1401` calls `adopt.plan` and then `adopt.write` directly. It does not pass through `tools/setup_intel.validate_config()` or `to_native()`. Scope those checks to callers that actually use the adapter. Validate any new CONTROL config against the current native engine source before replacing a file; the accepted flags are not established here. Preserve current localhost binding; any listener beyond localhost requires an API key.

## Preservation and next prerequisites

Preserve existing explicit MTP selection, the current Arc Auto drafter policy (including DFlash2 where supported), unrelated native tuning/config fields, and local model paths. Refuse a rewrite that would drop user settings or emit an unsupported native flag. Keep Arc vision behavior tied to actual native reader capability; do not infer it from upstream.

Before implementation, Main should reconcile the CONTROL entrypoint with the separate adapter, record accepted CLI flags from the actual native source, and confirm the config-write path preserves existing MTP/tuning and localhost/API-key settings. Then use host mocks for vendor/backend choice, local/incomplete GGUF paths, pack/config flow, preservation, and pre-write flag rejection. Desktop qualification separately requires the host app source, a packaged Control fixture, and an app-side dispatch test. This map provides no runtime, model-correctness, or performance qualification.
