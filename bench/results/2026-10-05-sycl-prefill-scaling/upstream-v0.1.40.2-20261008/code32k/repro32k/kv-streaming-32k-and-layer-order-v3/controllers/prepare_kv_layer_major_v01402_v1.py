"""Enable existing host-KV stage in the single-device layer-major path."""
from pathlib import Path
import datetime,difflib,hashlib,json
base=Path(__file__).parent
root=Path("/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05")
original=root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/prefill/prefill.cpp'
out=root/'build-sycl-kv-layer-major-v1-20261008/source';out.mkdir(parents=True)
receipt=base/'kv-layer-major-v01402-source-v1';receipt.mkdir(mode=0o700)
text=original.read_text()
before='''        if (m.ss->qsa_states[m.ss->qsa_ord0 + i].kv_mode != 0) {
            err = "prefill: layer-major requires all K/V pages resident in VRAM";
            return false;
        }'''
after='''        const auto& state = m.ss->qsa_states[m.ss->qsa_ord0 + i];
        if (state.kv_mode != 0 && state.kv_mode != 1) {
            err = "prefill: layer-major requires resident or streamed main K/V";
            return false;
        }
        if (state.kv_mode == 1 &&
            (!stage_own() || !m.ident_table || !m.stage.present() ||
             !state.host.present())) {
            err = "prefill: streamed layer-major requires owned identity staging";
            return false;
        }
        if (state.kv_mode == 1) {
            const char* prefetch = std::getenv("STRATA_KV_PREFETCH");
            if (prefetch && std::atoi(prefetch) != 0) {
                err = "prefill: streamed layer-major requires KV prefetch disabled";
                return false;
            }
        }'''
assert text.count(before)==1
changed=text.replace(before,after);p=out/'prefill.cpp';p.write_text(changed)
(receipt/'candidate.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),fromfile=str(original),tofile=str(p))))
record=dict(active=False,source_prepared=True,compiled=False,gpu_tested=False,adopted=False,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Only replace mode0-only layer-major rejection with mode0/mode1 and owned stage/no-prefetch requirements. Existing same-order stage/upload/append/attention and RAM-authoritative mirror are unchanged. No prefix-retention optimization yet.',
    source_basis='run_layer_major selects each single layer and calls run_impl over all prompt chunks. The already allocated full-context identity stage services its QSA; each prefix upload remains on the same in-order compute queue. Main-cache/MTP releases use previously validated graph retirement/restore callbacks. First candidate uses RAM residuals (R_GPU0) to leave enough room at256K geometry and across repeats.',
    required_gates=['Before candidate GPU use, complete current v3 chunk-major3x32K state/head/IDs/logprobs/MTP and normal shutdown/noXE.',
                    'First layer-major0-residual3x32K uses full main RAM release and MTP lease with actual graph retirement, flushed UR/L0 validation; compare all live main bytes/full head/output/counts.',
                    'Captures/diagnostics excluded from speed. Matched clean inputs>=32K, full262144 occupancy/repeat/restore/tail/refusals/later>=32K before adoption.'])
(receipt/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
