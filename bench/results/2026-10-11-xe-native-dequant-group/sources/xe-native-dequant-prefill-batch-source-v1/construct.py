from pathlib import Path
import hashlib, json, difflib
root = Path(__file__).resolve().parent
base = (root/'base/src/prefill/prefill.cpp').read_text()
s = base
anchor = '                    // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.\n'
new = '''                    // Xe source-only candidate: OFF unless explicitly requested. No non-Xe API references.
                    const bool native_batch = [&] {
#ifdef STRATA_ENABLE_SYCL
                        const char* v = std::getenv("STRATA_PREFILL_NATIVE_DEQUANT_BATCH");
                        return v && std::strtol(v, nullptr, 10) != 0 && lay.native && !use_mmq;
#else
                        return false;
#endif
                    }();
                    const uint8_t* nd_blob[16]{};
                    int nd_slot[16]{};
                    int nd_n = 0, nd_q0 = 0;
                    size_t nd_consumed = consumed;
                    uint64_t nd_ga = 0, nd_gd = 0, nd_da = 0, nd_dd = 0;
                    uint64_t nd_ge = 0, nd_de = 0, nd_gfe = 0, nd_dfe = 0, nd_flushes = 0, nd_reuse = 0, nd_released = 0;
                    auto native_flush = [&] {
                        if (!native_batch || nd_n == 0) return;
#ifdef STRATA_ENABLE_SYCL
                        const auto& f = lay.fmt[(size_t) l];
                        const size_t limit = std::numeric_limits<size_t>::max();
                        // NativeFmt supplies real row payloads, not padding or an inferred common type.
                        if (f.n_ff <= 0 || f.n_embd <= 0 || f.gu_row > limit / (size_t) f.n_ff ||
                            f.d_row > limit / (size_t) f.n_embd)
                            throw std::runtime_error("prefill native batch invalid source geometry");
                        const size_t gb = f.gu_row * (size_t) f.n_ff, db = f.d_row * (size_t) f.n_embd;
                        const size_t blob_bytes = (size_t) lay.blob_bytes(l);
                        if (f.bytes > blob_bytes || gb > f.up_off || f.up_off > f.down_off ||
                            gb > f.down_off - f.up_off || f.down_off > f.bytes || db > f.bytes - f.down_off ||
                            nd_q0 < 0 || nd_n > 16 || nd_q0 > g_expert_group - nd_n ||
                            (size_t) g_expert_group > limit / (size_t) GU_ELEMS / sizeof(uint16_t) ||
                            (size_t) g_expert_group > limit / (size_t) D_ELEMS / sizeof(uint16_t) ||
                            (size_t) f.n_ff > (size_t) D_ELEMS / (size_t) f.n_embd ||
                            (size_t) f.n_ff * (size_t) f.n_embd != (size_t) D_ELEMS ||
                            (size_t) GU_ELEMS != 2 * (size_t) D_ELEMS)
                            throw std::runtime_error("prefill native batch source/destination bounds");
                        strata::kernels::IqDequantGroup ng;
                        ng.n = nd_n;
                        for (int i = 0; i < nd_n; ++i) {
                            ng.gate[i] = nd_blob[i]; ng.up[i] = nd_blob[i] + f.up_off;
                            ng.down[i] = nd_blob[i] + f.down_off;
                            ng.gate_bytes[i] = gb; ng.up_bytes[i] = gb; ng.down_bytes[i] = db;
                        }
                        // Rebasing the destination makes API windows <=16 while preserving q in groups <=64.
                        uint16_t* gu = m.dq_gu + (size_t) nd_q0 * GU_ELEMS;
                        uint16_t* dn = m.dq_d + (size_t) nd_q0 * D_ELEMS;
                        const size_t gu_cap = (size_t) (g_expert_group - nd_q0) * GU_ELEMS * sizeof(uint16_t);
                        const size_t dn_cap = (size_t) (g_expert_group - nd_q0) * D_ELEMS * sizeof(uint16_t);
                        const bool ga = strata::kernels::iq_dequant_gu_group_f16(f.gu_type, ng, f.n_ff, f.n_embd,
                                                                                 gu, GU_ELEMS, gu_cap, m.cs);
                        if (transfer_counts) { PfRouteCensus::add(ga ? nd_ga : nd_gd); PfRouteCensus::add(ga ? nd_ge : nd_gfe, nd_n); }
                        if (!ga) for (int i = 0; i < nd_n; ++i) {
                            strata::kernels::iq_dequant_gu_f16(f.gu_type, nd_blob[i], nd_blob[i] + f.up_off,
                                                               f.n_ff, f.n_embd, gu + (size_t) i * GU_ELEMS, m.cs);
                            if (census) PfRouteCensus::add(census->native_dequant_gu);
                        }
                        const bool da = strata::kernels::iq_dequant_down_group_f16(f.d_type, ng, f.n_ff, f.n_embd,
                                                                                   dn, D_ELEMS, dn_cap, m.cs);
                        if (transfer_counts) { PfRouteCensus::add(da ? nd_da : nd_dd); PfRouteCensus::add(da ? nd_de : nd_dfe, nd_n); }
                        if (!da) for (int i = 0; i < nd_n; ++i) {
                            strata::kernels::iq_dequant_f16(f.d_type, nd_blob[i] + f.down_off, f.n_embd * f.n_ff,
                                                            dn + (size_t) i * D_ELEMS, m.cs);
                            if (census) PfRouteCensus::add(census->native_dequant_down);
                        }
                        // Checked markers follow BOTH readers. No slot or producer prefix is published early.
                        for (int i = 0; i < nd_n; ++i) if (nd_slot[i] >= 0) {
                            const int sl = nd_slot[i];
                            publication::require(strata::gpu::event_record(m.used[sl], m.cs), "strata::gpu::event_record", strata::gpu::last_error);
                            m.used_of[sl] = sl;
                            if (transfer_counts) PfRouteCensus::add(nd_released);
                        }
                        nd_n = 0;
                        if (transfer_counts) PfRouteCensus::add(nd_flushes);
                        if (nd_consumed > consumed) { consumed = nd_consumed; give_back(consumed); }
#endif
                    };
'''
assert s.count(anchor)==1
s=s.replace(anchor,new+anchor)
old='''                        if (lay.native) {
                            // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas'''
new='''                        if (native_batch) {
                            if (nd_n == 16) native_flush();
                            if (nd_n == 0) nd_q0 = q;
                            if (q != nd_q0 + nd_n) throw std::runtime_error("prefill native batch noncontiguous window");
                            nd_blob[nd_n] = blob_dev; nd_slot[nd_n] = slot; ++nd_n;
                        } else if (lay.native) {
                            // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas'''
assert s.count(old)==1;s=s.replace(old,new)
old='''                        if (slot >= 0) { publication::require(strata::gpu::event_record(m.used[slot], m.cs), "strata::gpu::event_record", strata::gpu::last_error); m.used_of[slot] = slot; }
                        if (q + 1 < G && j + 1 < order.size()) return true;'''
new='''                        if (!native_batch && slot >= 0) { publication::require(strata::gpu::event_record(m.used[slot], m.cs), "strata::gpu::event_record", strata::gpu::last_error); m.used_of[slot] = slot; }
                        if (native_batch && (nd_n == 16 || (stream_all && nd_consumed > consumed && nd_consumed - consumed >= (size_t) m.ring))) native_flush();
                        if (q + 1 < G && j + 1 < order.size()) return true;
                        if (native_batch) native_flush();   // weights must be expanded before the unchanged GEMMs'''
assert s.count(old)==1;s=s.replace(old,new)
old='''                                if (!stage_one(staged)) return false;'''
new='''                                if (native_batch) {
                                    const int32_t se = order[staged];
                                    const bool sr = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + se] >= 0;
                                    bool held = false;
                                    if (!sr) for (int i = 0; i < nd_n; ++i) held = held || nd_slot[i] == stage_next;
                                    if (held) { if (transfer_counts) PfRouteCensus::add(nd_reuse); native_flush(); }
                                }
                                if (!stage_one(staged)) return false;'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('[&] { if (group_gather) flush(); },','[&] { if (native_batch) native_flush(); if (group_gather) flush(); },')
old='''                                wait_issued(k);
                                if (group_gather) {'''
new='''                                if (native_batch && k - consumed >= (size_t) m.ring) native_flush();
                                wait_issued(k);
                                if (native_batch) nd_consumed = k + 1;
                                if (group_gather) {'''
assert s.count(old)==1;s=s.replace(old,new)
old='''                                if (!group_gather) {
                                    consumed = k;'''
new='''                                if (!group_gather && !native_batch) {
                                    consumed = k;'''
assert s.count(old)==1;s=s.replace(old,new)
old='''                    stager_done.complete();
                    pt.mark(kPfCombine, cs);'''
new='''                    if (native_batch) native_flush();
                    if (native_batch && transfer_counts)
                        std::fprintf(stderr, "[prefill-native-dequant-group-v1] layer=%lld gu_admitted=%llu gu_declined=%llu down_admitted=%llu down_declined=%llu gu_covered=%llu down_covered=%llu gu_fallback_experts=%llu down_fallback_experts=%llu flushes=%llu held_reuse_flushes=%llu released_slots=%llu\\n",
                                     (long long) l, (unsigned long long) nd_ga, (unsigned long long) nd_gd,
                                     (unsigned long long) nd_da, (unsigned long long) nd_dd, (unsigned long long) nd_ge,
                                     (unsigned long long) nd_de, (unsigned long long) nd_gfe, (unsigned long long) nd_dfe, (unsigned long long) nd_flushes, (unsigned long long) nd_reuse,
                                     (unsigned long long) nd_released);
                    stager_done.complete();
                    pt.mark(kPfCombine, cs);'''
assert s.count(old)==1;s=s.replace(old,new)
# Keep the full original consumer/stager block literally intact in the OFF arm.
old_start = base.index(anchor)
old_end = base.index('                    stager_done.complete();\n', old_start) + len('                    stager_done.complete();\n')
off_block = base[old_start:old_end]
s = s.replace('                    const uint8_t* nd_blob[16]{};', '                    if (native_batch) {\n                    const uint8_t* nd_blob[16]{};', 1)
end_marker = '                    stager_done.complete();\n                    pt.mark(kPfCombine, cs);'
assert s.count(end_marker) == 1
s = s.replace(end_marker, '                    stager_done.complete();\n                    } else {\n' + off_block + '                    }\n                    pt.mark(kPfCombine, cs);', 1)
# This is a text-only identity check, not executed behavior or timing qualification.
assert off_block in s
(root/'off-block-proof.json').write_text(json.dumps({'original_block_sha256': hashlib.sha256(off_block.encode()).hexdigest(), 'bytes': len(off_block.encode()), 'literal_original_off_arm': True}, indent=2)+'\n')
p=root/'replacement/src/prefill/prefill.cpp';p.write_text(s)
(root/'candidate.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/prefill/prefill.cpp',tofile='b/src/prefill/prefill.cpp')))
manifest={'status':'SOURCE_ONLY_UNEXECUTED','base_commit':'b3e1de06076b692614ef51512f5501838d4b6f2b','base_path':'src/prefill/prefill.cpp','opt_in':'STRATA_PREFILL_NATIVE_DEQUANT_BATCH=1','default':'OFF','api_dependency':'xe-native-dequant-group-api-source-v1; NOT yet qualified','files':{}}
for f in [root/'base/src/prefill/prefill.cpp',p,root/'candidate.patch',root/'construct.py',root/'off-block-proof.json']:
 manifest['files'][str(f.relative_to(root))]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
report = root.parent/'research-20261009/implementation-xe-native-dequant-prefill-batch-v1.txt'
if report.exists(): manifest['report'] = {'path': str(report), 'bytes': report.stat().st_size, 'sha256': hashlib.sha256(report.read_bytes()).hexdigest()}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
