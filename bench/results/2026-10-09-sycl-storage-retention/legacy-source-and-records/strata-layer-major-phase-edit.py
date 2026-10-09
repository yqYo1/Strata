from pathlib import Path
p = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/src/prefill/prefill.cpp')
s = p.read_text()
def replace(a,b):
    global s
    assert s.count(a)==1, (a[:120],s.count(a))
    s=s.replace(a,b)
replace('struct ExpertTransferTimer;', 'struct ExpertTransferTimer;\nstruct PfTimer;')
replace('    ExpertTransferTimer* transfer_context = nullptr;', '    ExpertTransferTimer* transfer_context = nullptr;\n    PfTimer* phase_context = nullptr;')
replace('kPfPleRead, kPfGrWrite, kPfCount };', 'kPfPleRead, kPfGrWrite, kPfLayerLoad, kPfResidual, kPfCount };')
replace('"gdn recurrence", "gdn out proj", "ple read wait", "hc write+norm"};', '"gdn recurrence", "gdn out proj", "ple read wait", "hc write+norm",\n                                        "layer weight preload", "residual copy"};')
replace('    double ms[kPfCount] = {};', '    double ms[kPfCount] = {};\n    double host_setup_ms = 0, host_sync_ms = 0, host_chunk_ms = 0;\n    double ple_read_ms = 0, ple_wait_ms = 0;')
start=s.index('        double total = 0.0;',s.index('    if (!m.transfer_context) transfers.report'))
end=s.index('        if (pe.on) {',start)
original_body=s[start:end]
body=original_body.replace('pt.ms', 'ms').replace('ms_since(t_start)', 'wall_ms').replace('stats_.ms_experts_host','staging_ms').replace('stats_.ms_ple','ple_ms')
replace('    ~PfTimer() {', '    void report(int64_t n, double wall_ms, double staging_ms, double ple_ms) {\n        if (!on) return;\n'+body+'    }\n    ~PfTimer() {')
replace(original_body, '''        pt.host_setup_ms += host_setup_ms; pt.host_sync_ms += host_sync_ms; pt.host_chunk_ms += host_chunk_ms;
        pt.ple_read_ms += ple_read_ms; pt.ple_wait_ms += ple_wait_ms;
        if (!m.phase_context) pt.report(n, ms_since(t_start), stats_.ms_experts_host, stats_.ms_ple);
''')
replace('            std::string pl;\n            for (int i = 0;', '            std::string pl;\n            char b[96];\n            for (int i = 0;')
replace('!m.compact || !m.src || std::getenv("STRATA_PREFILL_TIMING")) {', '!m.compact || !m.src) {')
replace('a full single-GPU compact FP16 path without phase markers', 'a full single-GPU compact FP16 path')
replace('    const auto started = Clock::now();\n    const auto old_stats = stats_;', '''    if (const char* preload = std::getenv("STRATA_PREFILL_PRELOAD_PLE"); preload && preload[0] == '1') {
        err = "prefill: PLE preload is unavailable for layer-major processing";
        return false;
    }
    const auto started = Clock::now();
    const auto old_stats = stats_;''')
replace('    transfers.residual_gpu_tokens = gpu_tokens;', '    transfers.residual_gpu_tokens = gpu_tokens;\n    PfTimer phases;')
replace('        float* gpu; int64_t gpu_tokens;', '        float* gpu; int64_t gpu_tokens; PfTimer* phases;')
replace('            m.residual_gpu = gpu; m.residual_gpu_tokens = gpu_tokens;', '            m.residual_gpu = gpu; m.residual_gpu_tokens = gpu_tokens; m.phase_context = phases;')
replace('m.transfer_context, m.residual_gpu, m.residual_gpu_tokens, on_chunk, on_stage_chunk};', 'm.transfer_context, m.residual_gpu, m.residual_gpu_tokens, m.phase_context, on_chunk, on_stage_chunk};')
replace('    m.residual_gpu = device_rows.get(); m.residual_gpu_tokens = gpu_tokens;', '    m.residual_gpu = device_rows.get(); m.residual_gpu_tokens = gpu_tokens; m.phase_context = &phases;')
replace('        const auto load_started = Clock::now();', '        phases.mark(kPfLayerLoad, strata::q_of(m.cs));\n        const auto load_started = Clock::now();')
replace('        transfers.serial_layer_load_ms += ms_since(load_started);', '        transfers.serial_layer_load_ms += ms_since(load_started);\n        phases.mark(kPfStart, strata::q_of(m.cs));')
replace('                const int64_t row = position - pos0;', '                phases.mark(kPfResidual, strata::q_of(m.cs));\n                const int64_t row = position - pos0;')
replace('                                         (size_t) count * D * sizeof(float), layer).wait();\n                return true;', '                                         (size_t) count * D * sizeof(float), layer).wait();\n                phases.mark(kPfStart, strata::q_of(m.cs));\n                return true;')
replace('    stats_.tokens = old_stats.tokens + n;', '    phases.mark(kPfStart, strata::q_of(m.cs));\n    m.cs->wait(); phases.fold();\n    phases.report(n, ms_since(started), stats_.ms_experts_host - old_stats.ms_experts_host,\n                  stats_.ms_ple - old_stats.ms_ple);\n    stats_.tokens = old_stats.tokens + n;')
replace('    PfTimer pt;', '    PfTimer local_phases;\n    PfTimer& pt = m.phase_context ? *m.phase_context : local_phases;')
replace('            const bool gpu_rows = m.transfer_context && c0 + T <= m.residual_gpu_tokens;', '            if (m.phase_context) pt.mark(kPfResidual, cs);\n            const bool gpu_rows = m.transfer_context && c0 + T <= m.residual_gpu_tokens;')
replace('                err = "prefill: the layer split\'s hand-off upload failed";\n                return false;\n            }', '                err = "prefill: the layer split\'s hand-off upload failed";\n                return false;\n            }\n            if (m.phase_context) pt.mark(kPfStart, cs);')
p.write_text(s)
