#!/usr/bin/env python3
"""Host-only source/header audit of the user-supplied resident-model reference."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
sys.path.insert(0, str(P / 'tools'))
from gguf_reader import GGUFFile


def pin(path):
    path = Path(path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def header(path):
    path = Path(path)
    before = path.stat()
    g = GGUFFile(path)
    with path.open('rb') as f:
        raw = f.read(g.data_start)
    assert len(raw) == g.data_start
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    arch = g.metadata['general.architecture']
    return dict(path=str(path), bytes=before.st_size, mtime_ns=before.st_mtime_ns,
        inode=before.st_ino, device=before.st_dev, architecture=arch,
        header_bytes=len(raw), header_sha256=hashlib.sha256(raw).hexdigest(),
        identity_scope='Stat identity plus full header SHA256; weight payload not hashed or read.',
        metadata={k: v for k, v in g.metadata.items() if k.startswith(arch + '.')
                  or k in ['general.name', 'general.size_label', 'general.file_type']},
        tensor_count=len(g.tensors), example_tensors=[dict(name=t.name, shape=t.shape,
            type=t.type_name) for t in g.tensors if t.name.startswith(('blk.0.', 'blk.3.'))])


def main():
    prof_path = A / 'hardware-resource-profile-v1.json'
    prof = json.loads(prof_path.read_text())
    window = prof['windows']['long_attention_envelope']
    llama = Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=llama, text=True).strip()
    assert revision == 'abeada335e2e78bd3fe63febafab7e900ce75810'
    source_paths = ['tools/llama-bench/llama-bench.cpp', 'src/models/qwen35.cpp',
                    'ggml/src/ggml-sycl/fattn.cpp', 'ggml/src/ggml-sycl/fattn-onednn.cpp']
    sources = []
    for rel in source_paths:
        original = subprocess.check_output(['git', 'show', revision + ':' + rel], cwd=llama)
        assert (llama / rel).read_bytes() == original
        sources.append(pin(llama / rel))
    dense = header('/data/llm_models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf')
    sparse = header('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/'
                    'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
    dm = dense['metadata']; sm = sparse['metadata']
    d_layers = dm['qwen35.block_count']; s_layers = sm['qwen4exp.block_count']
    d_gdn = d_layers - d_layers // dm['qwen35.full_attention_interval']
    s_gdn = s_layers - s_layers // sm['qwen4exp.full_attention_interval']
    d_heads = dm['qwen35.ssm.time_step_rank']; s_heads = sm['qwen4exp.ssm.time_step_rank']
    d_k = dm['qwen35.ssm.state_size']; s_k = sm['qwen4exp.ssm.state_size']
    d_v = dm['qwen35.ssm.inner_size'] // d_heads
    s_v = sm['qwen4exp.ssm.inner_size'] // s_heads
    record = dict(schema='strata-resident-dense-reference-audit-v1',
        source=pin(__file__), GPU_executed=False, fresh_benchmark_executed=False,
        user_reference=dict(provenance='User-supplied terminal result, 2026-10-16; not a Main reproduction.',
            command='llama/llama-bench -fa 1 -n 0 -p 131072 -ctk q8_0 -ctv q8_0 -m '
                    '/data/llm_models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf',
            build='abeada335 (11455)', device='Intel Arc B570 Graphics', driver='1.17.39395+14',
            actual_prompt_tokens=131072, generated_tokens=0, reported_model_bytes_GiB=5.55,
            reported_parameters_B=8.95, reported_ngl=-1, type_k='q8_0', type_v='q8_0',
            flash_attention=1, mean_tok_s=966.57, reported_spread_tok_s=10.59,
            individual_samples_available=False,
            enabled_build_features=['DNNL', 'F16', 'GRAPH', 'LEVEL_ZERO_API', 'VMM'],
            enabled_runtime_features=['DNN', 'FA_ONEDNN', 'MKL_FA', 'FLASH_ATTN', 'VMM',
                                     'FUSION', 'ESIMD', 'HOST_PINNED_MEM'],
            runtime_graph_enabled=False, actual_attention_kernel_route_observed=False),
        local_installed_binary=pin('/home/yayoi/.local/bin/llama/llama-bench'),
        matching_revision_source=dict(revision=revision, files=sources,
            proves_installed_binary_built_from_current_sources=False,
            bench_defaults=dict(batch=2048, microbatch=512, repetitions=5,
                full_prompt_warmup=True, clear_memory_before_each_sample=True),
            timing='Model/context creation and full-prompt warmup precede samples. '
                   'Each timed prompt uses random token IDs and ends with llama_synchronize. '
                   'Warmup/repetition defaults are source-derived, not recovered raw samples.'),
        model_headers=dict(dense=dense, strata=sparse),
        architectural_work_examples=dict(
            strata_expert_rows_per_8192_chunk_uniform_mean=8192 * sm['qwen4exp.expert_used_count'] / sm['qwen4exp.expert_count'],
            expert_rows_scope='Uniform selection assumption; actual expert row distribution is not measured here.',
            dense_GDN_layers=d_gdn, strata_GDN_layers=s_gdn,
            dense_GDN_heads=d_heads, strata_GDN_heads=s_heads,
            dense_GDN_state_elements=d_gdn*d_heads*d_k*d_v,
            strata_GDN_state_elements=s_gdn*s_heads*s_k*s_v,
            recurrent_state_element_ratio=(s_gdn*s_heads*s_k*s_v)/(d_gdn*d_heads*d_k*d_v),
            scope='Source/header-derived operation shapes, not measured runtime ratios or a speed prediction.'),
        existing_strata_profile=dict(receipt=pin(prof_path), case=prof['captured_case'],
            full_prefill_boundaries_proved=prof['full_prefill_boundaries_proved'],
            attention_envelope=window, host=prof['cpu'],
            long_QSA_service_s=prof['long_QSA_service_sum_ns']/1e9,
            clean_speed_reference=prof['clean_speed_reference']),
        interpretation=dict(
            metric='77.31% is compute-command time coverage inside the attention envelope, not arithmetic utilization.',
            prediction='Do not multiply resident-model tok/s by command coverage or infer throughput from A6B alone.',
            reference='966.57tok/s is a useful measured hardware reference, not a proved Strata ceiling or guaranteed target.',
            bottleneck='Prioritize inefficient GPU work, especially the measured33.963s QSA, and fragmented expert GEMMs; '
                       'this profile does not establish XMX/VRAM/SLM saturation.',
            timing_scope='Do not compare instrumented105.007s to clean tok/s. Clean Strata and supplied llama cases '
                         'also differ in actual prompt length, weights, warmup, placement and workload.'),
        retention='Compact source/header/result evidence only; no weight copies or raw trace generated.')
    out = A / 'dense-reference-audit-v1.json'
    assert not out.exists()
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(receipt=str(out), GPU_executed=False, recurrent_state_element_ratio=record[
        'architectural_work_examples']['recurrent_state_element_ratio'], strata_uniform_expert_rows=record[
        'architectural_work_examples']['strata_expert_rows_per_8192_chunk_uniform_mean'])))


if __name__ == '__main__':
    main()
