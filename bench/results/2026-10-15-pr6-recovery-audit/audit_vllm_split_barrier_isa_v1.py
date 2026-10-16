#!/usr/bin/env python3
"""Read existing artifacts only: tie the prior ISA to current embedded IR.

No compiler, runtime provider, GPU API, subprocess, or process inspection.
This establishes an input-image relationship, not the current JIT's output.
"""
import hashlib
import json
from pathlib import Path
import re

A = Path(__file__).resolve().parent


def pin(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def main():
    target = A / 'vllm-split-barrier-current-image-audit-v1.json'
    assert not target.exists()
    prior_path = A / 'own-vllm-integrated-isa-proof-v1.json'
    prior = json.loads(prior_path.read_text())
    assert prior['passed'] and prior['runtime_entries_match_all8']
    engine_path = A / 'own-qsa-bf16-prefix-tilepack-engine-build-v2.json'
    engine = json.loads(engine_path.read_text())
    assert engine['passed'] and engine['complete'] and engine['ownership_closed'] and not engine['active']
    binary = Path(engine['binary']['path'])
    assert pin(binary)['sha256'] == engine['binary']['sha256']
    image = Path(prior['runtime_image_mapping']['extracted_image'])
    image_pin = pin(image)
    assert image_pin['sha256'] == prior['runtime_image_mapping']['extracted_image_sha256']
    raw, executable = image.read_bytes(), binary.read_bytes()
    assert executable.count(raw) == 1
    offset = executable.index(raw)
    dis_path = A / 'own-vllm-integrated-isa-disassemble-v1.json'
    dis = json.loads(dis_path.read_text())
    assert dis['passed'] and dis['complete'] and dis['ownership_closed'] and not dis['active']
    ze_info = next(v for v in dis['disassembly_files'] if v['path'].endswith('/.ze_info'))
    assert pin(ze_info['path'])['sha256'] == ze_info['sha256'] == prior['ze_info_sha256']
    entries = {}
    for name, value in prior['compiled_entries'].items():
        path = Path(value['asm_path'])
        assert pin(path)['sha256'] == value['asm_sha256']
        lines = path.read_text().splitlines()
        signals = [(j, s.strip()) for j, s in enumerate(lines, 1) if 'signal barrier' in s]
        waits = [(j, s.strip()) for j, s in enumerate(lines, 1) if 'sync.bar ' in s]
        assert len(signals) == len(waits) == 1 and signals[0][0] < waits[0][0]
        dpas = [(j, s.strip()) for j, s in enumerate(lines, 1) if re.search(r'\bdpas\.', s)]
        assert dpas and all(signals[0][0] < j < waits[0][0] for j, _ in dpas)
        entries[name] = {
            'assembly': pin(path), 'metadata': value['metadata'],
            'static_signal_barrier_count': len(signals),
            'static_sync_bar_count': len(waits),
            'static_dpas_count': len(dpas),
            'signal': signals[0], 'wait': waits[0],
            'after_wait': [(j + 1, lines[j].strip()) for j in range(waits[0][0], min(waits[0][0] + 6, len(lines)))],
            'slm_message_text_matches': [s.strip() for s in lines if re.search(r'(?i)send\.slm|\.slm\.', s)],
            'fence_opcode_text_matches': [s.strip() for s in lines if re.search(r'(?i)\bfence\b', s)],
            'private_spill_proven_absent': False,
        }
    assert len(entries) == 8
    record = {
        'schema': 'strata-current-native16-image-prior-barrier-isa-v1',
        'source': pin(__file__), 'prior_proof': pin(prior_path),
        'prior_disassembly_receipt': pin(dis_path), 'engine_build_receipt': pin(engine_path),
        'current_engine': pin(binary), 'prior_image': image_pin,
        'prior_image_exactly_embedded_once': True, 'current_engine_file_offset': offset,
        'ze_info': pin(ze_info['path']), 'entries': entries,
        'GPU_executed': False, 'compiler_executed': False,
        'current_JIT_native_output_captured': False,
        'barrier_removal_proven_safe': False, 'barrier_cost_measured': False,
        'production_changed': False, 'performance_adopted': False,
        'interpretation': [
            'Existing eight-entry native16 IR image is byte-identical inside the current engine. Prior corresponding disassembly exists for both tile policies and all four 16-bit input/weight combinations.',
            'Every prior assembly entry has one static gateway signal and one sync.bar in the K loop with DPAS between them. Prior metadata declares SIMD16, GRF256 and one barrier resource; it is not a dynamic stall or occupancy measurement.',
            'No SLM message or fence opcode text was found in these prior disassemblies. That does not establish a universal helper contract or prove no private spill. The TLA private-memory warning remains a specific audit concern, not evidence of cross-subgroup shared storage.',
            'This is not a new current-runtime code capture. Same embedded IR alone does not attest every JIT option, runtime code patch, final native image or future compiler behavior.',
            'A private barrier experiment still requires a concrete reviewed change, Main-owned compiler/runtime correctness checks and separate speed samples. Model physical262144 consume and RESTORE remain separate unqualified gates.',
        ],
        'retention': {'owner': 'Main', 'consumer': 'Concrete Xe2 K-loop synchronization decision after current profile',
                      'next_review': 'Before any barrier experiment; keep compact facts and selected existing ISA, no new raw capture created.'},
    }
    target.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'receipt': str(target), 'entries': len(entries),
                      'same_image_offset': offset, 'GPU_executed': False}))


if __name__ == '__main__':
    main()
