from pathlib import Path
import collections
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True
B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
FROZEN = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010')
PACK = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
PRIMARY = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
C = W / 'bench/results/2026-10-10-current-native-pack-census'


def ident(p):
    return dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def stat(p):
    s = p.stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


class BoundedReader:
    def __init__(self, f):
        self.f = f
        self.total = 0

    def read(self, n):
        assert isinstance(n, int) and 0 <= n <= (32 << 20) - self.total, 'header read budget'
        data = self.f.read(n)
        assert len(data) == n, 'short header read'
        self.total += n
        return data

    def tell(self):
        return self.f.tell()


with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == '318711cf823eacb98fbe3d77f060080388da1487'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True).strip()
    assert not C.exists()
    reader_source = FROZEN / 'tools/gguf_reader.py'
    assert ident(reader_source)['sha256'] == '5ecf739bb3c2f07bc7f889f7f3a455e3dfdb40068409e10781a1a9b2be289b5e'
    loader = importlib.util.spec_from_file_location('frozen_pack_census_reader', reader_source)
    mod = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = mod
    loader.loader.exec_module(mod)
    manifest = PACK / 'native_experts.txt'
    manifest_bytes = manifest.read_bytes()
    assert 0 < len(manifest_bytes) < (64 << 10)
    lines = manifest_bytes.decode('utf-8').splitlines()
    assert lines[0].startswith('# strata native experts v3:')
    m = re.search(r'\(n_expert (\d+), total (\d+);', lines[0])
    assert m
    experts, declared_total = map(int, m.groups())
    assert experts == 512
    before = stat(PRIMARY)
    parsed = mod.GGUFFile.__new__(mod.GGUFFile)
    parsed.path = PRIMARY
    parsed.metadata = {}
    parsed.tensors = []
    parsed.version = 0
    parsed.alignment = 32
    with PRIMARY.open('rb') as fh:
        bounded = BoundedReader(fh)
        parsed._parse(bounded)
        header_bytes = bounded.total
    data_start = parsed._data_start
    assert header_bytes <= data_start <= (32 << 20)
    with PRIMARY.open('rb') as fh:
        header = fh.read(data_start)
    assert len(header) == data_start
    assert stat(PRIMARY) == before
    tensors = {t.name: t for t in parsed.tensors}
    assert len(tensors) == len(parsed.tensors), 'duplicate tensor names'
    rows = []
    cumulative = 0
    for line in lines[1:]:
        if not line or line.startswith('#'):
            continue
        values = line.split()
        assert len(values) == 8, 'this receipt is for the exact single-shard v3 pack'
        layer, gt, dt, packed_offset, blob, gate_off, up_off, down_off = map(int, values)
        assert layer == len(rows) and packed_offset == cumulative
        assert gt in (18, 21, 22, 23) and dt in (20, 42)
        roles = []
        for role, ty, shape, offset in (
            ('gate', gt, [2560, 640, 512], gate_off),
            ('up', gt, [2560, 640, 512], up_off),
            ('down', dt, [640, 2560, 512], down_off)):
            name = f'blk.{layer}.ffn_{role}_exps.weight'
            t = tensors[name]
            assert t.type_id == ty and t.shape == shape
            assert data_start + t.offset == offset
            nb = t.expected_bytes()
            assert nb is not None and nb % experts == 0
            assert offset <= before['bytes'] and nb <= before['bytes'] - offset
            roles.append(dict(role=role, tensor=name, type_id=ty, type_name=t.type_name,
                              shape=shape, absolute_offset=offset, tensor_bytes=nb,
                              bytes_per_expert=nb // experts,
                              block_geometry=mod.BLOCK_GEOMETRY[t.type_name]))
        assert sum(r['bytes_per_expert'] for r in roles) == blob
        assert roles[0]['bytes_per_expert'] == roles[1]['bytes_per_expert']
        rows.append(dict(layer=layer, gu_type=gt, down_type=dt, packed_offset=packed_offset,
                         packed_blob_bytes=blob, layer_all_experts_bytes=blob * experts,
                         roles=roles))
        cumulative += blob * experts
    assert len(rows) == 48 and cumulative == declared_total
    assert manifest.read_bytes() == manifest_bytes and stat(PRIMARY) == before
    pairs = []
    for key, count in sorted(collections.Counter((r['gu_type'], r['down_type']) for r in rows).items()):
        selected = [r for r in rows if (r['gu_type'], r['down_type']) == key]
        assert len({r['packed_blob_bytes'] for r in selected}) == 1
        pairs.append(dict(gu_type=key[0], down_type=key[1], layers=[r['layer'] for r in selected],
                          layer_count=count, experts=count * experts,
                          bytes_per_expert=selected[0]['packed_blob_bytes'],
                          all_layer_experts_bytes=sum(r['layer_all_experts_bytes'] for r in selected)))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = dict(created_utc=now, passed=True,
                   scope='Metadata-only census of current local native pack and primary GGUF header. No tensor payload, engine execution, GPU state or performance measurement. This is not an attestation that a model run loaded this pack.',
                   command=['python3', str(Path(__file__))], python=sys.version, pid=os.getpid(),
                   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   production_source_commit='495369cf4dc6e6da564095a1c26504dd3694e3f2',
                   reader_source=dict(path=str(reader_source), **ident(reader_source)),
                   layout_source=dict(path=str(FROZEN / 'src/kernels/cpu/expert_layout.cpp'), **ident(FROZEN / 'src/kernels/cpu/expert_layout.cpp')),
                   native_format_source=dict(path=str(FROZEN / 'src/kernels/cpu/native_expert.cpp'), **ident(FROZEN / 'src/kernels/cpu/native_expert.cpp')),
                   pack_manifest=dict(path=str(manifest), **ident(manifest)),
                   primary=dict(path=str(PRIMARY), **before, header_parsed_bytes=header_bytes,
                                header_plus_padding_bytes=data_start, header_plus_padding_sha256=hashlib.sha256(header).hexdigest()),
                   relevant_scalar_metadata={k: v for k, v in parsed.metadata.items()
                                             if isinstance(v, (int, float, bool)) and
                                             any(s in k for s in ('block_count', 'embedding_length', 'expert', 'head_count', 'key_length', 'value_length', 'ssm.', 'rope.dimension_count', 'context_length'))},
                   full_shard_hash_computed=False, header_read_budget_bytes=32 << 20,
                   expert_layers=48, experts_per_layer=experts, expert_blobs=48 * experts,
                   distinct_paired_formats=len(pairs), paired_formats=pairs,
                   total_all_layer_packed_bytes=cumulative,
                   maximum_packed_blob_bytes=max(r['packed_blob_bytes'] for r in rows),
                   layer_cache_MAXBLOB_times512_bytes=max(r['packed_blob_bytes'] for r in rows) * experts,
                   dequant_FP16_GU_output_bytes_per_expert=1280 * 2560 * 2,
                   dequant_FP16_Down_output_bytes_per_expert=2560 * 640 * 2,
                   per_layer=rows,
                   qualification='All144role descriptors match pack types, shapes and absolute offsets; per-role tensor extents lie within unchanged primary file; independent header block equations sum to all48declared blobs and contiguous packed total.',
                   limitations='No payload hash/correctness, allocator/cache/device-free-memory attestation, selected prefill route, expert row histogram, DMA/service timing or full-context inference. Layer cache bytes are size arithmetic only. Dequant output counts apply to the FP16 dequant route, not fused native or MMQ paths.')
    C.mkdir()
    (C / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (C / 'native_experts.txt').write_bytes(manifest_bytes)
    text = '''# Current native pack metadata census

CPU-only header inspection of the local IQ3_S pack on 2026-10-10. All 144 role descriptors across 48 layers match the primary GGUF header's types, shapes and absolute offsets. Per-role block equations sum to every declared expert blob and the contiguous packed total. No tensor payload or GPU state was read, and no engine/model run was performed.

| GU type | Down type | Layers | Bytes per expert |
| --- | --- | ---: | ---: |
'''
    for pair in pairs:
        text += f"| {mod.GGML_TYPES[pair['gu_type']]} ({pair['gu_type']}) | {mod.GGML_TYPES[pair['down_type']]} ({pair['down_type']}) | {pair['layer_count']} | {pair['bytes_per_expert']:,} |\n"
    text += f'''
The pack contains 512 experts per layer, {48 * experts:,} expert blobs, and {cumulative:,} total packed bytes. The maximum blob is {receipt['maximum_packed_blob_bytes']:,} bytes, giving {receipt['layer_cache_MAXBLOB_times512_bytes']:,} bytes for the frozen layer-cache slot-size scenario. This verifies current local metadata matching the historical size arithmetic; it does not prove that a future run loaded the pack or that its simultaneous VRAM allocations fit.

The previous 384-expert CPU capacity fixture covers selected IQ2_S/IQ4_NL experts in 15 layers. It cannot stand in for all paired formats. The production FP16 path writes 6,553,600 bytes of GU weights and 3,276,800 bytes of Down weights per dequantized expert, but layer-weight reuse alone does not prove that dequantization is reused: the source has a two-slot dequant ring and actual routed calls still need counting. Fused native/MMQ paths have different service and eligibility.

[The receipt](receipt.json) preserves all per-layer descriptors, source/manifest/header hashes, unchanged file identity and interpretation limits. The whole large shard was not hashed; the header hash covers only metadata plus padding. This is a next-control admission input, not an actual row-routing census, service result, payload qualification or full-context proof.
'''
    (C / 'README.md').write_text(text)
    (C / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    for args in (('diff', '--check'), ('add', str(C.relative_to(W))), ('diff', '--cached', '--check'),
                 ('commit', '-m', 'bench: pin all native pack role formats and dequant extents'), ('push',)):
        subprocess.run(['git', *args], cwd=W, check=True)
    print(json.dumps(dict(passed=True, paired_formats=pairs, total_bytes=cumulative,
                          header_parsed_bytes=header_bytes, receipt=ident(C / 'receipt.json'),
                          commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip())))
