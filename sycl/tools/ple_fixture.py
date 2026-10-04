#!/usr/bin/env python3
"""Generate ple_parity's artifact weights and ggml CPU graph capture.

The sparse dense.bin contains only the three regions read by ple_parity. It is
a diagnostic fixture, not a model pack. Requires numpy and pinned gguf-py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from gguf import GGUFReader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from strata_pack import canonicalise, pack_codes, reference_values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True, type=Path, help="Q2_0 shard 1")
    p.add_argument("--oracle", required=True, type=Path, help="ple_graph_oracle executable")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(str(args.gguf))
    tensors = {t.name: t for t in reader.tensors if t.name.startswith("blk.1.ple_")}
    def tensor(role):
        return tensors["blk.1.ple_" + role + ".weight"]
    def decode(role):
        t = tensor(role)
        return reference_values(t.tensor_type.name, t.data.tobytes()).astype("<f4").reshape(-1)
    key = tensor("key")
    if key.tensor_type.name != "Q2_0" or tensor("value").tensor_type.name != "BF16":
        raise ValueError("ple_parity requires the original Q2_0 key and BF16 value artifact")
    codes, scales, _, mapping = canonicalise(key.name, "Q2_0", key.data.tobytes())
    key_f32 = decode("key")
    if not np.array_equal(mapping.dec((codes, scales)).view(np.uint32), key_f32.view(np.uint32)):
        raise ValueError("canonical key differs from the original decoder")
    s16 = scales.astype("<f2")
    if not np.array_equal(s16.astype(np.float32), scales):
        raise ValueError("key scales are not exact FP16")
    regions = [(1017723776, pack_codes(codes, 2)), (1024277376, s16.tobytes()),
               (1025219456, decode("value").tobytes())]
    expected = [6553600, 819200, 26214400]
    with (args.out / "dense.bin").open("wb") as f:
        for (offset, data), n in zip(regions, expected):
            if len(data) != n: raise ValueError("unexpected fixture region size")
            f.seek(offset); f.write(data)
    N, HC, T, K, D = 2560, 4, 2, 4, 3
    rng = np.random.RandomState(7)
    emb = rng.normal(0, 0.5, T*N).astype("<f4")
    hidden = rng.normal(0, 0.5, T*HC*N).astype("<f4")
    hist = rng.normal(0, 0.2, (K-1)*D*HC*N).astype("<f4")
    with (args.out / "ple_in.bin").open("wb") as f:
        f.write(np.array([N, HC, T, K, D], dtype="<i4").tobytes())
        f.write(np.array([1e-6], dtype="<f4").tobytes())
        for a in [emb, hidden, key_f32, decode("value"), decode("norm_key"),
                  decode("norm_query"), decode("norm_conv"), decode("conv1d"), hist]:
            f.write(a.tobytes())
    subprocess.run([str(args.oracle.resolve()), str(args.out / "ple_in.bin"),
                    str(args.out / "ple_out.bin")], check=True)
    metadata = {"source_gguf": str(args.gguf.resolve()), "seed": 7, "tokens": T,
                "oracle": "pinned ggml CPU graph with real artifact weights dequantized to F32",
                "dense_bin": "sparse diagnostic fixture; only three PLE regions; not a model pack",
                "source_tensors": {name: hashlib.sha256(t.data.tobytes()).hexdigest()
                                   for name, t in tensors.items()},
                "capture_sha256": {name: hashlib.sha256((args.out/name).read_bytes()).hexdigest()
                                   for name in ["ple_in.bin", "ple_out.bin"]}}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("PASS PLE fixture generated:", args.out)


if __name__ == "__main__":
    main()
