"""Match traced prefill uploads to the IQ3_S model's independent pack index."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

INDEX_SHA256 = "b2a014878c3b61d6a6f915e194055a5e68ede34f9465dd1cb13249d902f008ac"
TOTAL_BYTES = 1_455_214_080
PTR = r"0x[0-9a-fA-F]+"
BEGIN = re.compile(rf"prefill canonical layer=(\d+) begin contiguous=([01]) src=({PTR})")
END = re.compile(rf"prefill canonical layer=(\d+) end contiguous=([01]) first=(\S+) dst=({PTR})")
RETURN = re.compile(r"<--- (ur\w+)\((.*)\) -> (UR_RESULT_\w+);")
COPY = re.compile(rf"\.hQueue = ({PTR}), \.blocking = (false|true|0|1), \.pDst = ({PTR}), "
                  rf"\.pSrc = ({PTR}), \.size = (\d+), ")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pack_layout(index):
    data = index.read_bytes()
    require(hashlib.sha256(data).hexdigest() == INDEX_SHA256, "unqualified pack index")
    layers = [[] for _ in range(48)]
    for line in data.decode().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        match = re.fullmatch(r"blk\.(\d+)\.(.*)", fields[0])
        size = int(fields[6])
        if not match or not size:
            continue
        layer, suffix = int(match[1]), match[2]
        if suffix in ("hc_attn_norm.weight", "hc_ffn_norm.weight") or (layer == 1 and suffix.startswith("ple_")):
            continue
        layers[layer].append((fields[0], int(fields[5]), size))
    for layer in layers:
        layer.sort()
    require(sum(map(len, layers)) == 672 and sum(size for rows in layers for _, _, size in rows) == TOTAL_BYTES,
            "unexpected per-layer tensor inventory")
    return layers


def compare_layer(rows, copies, src_base, dst_base, contiguous):
    """Consume whole published tensors in order; inside a call both address ranges must touch."""
    row = dst_offset = 0
    for src, dst, size in copies:
        remaining = size
        require(size > 0, "zero-byte submission")
        while remaining:
            require(row < len(rows), "upload crosses layer end")
            name, src_offset, payload = rows[row]
            require((src, dst) == (src_base + src_offset, dst_base + dst_offset),
                    f"wrong source or destination at {name}")
            require(remaining >= payload and (contiguous or remaining == payload),
                    f"partial tensor or merged baseline upload at {name}")
            remaining -= payload
            src += payload
            dst += payload
            dst_offset += (payload + 255) & ~255
            row += 1
    require(row == len(rows), "missing final tensor")


def verify(lines, layers, contiguous):
    layer = count = total = blocking = 0
    active = finished = False
    copies = []
    src_base = dst_base = queue = None
    for raw in lines:
        line = raw.strip()
        if line.startswith("prefill canonical layer="):
            begin = BEGIN.fullmatch(line)
            end = END.fullmatch(line)
            marker = begin or end
            require(marker and int(marker[1]) == layer < len(layers) and int(marker[2]) == contiguous,
                    "out-of-order layer or wrong arm")
            if begin:
                address = int(begin[3], 16)
                require(not active and address and src_base in (None, address), "changed snapshot or nested begin")
                src_base, active, copies = address, True, []
            else:
                address = int(end[4], 16)
                require(active and address and dst_base in (None, address) and address != src_base and
                        end[3] == layers[layer][0][0], "wrong bank anchor or unexpected end")
                dst_base = address
                compare_layer(layers[layer], copies, src_base, dst_base, contiguous)
                active = False
                layer += 1
        elif line.startswith("COMPONENT_ONLY canonical="):
            require(not finished and not active and layer == len(layers), "unexpected fixture completion")
            finished = True
        elif active and "<--- ur" in line:
            call = RETURN.search(line)
            require(call and call[3] == "UR_RESULT_SUCCESS", "failed or malformed API return in activation")
            name = call[1]
            if name == "urEnqueueUSMMemcpy":
                fields = COPY.match(call[2])
                require(fields, "missing typed USM copy arguments")
                handle, dst, src = (int(fields[i], 16) for i in (1, 3, 4))
                size = int(fields[5])
                require(handle and queue in (None, handle), "unexpected submission queue")
                queue = handle
                copies.append((src, dst, size))
                require(len(copies) <= len(layers[layer]), "too many submissions in one layer")
                count += 1
                total += size
                blocking += fields[2] in ("true", "1")
            elif name.startswith(("urEnqueue", "urCommandBuffer")):
                require(name in ("urEnqueueEventsWait", "urEnqueueEventsWaitWithBarrier",
                                 "urEnqueueEventsWaitWithBarrierExt"), "unexpected device work: " + name)
    require(finished and count == (217 if contiguous else 672) and total == TOTAL_BYTES,
            "incomplete fixture, unexpected submission count or byte total")
    return dict(layers=layer, copies=count, requested_bytes=total, blocking_copies=blocking)


def timing(stdout, contiguous):
    samples = []
    clock = None
    for line in stdout.splitlines():
        if line.startswith("prefill canonical clock "):
            m = re.fullmatch(r"prefill canonical clock median_ns=(\d+) p99_ns=(\d+)", line)
            require(m and clock is None, "unexpected timer calibration")
            clock = [int(m[1]), int(m[2])]
        if line.startswith("prefill canonical sweep="):
            m = re.fullmatch(r"prefill canonical sweep=(\d+) contiguous=([01]) layers=48 call_ms=([\d.]+)", line)
            require(m and int(m[1]) == len(samples) and int(m[2]) == contiguous, "wrong sweep or copy mode")
            ms = float(m[3])
            require(math.isfinite(ms) and 10 <= ms < 30000, "ineligible activation batch")
            samples.append(ms)
    require(clock and 0 <= clock[0] <= clock[1] < 5000 and clock[0] < 1000 and len(samples) == 4,
            "incomplete sweeps or timer overhead")
    return dict(clock_ns=clock, first_timed_sweep_ms=samples[0], subsequent_ms=samples[1:],
                subsequent_median_ms=statistics.median(samples[1:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("combined_fixture_log", type=Path)
    parser.add_argument("arm", choices=["rows", "runs"])
    args = parser.parse_args()
    with args.combined_fixture_log.open(encoding="utf-8") as trace:
        print(json.dumps(verify(trace, pack_layout(args.index), args.arm == "runs")))
