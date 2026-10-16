#!/usr/bin/env python3
"""UNRUN fixture source generator. Main runs it only after reviewing/applying the patch.

Read the actual selected source; extract Stager, inline issuer and release bodies.
Write a CPU-only fixture to a private output path supplied by Main. Never compile/run.
"""
from pathlib import Path
import argparse
import hashlib

def region(text, start, end):
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = (args.root / "src/prefill/prefill.cpp").read_text()
    stager = region(source, "struct Stager {", "\n// multi-GPU: the peer GPU's share")
    if "bool init(size_t blob_bytes, int nthreads) {" not in stager:
        raise RuntimeError("fixture targets guard-only Stager init; adapt explicitly after affinity composition")
    issuer = region(source, "        auto issue_until =", "        // D-5: the stream is issued")
    release = region(source, "void Prefill::release() {", "\nnamespace {\nconstexpr int64_t GEMM_SCRATCH")
    for required in ["Stager::work host-slot reuse", "Stager::issued_one host-slot DMA", "Stager::work device"]:
        if required not in stager:
            raise RuntimeError(f"missing actual Stager guard: {required}")
    if issuer.count("STRATA_PREFILL_STAGE_CHECK") != 4 or release.count("STRATA_PREFILL_STAGE_CHECK") != 2:
        raise RuntimeError("issuer/release guarded source differs; review and adapt fixture")
    release = release.replace("void Prefill::release()", "void extracted_release(FakeImpl* impl_)", 1)
    template = Path(__file__).with_name("cpu_fixture_template.cpp.in").read_text()
    fixture = template.replace("@STAGER@", stager).replace("@ISSUER@", issuer).replace("@RELEASE@", release)
    fixture = "// Actual source SHA256: " + hashlib.sha256(source.encode()).hexdigest() + "\n" + fixture
    args.output.write_text(fixture)
    print(f"CPU fixture source written, not compiled/run: {args.output}; sha256={hashlib.sha256(fixture.encode()).hexdigest()}")

if __name__ == "__main__":
    main()
