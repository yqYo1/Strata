#!/usr/bin/env python3
"""Bind pinned whole original generate.cpp to explicit native GPU metadata."""
import argparse
import hashlib
import json
from pathlib import Path


def replace(text, old, new, count=1):
    assert text.count(old) == count, old
    return text.replace(old, new)


def write(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = 'src/program/generate.cpp'
    inventory = json.loads((root / 'docs/sycl-audit/source-inventory.json').read_text())
    expected = next(x['upstream_sha256'] for x in inventory['files'] if x['path'] == source)
    data = (root / source).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, source
    text = data.decode()
    text = replace(text, 'cudaDeviceProp prop{};', 'strata::sycl_upstream::DeviceFacts prop;', 2)
    text = replace(text, 'cudaGetDeviceProperties(&prop, d)', 'strata::sycl_upstream::query_device_facts(d, &prop)')
    text = replace(text, 'prop.totalGlobalMem', 'prop.total_memory_bytes')
    text = replace(text, 'cudaDeviceProp dp{};', 'strata::sycl_upstream::DeviceFacts dp;')
    text = replace(text, 'cudaGetDeviceProperties(&dp, 0)', 'strata::sycl_upstream::query_device_facts(0, &dp)')
    text = replace(text, 'dp.name,', 'dp.name.c_str(),', 2)
    text = replace(text, 'cudaGetDeviceProperties(&prop, dev)', 'strata::sycl_upstream::query_device_facts(dev, &prop)')
    text = replace(text, 'prop.name[0] = 0;', 'prop.name.clear();')
    text = replace(text, 'dev, prop.name,', 'dev, prop.name.c_str(),')

    # Retain the original placement formula and expose its CUDA calibration
    # in the native diagnostic. Compute units do not pretend to be CUDA SMs.
    old = '''            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            if (cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) != cudaSuccess || khz <= 0) khz = 1800000;'''
    new = '''            strata::sycl_upstream::DeviceFacts facts;
            if (strata::sycl_upstream::query_device_facts(dev, &facts) == cudaSuccess) {
                sms = (int) facts.compute_units;
                khz = (int) facts.max_clock_mhz * 1000;
            }
            if (khz <= 0) khz = 1800000;'''
    text = replace(text, old, new)
    text = replace(text, '// SMs x GHz', '// Native compute units x GHz; original CUDA calibration is unvalidated here.')
    text = replace(text, 'CUDA%d %d SMs at %.2f GHz -> %.2f ms per layer, ',
                   'native GPU%d %d compute units at %.2f GHz -> %.2f ms per layer (CUDA-calibrated estimate), ')

    # Replace only the original startup backend diagnostic/admission block.
    # The same whole driver, feature branches, planner and engine calls remain.
    start = text.index('        cudaDeviceProp p{};')
    end = text.index('    strata::core::ArenaExpertSource arena_src;', start)
    assert text[start:end].count('strata::core::device_code_error()') == 1
    text = text[:start] + '''        strata::sycl_upstream::DeviceFacts facts;
        const bool named = cudaGetDevice(&dev) == cudaSuccess &&
            strata::sycl_upstream::query_device_facts(dev, &facts) == cudaSuccess;
        if (!named) cudaGetLastError();
        const char* name = named && !facts.name.empty() ? facts.name.c_str() : "(an unnamed GPU)";
        std::fprintf(stderr, "strata generate: native GPU %d: %s, architecture %s, compiled for %s\\n",
                     dev, name, named ? facts.architecture.c_str() : "?", strata::core::compiled_gpu_archs());
        if (named)
            std::fprintf(stderr, "strata generate: native driver %s, platform %s\\n",
                         facts.driver_version.c_str(), facts.platform_version.c_str());
        const std::string e = strata::core::device_code_error();
        if (!e.empty()) {
            std::fprintf(stderr, "strata generate: original native device image unavailable for %s (%s): %s - "
                                 "rebuild for this SYCL architecture\\n", name, facts.architecture.c_str(), e.c_str());
            return 1;
        }
    }
''' + text[end:]
    text = '#include "strata/sycl_upstream/device_metadata.hpp"\n' + text
    assert 'cudaDeviceProp' not in text and 'cudaGetDeviceProperties' not in text
    assert 'CUDART_VERSION' not in text and 'cudaDeviceGetAttribute' not in text
    args.output.mkdir(parents=True, exist_ok=True)
    generated = args.output / 'original_program_generate.cpp'
    write(generated, text)
    record = {source: {'original_sha256': expected, 'generated_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(),
                      'scope': 'Whole original driver. Four property sites and the placement hardware query use native facts; startup metadata/image diagnostics use native admission. Original multi-GPU CUDA timing calibration is retained but unvalidated for native compute units. Object compilation is not engine linkage or execution.'}}
    write(args.output / 'original-program-lowering.json', json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
