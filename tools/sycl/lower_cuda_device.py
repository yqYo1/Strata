#!/usr/bin/env python3
"""Bind original core device/arena logic to explicit native SYCL metadata."""
import argparse,hashlib,json,re
from pathlib import Path
from lower_cuda_products import lower

def function(text,name,body):
    # Locate the literal original function and retain its public signature.
    match=re.search(r'\b'+re.escape(name)+r'\([^;{}]*\)\s*\{',text)
    assert match,name
    start=match.end();depth=1;end=start
    while depth:
        depth += (text[end]=='{')-(text[end]=='}');end+=1
    return text[:start]+'\n'+body+'\n'+text[end-1:]

def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or path.read_text()!=text:path.write_text(text)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    inventory=json.loads((root/'docs/sycl-audit/source-inventory.json').read_text())
    hashes={x['path']:x['upstream_sha256'] for x in inventory['files']}
    records={}
    def checked(path):
        data=(root/path).read_bytes();assert hashlib.sha256(data).hexdigest()==hashes[path],path
        return data.decode()
    path='src/core/device.cu';text=checked(path)
    text=function(text,'compiled_gpu_archs','    return sycl_upstream::compiled_device_archs();')
    text=function(text,'device_summary','''    sycl_upstream::DeviceFacts facts;
    if (ordinal < 0 || ordinal >= device_count() ||
        sycl_upstream::query_device_facts(ordinal, &facts) != cudaSuccess) {
        cudaGetLastError(); return false;
    }
    name = facts.name;
    detail = sycl_upstream::device_summary_detail(facts);
    return true;''')
    text=function(text,'gpu_arch_problem','    return sycl_upstream::device_arch_problem(ordinal);')
    text=function(text,'device_code_error','''    int ordinal = 0;
    if (cudaGetDevice(&ordinal) != cudaSuccess) return "cannot read current native device";
    if (const auto why = gpu_arch_problem(ordinal); !why.empty()) return why;
    try {
        sycl::device device;
        check(sycl_upstream::cuda::inspect_device(ordinal, [&](const sycl::device& d) { device = d; }), "native device");
        sycl::context context;
        check(sycl_upstream::cuda::device_context(ordinal, &context), "native context");
        const auto id = sycl::get_kernel_id<sycl_upstream::OriginalDevicePoison>();
        const auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(context, {device}, {id});
        return bundle.has_kernel(id, device) ? "" : "original native poison image is absent for this device";
    } catch (const std::exception& e) { cudaGetLastError(); return e.what(); }''')
    text=function(text,'device_info','''    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) throw CudaError("no native Level Zero GPU device is present", -1);
    if (ordinal < 0 || ordinal >= count)
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " + std::to_string(count) + ")", -1);
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    sycl_upstream::DeviceFacts facts;
    check(sycl_upstream::query_device_facts(ordinal, &facts), "native device metadata");
    DeviceInfo d;
    d.ordinal = ordinal; d.name = facts.name; d.arch = facts.architecture;
    // The original public struct's CUDA-only capability/version fields stay
    // inapplicable (zero). Native compute units/versions have explicit facts.
    size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    d.free_bytes = free_b; d.total_bytes = total_b;
    if (const auto why = gpu_arch_problem(ordinal); !why.empty()) throw CudaError(why, -1);
    return d;''')
    # HIP-only definitions remain under their original inactive guards. Do not
    # define STRATA_USE_HIP or fabricate CUDA function/property declarations.
    text,record=lower(text,namespace='strata::core')
    launch=re.search(r'sycl_upstream::cuda_kernel::launch<true, false>\(.*?\}\);',text,re.S)
    assert launch and text.count('sycl_upstream::cuda_kernel::launch<')==1
    replacement='''(void) sycl_upstream::cuda::submit(nullptr,
                [=, strata_pointer = ((float*) base_ + b * threads), strata_count = (n - b * threads)](sycl::queue& q) {
                    return q.submit([&](sycl::handler& h) {
                        h.parallel_for<sycl_upstream::OriginalDevicePoison>(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, size_t(chunk) * threads), sycl::range<3>(1, 1, threads)),
                            [=](sycl::nd_item<3>) { poison_kernel(nullptr, strata_pointer, strata_count); });
                    });
                });'''
    text=text[:launch.start()]+replacement+text[launch.end():]
    text='#include "strata/sycl_upstream/device_metadata.hpp"\n'+text
    out=args.output/'original_core_device.cpp';write(out,text)
    records[path]={'original_sha256':hashes[path],'generated_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),**record,
                   'scope':'Original arena allocation/bump/destruction/poison loops retained. Native metadata and executable named-poison bundle replace backend-specific CUDA property/version/function-attribute queries.'}
    path='src/core/device_main.cpp';text=checked(path)
    start=text.index('#if defined(STRATA_USE_HIP)\n        std::printf("  HIP arch')
    end=text.index('#if defined(STRATA_USE_HIP) && defined(_WIN32)',start)
    text=text[:start]+'''        strata::sycl_upstream::DeviceFacts facts;
        if (strata::sycl_upstream::query_device_facts(d.ordinal, &facts) != cudaSuccess)
            throw std::runtime_error("cannot query native device metadata");
        std::printf("  SYCL architecture   %s (compiled for %s)\\n", facts.architecture.c_str(), strata::core::compiled_gpu_archs());
        std::printf("  compute units       %u, max clock %u MHz\\n", facts.compute_units, facts.max_clock_mhz);
        std::printf("  VRAM total / free   %s / %s\\n", human(d.total_bytes).c_str(), human(d.free_bytes).c_str());
        std::printf("  native driver       %s\\n", facts.driver_version.c_str());
        std::printf("  platform version    %s\\n", facts.platform_version.c_str());
''' + text[end:]
    text='#include "strata/sycl_upstream/device_metadata.hpp"\n'+text
    text=text.replace('numbered as HIP_VISIBLE_DEVICES /\\n"\n                        "                  CUDA_VISIBLE_DEVICES number them','numbered by the native Level Zero selector')
    out=args.output/'original_core_device_main.cpp';write(out,text)
    records[path]={'original_sha256':hashes[path],'generated_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                   'scope':'Original device/plan/arena CLI; native metadata display replaces CUDA-specific labels.'}
    write(args.output/'original-device-lowering.json',json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
