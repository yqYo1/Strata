"""Build an exhaustive device-helper comparison; do not execute it here."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = base/'qsa-vector-load-gpu'
out.mkdir(mode=0o700)
cpu = base/'qsa-vector-load-cpu'
receipt = json.loads((cpu/'record.json').read_text())
assert receipt['passed']
prefix = (cpu/'fixture.cpp').read_text().split('int main() {', 1)[0]
prefix += '#include <vector>\n#include <optional>\n#include <stdexcept>\n'
fixture = prefix + r'''
int main() try {
    constexpr int B = 2048;
    std::optional<sycl::device> selected;
    for (const auto &device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (device.get_backend() == sycl::backend::ext_oneapi_level_zero &&
            device.get_info<sycl::ext::intel::info::device::pci_address>() == "0000:05:00.0")
            selected = device;
    }
    if (!selected) throw std::runtime_error("B570 PCI device was not found");
    sycl::queue queue(*selected, [](sycl::exception_list errors) {
        for (auto error : errors) std::rethrow_exception(error);
    }, sycl::property::queue::in_order{});
    std::printf("device 0000:05:00.0: %s\n", selected->get_info<sycl::info::device::name>().c_str());
    std::fflush(stdout);
    std::vector<uint16_t> k16(B*HD), v16(B*HD), ks(B*HD/KV_Q8_GROUP), vs(B*HD/KV_Q8_GROUP);
    std::vector<int8_t> k8(B*HD), v8(B*HD);
    std::vector<float> output(B*64);
    QsaAttnPools pools;
    pools.k_pool = sycl::malloc_device<uint16_t>(k16.size(), queue);
    pools.v_pool = sycl::malloc_device<uint16_t>(v16.size(), queue);
    pools.k_q = sycl::malloc_device<int8_t>(k8.size(), queue);
    pools.v_q = sycl::malloc_device<int8_t>(v8.size(), queue);
    pools.k_scale = sycl::malloc_device<uint16_t>(ks.size(), queue);
    pools.v_scale = sycl::malloc_device<uint16_t>(vs.size(), queue);
    float *result = sycl::malloc_device<float>(output.size(), queue);
    if (!pools.k_pool || !pools.v_pool || !pools.k_q || !pools.v_q || !pools.k_scale || !pools.v_scale || !result)
        throw std::runtime_error("device allocation failed");
    const int8_t codes[8] = {-128,-127,-1,0,1,2,126,127};
    uint64_t mismatches=0, comparisons=0, non_nan_checks=0, nan_class_checks=0;
    for (unsigned first=0; first<65536; first+=B) {
        for (int row=0; row<B; ++row) {
            const unsigned pattern=first+row;
            const int d0=((pattern/3)%32)*8;
            for (int j=0; j<8; ++j) {
                k16[row*HD+d0+j]=uint16_t(pattern+8191*j);
                v16[row*HD+d0+j]=uint16_t(pattern+8191*j)^0x8000;
                k8[row*HD+d0+j]=codes[(j+pattern)%8];
                v8[row*HD+d0+j]=codes[(7-j+pattern)%8];
            }
            ks[row*(HD/KV_Q8_GROUP)+d0/KV_Q8_GROUP]=uint16_t(pattern);
            vs[row*(HD/KV_Q8_GROUP)+d0/KV_Q8_GROUP]=uint16_t(pattern)^0x8000;
        }
        queue.memcpy(const_cast<uint16_t*>(pools.k_pool),k16.data(),k16.size()*2);
        queue.memcpy(const_cast<uint16_t*>(pools.v_pool),v16.data(),v16.size()*2);
        queue.memcpy(const_cast<int8_t*>(pools.k_q),k8.data(),k8.size());
        queue.memcpy(const_cast<int8_t*>(pools.v_q),v8.data(),v8.size());
        queue.memcpy(const_cast<uint16_t*>(pools.k_scale),ks.data(),ks.size()*2);
        queue.memcpy(const_cast<uint16_t*>(pools.v_scale),vs.data(),vs.size()*2);
        queue.parallel_for(sycl::range<1>(B), [=](sycl::id<1> item) {
            const int row=item[0], d0=(((first+row)/3)%32)*8;
            float *values=result+row*64;
            for (int value=0; value<2; ++value) {
                original::load8_f16(pools,bool(value),row,d0,values+value*32);
                candidate::load8_f16(pools,bool(value),row,d0,values+value*32+8);
                original::load8_q8(pools,bool(value),row,d0,values+value*32+16);
                candidate::load8_q8(pools,bool(value),row,d0,values+value*32+24);
            }
        });
        queue.memcpy(output.data(), result, output.size()*sizeof(float));
        queue.wait_and_throw();
        for (int row=0; row<B; ++row) {
            const int d0=(((first+row)/3)%32)*8;
            for (int value=0; value<2; ++value) {
                const float *values=output.data()+row*64+value*32;
                for (int j=0; j<8; ++j) {
                    const float reference16=ieee_half((value?v16:k16)[row*HD+d0+j]);
                    const float reference8=float((value?v8:k8)[row*HD+d0+j]) *
                        ieee_half((value?vs:ks)[row*(HD/KV_Q8_GROUP)+d0/KV_Q8_GROUP]);
                    mismatches += !exact_or_nan(values[j],reference16);
                    mismatches += !exact_or_nan(values[8+j],reference16);
                    mismatches += !exact_or_nan(values[16+j],reference8);
                    mismatches += !exact_or_nan(values[24+j],reference8);
                    mismatches += !exact_or_nan(values[j],values[8+j]);
                    mismatches += !exact_or_nan(values[16+j],values[24+j]);
                    comparisons += 6;
                    non_nan_checks += !std::isnan(reference16)+!std::isnan(reference8);
                    nan_class_checks += std::isnan(reference16)+std::isnan(reference8);
                }
            }
        }
        std::printf("patterns completed: %u/65536, mismatches: %llu\n", first+B,
                    (unsigned long long)mismatches);
        std::fflush(stdout);
    }
    sycl::free(const_cast<uint16_t*>(pools.k_pool),queue);
    sycl::free(const_cast<uint16_t*>(pools.v_pool),queue);
    sycl::free(const_cast<int8_t*>(pools.k_q),queue);
    sycl::free(const_cast<int8_t*>(pools.v_q),queue);
    sycl::free(const_cast<uint16_t*>(pools.k_scale),queue);
    sycl::free(const_cast<uint16_t*>(pools.v_scale),queue);
    sycl::free(result,queue);
    std::printf("{\"patterns\":65536,\"comparisons\":%llu,\"non_nan_bit_checks\":%llu,"
                "\"nan_class_checks\":%llu,\"mismatches\":%llu}\n",
                (unsigned long long)comparisons,(unsigned long long)non_nan_checks,
                (unsigned long long)nan_class_checks,(unsigned long long)mismatches);
    return mismatches ? 1 : 0;
} catch (const std::exception &error) {
    std::fprintf(stderr,"fixture failed: %s\n",error.what());
    return 2;
}
'''
(out/'fixture.cpp').write_text(fixture)
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
argv=['/opt/intel/oneapi/compiler/2026.1/bin/icpx','-fsycl','-std=c++20','-O3',
      '-fp-model=precise','-fstrict-aliasing','-fsycl-default-sub-group-size=32',
      '-fsycl-device-code-split=per_kernel','-I'+str(root/'include'),str(out/'fixture.cpp'),
      '-o',str(out/'fixture')]
record={'scope':'Offline compilation of exact original/candidate helper GPU fixture; execution not performed',
        'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fixture_sha256':hashlib.sha256(fixture.encode()).hexdigest(),'argv':argv,
        'original_source_sha256':receipt['original_source_sha256'],
        'candidate_source_sha256':receipt['candidate_source_sha256']}
started=time.monotonic()
try:
    with (out/'compile.stdout').open('wb') as stdout,(out/'compile.stderr').open('wb') as stderr:
        completed=subprocess.run(argv,env=env,stdout=stdout,stderr=stderr,timeout=180)
    record['exit_code']=completed.returncode
    assert completed.returncode==0
    record['binary_sha256']=hashlib.sha256((out/'fixture').read_bytes()).hexdigest()
    record['passed']=True
finally:
    record['elapsed_seconds']=time.monotonic()-started
    record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
