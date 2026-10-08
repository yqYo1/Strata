"""Additional defined representation/atomic reads and supported-launch guards."""
from pathlib import Path
import datetime,difflib,hashlib,json
base=Path(__file__).parent
root=Path("/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05")
original=root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/kernels/cuda/kv_stream.dp.cpp'
previous=root/'build-sycl-kv-stream-safe-v2-20261008/source/kv_stream.dp.cpp'
out=root/'build-sycl-kv-stream-safe-v3-20261008/source'
out.mkdir(parents=True)
receipt=base/'kv-stream-safe-v01402-source-candidate-v3';receipt.mkdir(mode=0o700)
text=previous.read_text();changed=text
def replace(a,b):
    global changed
    assert changed.count(a)==1, a[:100]
    changed=changed.replace(a,b)
replace('            const int sl = m.page_table[b];',"""            // Another query lane can concurrently claim this page.
            const int sl =
                sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                    m.page_table[b]).load();""")
replace('''        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;''','''        // Only lane zero writes these counters, after the work-group phases.
        // Copy representations instead of aliasing the int32_t control storage.
        static_assert(sizeof(unsigned long long) == 8);
        const unsigned long long increments[3] = {
            (unsigned long long) placed, (unsigned long long) s_lookups, 1ull};
        for (int c = 0; c < 3; ++c) {
            unsigned long long value;
            std::memcpy(&value, m.ctl + 4 + 2 * c, sizeof(value));
            value += increments[c];
            std::memcpy(m.ctl + 4 + 2 * c, &value, sizeof(value));
        }''')
replace('''            const sycl::uint4 *src =
                reinterpret_cast<const sycl::uint4 *>(r.src[a] + b * r.len[a]);
            sycl::uint4 *dst =
                reinterpret_cast<sycl::uint4 *>(r.dst[a] + sl * r.len[a]);
#pragma unroll
            for (int i = item_ct1.get_local_id(2); i < r.len[a] / 16;
                 i += item_ct1.get_local_range(2)) dst[i] = src[i];''','''            const uint8_t* src = r.src[a] + b * r.len[a];
            uint8_t* dst = r.dst[a] + sl * r.len[a];
#pragma unroll
            for (int i = item_ct1.get_local_id(2); i < r.len[a] / 16;
                 i += item_ct1.get_local_range(2)) {
                // Preserve each 16-byte transaction without typed vector aliasing.
                sycl::uint4 value;
                std::memcpy(&value, src + 16 * i, sizeof(value));
                std::memcpy(dst + 16 * i, &value, sizeof(value));
            }''')
replace('#include <cstring>', '#include <cstring>\n#include <stdexcept>')
replace('''    if (n_q <= 0) return;
    if (s.n_head_kv''','''    if (n_q <= 0) return;
    const auto device = strata::q_of(stream)->get_device();
    const auto subgroups = device.get_info<sycl::info::device::sub_group_sizes>();
    if (!device.has(sycl::aspect::usm_host_allocations) ||
        device.get_info<sycl::info::device::max_work_group_size>() < RT ||
        std::find(subgroups.begin(), subgroups.end(), size_t{32}) == subgroups.end())
        throw std::runtime_error(
            "kv_stream: host USM, WG1024 and SG32 support are required");
    if (s.n_head_kv''')
p=out/'kv_stream.dp.cpp';p.write_text(changed)
(receipt/'candidate.diff').write_text(''.join(difflib.unified_diff(original.read_text().splitlines(True),changed.splitlines(True),fromfile=str(original),tofile=str(p))))
record=dict(active=False,source_prepared=True,compiled=False,gpu_tested=False,adopted=False,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    previous_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    changes=['Inherit v2 hit metadata atomics, host counter memcpy and four root property removals.',
             'Atomic page-table load paired with the existing atomic claim across concurrent query lanes.',
             'Device counter and 16-byte KV copy representation transfers through memcpy.',
             'Before resolve require host USM, WG1024 and subgroup32 capabilities. Runtime validates kernel-specific launch limits.'],
    scope='Candidate source only; no GPU or performance claim. Initial model gate must use chunk-major: existing layer-major rejects kv_mode1 before processing.')
(receipt/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

