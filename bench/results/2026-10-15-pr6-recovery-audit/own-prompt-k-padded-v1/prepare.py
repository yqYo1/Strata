from pathlib import Path
import hashlib,json,difflib
D=Path(__file__).resolve().parent
pins=json.loads((D/'snapshot.json').read_text())
files={}
for p,h in pins.items():
 b=(D/'base'/p).read_bytes();assert hashlib.sha256(b).hexdigest()==h
 files[p]=[b.decode(),b.decode()]
p='src/kernels/sycl/prompt_attention.cpp';s=files[p][0]
s=s.replace('#include <cstdlib>','#include <cstdlib>\n#include <stdexcept>\n#include <type_traits>')
s=s.replace('template <int Mode, bool Direct, bool Native = false, bool CellMajor = false>', '''// One genuine half array: no integer alias or pointer arithmetic across rows.
struct alignas(4) PaddedK {
  sycl::half values[HD * 40];
};
static_assert(alignof(PaddedK) >= 4 && sizeof(PaddedK) == HD * 40 * sizeof(sycl::half));
static_assert(std::is_standard_layout_v<PaddedK> && std::is_trivially_copyable_v<PaddedK>);
static_assert(sycl::is_device_copyable_v<PaddedK>);
template <int Mode, bool Direct, bool Native = false, int KMapping = 0>''')
s=s.replace('static_assert(!CellMajor || (Mode == 1 && Direct && !Native));','''static_assert(KMapping >= 0 && KMapping <= 2);
  static_assert(KMapping == 0 || (Mode == 1 && Direct && !Native));
  constexpr int KP = KMapping == 2 ? 40 : CH;''')
s=s.replace('''    sycl::local_accessor<sycl::half, 1> qhi(16 * HD, h), qlo(16 * HD, h),
        kv(CH * HD, h);''','''    sycl::local_accessor<sycl::half, 1> qhi(16 * HD, h), qlo(16 * HD, h);
    using KVStorage = std::conditional_t<KMapping == 2, PaddedK, sycl::half>;
    sycl::local_accessor<KVStorage, 1> kv(KMapping == 2 ? 1 : CH * HD, h);''')
s=s.replace('''          auto sg = it.get_sub_group();''','''          auto sg = it.get_sub_group();
          const auto kvp = [&]() {
            if constexpr (KMapping == 2) {
              // This points into the sole local object's single big half array.
              // Explicitly retain local address space for joint_matrix_load.
              return sycl::address_space_cast<sycl::access::address_space::local_space,
                  sycl::access::decorated::no>(&kv[0].values[0]);
            } else {
              return kv.template get_multi_ptr<sycl::access::decorated::no>();
            }
          }();''')
s=s.replace('if constexpr (CellMajor)', 'if constexpr (KMapping != 0)')
s=s.replace('kv[dim * CH + c]', 'kvp[dim * KP + c]').replace('kv[i] =','kvp[i] =')
s=s.replace('''                      kv.template get_multi_ptr<sycl::access::decorated::no>() +
                          k * CH + cell,
                      CH);''','''                      kvp + k * KP + cell,
                      KP);''')
s=s.replace('''                    kv.template get_multi_ptr<sycl::access::decorated::no>() +
                        k * HD + dim,''','''                    kvp + k * HD + dim,''')
a='''      if (cell_major && cell_major[0] == '1' && cell_major[1] == '\\0') {
        launch<1, true, false, true>'''
b='''      if (cell_major && cell_major[0] == '2' && cell_major[1] == '\\0') {
        // Preserve existing matrix support gate; never silently relabel arm 0/1.
        if (queue_for(stream).get_device().get_info<sycl::info::device::local_mem_size>() <
            size_t(60880 + 4096))
          throw std::runtime_error("prompt K pitch40 requires local memory >=64976 bytes");
        launch<1, true, false, 2>(q, p, ids, steps, cap, s, out, nq, stream);
        return;
      }
      if (cell_major && cell_major[0] == '1' && cell_major[1] == '\\0') {
        launch<1, true, false, 1>'''
assert a in s;s=s.replace(a,b)
assert 'CellMajor' not in s and 'k * CH + cell' not in s
files[p][1]=s
p='tests/sycl/prompt_k_cellmajor.cpp';s=files[p][0]
a='''    check(old==candidate,"logical K tile exact");
    for(int w:writes) check(w==1,"logical tile unique complete coverage");'''
b='''    check(old==candidate,"logical K tile exact");
    for(int w:writes) check(w==1,"logical tile unique complete coverage");
    constexpr int KP = 40;
    constexpr uint16_t sentinel = 0x7e55;
    std::vector<uint16_t> padded(HD*KP, sentinel);
    std::vector<int> padded_writes(HD*KP);
    for (int tid=0;tid<WG;++tid) for(int i=tid;i<HD*CH;i+=WG) {
      const int c=i/HD,d=i%HD;
      padded[d*KP+c]=half(rows[c]>=0 ? float(pool[rows[c]*HD+d]) : 0.f);
      ++padded_writes[d*KP+c];
    }
    for(int d=0;d<HD;++d) for(int c=0;c<KP;++c) {
      if(c<CH) {
        check(padded[d*KP+c]==old[d*CH+c], "pitch40 logical tile exact");
        check(padded_writes[d*KP+c]==1, "pitch40 unique complete coverage");
      } else {
        check(padded[d*KP+c]==sentinel && padded_writes[d*KP+c]==0,
              "pitch40 padding untouched");
      }
    }
    // Enumerate each QK matrix-load footprint: no padding may be consumed.
    for(int k=0;k<HD;k+=16) for(int cell : {0,16}) {
      check(((k*KP+cell)*sizeof(uint16_t))%4==0, "QK offset alignment");
      for(int r=0;r<16;++r) for(int c=0;c<16;++c) {
        check(cell+c<CH && (k+r)*KP+cell+c<HD*KP, "QK footprint bounds");
        check(padded[(k+r)*KP+cell+c]==old[(k+r)*CH+cell+c],
              "QK pitch40 consumed operand exact");
      }
    }'''
assert a in s;s=s.replace(a,b);files[p][1]=s
patch='';m={'status':'SOURCE_ONLY_UNBUILT_UNTESTED','base':'hashed exact working source snapshot in base/; no HEAD assumption','files':{}}
for p,(old,new) in files.items():
 out=D/'files'/p;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(new)
 m['files'][p]={'base_sha256':hashlib.sha256(old.encode()).hexdigest(),'proposal_sha256':hashlib.sha256(new.encode()).hexdigest()}
 if old!=new:patch+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+p,tofile='b/'+p))
(D/'source.diff').write_text(patch);m['diff_sha256']=hashlib.sha256(patch.encode()).hexdigest();(D/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
