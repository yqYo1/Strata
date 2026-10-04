from pathlib import Path
s=Path('/tmp/strata-sycl-goal-signindex-v2.cpp').read_text().replace('signindexed_iq3s','block_iq3_s')
s=s.replace('const uint16_t * GGML_RESTRICT qs = x[i].indices;','const uint8_t * GGML_RESTRICT qs = x[i].qs;\n        const uint8_t* qh=x[i].qh;\n        const uint8_t* signs=x[i].signs;')
for k in range(16):
 s=s.replace(f'signed_grid.data[qs[{k}]]',f'signed_grid.data[qs[{k}] | (((qh[ib32+{k//8}] >> {k%8}) & 1) << 8) | (((signs[ib32*4+{k//2}] >> {4*(k%2)}) & 15) << 9)]')
for v,u in ((1,1),(2,2),(3,4)):
 t=s.replace('unroll_count(2)',f'unroll_count({u})').replace('iq256_single_gu_rows_signindex_v2(',f'iq256_single_gu_rows_signraw_v{v}(')
 Path(f'/tmp/strata-sycl-goal-signraw-v{v}.cpp').write_text(t)
h=Path('/tmp/strata-sycl-goal-signindex-thread-bench.cpp').read_text().replace('iq256_single_gu_rows_signindex_v','iq256_single_gu_rows_signraw_v')
h=h.replace('sizeof(signindexed_iq3s)','sizeof(block_iq3_s)')
a=h.index('  auto pack=[&]');b=h.index('  auto p0=',a);h=h[:a]+'  auto pack=[&]{memcpy(packed.data(),raw.data(),raw.size());};\n'+h[b:]
Path('/tmp/strata-sycl-goal-signraw-thread-bench.cpp').write_text(h)
Path('/tmp/strata-sycl-goal-signraw-build.sh').write_text(Path('/tmp/strata-sycl-goal-signindex-build.sh').read_text().replace('signindex','signraw'))
