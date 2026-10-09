from pathlib import Path
import re
p=Path('/tmp/strata-upstream-arc-gdn-sync-probe.cpp');s=p.read_text()
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
original=(root/'sycl/src/prefill/kernels.dp.cpp').read_text()
start=re.search(r'__dpct_inline__\s+void\s+gdn_rec_kh_kernel\s*\(',original).start()
brace=original.index('{',start);depth=1;end=brace+1
while depth:
 depth+=(original[end]=='{')-(original[end]=='}');end+=1
body=original[start:end].replace('gdn_rec_kh_kernel','keyhead_rec')
helpers='''
constexpr int GDN_TB=8,VPK=HV/HK;
__dpct_inline__ void gdn_cp4(float* s,const float* g){*s=*g;}
__dpct_inline__ void gdn_cp16(float* s,const float* g){*reinterpret_cast<sycl::float4*>(s)=*reinterpret_cast<const sycl::float4*>(g);}
__dpct_inline__ void gdn_cp_commit(){}
__dpct_inline__ void gdn_cp_wait_prev(){}
'''
s=s.replace('\n}\nclass GdnPair;',helpers+body+'\n}\nclass GdnKH32Large;class GdnKH16;class GdnKH16Large;class GdnPair;')
needle=' q.parallel_for<GdnNorm>'
launch=''' auto kh=sycl::nd_range<3>(sycl::range<3>(1,4,16*4*32),sycl::range<3>(1,4,32));
 if(variant==5)q.parallel_for<GdnKH32Large>(kh,sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::keyhead_rec(state,h,gate,beta,y,T);});
 if(variant==6)q.parallel_for<GdnKH16>(kh,[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {probe::keyhead_rec(state,h,gate,beta,y,T);});
 if(variant==7)q.parallel_for<GdnKH16Large>(kh,sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {probe::keyhead_rec(state,h,gate,beta,y,T);});
'''
s=s.replace(needle,launch+needle)
s=s.replace('auto id=variant==3?sycl::get_kernel_id<GdnPair>():sycl::get_kernel_id<GdnPairLarge>();','''auto id=variant==3?sycl::get_kernel_id<GdnPair>():variant==4?sycl::get_kernel_id<GdnPairLarge>():variant==5?sycl::get_kernel_id<GdnKH32Large>():variant==6?sycl::get_kernel_id<GdnKH16>():sycl::get_kernel_id<GdnKH16Large>();''')
s=s.replace('for(int v:{3,4})','for(int v:{3,4,5,6,7})')
p.write_text(s)
