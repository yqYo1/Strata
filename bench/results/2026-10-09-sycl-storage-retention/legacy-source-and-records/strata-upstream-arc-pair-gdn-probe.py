from pathlib import Path
import shutil
p=Path('/tmp/strata-upstream-arc-gdn-sync-probe.cpp')
shutil.copy2(p,'/tmp/strata-upstream-arc-gdn-sync-probe-swizzled.cpp')
shutil.copy2('/tmp/strata-upstream-arc-gdn-sync-probe','/tmp/strata-upstream-arc-gdn-sync-probe-swizzled')
s=p.read_text().replace('#include <sycl/sycl.hpp>','#include <sycl/sycl.hpp>\n#include <sycl/ext/intel/experimental/grf_size_properties.hpp>')
new='''
__dpct_inline__ void pair_rec(float* state,const float* h,const float* gate,const float* beta,float* oc_out,int64_t T) {
 auto item=sycl::ext::oneapi::this_work_item::get_nd_item<3>();
 auto &sq=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(item.get_group());
 auto &sk=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(item.get_group());
 auto &rkv=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[4][CB]>(item.get_group());
 auto &ro=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[4][CB]>(item.get_group());
 const int c=item.get_local_id(2),rg=item.get_local_id(1),tid=rg*CB+c;
 const int head=item.get_group(2)/NCB,cb=item.get_group(2)%NCB,col=cb*CB+c,qh=head%HK;
 const size_t rs=(size_t)HV*S;float s[2][RPG];
 #pragma unroll
 for(int j=0;j<2;j++){
  const float* base=state+((size_t)((rg+j*2)*RPG)*HV+head)*S+col;
  #pragma unroll
  for(int r=0;r<RPG;r++)s[j][r]=base[r*rs];
 }
 float nq[2],nk[2],nv=0,ng=0,nb=0;
 auto fetch=[&](int64_t t){const float* ht=h+t*C;
  #pragma unroll
  for(int u=0;u<2;u++){nq[u]=ht[qh*S+tid+u*64];nk[u]=ht[HK*S+qh*S+tid+u*64];}
  nv=ht[2*HK*S+head*S+col];ng=gate[t*HV+head];nb=beta[t*HV+head];
 };
 if(T>0)fetch(0);
 for(int64_t t=0;t<T;t++){
  float cq[2],ck[2];
  #pragma unroll
  for(int u=0;u<2;u++){cq[u]=nq[u];ck[u]=nk[u];}
  const float cv=nv,cg=ng,cbt=nb;
  // The preceding output barrier has completed every read of the prior q/k.
  // The next stage barrier also precedes this token's writes of rkv and ro.
  #pragma unroll
  for(int u=0;u<2;u++){sq[tid+u*64]=cq[u];sk[tid+u*64]=ck[u];}
  item.barrier(sycl::access::fence_space::local_space);
  if(t+1<T)fetch(t+1);
  const float g=sycl::native::exp(cg);float kv[2]={0,0},out[2]={0,0};
  #pragma unroll
  for(int r=0;r<RPG;r++){
   #pragma unroll
   for(int j=0;j<2;j++)kv[j]=sycl::fma(s[j][r],sk[(rg+j*2)*RPG+r],kv[j]);
  }
  #pragma unroll
  for(int j=0;j<2;j++)rkv[rg+j*2][c]=kv[j];
  item.barrier(sycl::access::fence_space::local_space);
  const float kv_col=rkv[0][c]+rkv[1][c]+rkv[2][c]+rkv[3][c];
  const float delta=(cv-g*kv_col)*cbt;
  #pragma unroll
  for(int r=0;r<RPG;r++){
   #pragma unroll
   for(int j=0;j<2;j++){
    s[j][r]=sycl::fma(g,s[j][r],sk[(rg+j*2)*RPG+r]*delta);
    out[j]=sycl::fma(s[j][r],sq[(rg+j*2)*RPG+r],out[j]);
   }
  }
  #pragma unroll
  for(int j=0;j<2;j++)ro[rg+j*2][c]=out[j];
  item.barrier(sycl::access::fence_space::local_space);
  if(rg==0)oc_out[t*HV*S+head*S+col]=(ro[0][c]+ro[1][c]+ro[2][c]+ro[3][c])*sycl::rsqrt((float)S);
 }
 #pragma unroll
 for(int j=0;j<2;j++){
  float* base=state+((size_t)((rg+j*2)*RPG)*HV+head)*S+col;
  #pragma unroll
  for(int r=0;r<RPG;r++)base[r*rs]=s[j][r];
 }
}
'''
s=s.replace('\n}\nclass GdnLocal;',new+'\n}\nclass GdnPair;class GdnPairLarge;class GdnLocal;')
s=s.replace('else q.parallel_for<GdnShuffle>', 'else if(variant==2)q.parallel_for<GdnShuffle>')
needle=' q.parallel_for<GdnNorm>'
launch=''' auto paired=sycl::nd_range<3>(sycl::range<3>(1,2,48*4*32),sycl::range<3>(1,2,32));
 if(variant==3)q.parallel_for<GdnPair>(paired,[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::pair_rec(state,h,gate,beta,y,T);});
 if(variant==4)q.parallel_for<GdnPairLarge>(paired,sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::pair_rec(state,h,gate,beta,y,T);});
'''
s=s.replace(needle,launch+needle)
s=s.replace('for(int v:{1,2})','for(int v:{3,4})')
# After first launch, query resource requirements of the actual compiled candidate kernel.
s=s.replace(' old();fresh();q.wait_and_throw();',''' old();fresh();q.wait_and_throw();
 auto id=variant==3?sycl::get_kernel_id<GdnPair>():sycl::get_kernel_id<GdnPairLarge>();
 auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(),{q.get_device()},{id});
 auto kernel=bundle.get_kernel(id);
 printf("RESOURCE T=%lld variant=%d private=%llu spill=%llu\\n",(long long)T,variant,
 (unsigned long long)kernel.get_info<sycl::info::kernel_device_specific::private_mem_size>(q.get_device()),
 (unsigned long long)kernel.get_info<sycl::ext::intel::info::kernel_device_specific::spill_mem_size>(q.get_device()));
''')
p.write_text(s)
