#include <stdint.h>
#include <cuda_runtime.h>
constexpr int F=3000;
__device__ __forceinline__ int scan_warp(int x){
 int lane=threadIdx.x%32;
 for(int d=1;d<32;d*=2){int y=__shfl_up_sync(0xffffffffu,x,d);if(lane>=d)x+=y;}
 return x;
}
template<int NT> __device__ __forceinline__ int scan_block(int x,int* warp){
 int lane=threadIdx.x%32,w=threadIdx.x/32;
 x=scan_warp(x);if(lane==31)warp[w]=x;__syncthreads();
 int prior=0;for(int j=0;j<w;++j)prior+=warp[j];return prior+x;
}
template<int NT> __global__ void count_union(const uint32_t* bits,const int64_t* ids,const float* vals,uint32_t* mask,int* tiles,int NW,int NB){
 int u=blockIdx.y,t=threadIdx.x,w=blockIdx.x*NT+t;uint32_t v=0;
 if(w<NW){for(int j=0;j<16;++j)if(vals[u*16+j]>0)v|=bits[(u*F+ids[u*16+j])*NW+w];mask[u*NW+w]=v;}
 int n=__popc(v);for(int d=16;d;d/=2)n+=__shfl_down_sync(0xffffffffu,n,d);
 __shared__ int sums[NT/32];if(t%32==0)sums[t/32]=n;__syncthreads();
 if(t==0){int z=0;for(int j=0;j<NT/32;++j)z+=sums[j];tiles[u*NB+blockIdx.x]=z;}
}
template<int NT> __global__ void extract_union(const uint32_t* mask,const int* tiles,int* selected,int* counts,int H,int NW,int NB){
 int u=blockIdx.y,t=threadIdx.x,w=blockIdx.x*NT+t;uint32_t m=w<NW?mask[u*NW+w]:0;
 __shared__ int warp[NT/32];int n=__popc(m),s=scan_block<NT>(n,warp),base=0;

 if constexpr (NT==64){
  // One warp reduces the preceding tile counts for this CTA.
  __shared__ int tile_base;
  if(t<32){
   for(int j=t;j<(int)blockIdx.x;j+=32)base+=tiles[u*NB+j];
   for(int d=16;d;d/=2)base+=__shfl_down_sync(0xffffffffu,base,d);
   if(t==0)tile_base=base;
  }
  __syncthreads();
  base=tile_base;
 }else{
  for(int j=0;j<(int)blockIdx.x;++j)base+=tiles[u*NB+j];
 }
 if(t==0&&blockIdx.x==NB-1)counts[u]=base+tiles[u*NB+blockIdx.x];
 int at=base+s-n;while(m){int bit=__ffs(m)-1;selected[u*H+at++]=w*32+bit;m&=m-1;}
}
// One block per KV group. Each thread owns consecutive words, preserving
// token-order compaction while avoiding an inter-kernel count handoff.
template<int WORDS> __global__ void single_union(const uint32_t* bits,const int64_t* ids,const float* vals,int* selected,int* counts,int H,int NW){
 constexpr int NT=256;int u=blockIdx.x,t=threadIdx.x;uint32_t m[WORDS];int n=0;
 #pragma unroll
 for(int j=0;j<WORDS;++j){int w=t*WORDS+j;uint32_t v=0;if(w<NW){for(int f=0;f<16;++f)if(vals[u*16+f]>0)v|=bits[(u*F+ids[u*16+f])*NW+w];}m[j]=v;n+=__popc(v);}
 __shared__ int warp[8];int s=scan_block<NT>(n,warp),at=s-n;
 if(t==NT-1)counts[u]=s;
 #pragma unroll
 for(int j=0;j<WORDS;++j){uint32_t v=m[j];int w=t*WORDS+j;while(v){int bit=__ffs(v)-1;selected[u*H+at++]=w*32+bit;v&=v-1;}}
}
extern "C" int launch_union_warp(uintptr_t bits,uintptr_t ids,uintptr_t vals,uintptr_t mask,uintptr_t tiles,uintptr_t selected,uintptr_t counts,int U,int H,int mode,uint64_t stream){
 if(U<1||U>256||H<1||H>262144)return -1;int NW=(H+31)/32;
 if(mode==1){
  #define SINGLE(W) single_union<W><<<U,256,0,(cudaStream_t)stream>>>((uint32_t*)bits,(int64_t*)ids,(float*)vals,(int*)selected,(int*)counts,H,NW)
  if(NW<=256){SINGLE(1);}else if(NW<=512){SINGLE(2);}else if(NW<=1024){SINGLE(4);}else if(NW<=2048){SINGLE(8);}else if(NW<=4096){SINGLE(16);}else{SINGLE(32);}
 }else if(mode==0||mode==2||mode==3){
  #define DUAL(NT) {int NB=(NW+NT-1)/NT;count_union<NT><<<dim3(NB,U),NT,0,(cudaStream_t)stream>>>((uint32_t*)bits,(int64_t*)ids,(float*)vals,(uint32_t*)mask,(int*)tiles,NW,NB);extract_union<NT><<<dim3(NB,U),NT,0,(cudaStream_t)stream>>>((uint32_t*)mask,(int*)tiles,(int*)selected,(int*)counts,H,NW,NB);}
  if(mode==0){DUAL(128);}else if(mode==3){DUAL(64);}else{DUAL(256);}
 }else return -2;
 return (int)cudaGetLastError();
}
