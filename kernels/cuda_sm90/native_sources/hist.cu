#include <cuda_runtime.h>
#include <stdint.h>
__global__ void histogram(const float* a,int* out,int N){
 __shared__ int h[2048];for(int j=threadIdx.x;j<2048;j+=256)h[j]=0;__syncthreads();
 for(int i=blockIdx.x*256+threadIdx.x;i<N;i+=gridDim.x*256){float v=a[i];if(v>0)atomicAdd(h+(__float_as_uint(v)>>20),1);}
 __syncthreads();for(int j=threadIdx.x;j<2048;j+=256)out[blockIdx.x*2048+j]=h[j];
}
__global__ void reduce_hist(const int* h,int* out,int blocks){int j=blockIdx.x*256+threadIdx.x;if(j<2048){int z=0;for(int i=0;i<blocks;i++)z+=h[i*2048+j];out[j]=z;}}
__global__ void threshold(const int* h,int* cut,int K){if(threadIdx.x==0){int sum=0;int bin=0;for(int j=2047;j>=0;j--){sum+=h[j];if(sum>=K){bin=j;break;}}*cut=bin;}}
__global__ void count_ge(const float* a,int* counts,int N,const int* cut){
 int t=threadIdx.x,start=blockIdx.x*2048,n=0;int c=*cut;
 for(int r=0;r<8;r++){int i=start+r*256+t;if(i<N){float v=a[i];n+=(v>0&&int(__float_as_uint(v)>>20)>=c);}}
 for(int s=16;s;s/=2)n+=__shfl_down_sync(0xffffffff,n,s);__shared__ int warps[8];if(t%32==0)warps[t/32]=n;__syncthreads();if(t==0){int z=0;for(int j=0;j<8;j++)z+=warps[j];counts[blockIdx.x]=z;}
}
__global__ void collect_ge(const float* a,const long long* prefix,long long* ids,float* values,int N,const int* cut){
 int t=threadIdx.x,lane=t%32,w=t/32,start=blockIdx.x*2048,c=*cut;__shared__ int nw[8],offset;if(t==0)offset=0;__syncthreads();long long base=blockIdx.x?prefix[blockIdx.x-1]:0;
 for(int r=0;r<8;r++){int i=start+r*256+t;float v=i<N?a[i]:0;bool keep=v>0&&int(__float_as_uint(v)>>20)>=c;unsigned mask=__ballot_sync(0xffffffff,keep);if(lane==0)nw[w]=__popc(mask);__syncthreads();int before=0;for(int j=0;j<w;j++)before+=nw[j];int local=__popc(mask&((1u<<lane)-1u));if(keep){long long at=base+offset+before+local;ids[at]=i;values[at]=v;}__syncthreads();if(t==0){for(int j=0;j<8;j++)offset+=nw[j];}__syncthreads();}
}
extern "C" int hist(uint64_t a,uint64_t out,int N,int blocks,uint64_t stream){histogram<<<blocks,256,0,(cudaStream_t)stream>>>((float*)a,(int*)out,N);return cudaGetLastError();}
extern "C" int reduce_cut(uint64_t h,uint64_t total,uint64_t cut,int blocks,int K,uint64_t stream){reduce_hist<<<8,256,0,(cudaStream_t)stream>>>((int*)h,(int*)total,blocks);threshold<<<1,1,0,(cudaStream_t)stream>>>((int*)total,(int*)cut,K);return cudaGetLastError();}
extern "C" int count(uint64_t a,uint64_t counts,int N,uint64_t cut,uint64_t stream){count_ge<<<(N+2047)/2048,256,0,(cudaStream_t)stream>>>((float*)a,(int*)counts,N,(int*)cut);return cudaGetLastError();}
extern "C" int collect(uint64_t a,uint64_t prefix,uint64_t ids,uint64_t values,int N,uint64_t cut,uint64_t stream){collect_ge<<<(N+2047)/2048,256,0,(cudaStream_t)stream>>>((float*)a,(long long*)prefix,(long long*)ids,(float*)values,N,(int*)cut);return cudaGetLastError();}
