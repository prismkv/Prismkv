#include <cuda_runtime.h>
#include <stdint.h>
__global__ void pk(const unsigned short* q,float* a){int r=blockIdx.x,d=threadIdx.x;float v=__uint_as_float((unsigned)q[r*256+d]<<16);a[r*512+d]=v;a[r*512+256+d]=v*v;}
extern "C" int pack(uint64_t q,uint64_t a,int groups,uint64_t stream){pk<<<groups*6,256,0,(cudaStream_t)stream>>>((unsigned short*)q,(float*)a);return cudaGetLastError();}
__global__ void membership(const int* ids,const int* offsets,unsigned* bits,int H,int C,int NW){int g=blockIdx.y,f=blockIdx.x;for(int j=offsets[g*3001+f]+threadIdx.x;j<offsets[g*3001+f+1];j+=256){int t=ids[g*C+j];atomicOr(bits+(g*3000+f)*NW+t/32,1u<<(t%32));}}
extern "C" int build_bits(uint64_t ids,uint64_t offsets,uint64_t bits,int H,int C,uint64_t stream){membership<<<dim3(3000,4),256,0,(cudaStream_t)stream>>>((int*)ids,(int*)offsets,(unsigned*)bits,H,C,(H+31)/32);return cudaGetLastError();}
