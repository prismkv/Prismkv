#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
__device__ float red(float a,float* sh,bool mx){int t=threadIdx.x;for(int s=16;s;s/=2){float b=__shfl_down_sync(0xffffffff,a,s);a=mx?fmaxf(a,b):a+b;}if(t%32==0)sh[t/32]=a;__syncthreads();if(t<32){a=t<8?sh[t]:(mx?-INFINITY:0);for(int s=16;s;s/=2){float b=__shfl_down_sync(0xffffffff,a,s);a=mx?fmaxf(a,b):a+b;}if(t==0)sh[0]=a;}__syncthreads();a=sh[0];__syncthreads();return a;}
__device__ void best(float &a,int &i,float b,int j){if(b>a||(b==a&&j<i)){a=b;i=j;}}
__global__ void fused(const float* prod,const float* ln,const int* ct,int64_t* ids,float* vals,int* raw){
 int u=blockIdx.x,t=threadIdx.x;__shared__ float sh[8];__shared__ int si[8];float agg[12];int ix[12];
 for(int j=0;j<12;j++){int f=t+j*256;bool ok=f<3000&&ct[u*3000+f]>0;agg[j]=ok?0:-INFINITY;ix[j]=ok?f:INT_MAX;}
 for(int h=0;h<6;h++){float a[12],mx=-INFINITY;for(int j=0;j<12;j++){int f=ix[j];a[j]=f!=INT_MAX?prod[(u*6+h)*3000+f]+ln[u*3000+f]:-INFINITY;mx=fmaxf(mx,a[j]);}mx=red(mx,sh,true);float den=0;for(int j=0;j<12;j++){a[j]=isfinite(mx)?expf(a[j]-mx):0;den+=a[j];}den=red(den,sh,false);for(int j=0;j<12;j++)if(ix[j]!=INT_MAX)agg[j]+=den>0?a[j]/den:0;}
 if(t<16){ids[u*16+t]=0;vals[u*16+t]=0;}int total=0;
 for(int n=0;n<16;n++){float a=-INFINITY;int f=INT_MAX;for(int j=0;j<12;j++)best(a,f,agg[j],ix[j]);for(int s=16;s;s/=2){float b=__shfl_down_sync(0xffffffff,a,s);int k=__shfl_down_sync(0xffffffff,f,s);best(a,f,b,k);}if(t%32==0){sh[t/32]=a;si[t/32]=f;}__syncthreads();if(t<32){a=t<8?sh[t]:-INFINITY;f=t<8?si[t]:INT_MAX;for(int s=16;s;s/=2){float b=__shfl_down_sync(0xffffffff,a,s);int k=__shfl_down_sync(0xffffffff,f,s);best(a,f,b,k);}if(t==0)si[0]=f;}__syncthreads();int win=si[0];if(t==0&&win!=INT_MAX){ids[u*16+n]=win;vals[u*16+n]=1;total+=ct[u*3000+win];}for(int j=0;j<12;j++)if(ix[j]==win){ix[j]=INT_MAX;agg[j]=-INFINITY;}__syncthreads();}if(t==0)raw[u]=total;
}
extern "C" int fused_norm_top(uintptr_t prod,uintptr_t ln,uintptr_t ct,uintptr_t ids,uintptr_t vals,uintptr_t raw,int groups,uint64_t stream){fused<<<groups,256,0,(cudaStream_t)stream>>>((float*)prod,(float*)ln,(int*)ct,(int64_t*)ids,(float*)vals,(int*)raw);return(int)cudaGetLastError();}
