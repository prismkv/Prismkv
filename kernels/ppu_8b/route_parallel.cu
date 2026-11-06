#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
__device__ float row_reduce(float a,float* w,bool mx){
 int t=threadIdx.x%256;
 for(int o=16;o;o/=2){float b=__shfl_down_sync(0xffffffff,a,o);a=mx?fmaxf(a,b):a+b;}
 if(t%32==0)w[t/32]=a;__syncthreads();
 if(t<32){a=t<8?w[t]:(mx?-INFINITY:0);for(int o=16;o;o/=2){float b=__shfl_down_sync(0xffffffff,a,o);a=mx?fmaxf(a,b):a+b;}if(t==0)w[0]=a;}
 __syncthreads();a=w[0];__syncthreads();return a;
}
__device__ void pm(float &a,int &i,float b,int j){if(b>a||(b==a&&j<i)){a=b;i=j;}}
__global__ void fused_parallel(const float* prod,const float* ln,const int* ct,int64_t* ids,float* vals,int* raw){
 int u=blockIdx.x,t=threadIdx.x,h=t/256,r=t%256;
 __shared__ float prob[4][3000],rowwork[32],w[32];__shared__ int wi[32];
 float v[12],mx=-INFINITY;
 for(int j=0;j<12;j++){int b=r+j*256;float a=b<3000&&ct[u*3000+b]>0?prod[(u*4+h)*3000+b]+ln[u*3000+b]:-INFINITY;v[j]=a;mx=fmaxf(mx,a);}
 mx=row_reduce(mx,rowwork+h*8,true);float den=0;
 for(int j=0;j<12;j++){v[j]=isfinite(mx)?expf(v[j]-mx):0;den+=v[j];}den=row_reduce(den,rowwork+h*8,false);
 for(int j=0;j<12;j++){int b=r+j*256;if(b<3000)prob[h][b]=den>0?v[j]/den:0;}
 __syncthreads();float agg[3];int id[3];
 for(int j=0;j<3;j++){int b=t+j*1024;bool ok=b<3000&&ct[u*3000+b]>0;float a=0;if(ok)for(int hh=0;hh<4;hh++)a+=prob[hh][b];agg[j]=ok?a:-INFINITY;id[j]=ok?b:INT_MAX;}
 if(t<16){ids[u*16+t]=0;vals[u*16+t]=0;}int total=0;
 for(int n=0;n<16;n++){
  float a=-INFINITY;int b=INT_MAX;for(int j=0;j<3;j++)pm(a,b,agg[j],id[j]);
  for(int o=16;o;o/=2){float z=__shfl_down_sync(0xffffffff,a,o);int k=__shfl_down_sync(0xffffffff,b,o);pm(a,b,z,k);}
  if(t%32==0){w[t/32]=a;wi[t/32]=b;}__syncthreads();
  if(t<32){a=w[t];b=wi[t];for(int o=16;o;o/=2){float z=__shfl_down_sync(0xffffffff,a,o);int k=__shfl_down_sync(0xffffffff,b,o);pm(a,b,z,k);}if(t==0)wi[0]=b;}
  __syncthreads();int win=wi[0];if(t==0&&win!=INT_MAX){ids[u*16+n]=win;vals[u*16+n]=1;total+=ct[u*3000+win];}
  for(int j=0;j<3;j++)if(id[j]==win){agg[j]=-INFINITY;id[j]=INT_MAX;}__syncthreads();
 }if(t==0)raw[u]=total;
}
extern "C" int fused_norm_top(uintptr_t prod,uintptr_t ln,uintptr_t ct,uintptr_t ids,uintptr_t vals,uintptr_t raw,uint64_t stream){
 fused_parallel<<<8,1024,0,(cudaStream_t)stream>>>((float*)prod,(float*)ln,(int*)ct,(int64_t*)ids,(float*)vals,(int*)raw);return(int)cudaGetLastError();
}
