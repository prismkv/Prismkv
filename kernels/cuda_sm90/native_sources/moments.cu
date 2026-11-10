#include <cuda_runtime.h>
#include <stdint.h>
// Each dimension owns its local Welford recurrence. A bucket-relative anchor
// avoids loss of small differences when a bucket has a large shared offset.
__global__ void partial(const float* k,const int64_t* ids,const int64_t* off,float* tmp,int S){
 int f=blockIdx.x,s=blockIdx.y,d=threadIdx.x;
 int begin=off[f],end=off[f+1],nall=end-begin;
 int lo=begin+(long long)nall*s/S,hi=begin+(long long)nall*(s+1)/S;
 float anchor=nall?k[ids[begin]*256+d]:0.f,mean=0.f,m2=0.f;int n=0;
 for(int j=lo;j<hi;++j){float x=k[ids[j]*256+d]-anchor; ++n;float delta=x-mean;mean+=delta/n;m2+=delta*(x-mean);}
 int base=(f*S+s)*513;tmp[base+d]=mean;tmp[base+256+d]=m2;if(d==0)tmp[base+512]=(float)n;
}
__global__ void finish(const float* k,const int64_t* ids,const int64_t* off,const float* tmp,float* mu,float* var,int S){
 int f=blockIdx.x,d=threadIdx.x;float mean=0.f,m2=0.f,n=0.f;
 for(int s=0;s<S;++s){int base=(f*S+s)*513;float nn=tmp[base+512];if(nn>0.f){float mm=tmp[base+d],vv=tmp[base+256+d],delta=mm-mean,total=n+nn;mean+=delta*(nn/total);m2+=vv+delta*delta*(n*nn/total);n=total;}}
 float anchor=off[f+1]>off[f]?k[ids[off[f]]*256+d]:0.f;
 mu[f*256+d]=n?anchor+mean:0.f;var[f*256+d]=n?fmaxf(m2/n,0.f):0.f;
}
extern "C" int moments(uint64_t k,uint64_t ids,uint64_t off,uint64_t tmp,uint64_t mu,uint64_t var,int S,uint64_t stream){
 if(S<1||S>32)return -1;
 partial<<<dim3(3000,S),256,0,(cudaStream_t)stream>>>((float*)k,(int64_t*)ids,(int64_t*)off,(float*)tmp,S);
 finish<<<3000,256,0,(cudaStream_t)stream>>>((float*)k,(int64_t*)ids,(int64_t*)off,(float*)tmp,(float*)mu,(float*)var,S);
 return cudaGetLastError();
}
