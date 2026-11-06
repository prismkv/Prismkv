#include <cuda_runtime.h>
#include "hggc_bf16.h"
#include <stdint.h>
#include <math.h>
constexpr int D=128,E=4,Q=4;
template<int MODE> __global__ void part(const bfloat16* q,const bfloat16* k,const bfloat16* v,const int* ids,const int* cnt,float* state,int H,int T,int KS,int VS,int S){
 int g=blockIdx.x,p=blockIdx.y,lane=threadIdx.x%32,w=threadIdx.x/32;
 float qr[Q][E],av[Q][E],m[Q],z[Q];
 #pragma unroll
 for(int a=0;a<Q;++a){m[a]=-INFINITY;z[a]=0;
  #pragma unroll
  for(int d=0;d<E;++d){qr[a][d]=__bfloat162float(q[(g*Q+a)*D+lane*E+d]);av[a][d]=0;}}
 int old=cnt[g];
 int span=((old+T+S*4-1)/(S*4))*4;
 int begin=MODE==0?p*4+w:p*span+w, end=MODE==0?old+T:min(old+T,(p+1)*span), step=MODE==0?S*4:4;
 for(int j=begin;j<end;j+=step){int pos=j<old?ids[g*H+j]:H+j-old;float kk[E],vv[E];
  if(MODE==2){
   uint2 ka=*(const uint2*)(k+(int64_t)g*KS+pos*D+lane*E);uint2 va=*(const uint2*)(v+(int64_t)g*VS+pos*D+lane*E);
   kk[0]=__uint_as_float(ka.x<<16);kk[1]=__uint_as_float(ka.x&0xffff0000u);kk[2]=__uint_as_float(ka.y<<16);kk[3]=__uint_as_float(ka.y&0xffff0000u);
   vv[0]=__uint_as_float(va.x<<16);vv[1]=__uint_as_float(va.x&0xffff0000u);vv[2]=__uint_as_float(va.y<<16);vv[3]=__uint_as_float(va.y&0xffff0000u);
  }else{
  #pragma unroll
  for(int d=0;d<E;++d){kk[d]=__bfloat162float(k[(int64_t)g*KS+pos*D+lane*E+d]);vv[d]=__bfloat162float(v[(int64_t)g*VS+pos*D+lane*E+d]);}
  }
  #pragma unroll
  for(int a=0;a<Q;++a){float dot=0;
   #pragma unroll
   for(int d=0;d<E;++d)dot+=qr[a][d]*kk[d];
   #pragma unroll
   for(int sh=16;sh>0;sh/=2)dot+=__shfl_down_sync(0xffffffffu,dot,sh);
   float score=__shfl_sync(0xffffffffu,dot,0)*0.08838834764831845f;
   float nm=fmaxf(m[a],score),alpha=__expf(m[a]-nm),prob=__expf(score-nm);
   #pragma unroll
   for(int d=0;d<E;++d)av[a][d]=av[a][d]*alpha+prob*vv[d];
   z[a]=z[a]*alpha+prob;m[a]=nm;
  }
 }
 __shared__ float sm[Q][4],sz[Q][4],sv[Q][4][D];
 #pragma unroll
 for(int a=0;a<Q;++a){if(lane==0){sm[a][w]=m[a];sz[a][w]=z[a];}
  #pragma unroll
  for(int d=0;d<E;++d)sv[a][w][lane*E+d]=av[a][d];}
 __syncthreads();
 if(w==0){
  #pragma unroll
  for(int a=0;a<Q;++a){float mx=-INFINITY,den=0,ww[4];for(int b=0;b<4;++b)mx=fmaxf(mx,sm[a][b]);
   for(int b=0;b<4;++b){ww[b]=mx!=-INFINITY?__expf(sm[a][b]-mx):0;den+=ww[b]*sz[a][b];}
   int at=((g*Q+a)*S+p)*(D+2);if(lane==0){state[at]=mx;state[at+1]=den;}
   #pragma unroll
   for(int d=0;d<E;++d){float val=0;for(int b=0;b<4;++b)val+=ww[b]*sv[a][b][lane*E+d];state[at+2+lane*E+d]=val;}
  }
 }
}
__global__ void merge(const float* state,bfloat16* out,int S){
 int h=blockIdx.x,t=threadIdx.x;__shared__ float weights[128],den;
 if(t<32){float mx=-INFINITY;for(int j=t;j<S;j+=32)mx=fmaxf(mx,state[(h*S+j)*(D+2)]);
  for(int sh=16;sh>0;sh/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,sh));
  float z=0;for(int j=t;j<S;j+=32){float w=mx!=-INFINITY?__expf(state[(h*S+j)*(D+2)]-mx):0;weights[j]=w;z+=w*state[(h*S+j)*(D+2)+1];}
  for(int sh=16;sh>0;sh/=2)z+=__shfl_xor_sync(0xffffffffu,z,sh);if(t==0)den=z;}
 __syncthreads();float val=0;for(int j=0;j<S;++j)val+=weights[j]*state[(h*S+j)*(D+2)+2+t];out[h*D+t]=__float2bfloat16(den>0?val/den:0);
}
extern "C" int attention_tuned(uintptr_t q,uintptr_t k,uintptr_t v,uintptr_t ids,uintptr_t cnt,uintptr_t state,uintptr_t out,int H,int T,int KS,int VS,int S,int mode,uint64_t stream){
 if(S<1||S>128||H<1||T<0||KS<(H+T)*D||VS<(H+T)*D)return -1;
 #define PART(M) part<M><<<dim3(8,S),128,0,(cudaStream_t)stream>>>((const bfloat16*)q,(const bfloat16*)k,(const bfloat16*)v,(const int*)ids,(const int*)cnt,(float*)state,H,T,KS,VS,S)
 if(mode==0){PART(0);}else if(mode==1){PART(1);}else if(mode==2){PART(2);}else return -2;
 merge<<<32,128,0,(cudaStream_t)stream>>>((const float*)state,(bfloat16*)out,S);return(int)cudaGetLastError();
}
