#include <cuda_runtime.h>
#include <cuda_bf16.h>
using bfloat16 = __nv_bfloat16;
#include <stdint.h>
#include <math.h>
constexpr int D=256,E=8,Q=6;
template<int MODE> __global__ void part(const bfloat16* q,const bfloat16* k,const bfloat16* v,const int* ids,const int* cnt,float* state,const int* histories,const int* lengths,int stride,int capacity,int S){
 int g=blockIdx.x,p=blockIdx.y,lane=threadIdx.x%32,w=threadIdx.x/32;
 float qr[Q][E],av[Q][E],m[Q],z[Q];
 #pragma unroll
 for(int a=0;a<Q;++a){m[a]=-INFINITY;z[a]=0;
  #pragma unroll
  for(int d=0;d<E;++d){qr[a][d]=__bfloat162float(q[(g*Q+a)*D+lane*E+d]);av[a][d]=0;}}
 const int slot=g/4, H=histories[slot], T=lengths[slot]-H;
 int old=T>0?cnt[g]:0;
 int span=((old+T+S*4-1)/(S*4))*4;
 int begin=MODE==0?p*4+w:p*span+w, end=MODE==0?old+T:min(old+T,(p+1)*span), step=MODE==0?S*4:4;
 for(int j=begin;j<end;j+=step){int pos=j<old?ids[g*stride+j]:H+j-old;float kk[E],vv[E];
  if(MODE==2){
   #pragma unroll
   for(int j=0;j<2;j++){
    uint2 ka=*(const uint2*)(k+((int64_t)slot*capacity+pos)*4*D+(g%4)*D+lane*E+j*4);uint2 va=*(const uint2*)(v+((int64_t)slot*capacity+pos)*4*D+(g%4)*D+lane*E+j*4);
    kk[j*4]=__uint_as_float(ka.x<<16);kk[j*4+1]=__uint_as_float(ka.x&0xffff0000u);kk[j*4+2]=__uint_as_float(ka.y<<16);kk[j*4+3]=__uint_as_float(ka.y&0xffff0000u);
    vv[j*4]=__uint_as_float(va.x<<16);vv[j*4+1]=__uint_as_float(va.x&0xffff0000u);vv[j*4+2]=__uint_as_float(va.y<<16);vv[j*4+3]=__uint_as_float(va.y&0xffff0000u);
   }
  }else{
  #pragma unroll
  for(int d=0;d<E;++d){kk[d]=__bfloat162float(k[((int64_t)slot*capacity+pos)*4*D+(g%4)*D+lane*E+d]);vv[d]=__bfloat162float(v[((int64_t)slot*capacity+pos)*4*D+(g%4)*D+lane*E+d]);}
  }
  #pragma unroll
  for(int a=0;a<Q;++a){float dot=0;
   #pragma unroll
   for(int d=0;d<E;++d)dot+=qr[a][d]*kk[d];
   #pragma unroll
   for(int sh=16;sh>0;sh/=2)dot+=__shfl_down_sync(0xffffffffu,dot,sh);
   float score=__shfl_sync(0xffffffffu,dot,0)*0.0625f;
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


// Slot-major [slot, position, kv_group, head_dimension] cache shared with Flash.
__global__ void append_slots(const unsigned* k, const unsigned* v,
                             unsigned* keys, unsigned* values,
                             const int* positions, int capacity) {
    const int slot = blockIdx.x;
    const int position = positions[slot];
    if (position < 0 || position >= capacity) return;
    constexpr int words = 4 * D / 2;
    const int64_t dst = ((int64_t)slot * capacity + position) * words;
    for (int i = threadIdx.x; i < words; i += blockDim.x) {
        keys[dst+i] = k[slot*words+i];
        values[dst+i] = v[slot*words+i];
    }
}

extern "C" int batch_append(uint64_t k, uint64_t v, uint64_t keys, uint64_t values,
                            uint64_t positions, int slots, int capacity, uint64_t stream) {
    append_slots<<<slots,256,0,(cudaStream_t)stream>>>(
        (const unsigned*)k,(const unsigned*)v,(unsigned*)keys,(unsigned*)values,
        (const int*)positions,capacity);
    return cudaGetLastError();
}

extern "C" int batch_av(uint64_t q, uint64_t k, uint64_t v, uint64_t ids,
                        uint64_t count, uint64_t state, uint64_t out, uint64_t histories,
                        uint64_t lengths, int groups, int stride, int capacity, int splits,
                        uint64_t stream) {
    if (groups<4 || groups>256 || groups%4 || splits<1 || splits>128) return -1;
    part<2><<<dim3(groups,splits),128,0,(cudaStream_t)stream>>>(
        (const bfloat16*)q,(const bfloat16*)k,(const bfloat16*)v,(const int*)ids,
        (const int*)count,(float*)state,(const int*)histories,(const int*)lengths,
        stride,capacity,splits);
    merge<<<groups*6,256,0,(cudaStream_t)stream>>>((const float*)state,(bfloat16*)out,splits);
    return cudaGetLastError();
}
