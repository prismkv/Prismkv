"""PPU8B synthetic standalone correctness witness, not a latency benchmark.

Install the native-check fixture dependency: pip install -e '.[native]'.
Compile with --build-dir, or pass a previously built --library.
"""
from pathlib import Path
import argparse,json,math,time
import numpy as np
import torch
from prismkv.core import FeatureIndex
from prismkv.native8b import Native8B,compile_ppu

@torch.no_grad()
def indices(K,kind):
 H=K.shape[1];result=[]
 for g in range(8):
  if kind=='empty':fid=np.empty(0,np.int64);tid=np.empty(0,np.int64)
  elif kind=='ties':tid=np.arange(H,dtype=np.int64);fid=tid%min(H,64)
  elif kind=='few':tid=np.arange(H,dtype=np.int64);fid=(tid+g)%7
  else:
   tid=np.repeat(np.arange(H,dtype=np.int64),3)
   fid=np.stack([(np.arange(H)+g*17+offset)%3000 for offset in [0,137,811]],axis=-1).reshape(-1)
  lists=[torch.as_tensor(tid[fid==f],device='cuda') for f in range(3000)]
  ids=torch.as_tensor(tid,device='cuda');features=torch.as_tensor(fid,device='cuda')
  n=torch.bincount(features,minlength=3000);sums=torch.zeros(3000,128,device='cuda');squares=torch.zeros_like(sums)
  if len(ids):
   raw=K[g,ids].float();sums.index_add_(0,features,raw);squares.index_add_(0,features,raw.square())
  mu=sums/n.clamp_min(1)[:,None];var=(squares/n.clamp_min(1)[:,None]-mu.square()).clamp_min(0)
  if kind=='ties':mu.zero_();var.zero_()
  result.append(FeatureIndex(H,K[g].float().mean(0),n,mu,var,lists))
 return result

@torch.no_grad()
def one(H,tail,kind,seed):
 torch.manual_seed(seed)
 K=torch.randn(8,H+tail,128,device='cuda',dtype=torch.bfloat16)
 V=torch.randn_like(K);Q=torch.randn(8,4,128,device='cuda',dtype=torch.bfloat16)
 entries=indices(K[:,:H],kind);operator=Native8B(entries,LIB);operator.route(Q);torch.cuda.synchronize()
 expected=[];rows=[]
 for g,e in enumerate(entries):
  live=e.counts>0
  logits=Q[g].float()@e.mean.T/math.sqrt(128)+Q[g].float().square()@e.variance.T/256+e.counts.clamp_min(1).float().log()
  logits[:,~live]=-torch.inf
  score=logits.softmax(-1).mean(0) if bool(live.any()) else torch.zeros(3000,device='cuda')
  f=score.argsort(descending=True,stable=True);f=f[live[f]][:16]
  ids=torch.cat([e.lists[int(b)] for b in f]).unique(sorted=True) if len(f) else torch.empty(0,device='cuda',dtype=torch.long)
  actual=operator.ids[g,:int(operator.count[g])].long()
  assert torch.equal(ids,actual),('ID mismatch',kind,H,g)
  assert torch.equal(operator.features[g,:len(f)],f),('feature mismatch',kind,H,g)
  full_ids=torch.cat([ids,torch.arange(H,H+tail,device='cuda')])
  expected.append((Q[g].float()@K[g,full_ids].float().T/math.sqrt(128)).softmax(-1)@V[g,full_ids].float())
  rows.append(dict(group=g,unique=len(ids),memberships=e.memberships))
 expected=torch.stack(expected)
 errors=[]
 for splits in [32,64,128]:
  for schedule in [0,2]:
   actual=operator.attention(Q,K,V,splits=splits,schedule=schedule)
   torch.cuda.synchronize()
   err=float((actual.float()-expected).norm()/expected.norm().clamp_min(1e-8))
   max_abs=float((actual.float()-expected).abs().max())
   assert torch.isfinite(actual).all() and err<.01 and max_abs<.05,(kind,H,splits,schedule,err,max_abs)
   errors.append(dict(splits=splits,schedule=schedule,relative_L2=err,max_absolute=max_abs))
 result=dict(H=H,tail=tail,kind=kind,exact_IDs=True,exact_features=True,groups=rows,AV_errors=errors,
             owned_buffer_bytes=operator.buffer_bytes())
 del K,V,Q,entries,operator;torch.cuda.synchronize()
 return result

def main():
 p=argparse.ArgumentParser()
 choice=p.add_mutually_exclusive_group(required=True)
 choice.add_argument('--library',type=Path)
 choice.add_argument('--build-dir',type=Path)
 p.add_argument('--sdk-root',default='/usr/local/PPU_SDK')
 p.add_argument('--output',type=Path,default=Path('output/native8b_validation.json'))
 a=p.parse_args()
 global LIB
 LIB=a.library if a.library else compile_ppu(Path(__file__).resolve().parents[1]/'kernels/ppu_8b',a.build_dir,a.sdk_root)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.reset_peak_memory_stats();rows=[]
 for H,tail,kind in [(32,1,'ties'),(65,7,'few'),(257,31,'random'),(257,129,'empty'),(1025,3,'random'),(65536,17,'random')]:
  rows.append(one(H,tail,kind,90+H))
 result=dict(passed=True,torch=torch.__version__,device=torch.cuda.get_device_name(),cases=rows,
   pytorch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
   pytorch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
   scope='synthetic standalone route/union/AV; one-layer memory, not full-model peak; no latency claim')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(result,indent=2),encoding='utf8')
 print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
