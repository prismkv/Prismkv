import torch,time,types

class Router:
 def __init__(self,K,H,B,mode,opt):
  self.H=H;self.B=(B//16)*16 if mode=='quest' else B;self.mode=mode;self.opt=opt
  self.ids=torch.empty((8,H),device=K.device,dtype=torch.int32);self.count=torch.full((8,),self.B,device=K.device,dtype=torch.int32)
  self.state=torch.empty((32,128,130),device=K.device,dtype=torch.float32);self.out=torch.empty((8,4,128),device=K.device,dtype=K.dtype)
  self.offset=torch.arange(16,device=K.device)
  if mode=='quest':
   pages=K[:,:H].reshape(8,H//16,16,128);self.lo=pages.amin(2).transpose(1,2).contiguous();self.hi=pages.amax(2).transpose(1,2).contiguous()
  else:
   self.channels=K[:,:H].float().abs().mean(1).topk(32,dim=-1).indices
   self.small=K[:,:H].gather(2,self.channels[:,None,:].expand(8,H,32)).transpose(1,2).contiguous()
 def route(self,Q):
  if self.mode=='quest':
   s=torch.bmm(Q.clamp_min(0),self.hi).float()+torch.bmm(Q.clamp_max(0),self.lo).float()
   ix=(s*(128**-.5)).softmax(-1).mean(1).topk(self.B//16,dim=-1,sorted=False).indices
   chosen=(ix[:,:,None]*16+self.offset).reshape(8,self.B)
  else:
   q=Q.gather(2,self.channels[:,None,:].expand(8,4,32));s=torch.bmm(q,self.small).float()*(128**-.5)
   chosen=s.softmax(-1).mean(1).topk(self.B,dim=-1,sorted=False).indices
  self.ids[:,:self.B].copy_(chosen)
 def av(self,Q,K,V):return self.opt.av(self,Q,K,V,(128,2))
 def run(self,Q,K,V):self.route(Q);return self.av(Q,K,V)
 def check(self,Q,K,V):
  out=self.run(Q,K,V).clone();ref=[]
  for g in range(8):
   ix=self.ids[g,:self.B].long();assert ix.unique().numel()==self.B and int(ix.max())<self.H
   ix=torch.cat([ix,torch.arange(self.H,K.shape[1],device=K.device)])
   kk=K[g,ix].float();vv=V[g,ix].float();ref.append((Q[g].float()@kk.T*(128**-.5)).softmax(-1)@vv)
  ref=torch.stack(ref);err=float((out.float()-ref).norm()/ref.norm());assert err<.02,err
  return dict(relative_L2=err,old_count=self.B,H=self.H)
