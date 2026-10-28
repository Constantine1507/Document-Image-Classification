"""Small PyTorch neural corrections trained on frozen CNN probabilities only."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ProbabilityCorrector(nn.Module):
    """Temperature, mixture, bias and optional regularized neural residual."""
    def __init__(self, config):
        """Initialize corrections close to the original arithmetic ensemble."""
        super().__init__();self.config=dict(config);self.method=config['method']
        self.log_temperature=nn.Parameter(torch.zeros(2))
        self.weight_logits=nn.Parameter(torch.zeros(2,16 if self.method=='classwise' else 1))
        self.bias=nn.Parameter(torch.zeros(16),requires_grad=self.method not in {'scalar'})
        self.register_buffer('decision_offset',torch.zeros(16))
        self.residual=None
        if self.method=='linear':
            self.residual=nn.Linear(64,16,bias=False)
            nn.init.zeros_(self.residual.weight)
        elif self.method=='mlp':
            self.residual=nn.Sequential(nn.Linear(71,config.get('hidden',32)),nn.GELU(),
                                        nn.Dropout(config.get('dropout',.3)),nn.Linear(config.get('hidden',32),16,bias=False))
            nn.init.zeros_(self.residual[-1].weight)

    def features(self, probability):
        """Create class probabilities and centered log probabilities without IDs or metadata."""
        logs=probability.clamp_min(1e-8).log()
        centered=(logs-logs.mean(2,keepdim=True))/4.
        features=torch.cat([probability.flatten(1),centered.flatten(1)],dim=1)
        if self.method=='mlp':
            top=probability.topk(2,dim=2).values
            entropy=-(probability*logs).sum(2)/math.log(16)
            agreement=(probability[:,0].argmax(1)==probability[:,1].argmax(1)).to(probability.dtype)[:,None]
            features=torch.cat([features,entropy,top[:,:,0],top[:,:,0]-top[:,:,1],agreement],dim=1)
        return features

    def forward(self, probability):
        """Return logits, including all normalization-dependent calibration effects."""
        temperature=self.log_temperature.clamp(math.log(.5),math.log(5.)).exp()
        calibrated=(probability.clamp_min(1e-12).log()/temperature[None,:,None]).softmax(2)
        weights=self.weight_logits.softmax(0)
        mixed=(calibrated*weights[None]).sum(1)
        logits=mixed.clamp_min(1e-12).log()+(self.bias-self.bias.mean())+self.decision_offset
        if self.residual is not None:logits=logits+self.residual(self.features(probability))
        return logits

    def penalty(self):
        """Shrink corrections; separate classwise and dense residual penalties."""
        penalty=.01*self.log_temperature.square().mean()+.05*self.bias.square().mean()
        if self.method=='classwise':penalty=penalty+self.config['regularization']*self.weight_logits.square().mean()
        elif self.residual is not None:
            penalty=penalty+self.config['regularization']*sum(p.square().sum() for p in self.residual.parameters())
        return penalty


def fixed_predict(probability, method):
    """Evaluate fixed, untrained baselines in the same sample order."""
    if method=='baseline':return probability.mean(1)
    if method=='convnext':return probability[:,0]
    if method=='efficientnet':return probability[:,1]
    logits=np.log(np.clip(probability,1e-12,1)).mean(1)
    logits-=logits.max(1,keepdims=True);result=np.exp(logits)
    return result/result.sum(1,keepdims=True)


@torch.no_grad()
def optimize_decision_bias(model, inputs, targets, config):
    """Optimize only training-row accuracy with small bounded class-bias changes."""
    logits=model(inputs).detach().clone();offset=torch.zeros(16,dtype=inputs.dtype)
    grid=torch.tensor([-.25,-.15,-.1,-.05,-.025,0,.025,.05,.1,.15,.25],dtype=inputs.dtype)
    penalty=config.get('decision_penalty',.002)
    for iteration in range(4):
        changed=False
        for cls in range(16):
            other=logits.clone();other[:,cls]=-torch.inf
            other_value,other_class=other.max(1)
            choices=(offset[cls]+grid/(1+iteration//2)).clamp(-.5,.5).unique()
            score=logits[:,cls,None]+choices[None]-offset[cls]
            # Exact ties respect torch.argmax's lowest-class-index tie break.
            choose=(score>other_value[:,None])|((score==other_value[:,None])&(cls<other_class[:,None]))
            predictions=torch.where(choose,cls,other_class[:,None])
            objective=(predictions==targets[:,None]).to(inputs.dtype).mean(0)-penalty*choices.square()
            best=choices[objective.argmax()]
            if abs(float(best-offset[cls]))>1e-12:changed=True
            logits[:,cls]+=best-offset[cls];offset[cls]=best
        if not changed:break
    model.decision_offset.copy_(offset-offset.mean())


def fit(config, probability, labels, seed=42):
    """Fit using training rows only; no validation labels or early-stopping leak."""
    if config['method'] in {'baseline','geometric','convnext','efficientnet'}:return None
    torch.set_num_threads(2);torch.manual_seed(seed)
    dtype=torch.float32 if config['method']=='mlp' else torch.float64
    model=ProbabilityCorrector(config).to(dtype=dtype)
    x=torch.as_tensor(probability,dtype=dtype);y=torch.as_tensor(labels,dtype=torch.long)
    if config['method']=='mlp':
        optimizer=torch.optim.AdamW(model.parameters(),lr=.015,weight_decay=0)
        for epoch in range(config['epochs']):
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y)+model.penalty()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite neural residual loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            for group in optimizer.param_groups:group['lr']=.001+.014*.5*(1+math.cos(math.pi*(epoch+1)/config['epochs']))
    else:
        optimizer=torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad],lr=.7,max_iter=100,
                                     history_size=20,tolerance_grad=1e-7,line_search_fn='strong_wolfe')
        def closure():
            """Evaluate the regularized calibration likelihood and its gradients."""
            optimizer.zero_grad();loss=F.cross_entropy(model(x),y)+model.penalty()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite calibration loss')
            loss.backward();return loss
        optimizer.step(closure)
    model.eval()
    if config['method']=='accuracy_bias':optimize_decision_bias(model,x,y,config)
    return model


@torch.inference_mode()
def predict(config, model, probability):
    """Return finite probability rows, optionally shrunk toward the original ensemble."""
    if model is None:return fixed_predict(probability,config['method'])
    model.eval();dtype=next(model.parameters()).dtype
    outputs=[]
    for start in range(0,len(probability),2048):
        logits=model(torch.as_tensor(probability[start:start+2048],dtype=dtype))
        outputs.append(logits.double().softmax(1).cpu().numpy())
    result=np.concatenate(outputs)
    blend=config.get('baseline_blend',0.)
    result=(1-blend)*result+blend*probability.mean(1)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1),1,atol=1e-5):raise ValueError('Invalid predicted probabilities')
    return result
