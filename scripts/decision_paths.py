"""Actual fixed-prefix and incremental decision paths, no backfilled latency."""
from contextlib import contextmanager
import math,torch
from torch import nn
from transformers.masking_utils import create_causal_mask,create_sliding_window_causal_mask
from train_readout import ChoiceHead,qwen_prompt
from evaluate_readout import PrefixRunner,candidate_embeddings,predict_head
from matched_common import DEPTHS,HEAD_ROOT

@contextmanager
def limited_layers(model,depth):
    original=model.model.layers
    model.model.layers=nn.ModuleList(list(original)[:depth])
    try:yield
    finally:model.model.layers=original

@torch.inference_mode()
def native_scores(model,tok,row,device,depth):
    prompt=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    pids=tok(prompt,add_special_tokens=False).input_ids
    full=[tok(prompt+label+tok.eos_token,add_special_tokens=False).input_ids for label in row['labels']]
    common=len(pids)
    for ids in full:
        k=0
        while k<min(len(pids),len(ids)) and pids[k]==ids[k]:k+=1
        common=min(common,k)
    if common<1:raise RuntimeError('no shared prefill context')
    x=torch.tensor([pids[:common]],device=device)
    scores=[];forward_tokens=common
    with limited_layers(model,depth):
        root=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=True)
        cache=root.past_key_values;first=torch.log_softmax(root.logits[0,-1].float(),dim=-1)
        for ids in full:
            tail=ids[common:]
            if not tail:raise RuntimeError('candidate must include a nonempty completion')
            value=first[tail[0]]
            if len(tail)>1:
                cache.crop(common)
                for j in range(1,len(tail)):
                    branch=torch.tensor([[tail[j-1]]],device=device)
                    mask=torch.ones((1,common+j),dtype=torch.long,device=device)
                    out=model(input_ids=branch,attention_mask=mask,past_key_values=cache,use_cache=True)
                    value=value+torch.log_softmax(out.logits[0,-1].float(),dim=-1)[tail[j]]
                forward_tokens+=len(tail)-1
            scores.append(value)
    logits=torch.stack(scores);probs=torch.softmax(logits,dim=-1);j=int(logits.argmax())
    return {'prediction':row['labels'][j],'confidence':float(probs[j]),'probabilities':{lab:float(probs[i]) for i,lab in enumerate(row['labels'])},'log_likelihoods':{lab:float(logits[i]) for i,lab in enumerate(row['labels'])},'executed_layers':depth,'forward_token_blocks':depth*forward_tokens,'shared_prefill':True,'candidate_scoring':'one shared prefill, tokenwise cached branches; BF16 default SDPA'}

def load_heads(run,condition,device):
    heads={}
    for d in DEPTHS:
        h=ChoiceHead().to(device).eval();h.load_state_dict(torch.load(run/HEAD_ROOT/condition/f'head_layer{d}.pt',weights_only=True,map_location=device));heads[d]=h
    return heads

def fixed_head(model,heads,tok,row,device,depth):
    x=torch.tensor([qwen_prompt(tok,row['prompt'])],device=device)
    h=PrefixRunner(model,depth)(x)
    emb=model.get_input_embeddings().weight.detach()
    pred,conf,probs=predict_head(heads[depth],h,candidate_embeddings(tok,emb,row,device),row['labels'])
    return {'prediction':pred,'confidence':conf,'probabilities':probs,'executed_layers':depth,'forward_token_blocks':depth*x.numel(),'shared_prefill':True}

def rule_accept(probs,previous,kind,tau):
    vals=sorted(probs.values(),reverse=True);label=max(probs,key=probs.get)
    if kind=='confidence':return vals[0]>=tau
    if kind=='margin':return vals[0]-vals[1]>=tau
    if kind=='entropy':return -sum(p*math.log(max(p,1e-30)) for p in vals)/math.log(len(vals))<=tau
    if kind=='agreement':return label==previous and vals[0]>=tau
    raise ValueError(kind)

@torch.inference_mode()
def incremental_head(model,heads,tok,row,device,rule):
    ids=qwen_prompt(tok,row['prompt']);x=torch.tensor([ids],device=device);positions=torch.arange(x.shape[1],device=device)[None]
    h=model.model.embed_tokens(x);kwargs={'config':model.config,'inputs_embeds':h,'attention_mask':torch.ones_like(x),'past_key_values':None,'position_ids':positions}
    masks={'full_attention':create_causal_mask(**kwargs)}
    if model.model.has_sliding_layers:masks['sliding_attention']=create_sliding_window_causal_mask(**kwargs)
    rotary=model.model.rotary_emb(h,positions)
    c=candidate_embeddings(tok,model.get_input_embeddings().weight.detach(),row,device);previous=None;checks=[]
    for i,layer in enumerate(model.model.layers):
        h=layer(h,attention_mask=masks[model.config.layer_types[i]],position_ids=positions,past_key_values=None,use_cache=False,position_embeddings=rotary)
        d=i+1
        if d not in (4,8,12,16,20,24,28):continue
        label,conf,probs=predict_head(heads[d],h[0,-1].float(),c,row['labels']);checks.append(d)
        if d==28 or (d>=8 and rule_accept(probs,previous,rule['kind'],rule['threshold'])):
            return {'prediction':label,'confidence':conf,'probabilities':probs,'executed_layers':d,'forward_token_blocks':d*x.numel(),'head_checks':checks,'shared_prefill':True}
        previous=label
    raise RuntimeError('no exit')
