"""Shared frozen model names, I/O, and strict decision semantics."""
from pathlib import Path
import hashlib,json,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel

DEPTHS=[2,4,8,12,16,20,24,28]
HEAD_ROOT='checkpoints/matched_heads_unmerged'
POLICY_PREFIX='matched_policy_unmerged_'
CONDITIONS=['direct','matched_sft_answer_a1_seed20281028','matched_sft_trace_a0_seed20281028','matched_sft_trace_a0.5_seed20281028','matched_sft_trace_a1_seed20281028','matched_msd_a0_seed20281028','matched_msd_a0.5_seed20281028','matched_msd_a1_seed20281028']

def read(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_model(run,condition,device):
    path=Path(run)/'models/Qwen3-1.7B'
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    if condition!='direct':
        model=PeftModel.from_pretrained(model,str(Path(run)/'results'/condition),is_trainable=False).eval()
        # Preserve the exact LoRA projection path used for checkpoint dev
        # selection. BF16 weight merging failed the fixed precision tolerance.
        # get_base_model retains active LoRA layers but exposes HF's stable
        # decoder hierarchy to hooks and incremental execution.
        model=model.get_base_model().eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

def load_tokenizer(run):return AutoTokenizer.from_pretrained(Path(run)/'models/Qwen3-1.7B',local_files_only=True,use_fast=True)
def model_manifest(run):
    out={}
    for c in CONDITIONS[1:]:
        p=Path(run)/'results'/c
        d=json.loads((p/'train_log.json').read_text());pr=json.loads((p/'training_provenance.json').read_text())
        out[c]={'adapter_sha256':sha(p/'adapter_model.safetensors'),'selected_step':d['selected_checkpoint_step'],'dev_accuracy':d['dev']['accuracy'],'train_sha256':pr['train_sha256'],'steps':len(d['steps'])}
    return out
