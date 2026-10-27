#!/usr/bin/env python3
"""Single-GPU LoRA pilot for answer-only SFT and paper-spec on-policy reverse KL.

Teacher/student token positions are aligned on a single sampled student rollout.
MSD computes the exact full-vocabulary reverse KL per response token. This is a
transparent pilot implementation, not author-provided code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prompt_message(tokenizer, prompt: str, thinking: bool, add_generation_prompt: bool = True) -> str:
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=thinking)
    except TypeError:
        # Do not silently downgrade the mode: Qwen3's template must expose this arg.
        raise RuntimeError("tokenizer chat template does not support enable_thinking")


def gen_ids(model, tokenizer, prompt: str, thinking: bool, max_new: int, temperature: float, device):
    text = prompt_message(tokenizer, prompt, thinking)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        kwargs={"input_ids":ids,"attention_mask":torch.ones_like(ids),"max_new_tokens":max_new,"do_sample":(temperature > 0),"top_p":1.0,"top_k":0,"pad_token_id":tokenizer.eos_token_id}
        if temperature > 0: kwargs["temperature"]=temperature
        out = model.generate(**kwargs)
    if was_training:
        model.train()
    new_ids = out[0, ids.shape[1]:].detach()
    return new_ids, tokenizer.decode(new_ids, skip_special_tokens=False), int(new_ids.numel())


def gen(model, tokenizer, prompt: str, thinking: bool, max_new: int, temperature: float, device):
    _, text, n = gen_ids(model, tokenizer, prompt, thinking, max_new, temperature, device)
    return text, n


def extract_think(answer: str):
    a = answer.find("<think>")
    b = answer.find("</think>")
    if b < 0:
        return None, None
    trace = answer[a + len("<think>"):b] if a >= 0 else answer[:b]
    solution = answer[b + len("</think>"):]
    return trace, solution


def strict_correct(text: str, expected: str) -> bool:
    # Last non-empty line must exactly match one target label. No normalization or repair.
    lines = [x.strip() for x in text.replace("<|im_end|>", "").replace("<|endoftext|>", "").splitlines() if x.strip()]
    return bool(lines and lines[-1] == expected)


def parse_direct_label(text, labels):
    """Whitespace and one terminal EOS only; no answer extraction or repair."""
    s=text.strip()
    for eos in ("<|im_end|>", "<|endoftext|>"):
        if s.endswith(eos):
            s=s[:-len(eos)].strip()
            break
    return s if s in labels else None


def tokenize_pair(tokenizer, prompt, completion, thinking):
    msgs = [{"role":"user","content":prompt},{"role":"assistant","content":completion}]
    try:
        full = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, enable_thinking=thinking)
    except TypeError:
        raise RuntimeError("tokenizer chat template does not support enable_thinking")
    prefix = tokenizer(prompt_message(tokenizer, prompt, thinking), add_special_tokens=False).input_ids
    lcp = 0
    while lcp < min(len(prefix), len(full)) and prefix[lcp] == full[lcp]:
        lcp += 1
    # lcp is the true first response-token prediction position even if BPE merges
    # across the assistant boundary. The caller labels positions lcp onward.
    return prefix, full, lcp


def response_tokens(tokenizer, prompt, completion, thinking):
    """Tokenize the exact teacher response stream with the template intact."""
    msgs=[{"role":"user","content":prompt},{"role":"assistant","content":completion}]
    ids=tokenizer.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False,enable_thinking=thinking)
    if hasattr(ids,"input_ids"): ids=ids.input_ids
    if hasattr(ids,"ids"): ids=ids.ids
    if isinstance(ids,torch.Tensor): ids=ids.detach().cpu().tolist()
    ids=list(ids)
    prefix=tokenizer(prompt_message(tokenizer,prompt,thinking),add_special_tokens=False).input_ids
    lcp=0
    while lcp<min(len(prefix),len(ids)) and prefix[lcp]==ids[lcp]: lcp+=1
    return ids,lcp


def valid_trace(answer, row):
    trace, solution = extract_think(answer)
    if trace is None or len([x for x in trace.splitlines() if x.strip()]) < 2:
        return False
    if not strict_correct(solution, row["answer"]):
        return False
    # Executable factual grounding screen: every nontrivial numeric or symbolic atom
    # in the trace must be present in prompt or verifier certificate.
    source = row["prompt"] + " " + " ".join(row["certificate"])
    atoms = set(x for x in __import__("re").findall(r"\b(?:\d+|[A-Z][A-Z0-9_]{1,})\b", trace))
    allowed = set(__import__("re").findall(r"\b(?:\d+|[A-Z][A-Z0-9_]{1,})\b", source)) | {row["answer"]}
    return atoms.issubset(allowed)


def response_prompt_ids(tokenizer, prompt, thinking, device):
    s = prompt_message(tokenizer, prompt, thinking)
    return tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)


def masked_teacher_context(prompt_ids, trace_ids, tokenizer, alpha):
    """Build x,z_<=floor(alpha*m) within Qwen's explicit think block."""
    m=len(trace_ids); k=int(alpha*m)
    think_open=tokenizer.convert_tokens_to_ids("<think>")
    think_close=tokenizer.convert_tokens_to_ids("</think>")
    if think_open is None or think_close is None or think_open==tokenizer.unk_token_id or think_close==tokenizer.unk_token_id:
        raise RuntimeError("Qwen think boundary tokens are missing from tokenizer vocabulary")
    opens=(prompt_ids[0]==think_open).nonzero(as_tuple=False).flatten()
    closes=(prompt_ids[0]==think_close).nonzero(as_tuple=False).flatten()
    if len(opens)!=1 or len(closes)!=1 or int(opens[0])>=int(closes[0]):
        raise RuntimeError("expected one ordered empty think block in the no-think prompt")
    if k==0: return prompt_ids
    oi=int(opens[0]);ci=int(closes[0])
    prefix=prompt_ids[:,:oi+1]
    trace=torch.tensor([trace_ids[:k]],device=prompt_ids.device,dtype=prompt_ids.dtype)
    if k==m:
        return torch.cat([prefix,trace,prompt_ids[:,ci:]],dim=1)
    return torch.cat([prefix,trace],dim=1)


def masked_sequence_loss(model, tokenizer, rows, device, mode, alpha=1.0, rollout_temp=1.0, max_prompt=1536, max_response=256, teacher=None):
    losses=[]; metrics={"rollout_tokens":0,"kl_tokens":0,"sft_target_tokens":0,"student_forward_tokens":0,"teacher_forward_tokens":0,"teacher_entropy":0.0}
    for row in rows:
        p = row["prompt"]
        prompt_ids = response_prompt_ids(tokenizer,p,False,device)
        if prompt_ids.shape[1] > max_prompt:
            prompt_ids = prompt_ids[:, -max_prompt:]
        if mode in {"sft_answer", "sft_trace"}:
            if mode == "sft_answer":
                fullids,response_start=response_tokens(tokenizer,p,row["answer"],False)
            else:
                trace_ids = row.get("teacher_trace_token_ids")
                solution_ids = row.get("teacher_solution_token_ids")
                if trace_ids is None or solution_ids is None:
                    raise RuntimeError("exact teacher trace/solution token ids missing; reconstructing from text is disallowed")
                # Paper Appendix A.3 targets z_{k+1:m} followed by solution s.
                k = int(alpha*len(trace_ids))
                suffix=trace_ids[k:]
                if suffix:
                    # With a nonempty CoT suffix, use Qwen's open think-mode
                    # assistant prefix and explicitly train both boundary tokens.
                    prefix=tokenizer(prompt_message(tokenizer,p,True),add_special_tokens=False).input_ids
                    think_open=tokenizer.convert_tokens_to_ids("<think>")
                    think_close=tokenizer.convert_tokens_to_ids("</think>")
                    tail_ids=[think_open]+suffix+[think_close]+solution_ids
                else:
                    # At alpha=1 only the solution remains, so match the
                    # student direct/no-think prompt exactly.
                    prefix=tokenizer(prompt_message(tokenizer,p,False),add_special_tokens=False).input_ids
                    tail_ids=solution_ids
                fullids=list(prefix)+list(tail_ids)
                response_start=len(prefix)
            ids=torch.tensor([fullids],device=device)
            logits=model(input_ids=ids,use_cache=False).logits[:, :-1, :].float()
            labels=ids[:,1:]
            offset=max(0,response_start-1)
            end=labels.shape[1]
            loss=F.cross_entropy(logits[:,offset:end,:].reshape(-1,logits.size(-1)),labels[:,offset:end].reshape(-1))
            metrics["sft_target_tokens"]+=int(end-offset)
            metrics["student_forward_tokens"]+=int(ids.numel())
            losses.append(loss)
            continue
        # Generate a student no-think on-policy rollout, then score that same y
        # under student πθ(y|x) and teacher πT(y|x,z<=kα).
        with torch.no_grad():
            sampled_ids, sampled, _ = gen_ids(model,tokenizer,p,False,max_response,rollout_temp,device)
        metrics["rollout_tokens"] += int(sampled_ids.numel())
        if mode == "msd":
            ttrace = row["teacher_trace"]
            ids_trace = row.get("teacher_trace_token_ids")
            if ids_trace is None:
                raise RuntimeError("exact teacher trace token ids missing; reconstructing from text is disallowed")
            t_input=masked_teacher_context(prompt_ids, row["teacher_trace_token_ids"], tokenizer, alpha)
        else:
            teacher_prompt_ids = prompt_ids
            t_input = prompt_ids
        yids=sampled_ids.unsqueeze(0).to(device)
        if yids.numel()==0:
            continue
        # Use identical student sampled token IDs after both contexts.
        student_full=torch.cat([prompt_ids,yids],dim=1)
        teacher_full=torch.cat([t_input,yids],dim=1)
        metrics["student_forward_tokens"]+=int(student_full.numel())
        metrics["teacher_forward_tokens"]+=int(teacher_full.numel())
        with torch.no_grad():
            teacher_model=teacher if teacher is not None else model
            tlogits=teacher_model(input_ids=teacher_full,use_cache=False).logits[:, t_input.shape[1]-1:-1,:].float()
        slogits=model(input_ids=student_full,use_cache=False).logits[:,prompt_ids.shape[1]-1:-1,:].float()
        logq=F.log_softmax(slogits,dim=-1)
        # Exact reverse KL: sum_v q_student(v) [log q_student(v)-log p_teacher(v)]
        logp_t=F.log_softmax(tlogits,dim=-1)
        token_kl=(F.softmax(slogits,dim=-1)*(logq-logp_t)).sum(-1)
        # Equation (1) uses a sum over response positions and an expectation
        # over sampled examples; preserve that reduction rather than token-mean.
        losses.append(token_kl.sum())
        metrics["kl_tokens"] += token_kl.numel()
        metrics["teacher_entropy"] += float((-(F.softmax(tlogits,dim=-1)*logp_t).sum(-1)).mean().detach().cpu())
    if not losses:
        raise RuntimeError("empty training microbatch")
    return torch.stack(losses).mean(),metrics


def evaluate(model, tokenizer, rows, device, max_new=128, thinking=False, temperature=1.0, seed=20281028):
    good=0; valid=0;tokens=[]; latencies=[]; by_family={};predictions=[]
    for ix,row in enumerate(rows):
        # Identical per-item dev draw at every scheduled checkpoint. Preserve
        # training RNG so checkpoint evaluation cannot change future rollouts.
        torch.manual_seed(seed+ix);torch.cuda.manual_seed_all(seed+ix)
        start=time.perf_counter()
        out,n=gen(model,tokenizer,row["prompt"],thinking,max_new,temperature,device)
        if device.type=="cuda": torch.cuda.synchronize(device)
        latencies.append(time.perf_counter()-start); tokens.append(n)
        label=parse_direct_label(out,row['labels']);ok=label==row["answer"];good+=ok;valid+=int(label is not None)
        predictions.append({'id':row['id'],'family':row['family'],'prediction_raw':out,'prediction':label,'gold':row['answer'],'correct':ok,'schema_valid':label is not None,'generated_tokens':n,'latency_s':latencies[-1]})
        rec=by_family.setdefault(row["family"],[0,0]);rec[0]+=int(ok);rec[1]+=1
    return {"n":len(rows),"accuracy":good/max(1,len(rows)),"correct":good,"schema_valid_rate":valid/max(1,len(rows)),"mean_generated_tokens":sum(tokens)/max(1,len(tokens)),"p50_latency_s":float(torch.tensor(latencies).median()) if latencies else None,"p95_latency_s":float(torch.quantile(torch.tensor(latencies),.95)) if latencies else None,"by_family":{k:{"correct":v[0],"n":v[1],"accuracy":v[0]/v[1]} for k,v in by_family.items()},"predictions":predictions}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",type=Path,required=True); ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--method",choices=["baseline","sft_answer","sft_trace","msd"],required=True)
    ap.add_argument("--alpha",type=float,default=1.0);ap.add_argument("--steps",type=int,default=64);ap.add_argument("--seed",type=int,default=20261025)
    ap.add_argument("--train-file",type=Path,default=None);ap.add_argument("--student-response-cap",type=int,default=256)
    ap.add_argument("--run-name",type=str,default=None)
    ap.add_argument("--gpu",type=int,default=0);ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args()
    if not 0.0 <= a.alpha <= 1.0: raise ValueError("alpha must be in [0, 1]")
    if a.steps < 1: raise ValueError("steps must be positive")
    random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    device=torch.device(f"cuda:{a.gpu}")
    tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").to(device)
    if not tokenizer.chat_template or "enable_thinking" not in tokenizer.chat_template: raise RuntimeError("expected Qwen3 thinking chat template")
    # Fail fast on the actual Qwen3 generation convention used by this run.
    qwen_open=tokenizer.convert_tokens_to_ids("<think>");qwen_close=tokenizer.convert_tokens_to_ids("</think>")
    think_prompt=prompt_message(tokenizer,"fixture",True)
    no_think_prompt=prompt_message(tokenizer,"fixture",False)
    if qwen_open==tokenizer.unk_token_id or qwen_close==tokenizer.unk_token_id or not no_think_prompt.endswith("<think>\n\n</think>\n\n") or think_prompt.endswith("<think>\n") or not think_prompt.endswith("assistant\n"):
        raise RuntimeError("unexpected Qwen3 think/no-think generation prompt boundary")
    print(json.dumps({"model":str(a.model),"mode_teacher":prompt_message(tokenizer,"fixture",True)[:300],"mode_student":prompt_message(tokenizer,"fixture",False)[:300],"layers":model.config.num_hidden_layers,"gpu":torch.cuda.get_device_name(device)},ensure_ascii=False),flush=True)
    train_path=(a.train_file if a.train_file is not None and a.train_file.is_absolute() else a.run/a.train_file) if a.train_file is not None else (a.run/"data/train_teacher_verified.jsonl" if a.method in {"msd","sft_trace","sft_answer"} and (a.run/"data/train_teacher_verified.jsonl").exists() else a.run/"data/train.jsonl")
    train=read_jsonl(train_path);dev=read_jsonl(a.run/"data/dev.jsonl")
    run_name=a.run_name or (f"smoke_{a.method}_a{a.alpha:g}" if a.dry_run else f"{a.method}_seed{a.seed}")
    out=a.run/"results"/run_name
    if out.exists() and (out/"adapter_model.safetensors").exists(): raise RuntimeError(f"refusing to overwrite existing adapter {out}")
    out.mkdir(parents=True,exist_ok=True)
    if a.dry_run:
        if not train or not dev: raise RuntimeError("dry-run requires nonempty train and dev data")
        train=train[:1];dev=dev[:1];a.steps=1
    if a.method=="baseline":
        metrics=evaluate(model,tokenizer,dev,device,thinking=False,temperature=1.0)
        (a.run/"results").mkdir(exist_ok=True,parents=True)
        (a.run/"results/baseline_dev.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"dev":metrics},ensure_ascii=False),flush=True);return
    cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],task_type="CAUSAL_LM")
    model=get_peft_model(model,cfg);model.train()
    # Freeze immutable teacher as a second same-initialization replica only for MSD.
    teacher=None
    if a.method=="msd":
        teacher=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa").to(device).eval()
        for p in teacher.parameters():p.requires_grad_(False)
    elif a.method in {"sft_answer","sft_trace"}:
        pass
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=0.01)
    logs=[]; t0=time.time(); best_dev=None; best_state=None; best_step=None
    for step in range(a.steps):
        rows=([train[0]] if a.dry_run else [train[(step*4+j)%len(train)] for j in range(4)])
        if a.method in {"msd","sft_trace"} and any("teacher_trace" not in r for r in rows):
            raise RuntimeError("teacher_trace missing; run and validate teacher collection before MSD")
        opt.zero_grad(set_to_none=True);micro_losses=[];met_total={"rollout_tokens":0,"kl_tokens":0,"sft_target_tokens":0,"student_forward_tokens":0,"teacher_forward_tokens":0,"teacher_entropy":0.0}
        # Stream four one-example microbatches to bound activation memory while
        # preserving the configured effective batch size and mean loss.
        for row in rows:
            loss,met=masked_sequence_loss(model,tokenizer,[row],device,a.method,alpha=a.alpha,max_response=a.student_response_cap,teacher=teacher)
            (loss/len(rows)).backward();micro_losses.append(float(loss.detach().cpu()))
            for k in met_total: met_total[k]+=met.get(k,0)
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();opt.zero_grad(set_to_none=True)
        rec={"step":step+1,"loss":float(sum(micro_losses)/len(micro_losses)),"gradient_norm":float(grad_norm.detach().cpu()),"rollout_tokens":int(met_total.get("rollout_tokens",0)),"kl_tokens":int(met_total.get("kl_tokens",0)),"teacher_entropy":float(met_total.get("teacher_entropy",0.0)/len(rows)),"seconds":time.time()-t0,"cuda_peak_allocated":torch.cuda.max_memory_allocated(device),"effective_batch_size":len(rows),"microbatch_size":1};logs.append(rec)
        for k in ("sft_target_tokens","student_forward_tokens","teacher_forward_tokens"):rec[k]=int(met_total[k])
        with (out/'steps.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(rec)+'\n')
        if (step+1)%8==0 or step+1==a.steps: print(json.dumps(rec),flush=True)
        if (step+1)%16==0 or step+1==a.steps:
            model.eval()
            with torch.random.fork_rng(devices=[a.gpu]):
                cur=evaluate(model,tokenizer,dev,device,thinking=False,temperature=1.0,seed=a.seed)
            rec["dev_accuracy_at_checkpoint"]=cur["accuracy"]
            preds=cur.pop('predictions')
            (out/f'dev_step{step+1:03d}_predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in preds),encoding='utf-8')
            (out/f'dev_step{step+1:03d}_metrics.json').write_text(json.dumps(cur,indent=2)+'\n',encoding='utf-8')
            if best_dev is None or cur["accuracy"]>best_dev["accuracy"]:
                best_dev=cur;best_state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()};best_step=step+1
            model.train()
    if best_state is not None: set_peft_model_state_dict(model,best_state)
    model.eval(); metrics=best_dev
    model.save_pretrained(out);tokenizer.save_pretrained(out)
    (out/"train_log.json").write_text(json.dumps({"steps":logs,"dev":metrics,"selected_checkpoint_step":best_step,"elapsed_s":time.time()-t0,"gpu_hours":(time.time()-t0)/3600,"peak_memory_bytes":torch.cuda.max_memory_allocated(device),"method_label":a.method,"paper_equation_1_core_objective":a.method=="msd","full_paper_reproduction":False,"implementation_note":"single-node Qwen3-1.7B synthetic decision pilot with LoRA; differs from paper model scale, task domain, optimizer budget and infrastructure","train_file":str(train_path),"n_train_rows":len(train),"alpha":a.alpha if a.method in {"msd","sft_trace"} else None,"student_response_cap":a.student_response_cap,"lora":{"r":16,"alpha":32,"dropout":0.05},"seed":a.seed,"dry_run":a.dry_run,"optimizer":{"name":"AdamW","lr":1e-5,"weight_decay":0.01}},indent=2)+"\n")
    provenance={"train_sha256":hashlib.sha256(train_path.read_bytes()).hexdigest(),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"exposed_train_ids":[train[(step*4+j)%len(train)]['id'] for step in range(a.steps) for j in range(1 if a.dry_run else 4)],"total_sft_target_tokens":sum(r['sft_target_tokens'] for r in logs),"total_student_rollout_tokens":sum(r['rollout_tokens'] for r in logs),"total_kl_tokens":sum(r['kl_tokens'] for r in logs),"total_student_forward_tokens":sum(r['student_forward_tokens'] for r in logs),"total_teacher_forward_tokens":sum(r['teacher_forward_tokens'] for r in logs),"selection":"strict dev generation accuracy; fixed per-item draw; tie earliest checkpoint","token_budget_matched":False,"example_exposures_and_updates_matched":not a.dry_run}
    (out/'training_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({"dev":metrics,"elapsed_s":time.time()-t0,"gpu_hours":(time.time()-t0)/3600},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
