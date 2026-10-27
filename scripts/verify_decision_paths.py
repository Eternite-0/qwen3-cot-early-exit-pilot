#!/usr/bin/env python3
"""Fresh fixtures for cached candidate scoring, real exits, precision and permutations."""
from pathlib import Path
import argparse,json,time,torch
from peft import PeftModel
from matched_common import load_model,load_tokenizer,write,DEPTHS
from decision_paths import native_scores,load_heads,fixed_head,limited_layers
from train_readout import qwen_prompt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args();dev=torch.device(f'cuda:{a.gpu}');tok=load_tokenizer(a.run);model=load_model(a.run,'direct',dev)
    report={'fixture':'new artificial prompts only; no held-out rows','native_attention_backend':'BF16 default SDPA; shared full-logit prefill and tokenwise cached branches','cached_sum_loglik_atol':.15,'cached_sum_loglik_rtol':.005,'cases':[]}
    cases=[{'prompt':'Choose A or AB. Return only the label.','labels':['A','AB']},{'prompt':'Return a direction label.','labels':['NORTH','NORTHSTAR','SOUTH']},{'prompt':'Return a permitted numeric label.','labels':['42','421','-1']},{'prompt':'Return one label.','labels':['yes','no']}]
    with torch.inference_mode():
        for row in cases:
            cached=native_scores(model,tok,row,dev,28)
            prompt=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            pids=tok(prompt,add_special_tokens=False).input_ids;scores=[]
            for label in row['labels']:
                ids=tok(prompt+label+tok.eos_token,add_special_tokens=False).input_ids
                assert ids[:len(pids)]==pids
                x=torch.tensor([ids],device=dev);o=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False)
                logits=o.logits[0,len(pids)-1:-1].float();target=x[0,len(pids):]
                scores.append(torch.log_softmax(logits,-1).gather(1,target[:,None]).sum())
            probs=torch.softmax(torch.stack(scores),-1)
            delta=max(abs(float(probs[i])-cached['probabilities'][lab]) for i,lab in enumerate(row['labels']))
            print(json.dumps({'fixture_labels':row['labels'],'reference_probs':probs.tolist(),'cached_probs':cached['probabilities'],'probability_max_abs':delta,'reference_ll':[float(z) for z in scores],'cached_ll':cached['log_likelihoods']}),flush=True)
            assert delta<=.025
            score_delta=max(abs(float(scores[i])-cached['log_likelihoods'][lab]) for i,lab in enumerate(row['labels']))
            assert all(abs(float(scores[i])-cached['log_likelihoods'][lab])<=.15+.005*abs(float(scores[i])) for i,lab in enumerate(row['labels']))
            report['cases'].append({'labels':row['labels'],'cache_vs_full_probability_max_abs':delta,'cache_vs_full_loglik_max_abs':score_delta,'same_top1':row['labels'][int(probs.argmax())]==cached['prediction']})
        counts={d:0 for d in range(1,29)};hooks=[]
        for d,layer in enumerate(model.model.layers,1):
            def hook(module,inputs,output,d=d):counts[d]+=1
            hooks.append(layer.register_forward_hook(hook))
        try:native_scores(model,tok,cases[0],dev,4)
        finally:
            for h in hooks:h.remove()
        assert all(v==0 for d,v in counts.items() if d>4)
        report['native_layer4_actual_block_invocations']=counts
        equal=native_scores(model,tok,{'prompt':'Choose a label.','labels':['yes','yes']},dev,28)
        assert equal['probabilities']['yes']==.5
        report['tie_rule']='first original candidate (torch.argmax); identical candidates give equal scores'
        heads=load_heads(a.run,'direct',dev);row=cases[1]
        z=fixed_head(model,heads,tok,row,dev,8)
        swapped={**row,'labels':list(reversed(row['labels']))};rev=fixed_head(model,heads,tok,swapped,dev,8)
        diff=max(abs(z['probabilities'][k]-rev['probabilities'][k]) for k in row['labels']);assert diff<2e-6
        report['head_candidate_permutation_probability_max_abs']=diff
        x=torch.tensor([qwen_prompt(tok,cases[0]['prompt'])],device=dev);h=model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).last_hidden_state[:,-1]
        fp32=torch.nn.functional.linear(h.float(),model.lm_head.weight.float());bf16=model.lm_head(h).float();head_diff=float((fp32-bf16).abs().max())
        report['fp32_vs_bf16_head_max_abs']=head_diff;assert head_diff<.25
    merge_report=[]
    for condition in ('matched_sft_answer_a1_seed20281028','matched_msd_a1_seed20281028'):
        raw=load_model(a.run,'direct',dev);adapter=PeftModel.from_pretrained(raw,str(a.run/'results'/condition),is_trainable=False).eval()
        with torch.inference_mode():
            before=adapter(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.float()
            unwrapped=adapter.get_base_model()(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.float()
            unwrapped_delta=float((before-unwrapped).abs().max());assert unwrapped_delta==0.
            merged=adapter.merge_and_unload().eval();after=merged(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=1).logits.float()
        difference=float((before-after).abs().max())
        merge_report.append({'condition':condition,'max_abs_logit_delta':difference,'same_first_argmax':int(before[0,-1].argmax())==int(after[0,-1].argmax()),'atol':.5,'merge_accepted':difference<=.5,'chosen_unwrapped_HF_with_active_LoRA_max_abs':unwrapped_delta})
        del before,unwrapped,after,merged,adapter,raw
        torch.cuda.empty_cache()
    report['merge_precision_audit']=merge_report
    report['inference_choice']='active unmerged LoRA projections in HF causal LM; exact fixture equivalence to PeftModel; BF16 weight merge not used'
    report['status']='PASS';write(a.run/'results/matched_decision_path_fixture.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
