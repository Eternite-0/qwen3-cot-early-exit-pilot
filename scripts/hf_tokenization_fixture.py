#!/usr/bin/env python3
"""HF continuation tokenization and score alignment fixture.

Run locally with transformers, or pass --model-dir on the server. No benchmark
items are read. This fixture explicitly reveals boundary-merge mismatches in
the old independent-prefix/suffix tokenization rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-cases", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("fixture_report.json"))
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
    cases = [
        ("Decision:\n", ["YES", " NO", "NO", "ACCEPT", "ACCEPTED"]),
        ("Outcome: ", ["North", "Northstar", "South", "Southbound"]),
        ("Final answer=", ["42", " 42", "421", "-1"]),
        ("Label\n", ["A", "B", "ALLOW", "DENY"]),
    ]
    report = {"model": str(args.model_dir), "dtype": str(next(model.parameters()).dtype), "cases": [], "hf_vs_manual_max_abs": 0.0, "right_padding_max_abs": 0.0, "left_padding_max_abs": 0.0}
    mismatch_count = 0
    count = 0
    for prefix, candidates in cases:
        pids = tok(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        for candidate in candidates:
            # Canonical rule: tokenize the concatenated string; suffix positions
            # are whatever tokenization of that concatenation yields beyond the
            # longest common token prefix with the prompt-only encoding.
            full_ids = tok(prefix + candidate, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
            lcp = 0
            while lcp < min(len(pids), len(full_ids)) and pids[lcp] == full_ids[lcp]:
                lcp += 1
            old_sids = tok(candidate, add_special_tokens=False)["input_ids"]
            boundary_merge = lcp != len(pids)
            mismatch_count += int(boundary_merge)
            count += 1
            # HF causal LM logits at position t predict input token t+1. Compare
            # teacher-forced sum against explicit gather from standard HF output.
            ids = torch.tensor([full_ids], device=args.device)
            with torch.inference_mode():
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                logits = out.logits.float()
                positions = torch.arange(lcp - 1, len(full_ids) - 1, device=args.device)
                targets = ids[0, lcp:]
                manual = torch.log_softmax(logits[0, positions], dim=-1).gather(1, targets[:, None]).sum()
                hf_labels = ids.clone()
                hf_labels[:, :lcp] = -100
                hf_loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=hf_labels, use_cache=False).loss.float()
                hf_sum = -hf_loss * (len(full_ids) - lcp)
                hidden = model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
                standard = out.logits
                bf16_head = model.lm_head(hidden).float()
                fp32_head = F.linear(hidden.float(),model.lm_head.weight.float())
                head_delta = float((bf16_head-fp32_head).abs().max().cpu())
                hf_head_delta = float((standard.float()-bf16_head).abs().max().cpu())
                delta = float((manual - hf_sum).abs().cpu())
            report["hf_vs_manual_max_abs"] = max(report["hf_vs_manual_max_abs"], delta)
            report["cases"].append({"prefix": prefix, "candidate": candidate, "prefix_ids": pids, "full_ids": full_ids, "lcp_tokens": lcp, "suffix_ids_canonical": full_ids[lcp:], "suffix_ids_old_independent": old_sids, "boundary_merge_detected": boundary_merge, "hf_loss_sum_abs_diff": delta, "bf16_vs_fp32_head_max_abs":head_delta,"hf_logits_vs_bf16_head_max_abs":hf_head_delta})
            if len(report["cases"]) >= args.max_cases:
                break
        if len(report["cases"]) >= args.max_cases:
            break
    # Right-padding test: standard HF attention_mask logits must match unpadded
    # per-row runs at every non-padding position, and prediction labels use t-1.
    testseq=[]
    for prefix,cands in cases[:2]:
        pids=tok(prefix,add_special_tokens=False).input_ids
        for cand in cands[:2]:
            testseq.append(pids+tok(cand,add_special_tokens=False).input_ids)
    maxlen=max(map(len,testseq)); pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batch=torch.full((len(testseq),maxlen),pad,dtype=torch.long,device=args.device); mask=torch.zeros_like(batch)
    for i,s in enumerate(testseq): batch[i,:len(s)]=torch.tensor(s,device=args.device);mask[i,:len(s)]=1
    with torch.inference_mode():
        bo=model(input_ids=batch,attention_mask=mask,use_cache=False).logits.float()
        for i,s in enumerate(testseq):
            so=model(input_ids=torch.tensor([s],device=args.device),attention_mask=torch.ones((1,len(s)),device=args.device),use_cache=False).logits.float()
            delta=float((bo[i,:len(s)]-so[0]).abs().max().cpu());report["right_padding_max_abs"]=max(report["right_padding_max_abs"],delta)
    # Left-padding control for decoder-only batches.
    left_batch=torch.full((len(testseq),maxlen),pad,dtype=torch.long,device=args.device);left_mask=torch.zeros_like(left_batch);offsets=[]
    for i,s in enumerate(testseq):
        off=maxlen-len(s);offsets.append(off);left_batch[i,off:]=torch.tensor(s,device=args.device);left_mask[i,off:]=1
    with torch.inference_mode():
        lo=model(input_ids=left_batch,attention_mask=left_mask,use_cache=False).logits.float()
        for i,s in enumerate(testseq):
            so=model(input_ids=torch.tensor([s],device=args.device),attention_mask=torch.ones((1,len(s)),device=args.device),use_cache=False).logits.float()
            delta=float((lo[i,offsets[i]:]-so[0]).abs().max().cpu());report["left_padding_max_abs"]=max(report["left_padding_max_abs"],delta)
    report["boundary_merges_detected"] = mismatch_count
    report["n_cases"] = count
    report["tolerance"] = {"hf_manual_atol": 2e-3, "hf_manual_rtol": 2e-3}
    if report["hf_vs_manual_max_abs"] > 2e-3:
        raise AssertionError(f"HF/manual prediction alignment mismatch {report['hf_vs_manual_max_abs']}")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary={k:v for k,v in report.items() if k!="cases"}
    summary["bf16_vs_fp32_head_max_abs"] = max(x["bf16_vs_fp32_head_max_abs"] for x in report["cases"])
    summary["hf_logits_vs_bf16_head_max_abs"] = max(x["hf_logits_vs_bf16_head_max_abs"] for x in report["cases"])
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
