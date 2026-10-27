#!/usr/bin/env python3
"""Validate Qwen teacher-context boundaries for alpha=0/partial/1."""
from pathlib import Path
import argparse,json,torch
from transformers import AutoTokenizer
from train_pilot import response_prompt_ids,masked_teacher_context

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,use_fast=True)
    prompt=response_prompt_ids(tok,'Return a label.',False,'cpu')
    trace=[tok.convert_tokens_to_ids(x) for x in ('A','B','C','D')]
    contexts=[masked_teacher_context(prompt,trace,tok,x) for x in (0.0,0.5,1.0)]
    close=tok.convert_tokens_to_ids('</think>')
    assert torch.equal(contexts[0],prompt)
    assert close not in contexts[1][0].tolist()
    assert close in contexts[2][0].tolist()
    assert contexts[1][0,-len(trace[:2]):].tolist()==trace[:2]
    print(json.dumps({'alpha0_equals_student_prompt':True,'partial_context_leaves_think_open':True,
                      'alpha1_context_contains_close_marker':True,'context_lengths':[int(x.shape[1]) for x in contexts]}))
if __name__=='__main__':main()
