#!/usr/bin/env python3
"""Generate a separate, unviewed synthetic test from an independent seed.

This was unviewed during development but remains synthetic and target-informed;
it cannot support a broad external-generalization claim.
"""
from pathlib import Path
import argparse
from generate_tasks import FAMILIES, instance, force_answer, solve, render, digest, verify, VERSION, ANSWER_SPACES, write_jsonl
import random, json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=20281025);ap.add_argument('--n',type=int,default=1000);ap.add_argument('--template',default='template_e');ap.add_argument('--quiet',action='store_true');a=ap.parse_args()
    rng=random.Random(a.seed); rows=[];seen=set();family_i={f:0 for f in FAMILIES}
    for i in range(a.n):
        family=FAMILIES[i%len(FAMILIES)]; local_i=family_i[family]; family_i[family]+=1
        target=ANSWER_SPACES[family][local_i%len(ANSWER_SPACES[family])]
        p=force_answer(family,instance(family,rng,local_i),target); answer,cert=solve(family,p)
        prompt,labels=render(family,p,a.template)
        sid=digest(f'{VERSION}|untouched|{family}|{json.dumps(p,sort_keys=True)}')[:20]
        split_label='synthetic_template_transfer_check_v6_unopened' if 'v6_unopened' in a.out.name else ('synthetic_template_transfer_check_v5_unopened' if 'v5_unopened' in a.out.name else ('synthetic_balanced_template_transfer_check_v3' if 'v3_balanced' in a.out.name else 'synthetic_template_transfer_check'))
        row={'id':sid,'source_id':sid,'split':split_label,'family':family,'template_family':a.template,'parameters':p,'prompt':prompt,'answer':answer,'labels':labels,'certificate':cert,'difficulty':{'family':family,'input_characters':len(prompt)},'generator_version':VERSION,'generator_seed':a.seed,'verifier':'generate_tasks.py::solve/v1'}
        if answer!=target or not verify(row) or sid in seen: raise RuntimeError(f'invalid external synthetic item {i}')
        seen.add(sid);rows.append(row)
    write_jsonl(a.out,rows)
    summary={'n':len(rows),'seed':a.seed,'families':{f:sum(r['family']==f for r in rows) for f in FAMILIES},'answers_by_family':{f:{label:sum(r['family']==f and r['answer']==label for r in rows) for label in ANSWER_SPACES[f]} for f in FAMILIES},'answer_content_not_read_or_evaluated':True}
    if a.quiet:
        # Keep the held-out labels out of routine generator output and sidecar
        # summaries; the post-generation verifier reports only pass/fail/hash.
        safe_summary={'n':len(rows),'seed':a.seed,'template':a.template,
                      'families':{f:sum(r['family']==f for r in rows) for f in FAMILIES},
                      'held_out_items_not_rendered_or_summarized':True}
        Path(a.out).with_suffix('.summary.json').write_text(json.dumps(safe_summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    else:
        Path(a.out).with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(summary))
if __name__=='__main__':main()
