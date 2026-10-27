#!/usr/bin/env python3
"""Dev-only diagnostic for the frozen symbolic grounding screen."""
import json,re,sys,pathlib,collections
root=pathlib.Path(__file__).resolve().parents[1]
raw=[json.loads(x) for x in (root/'results/teacher_cap_diagnostic_dev.jsonl').read_text(encoding='utf-8').splitlines()]
dev={x['id']:x for x in map(json.loads,(root/'data/dev.jsonl').read_text(encoding='utf-8').splitlines())}
pat=r'\b(?:\d+|[A-Z][A-Z0-9_]{1,})\b';summary=collections.defaultdict(collections.Counter)
examples={}
for x in raw:
 if not x['trace_closed']:continue
 row=dev[x['id']];trace=x['cot_raw'].split('</think>',1)[0].replace('<think>','')
 source=row['prompt']+' '+' '.join(row['certificate']);allowed=set(re.findall(pat,source))|{row['answer']};atoms=set(re.findall(pat,trace));missing=atoms-allowed
 summary[x['family']]['closed']+=1;summary[x['family']]['screened']+=int(x['trace_screen_pass']);summary[x['family']]['numeric_atom_violations']+=int(any(t.isdigit() for t in missing))
 if missing and x['family'] not in examples:examples[x['family']]={'id':x['id'],'missing_atom_count':len(missing),'answer_correct':x['cot_answer_correct'],'screened':x['trace_screen_pass']}
out={'protocol':'dev-only analysis of existing cap diagnostic; no threshold or data changed','by_family':{f:dict(c) for f,c in summary.items()},'first_missing_atom_examples':examples,'policy':'retain original screen unchanged for the frozen teacher collection'}
(root/'results/teacher_trace_screen_audit_dev.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'dev_only_audit_saved':True,'families':len(summary),'protocol_threshold_changed':False}))
