#!/usr/bin/env python3
"""Train-head evaluation with exact block outputs and a real incremental stopper."""
from __future__ import annotations
import argparse, json, random, statistics, time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from train_readout import ChoiceHead, read, qwen_prompt, DEPTHS

class StopPrefix(Exception):
    pass

class PrefixRunner:
    """Capture the decoder output before Qwen3's final RMSNorm and stop there."""
    def __init__(self, model, depth):
        self.model, self.depth, self.hidden = model, depth, None
    def _hook(self, module, inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        self.hidden = value[0, -1].float()
        raise StopPrefix()
    def __call__(self, input_ids):
        hook = self.model.model.layers[self.depth - 1].register_forward_hook(self._hook)
        try:
            try:
                self.model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, return_dict=True)
            except StopPrefix:
                pass
        finally:
            hook.remove()
        if self.hidden is None:
            raise RuntimeError(f"layer {self.depth} hook did not fire")
        return self.hidden

def candidate_embeddings(tok, embedding, row, device):
    out = torch.zeros((1, len(row['labels']), embedding.shape[1]), device=device)
    for j, label in enumerate(row['labels']):
        ids = tok(str(label), add_special_tokens=False).input_ids
        out[0, j] = embedding[torch.tensor(ids)].float().mean(0)
    return out

def predict_head(head, hidden, candidates, labels):
    logits = head(hidden[None], candidates)[0].float()
    probs = torch.softmax(logits, dim=-1)
    j = int(logits.argmax().item())
    return labels[j], float(probs.max().item()), {str(labels[k]): float(probs[k].item()) for k in range(len(labels))}

def incremental_dynamic(model, heads, tok, embedding, row, device, threshold):
    """One prefill and one incremental block loop; halt as soon as frozen rule fires."""
    ids = qwen_prompt(tok, row['prompt'])
    input_ids = torch.tensor([ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    hidden = model.model.embed_tokens(input_ids)
    kw = {'config': model.config, 'inputs_embeds': hidden, 'attention_mask': attention_mask,
          'past_key_values': None, 'position_ids': positions}
    masks = {'full_attention': create_causal_mask(**kw)}
    if model.model.has_sliding_layers:
        masks['sliding_attention'] = create_sliding_window_causal_mask(**kw)
    rotary = model.model.rotary_emb(hidden, positions)
    previous = None
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=masks[model.config.layer_types[i]],
                       position_ids=positions, past_key_values=None, use_cache=False,
                       position_embeddings=rotary)
        depth = i + 1
        if depth not in (4, 8, 12, 16, 20, 24, 28):
            continue
        cand = candidate_embeddings(tok, embedding, row, device)
        label, confidence, _ = predict_head(heads[depth], hidden[0, -1].float(), cand, row['labels'])
        if depth == 28 or (depth >= 8 and label == previous and confidence >= threshold):
            return label, confidence, depth
        previous = label
    raise RuntimeError('incremental loop ended without a final exit')

def load_heads(run, device):
    heads = {}
    for depth in DEPTHS:
        head = ChoiceHead().to(device).eval()
        head.load_state_dict(torch.load(run / f'checkpoints/readout_heads/head_layer{depth}.pt', map_location=device, weights_only=True))
        heads[depth] = head
    return heads

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True); ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--split', default='pilot_holdout'); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--repeats', type=int, default=5); ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--output-stem', default='readout_pilot_holdout_valid'); ap.add_argument('--calibrate-only', action='store_true'); ap.add_argument('--smoke-dynamic-dev', type=int, default=0); ap.add_argument('--skip-warmup', action='store_true')
    a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); device = torch.device(f'cuda:{a.gpu}')
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(a.model), local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    embedding = model.get_input_embeddings().weight.detach(); heads = load_heads(a.run, device)
    runners = {d: PrefixRunner(model, d) for d in DEPTHS}
    dev_rows = read(a.run / 'data/dev.jsonl'); dev_pred = {str(d): {} for d in DEPTHS}; dev_conf = {str(d): {} for d in DEPTHS}
    # Dev-only warmup and predictions using the same pre-final-norm block state as training.
    for d in DEPTHS:
        for row in dev_rows:
            ids = qwen_prompt(tok, row['prompt']); x = torch.tensor([ids], device=device)
            with torch.inference_mode():
                hidden = runners[d](x)
                label, conf, _ = predict_head(heads[d], hidden, candidate_embeddings(tok, embedding, row, device), row['labels'])
            dev_pred[str(d)][row['id']] = label; dev_conf[str(d)][row['id']] = conf
    reference = sum(dev_pred['28'][r['id']] == r['answer'] for r in dev_rows) / len(dev_rows)
    eligible = []
    for ti in range(50, 96, 5):
        tau = ti / 100; choices = []
        for row in dev_rows:
            exits = [d for d in (8, 12, 16, 20, 24) if dev_pred[str(d)][row['id']] == dev_pred[str(d-4)][row['id']] and dev_conf[str(d)][row['id']] >= tau]
            d = exits[0] if exits else 28
            choices.append((d, dev_pred[str(d)][row['id']] == row['answer']))
        accuracy = sum(ok for _, ok in choices) / len(choices); mean_layers = sum(d for d, _ in choices) / len(choices)
        if accuracy >= reference - 0.02 and mean_layers <= 28 * 0.8:
            eligible.append({'threshold': tau, 'dev_accuracy': accuracy, 'reference_dev_accuracy': reference, 'mean_executed_layers': mean_layers})
    selected = sorted(eligible, key=lambda x: (x['mean_executed_layers'], -x['threshold']))[0] if eligible else None
    policy = {'policy': 'single prefill; incrementally execute blocks; from depth 8 exit at d if heads at d and d-4 agree and max softmax at d >= tau; else finish at 28',
              'check_depths': [4, 8, 12, 16, 20, 24, 28],
              'selection': 'dev accuracy >= depth28 accuracy - 0.02 AND mean executed layers <= 22.4 (20% below 28); among eligible minimize mean layers; ties choose larger tau',
              'selected': selected, 'eligible_thresholds': eligible, 'dev_n': len(dev_rows),
              'depth28_dev_accuracy': reference, 'latency': 'one incremental decoder-layer loop; no full-depth backfill'}
    policy_path = a.run / 'configs/dynamic_stopper_dev_frozen.json'; policy_path.parent.mkdir(exist_ok=True)
    if a.calibrate_only:
        policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(policy, ensure_ascii=False), flush=True); return
    if a.smoke_dynamic_dev:
        if not policy_path.exists(): raise RuntimeError('freeze dev stopper policy first')
        tau = selected['threshold'] if selected else 0.55
        smoke_rows = dev_rows[:a.smoke_dynamic_dev]; smoke = []
        for row in smoke_rows:
            if device.type == 'cuda': torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode(): label, conf, depth = incremental_dynamic(model, heads, tok, embedding, row, device, tau)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            smoke.append({'id': row['id'], 'prediction': label, 'gold': row['answer'], 'correct': label == row['answer'], 'depth': depth, 'latency_s': time.perf_counter()-t0})
        print(json.dumps({'split':'dev_smoke_only','threshold':tau,'n':len(smoke),'mean_executed_layers':sum(x['depth'] for x in smoke)/max(1,len(smoke)),'rows':smoke},ensure_ascii=False),flush=True); return
    if not policy_path.exists():
        raise RuntimeError('run --calibrate-only on dev and freeze dynamic_stopper_dev_frozen.json before test evaluation')
    frozen = json.loads(policy_path.read_text(encoding='utf-8'))
    if frozen != policy:
        raise RuntimeError('dev-selected policy differs from frozen file; refusing test evaluation')
    rows = read(a.run / f'data/{a.split}.jsonl')
    # 10 warmups on development rows for each fixed-depth path.
    if not a.skip_warmup:
        for d in DEPTHS:
            for row in dev_rows[:10]:
                ids = qwen_prompt(tok, row['prompt']); x = torch.tensor([ids], device=device)
                with torch.inference_mode(): _ = runners[d](x)
    fixed = {d: [] for d in DEPTHS}; dynamic = []; start = time.time()
    for rep in range(a.repeats):
        schedule = [(d, ix) for d in DEPTHS for ix in range(len(rows))]
        random.Random(a.seed + rep * 101).shuffle(schedule)
        for d, ix in schedule:
            row = rows[ix]; ids = qwen_prompt(tok, row['prompt']); x = torch.tensor([ids], device=device)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                hidden = runners[d](x)
                label, conf, probs = predict_head(heads[d], hidden, candidate_embeddings(tok, embedding, row, device), row['labels'])
            if device.type == 'cuda': torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
            fixed[d].append({'id': row['id'], 'family': row['family'], 'gold': row['answer'], 'prediction': label, 'correct': label == row['answer'], 'confidence': conf, 'probabilities': probs, 'repeat': rep, 'latency_s': dt})
        if selected:
            order = list(range(len(rows))); random.Random(a.seed + rep * 7919).shuffle(order)
            for ix in order:
                row = rows[ix]
                if device.type == 'cuda': torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                with torch.inference_mode(): label, conf, depth = incremental_dynamic(model, heads, tok, embedding, row, device, selected['threshold'])
                if device.type == 'cuda': torch.cuda.synchronize(device)
                dynamic.append({'id': row['id'], 'family': row['family'], 'gold': row['answer'], 'prediction': label, 'correct': label == row['answer'], 'confidence': conf, 'executed_layers': depth, 'exit_depth': depth, 'repeat': rep, 'latency_s': time.perf_counter() - t0})
        print(json.dumps({'repeat': rep + 1, 'elapsed_s': time.time() - start}), flush=True)
    report = {'protocol': f'{a.split}, exact hooked block output, one-shot deterministic quality, {a.repeats} latency repeats', 'n_items': len(rows), 'repeats': a.repeats, 'dynamic_policy': frozen['selected'], 'depths': {}, 'dynamic': None, 'elapsed_s': time.time() - start, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device)}
    predictions = []
    for d, vals in fixed.items():
        unique = {x['id']: x for x in vals if x['repeat'] == 0}; lat = sorted(x['latency_s'] for x in vals); fam = {}
        for x in unique.values():
            z = fam.setdefault(x['family'], [0, 0]); z[0] += int(x['correct']); z[1] += 1
        report['depths'][str(d)] = {'n_items': len(unique), 'n_item_repeat': len(vals), 'correct': sum(x['correct'] for x in unique.values()), 'accuracy': sum(x['correct'] for x in unique.values()) / len(unique), 'p50_latency_s': statistics.median(lat), 'p95_latency_s': lat[min(len(lat)-1, int(.95*len(lat)))], 'mean_executed_layers': d, 'by_family': {k: {'correct':v[0], 'n':v[1], 'accuracy':v[0]/v[1]} for k,v in fam.items()}}
        predictions.extend({'condition': f'fixed_{d}', **x} for x in vals)
    if dynamic:
        unique = {x['id']: x for x in dynamic if x['repeat'] == 0}; lat = sorted(x['latency_s'] for x in dynamic); fam = {}
        for x in unique.values():
            z = fam.setdefault(x['family'], [0, 0]); z[0] += int(x['correct']); z[1] += 1
        report['dynamic'] = {'n_items':len(unique), 'n_item_repeat':len(dynamic), 'correct':sum(x['correct'] for x in unique.values()), 'accuracy':sum(x['correct'] for x in unique.values())/len(unique), 'mean_executed_layers':sum(x['executed_layers'] for x in unique.values())/len(unique), 'p50_latency_s':statistics.median(lat), 'p95_latency_s':lat[min(len(lat)-1,int(.95*len(lat)))], 'by_family':{k:{'correct':v[0],'n':v[1],'accuracy':v[0]/v[1]} for k,v in fam.items()}}
        predictions.extend({'condition':'dynamic', **x} for x in dynamic)
    out = a.run / 'results'; out.mkdir(exist_ok=True)
    (out / f'{a.output_stem}_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (out / f'{a.output_stem}_predictions.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in predictions), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
