# Qwen3 CoT Internalization and Early-Exit Pilot

This repository contains the code, protocol notes, and figures for a small **synthetic generator-transfer pilot** studying whether CoT internalization can support shallow decision readouts and early exits in Qwen3-1.7B.

The experiment follows the requested main line: answer-only SFT and an MSD-style on-policy reverse-KL surrogate, followed by candidate-conditioned intermediate readouts and dynamic early-exit checks. It does not implement a loop architecture and is not a reproduction of LEV or Jeva.

## What the pilot found

The matched held-out pilot used 500 generated decision items, one training seed, and five interleaved timing repeats per item.

| condition | accuracy | p50 latency | p95 latency | mean output tokens |
| --- | ---: | ---: | ---: | ---: |
| Original direct | 59.48% | 103.66 ms | 136.75 ms | 3.29 |
| Answer-only SFT | 69.16% | 203.40 ms | 269.89 ms | 3.32 |
| MSD-style α=1 | 70.92% | 204.05 ms | 307.30 ms | 3.39 |

MSD-style α=1 was 1.76 percentage points above matched answer-only SFT, with paired 95% CI [-0.33, 3.89] pp. The preregistered expansion gate required at least +3 pp with a non-negative lower bound, so the pilot was not expanded. The latency gate also failed.

Candidate-conditioned heads at layers 2 and 8 collapsed toward fixed labels. The best descriptive trained-head point was layer 24, but the same-readout comparison against its layer-28 reference improved p50 latency by about 13.8%, below the 20% target, and its paired quality interval did not establish the planned non-inferiority margin. No dynamic rule was eligible for final testing. AG News was only a 120-item narrow regression check.

These results do not establish JevBench performance, broad capability retention, real robot-intervention effectiveness, or failure of the original MSD paper. The implementation is explicitly an **MSD-style on-policy full-vocabulary KL surrogate pilot**, not a verified full-paper reproduction. The synthetic splits also have latent task overlap after decorative case identifiers are removed; this limits generalization claims.

## Repository contents

- `scripts/`: data generation, teacher collection, training, readout, exit evaluation, analysis, and plotting code.
- `figures/`: editable SVG and PDF figures plus 300 dpi PNG previews generated from saved pilot results.
- `METHOD_AUDIT.md`, `RESULT_INTERPRETATION.md`, and `NEXT_EXPERIMENT.md`: method correspondence, interpretation, and a prospective recovery plan.
- `protocol_addendum_*.md`: frozen implementation and measurement notes.
- `results/summary.json`: a small, path-free summary of the reported results.

Large or sensitive artifacts are intentionally excluded: model weights and tokenizer files, adapters and head checkpoints, raw synthetic data, teacher traces, per-item predictions, logs, caches, archives, manifests, and SSH/synchronization helpers. They are not needed to inspect the code or reported conclusions.

## Reproduction notes

The scripts expect a local run directory containing a user-supplied Qwen3-1.7B checkpoint, tokenizer, generated data, and saved training/evaluation outputs. They do not download a model or access a server. Reproduction must use a new timestamped directory and must keep final evaluation data independent from model or threshold selection.

```bash
python -m pip install -r requirements.txt
python scripts/generate_tasks.py --help
python scripts/train_pilot.py --help
python scripts/run_matched_training.py --help
python scripts/analyze_matched.py --help
```

The saved pilot used Python 3.11, PyTorch 2.8, Transformers 5.10, PEFT 0.19, BF16 Qwen3-1.7B, SDPA attention, LoRA rank 16, α∈{0, 0.5, 1}, 64 updates, and one seed. These values describe the saved run; they are not a recommendation to reuse its compromised synthetic split.

## License and attribution

Code is released under the MIT License. Qwen3 weights and tokenizer remain subject to their upstream license and are not redistributed here. The MSD comparison is based on the public paper [Masked Self-Distillation](https://arxiv.org/abs/2607.22629); no author-owned implementation was verified for this pilot.
