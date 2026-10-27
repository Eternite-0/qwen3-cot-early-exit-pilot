# Reproduction guide

This is a code-and-analysis repository. It does not contain model weights, tokenizer files, raw data, teacher CoT, checkpoints, or per-item outputs. Supply those artifacts from an authorized source and work in a new timestamped run directory.

## Environment

The saved pilot used Python 3.11, PyTorch 2.8.0 with CUDA 12.8, Transformers 5.10.2, PEFT 0.19.1, Accelerate 1.13.0, and an RTX 4090. Matching versions reduce numerical and timing differences but do not guarantee identical GPU results.

```bash
python -m pip install -r requirements.txt
```

## Recommended order

1. Create independent train, development, and holdout splits with `scripts/generate_tasks.py` and audit latent instance keys before rendering text.
2. Run the tokenization and decision-path fixtures: `hf_tokenization_fixture.py`, `verify_decision_paths.py`, and `verify_msd_mask.py`.
3. Collect and screen teacher traces with `collect_teacher.py` and `collect_train_traces.py`. The historical lexical grounding screen is not a proof of step-by-step reasoning.
4. Train answer-only SFT and MSD-style conditions in fresh output directories with `run_matched_training.py`.
5. Train candidate-conditioned heads, run fixed and dynamic exit evaluation, and measure the head/stopper overhead in the same forward pass.
6. Analyze and plot only saved outputs with `analyze_matched.py`, `analyze_layer_attribution.py`, and `plot_matched_presentation.py`.

All final test choices, thresholds, and checkpoint selection rules must be frozen before reading final holdout outcomes. The historical pilot used one seed and did not pass its expansion gate; repeating it does not repair the data-isolation or method-fidelity limitations documented in `METHOD_AUDIT.md`.

## Reproduction boundary

The reported figures are reconstructions of saved pilot outputs. Recreating the exact numbers requires the excluded local artifacts and the original software/model revision. This repository intentionally provides no credentials, remote paths, synchronization code, or private data.
