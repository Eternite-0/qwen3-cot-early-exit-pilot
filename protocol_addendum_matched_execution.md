# Matched pilot execution and selection freeze

Frozen 2026-10-26 Asia/Shanghai after train-only teacher collection and before any non-smoke matched training or v6 model evaluation. Original `preregistration.md` remains unchanged. This addendum governs the new matched pilot rather than the earlier unmatched answer-only control.

## Data and estimand

The merged teacher-screened corpus has 320 of the first 600 original train instances (SHA-256 recorded by `results/teacher_train_collection_merged.json`). Family counts are evidence 71, intent 73, multihop 45, routing 91, rules 36, time/numeric 4. Teacher generation produced 444 closed traces and 415 correct final answers. The screen establishes symbolic/numeric grounding and a correct final label; it is not a step-level logical proof, human validation, or proof that every trace is reliable. These limitations apply to all matched methods equally. Do not add rejected or held-out rows to any training condition.

The estimand is transfer among the generator's six specified task families and a new surface template/parameter seed. Generator shortcuts (e.g. forced routing feasibility and direct-edge reachable cases) and repeated latent graph/intent structures can limit difficulty and independence despite unique case IDs. Report any latent overlap audit, weak cells, and family outcomes. This is not an external JevBench or real-robot estimate.

## Execution clarifications

Run all seven conditions with 64 updates, effective batch 4, first 256 accepted rows in the same frozen original-index order, LoRA/optimizer and seed as in the teacher training addendum. The identical dataset, number of example exposures, and updates are matched. Target/rollout tokens and actual compute are measured, not equalized; do not label this experiment a token-matched comparison. SFT uses response-token-mean cross entropy; MSD uses the equation-(1) sum of per-response-token full-vocabulary reverse KL. This normalization difference is a declared implementation detail. Each model remains the same Qwen/Qwen3-1.7B revision; no model swap is counted as distillation gain.

Scheduled checkpoint selection is at 16/32/48/64 updates on the original 300-row dev set, with the same per-item sampling seed at each checkpoint, temperature 1, top-p 1, top-k disabled and cap 128. Use strict direct-label schema: trim whitespace and one terminal EOS, then require exact equality to an allowed label, with no extraction or repair. Choose highest strict dev accuracy; tie chooses earliest step. Save checkpoint dev predictions. Dev evaluation RNG is isolated from subsequent training sampling. This replaces the earlier permissive last-line correctness for new matched checkpoint selection; the historical teacher grounding screen is unchanged.

MSD α=1 is the primary complete-internalization pilot versus matched answer-only SFT; α=.5 is secondary; α=0 is a no-information sanity control. All seven model conditions are evaluated once on v6 after selection is frozen. For later readout diagnostics select one of MSD α=.5/1 on dev only (higher strict accuracy; tie selects α=1). Train the same candidate-conditioned heads for original, answer-only, and dev-selected MSD, using the same 320 accepted examples, epochs and dev policy. Keep native readout and trained readout separate.

The minimum expansion gate stays +3 percentage points for MSD α=1 over matched answer-only SFT on the held-out transfer pilot, with a paired 95% interval excluding clearly harmful changes. No 20k expansion unless that gate and teacher/data coverage justify it. One training seed is a feasibility pilot; no multi-seed robustness claim.

## Resources and validation

Two existing RTX 4090s are available. The actual teacher collection used about 4861/4882 seconds on the two GPUs (2.706 summed GPU-hours), peak allocated memory about 3.56 GB per replica. Step smoke fits on one GPU: SFT peaks 4.17–6.32 GB and MSD about 7.53 GB for the short response fixture. Formal response cap is 256; retain any allocation failure logs and do not mislabel an implementation failure as a method result.

For this continuation, permit up to 8 additional GPU-hours for matched training, fixed heads, dev calibration, and frozen evaluation, and up to 8 GiB incremental files. This extends the earlier run-local estimate, using only already authorized local/server resources. Log actual GPU time. If resources are insufficient, omit repeated training seeds and secondary controls first, report statuses, and keep the test closed until condition selection is final. No paid resources or shared environment changes.

The held-out artifact is v6: SHA-256 `12152f0add59b37e0107774ec739138a4da799243fffed16108411ae94d45c29`, 559786 bytes. Historical pilot/v2/v3 and public JevBench examples are diagnostics already viewed and are not an untouched final test. Do not regenerate v6 or use its outcomes to choose training, prompts, or early-exit thresholds.
