# 新 run 预注册：CoT 内化、训练 readout 与逐层早退

冻结时间：2026-10-25（Asia/Shanghai），在生成/查看任何本 run 数据答案前。实验目录：`runs/jev_cot_early_exit_20261025_2109/`；服务器目录固定为 `<timestamped run directory>/`。旧 run 与其冻结预注册保持原样。旧 JevBench 231 项已经查看，只作为已知公开诊断资料，不再称为 untouched final test。

## 问题、估计对象与范围

估计在程序可验证、目标任务启发的六类合成 typed-decision 任务中，把完整 Qwen3 推理轨迹通过 masked self-distillation 内化到 no-think student 后，是否能以更少真实生成 token 保持答案正确率；以及训练型候选条件化中间层 readout/逐层 stopper 是否能降低同一次前向的端到端延迟。外部 JevBench 迁移仅属后续已知公开诊断。本实验不验证真实机器人干预，不以 looped model、LEV 或 Jeva 为目标。

## 论文定义与实现忠实度

算法依据 Kalwar et al., *Masked Self-Distillation: Internalizing the Chain-of-Thought in Language Models*, arXiv:2607.22629v3（2026-09-30），正文 §3、式 (1)–(4)，附录 A.1、A.3。原文令 teacher trace 为 `z=(z1…zm)`，`kα=floor(αm)`；teacher 条件为 `x,z≤kα,y<t`，student 条件为 `x,y<t`，`y~πSθ(.|x)` 是 student on-policy rollout；目标为逐 token `Σt DKL(πSθ(.|x,y<t) || πT(.|x,z≤kα,y<t))`，全词表上的 reverse KL，梯度只作用于 student。α 是被 teacher 观察、student 被遮蔽并需内化的 think trace 前缀比例：α=0 无掩码，α=1 完整掩码。mask 依据 teacher 生成的 `<think>…</think>` 边界；solution token 紧接 trace。Teacher 与 student 从同一 checkpoint 权重初始化，teacher think、student no-think。SFT proxy 使用固定 teacher completion 的未遮蔽 trace 后缀及 solution 作 token cross-entropy；它是 off-policy 对照，不称为 MSD。

论文参数报告：MSD Qwen3-4B/8B 使用 200 steps、全局 question batch 16、AdamW LR 1e-6、无 warmup、weight decay 0.01、clip 1；单问题 4B 8 rollouts、8B 5 rollouts；采样 temperature 1/top-p 1/top-k -1，单机 2–8 Gaudi2。论文附录 SFT 是 30k steps、LR 1e-5，故本 pilot 的预算匹配 SFT 是额外的工程控制，不是作者 SFT replication。

官方实现审计：在 arXiv HTML v3 正文、附录及公开 GitHub repository 搜索中未发现可归属此论文的官方代码仓库；GitHub code search API 要求登录，不能据此断言不存在。源记录：`https://arxiv.org/html/2607.22629v3`、`https://api.github.com/search/repositories?q=%22Masked+Self-Distillation%22`（2026-10-25）；本 run 对算法规范以可读正文/附录为准，明确标记为 paper-spec implementation，非官方代码复现。任何训练脚本如没有逐 token teacher/student logits reverse KL 就必须标为 SFT proxy。

## 模型与资源预算

- 主初始化：公开后训练 `Qwen/Qwen3-1.7B`，Transformers 原生 Qwen3 CausalLM；HF API 实测 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，Base 初始化关联 `Qwen/Qwen3-1.7B-Base`。新 run 下载后保存模型卡、config、tokenizer/template 文件和每个权重 shard 的 SHA-256。chat template 需明确设置 `enable_thinking=True` 给 teacher、`False` 给 student，并在 fixture 中验证渲染内容和标签。
- Base 对照：缓存 `Qwen/Qwen3-1.7B-Base` revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`，权重 SHA-256 `6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841`；与 post-trained checkpoint 差异只报告为模型差异，不能算蒸馏收益。
- 服务器：RTX 4090 ×2，单卡 24,564 MiB；Driver 595.58.03，PyTorch 2.8.0+cu128 / CUDA build 12.8，Transformers 5.10.2、PEFT 0.19.1、Accelerate 1.13.0；旧 venv 无 datasets/trl/matplotlib/scipy。`/ai` 当场可用 64 GiB，根 `/` 仅 11 GiB。run、HF cache、临时 checkpoint 均放 `/ai`，不得写根盘默认缓存。训练最多占 1 GPU 做 student、另 1 GPU 做 frozen teacher；先验证峰值显存和吞吐。
- 新合成数据固定 2,000 train、300 dev、500 target-informed pilot holdout；按潜在实例分配 split 后生成文本。生成器 seed 20261025，家族与模板按 split 固定。额外“untouched external”只在后续另建封存文件，不在此 run 用测试反馈调参。
- 计算预算上限：模型单次下载约 3.5 GiB；数据低于 0.2 GiB；checkpoint LoRA 每法不超过 1 GiB；本轮最多 6 GPU-hours 和 `/ai` 10 GiB 新增落盘。实测阶段计时后更新估时；超限先停止重复 seed/α 网格，不删除缓存或他人文件。

## 数据、过滤和泄漏控制

六类规则/例外、多步路由、时间数值、多跳依赖、证据充分性、意图抽取；由确定性 solver 生成答案/certificate。按潜在参数实例、任务族与 split 专属模板族切分；train/dev/pilot 的 prompt 模板分别隔离，精确 hash 和 source_id 跨 split 不重叠。记录参数、seed、solver verifier、answer、certificate、模板、难度代理。dev 用于训练预算/teacher 信号质量检查与早退门槛；pilot_holdout 只在超参、epoch/checkpoint、阈值和规则冻结后运行一次。另创建 seed 20271025、template d 的独立 synthetic test；先由符号 verifier 检查并只保存、不打开题目或标签，所有方法与阈值冻结后方可评测。该集合只估计 generator-defined synthetic transfer，不称外部真实泛化。公开 JevBench 结果从不参与训练或选择。

Teacher 的 direct 与 explicit-CoT 使用相同 300 条 dev 上比较，之后在 train 生成 teacher 轨迹；两者 temperature=1, top-p=1, top-k=0、max-new=256，结果按数据 seed 和 generation seed 固定。保留条件：严格 parse、答案匹配 solver、答案 schema 合法；trace 必须可提取且至少 2 个非空推理步骤，所有可机读数字/实体提及只能取自输入或 solver certificate；抽样人工盲审 100 条（若没有可完成的独立人工审阅，则标注步骤可靠性“未证实”，不得称 verified CoT）。最终答案正确不自动表示步骤可靠。matched-data answer-only SFT 与 MSD 使用完全相同的 teacher 样本、过滤集、初始化和 token budget。

## 条件、训练与选择

所有条件使用同一冻结 post-trained 初始化、LoRA 配置、同一训练实例次序及同等 student token/optimizer-step budget，teacher 始终冻结。先以一个预注册 seed 20261025 做 pilot：原模型 direct/no-think、原模型 explicit CoT/think、answer-only SFT、matched-data teacher suffix SFT proxy、on-policy reverse-KL MSD α∈{0,0.5,1.0}。主训练每条件最多 64 optimizer steps，microbatch 1、梯度累积 4（有效 batch 4）、LoRA r=16/alpha=32、dropout 0.05、AdamW LR 1e-5、max context 1536、response cap 256；温度 1、top-p 1、top-k 禁用；定期按固定 dev loss 保存，选 dev 答案准确率最高 checkpoint，平局取更早 step。实际 tokenizer 长度、吞吐和峰值显存先测再启动，若预算不符则在 pilot holdout 未访问前按记录理由统一缩小所有方法步数。

最小继续扩大门槛：pilot holdout 上完整深度 no-think MSD 相对 matched-data answer-only SFT 至少 +3 percentage points accuracy，且 paired 95% CI 不与明显伤害相容；否则保留无效/不确定 pilot，不以公开题调参。pilot 只用一个 seed，正式扩展前要求资源允许时至少 3 seeds。

轻量 readout：冻结 backbone，在 dev 之外的训练 split 隐藏状态上训练候选语义条件化 pairwise/choice head；标签集合和排列可变，不按固定标签编号训练。层深 2（探索）、4/8/12/16/20/24、28；所有出口共享初始化。若 train/dev 上相对原生输出有预注册学习信号，才进入递增逐层 dynamic stopper。Stopper/head 开销、每层 KV/状态开销均计入；必须在单次前向完成，不可全深度回填浅层时延。

## 读出、测试模式、指标

先运行 `hf_tokenization_fixture.py` 检查完整 HF 拼接 token 序列与旧独立 token 追加的差异、预测位移、右 padding、单/多 token、共前缀、并列、FP32/BF16 head。固定容差：同 dtype logits `atol=2e-3, rtol=2e-3`；FP32 与 BF16 head 的 label log-score 差须记录，分数排序变化计为不稳定。实测 17 个候选里 5 个发生边界 token merge；HF NLL 与手工 shift gather 最大差为 0；右补齐 batch 与逐题 logits 最大差 0.3125；BF16 与 FP32 head 最大 logit 差 0.06254。因此评分采用完整拼接后 tokenization、显式响应预测位移和逐候选无 padding（或另行验证通过的左 padding）；旧输出不修补。自由生成主准确率用严格解析，不自动修复答案；中间 readout 单独列为训练头或未训练 head。另测真实 direct-answer 和显式 CoT 自回归生成延迟/token 数；输出严格按同一 prompt 序列化。

主指标 accuracy、严格 schema validity、平均/分位真实输出 token、token/s、串行 batch-1 warm p50/p95 latency、峰值显存、实际执行层数；总体和 task family/难度/候选数/上下文长度分层，题目 paired bootstrap 95% CI。候选条件每组先 10 次 warm-up，再随机交错，至少 5 次硬件时延重复。训练曲线保留 loss/KL/gradient norm/GPU 小时。动态出口额外报告 risk-coverage、平均层数、head/stopper 时间。一般能力回归先查公开独立验证集及许可；无可验证数据则标未测。无可靠轨迹标签则生成干预未测。

非劣判据：准确率相对同初始化完整 direct 基线最多下降 2 pp，配对 95% 区间下界须高于 -2 pp；p50 至少改善 20%，p95 不恶化。层出口先由 dev 的 paired disagreement/cluster 分布做样本量评估。所有阈值仅 dev 选择。任何在 pilot_holdout 上的结论明确为 target-informed synthetic pilot，不声称 JevBench/general deployment 泛化。

## 运行状态与预注册偏差

当前只冻结计划与来源；在 fixture、checkpoint revision/hash、数据去重/solver 校验、teacher 质量及实测吞吐/预算核验后才开始相关阶段。任何偏差写入 `protocol_deviations.md`，不得修改本文件以追溯覆盖。
