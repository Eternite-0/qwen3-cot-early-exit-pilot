# 结果解释与归因

本轮确实执行了参数训练，存储不是已证实的训练障碍；但**尚未验证 CoT 内化能带来满足预注册标准的浅层加速**。MSD1相对answer-only的+1.76pp区间跨0且未达到+3pp扩展门槛，未扩20k；不能将这个有限pilot解释为原论文方法失败。

原始direct已经只生成3.29个token，MSD1为3.39个；完整深度p50分别103.66/204.05ms，answer-only为203.40ms。当前active unmerged LoRA后端成本包含在内，MSD1没有比原始direct更快或比answer-only更少生成token。72题explicit-CoT对照平均544.14token、p50为18.09秒，但36.11%请求触及768token上限；其60.28%准确率属于这72题和该截断预算，不能当成无限预算teacher能力或与500题总均值直接比较。CoT与no-think间的巨大token差不能全部归因于参数蒸馏。

固定出口的名义成功点是MSD1/SFT的24层trained head，相对原始direct生成参考包含了readout/服务路径变化。MSD1 L24为67.40%、p50/p95为57.92/60.18ms；同readout自身L28为65.00%、67.17/69.78ms。真正保持backbone/readout相同的截层p50改善约13.8%，未达到20%；质量差+2.40pp，marginal95%CI[-1.59,+6.47]。同为24层trained head，MSD1−answer-only仅+0.60pp，CI[-1.44,+2.68]，不支持将快速决策归因于MSD的独特增益。

另补充了**查看主结果后**的全48出口对自身完整生成参考的描述分析，未重选规则或改主检验：MSD1 L24相对自身完整生成质量差−3.52pp，marginal95%CI[-6.98,−0.10]，未达到2pp非劣标准。见 `results/own_generation_reference_metrics.json/.csv` 和 `post_result_reference_analysis.md`；这不是新增预注册确认检验。

两层和八层MSD trained head均为42.60%。五个任务族全部输出固定标签，intent中84.34%输出A17；它不是已学会可靠浅层决策的证据，详见label诊断。动态规则已经在dev真实增量计时，因未过门槛没有入选test策略。

原train池与v6有153/500题精确潜在参数重复；进一步按实际训练暴露计，adapter前256行覆盖43/500题，head320行覆盖53/500题，见 `results/actual_exposure_test_overlap.json`。探索性排除所有历史已见参数后剩298题，MSD1−answer-only为+2.82pp，CI[+0.27,+5.50]，但其中rules族为0题，且这是目标启发的探索子集，不能替代主扩展门槛或修复原隔离设计。

AG News120上，MSD1相对原始direct为+4.17pp，CI[+1.17,+7.50]；仅说明该窄任务未观察到下降，不证明一般能力全面保持。完整资源、全部alpha/SFT proxy及失败日志均保留。数据隔离、teacher步骤可靠性和方法完整梯度核实等后续最小修正见 `NEXT_EXPERIMENT.md`。
