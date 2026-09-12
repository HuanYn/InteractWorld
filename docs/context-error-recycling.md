# Native97 因果历史误差回收（实验性）

目标是检验训练时只见干净历史、推理时收到有误差历史的差异，是否影响后半段人物和场景稳定性。
这是一项尚待 GPU/画质验证的独立训练方法，不是已经解决人物形变的结果报告。

## 借鉴来源与实现范围

参考 [NVlabs/LongLive 的 error-recycling 实现](https://github.com/NVlabs/LongLive/blob/6b36d20ec6f7958d29d11a704dfa64611a9f2572/model/diffusion.py)，
固定来源 commit `6b36d20ec6f7958d29d11a704dfa64611a9f2572`。
本项目只独立实现其中的历史条件误差回收思想，保留 ABot/Wan 模型代码及第三方归属；
不加载 LongLive 或 LingBot 权重，不将上游展示效果写成本项目结果。

- 主体仍是自己的 Action Adapter + rank16 LoRA，接入 ABot 的 block-causal Wan 模型。
- 输入支持原49帧与 native97帧；97帧必须由连续 RGB 独立编码为25个 latent，分为首帧加8个3-latent块，不能拼接旧49帧缓存。
- 新预测误差为 `error = (noisy - sigma * predicted_flow) - GT_clean`，从真实前向的 detached 预测逐块计算。
- 每次先从旧误差库构造 `clean_x`，完成当前前向后才将当前误差入库。首 latent 不扰动、不入库，loss 仍排除首帧。
- 只修改独立历史条件，保持 future noisy/flow target/sigma、prompt、动作及数据 RNG 不变；原因果 mask 只让 noisy 块看已完成的历史，不看当前或未来 GT 块。
- 误差块以 CPU BF16 存储，按 sigma 入桶，历史注入跨非空桶采样；独立 CPU RNG 与完整误差库一起保存和恢复。

这是 **context-only error recycling**：不是完整 SVI、DMD、CD、位置误差桶、分布式共享误差库，
也不是让模型在线完成15秒自 rollout 后反传。当前不扰动 noisy 分支，因此不套用那些同时改写 noisy 输入与 velocity 目标的完整配方。
它可能减轻对干净历史的过度依赖，但也可能放大已有错误，不能保证提升身份保持、视角或画质。

## 同源配对配置

模板为 `configs/train/causal_context_error_recycling_example.yaml`。
先将 `/project` 路径替换为自己的绝对路径，填入真实父 checkpoint SHA256 和 manifest SHA256。
必须使用自己已经训练的 **native97、absolute-index resampled Action checkpoint**，以及它绑定的同一
window97 manifest、feature index、receipt、静态文本和动作尺度；不要改写旧 checkpoint 内的来源来通过检查。
示例路径中的 step120 只是已有父模型位置示意，源码不附带权重或缓存。

复制模板为 candidate/control 两份；对照仅改变 `error_recycling.enabled: false` 与独立的新 `training.output_dir`。
两边保留同一父权重、数据顺序、训练 seed42、学习率、batch1×accum8、200步上限和每20步 checkpoint。
候选参数为 `clean_prob=0.25`、`context_inject_prob=0.5`、`error_scale=0.25`，独立 seed1419；
warmup 为16次前向观测，即2个 optimizer steps。实际是否发生注入，以累计指标为准。
误差库全局上限64MiB，每个 sigma 桶最多8个块；这是 CPU 缓冲上限，不是整项训练的显存需求。

开启时 checkpoint stage 为 `causal_context_error_recycling_v1`，buffer method 为 `context_error_recycling_v1`。
关闭时保留 `causal_teacher_forcing_v1` 与旧配置序列化行为。开启/关闭不是严格 resume 可变参数。

## 20步执行段与严格恢复

本例不由旧版 `configure_reproduction.py` 自动生成，需要单独填写以上模板。先做不加载模型的 CPU 检查：

```bash
python train_causal_teacher_forcing.py --config /project/configs/recycling.yaml --dry-run
```

下面复用 [README 的 `run_gpu` 函数](../README.md#3-串行缓存和训练)，只适用于操作者有权使用、通过门禁的独立单卡机器。
每个长任务放在自己的 screen/tmux 中，并单独 tee 日志；公开代码不附带私人授权、预算或自动调度器。

```bash
# YAML max_steps 始终为200；执行段到20停下并保存，不修改采样总长度或训练配置哈希。
run_gpu train_causal_teacher_forcing.py --config /project/configs/recycling.yaml \
  --stop-after-step 20 2>&1 | tee /project/logs/recycling-first20.log

# 首段通过资源和实际输出检查后，用同一份 YAML 和本运行自己的 checkpoint 继续。
run_gpu train_causal_teacher_forcing.py --config /project/configs/recycling.yaml \
  --resume /project/runs/causal-context-recycling/checkpoints/step-0000020.pt \
  2>&1 | tee /project/logs/recycling-resume20.log
```

Control 运行同样的两段命令，改用它自己的 YAML、输出和日志路径。
不要为了分段把 `--max-steps` 改为20：那会改变配置和 absolute-index 采样总长度，不是本例的严格恢复流程。
`execution_result.json` 在正常段结束时记录 `segment_complete`、`execution_step_cap`、真实 step/microcursor 与 checkpoint；
到200才记录完整结束。严格恢复校验 stage、配置、哈希、父来源与误差观测数，并恢复 optimizer、数据游标、
Python/NumPy/Torch/CUDA RNG 和误差库独立 RNG；保留 best1+last2，best 只按训练 loss 选择。

## 最小结果记录与边界

两边都记录同步的 `optimizer_step_seconds`、峰值显存和 loss；候选另记录误差库字节数、条目/桶数、
`total_injected_forwards`、`total_injected_blocks` 和恢复状态。没有实际注入的短跑不能证明该机制执行过。
CPU 测试涵盖首图/target 不变、旧误差先用后更新，以及分段恢复；它们不能证明5B训练能装进某张显卡。

新 checkpoint 不会自动进入旧 Demo。评测必须明确支持并绑定它的真实 stage、配置、父来源和权重 SHA，
不能改名成 Action、MoBA 或旧 Causal 权重绕过检查。比较使用相同首图、静态 prompt、seed 和15秒动作：
观看人物/场景稳定性，并检查 correct 的 weighted MSE 是否同时低于 zero 与 shuffled。
数值条件通过仍不代表动作语义、泛化或长期画质已通过；发布运行结果前不能宣称收益或实时性能。
