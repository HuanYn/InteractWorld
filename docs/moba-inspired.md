# MoBA 启发的双向辅助训练（实验性）

目标是缓解转为因果生成后画质退化。参考 [LingBot-World 2.0 论文 3.2 节](https://arxiv.org/html/2607.07534v1#S3.SS2)
保留双向训练信号的思路，在原 Wan2.2 因果模型上独立实现，不加载 LingBot 权重或复制其训练代码。
本页不是画质改善报告：GPU 运行、峰值显存和固定15秒效果均待验证。

## 实际实现

`L = L_TF + lambda_bid * L_BID`，建议首个有限对照使用 `0` 与 `0.1`。

- 两部分共享权重、样本、噪声、时间步、初始图像、8维数值动作和静态场景描述。
- TF 沿用 clean/noisy 因果可见性；BID 看同一 noisy 序列的全双向上下文，不能传入 clean_x 或 KV。
- 两次 forward/backward 顺序执行，累加同一组梯度后按梯度积累策略更新，避免同时保留两套激活。
- 初始 latent 固定为干净条件、时间步0，并从两项 loss 中排除。
- 双向 mask 独立于原因果 mask 缓存；默认训练和 KV 推理仍走原模式。
- 新方法标识 `moba_inspired_sequential_bid_v1`，checkpoint stage 为 `causal_moba_regularized_v1`；
  不允许把它伪装成旧 TF checkpoint 或静默混用恢复配置。

这不是官方 packed 多分支 mask、逐 chunk 动态文本或完整 MoBA 训练配方。
没有 CD 的一致性目标、DMD 的 score-difference 目标，也没有完成长时自生成历史训练。
训练中双向可见未来 noisy token 是辅助去噪目标，不是让因果推理看到真实未来。

## 使用边界

入口 `train_causal_moba.py`；模板 `configs/train/causal_moba_example.yaml`。
先替换模板里的 `/project` 路径，指向自己已训练的 Action checkpoint、相同静态文本缓存及真实哈希。
无 `--launch` 时仅检查配置并输出 CPU 计划，不下载模型、不运行 GPU。

```bash
python train_causal_moba.py --config /absolute/path/to/causal-moba.yaml
```

首次 GPU 使用需要与现有训练相同的明确 GPU 门禁参数和独立的新输出目录。
weight=0 的对照必须与候选保持同一父 checkpoint、噪声、数据次序、学习率和更新数，
不能与来源不同的历史 TF 运行直接比较。顺序计算降低激活重叠，但不保证适合任意显存。
验证通过后再接入正式生成；旧展示服务不会自动接受新 stage 或替换正在使用的模型。

验收沿用冻结首帧、静态文字、seed、动作的15秒原视频，分别检查形变与场景漂移；
动作对照要求正确动作 weighted MSE 同时低于 zero 和 shuffled。
loss 下降、CPU 测试通过和代码可启动均不能代替上述结果。
