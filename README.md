# InterActWorld — ABot-inspired training reproduction

从官方 **Wan2.2-TI2V-5B 基座**训练动作 Adapter + LoRA、因果模型和 LongForcing-lite，
最后生成动作条件视频。不使用官方 ABot 成品模型替代自己的训练。

本仓库公开的是可复现的代码与配置，**不是已经达到稳定画质的产品**。
当前旧预览仍有人物形变、重复纹理和条纹；稳定展示效果尚未交付，也未达到实时交互。
LongForcing-lite 是本项目的少步 endpoint 蒸馏实现，**不是完整复现官方 LongForcing/DMD**。
代码运行成功、训练 loss 下降或视频达到 15 秒，都不等于画质和动作控制已经通过。

week LongForcing 配置使用 reentrant 梯度检查点，保留编译的 FlexAttention，
并显式保存每次调用的张量输入和注意力状态；其他训练配置的梯度检查点模式不变。
FlexAttention 为已知窗口形状保留有限的编译容量，并禁止静默退回高显存的普通执行路径；
这不改变注意力数学，也不是完整训练或画质通过的保证。
若中断前尚无 checkpoint，保留失败目录并从父权重新建实验；不能将日志中的步数当作可恢复权重。

不包含权重、数据、生成素材、训练日志、服务器配置或操作方授权记录。

## 异步展示界面

`scripts/serve_abot_demo.py` 提供预置场景、15秒真实动作时间线、任务队列、状态、播放和下载。
每次生成调用本项目自训 checkpoint，不以历史录像代替新任务。视频上方照片与实际静态文本保留，
提示词铺满照片右侧；左下 WASD、右下方向键显示模型收到的动作。
界面默认仅预览；实际 GPU 生成需要部署方提供独立授权/预算门禁和完整模型资产。
运行方式与验证边界见 [异步 Demo 文档](training/demo/README.md)。
这仍是实验原型，不能据此宣称稳定15秒画质、可靠动作控制或实时性能。

动作残差尺度现在贯通训练、LongForcing replay 与视频推理，并随 checkpoint 保存；
旧 checkpoint 缺省仍按 1.0 读取，不会被悄悄改成新值。新建 Action 修复实验可用
`--warm-start-from /path/to/action-checkpoint.pt` 只继承模型权重，重新初始化 optimizer、
RNG 和 step=0；这不同于严格的同运行 `--resume`，也不同于旧 gate20 的 `--initialize-from`。
修改尺度或学习率必须使用独立输出目录，不能冒充从父 checkpoint 的步数继续。
`action_teacher_5090_repair_r003.yaml` 是实验候选，使用非零动作尺度和零 LoRA 学习率，
**不是画质或控制已通过的推荐预设**；下游贯通的 CPU 测试不等于 GPU/15 秒视觉验收。

`cache_scene_static_prompts.py` 可只编码原始 `scene_static` 字段，新增独立文本缓存，
不重写视频/动作缓存，也不在缺失时退回带动作叙述的 narrative。
Action 的可选 `data.prompt_cache_path` 使用该缓存及其哈希收据；只有 weights-only
warm start 允许显式改变文本缓存，严格 resume 仍检查完整配置和内容哈希。
`action_teacher_5090_repair_r004.yaml` 用于检验动态叙述是否削弱按键条件：
它保持 R003 的尺度与学习率，仅替换文本条件，属于未完成画质/控制验证的实验配置。

Causal 训练现要求静态文本 sidecar 的策略、路径、缓存/收据哈希及 `action_scale`
与父 Action checkpoint 一致，不一致直接报错，不会静默退回旧 narrative。
LongForcing 的自生成历史与短窗口 replay 现共用该静态文本缓存，且严格继承
Action/Causal 父模型的文本条件及动作尺度；旧版未声明缓存的配置仍保留原行为。
Demo 素材从同一缓存收据读取准确文字，并绑定来源 episode；评测拒绝缺失或不一致的
文本缓存/收据/文字。以上是 CPU 已覆盖的接口修复，尚未完成新下游 GPU 训练和画质验证。
R005 是仅将 LoRA 学习率从 0 调至 2e-6 的候选实验，尚未通过画质、控制或 15 秒稳定性验证。
新增独立的 [MoBA 启发双向辅助训练](docs/moba-inspired.md)：共享因果骨干，顺序计算
TF 与双向 flow loss。它是实验性简化实现，不是官方 packed MoBA；GPU 训练、显存和
15秒画质收益尚未验证。一致性蒸馏 CD 和完整 DMD 仍未实现。

Action 可选重采样工厂 `training.data.action_resampled:build_resampled_action_teacher_dataloader`
按绝对样本序号重采样窗口、噪声和时间步，避免旧循环反复使用同一条件。
从旧采样 checkpoint 切换时使用 `--sampling-transition-from`，严格核对来源，
只加载权重并新建 optimizer/RNG/step0；不要将采样切换冒充原运行的严格 resume。

新增实验性 [native97 因果历史误差回收](docs/context-error-recycling.md)：在因果训练的
独立历史条件中重用旧预测残差，保留干净首图，不改变 noisy 输入、flow 目标或数值动作。
97帧缓存使用内部静态文本特征，`prompt_cache_path: null` 在此契约下不是旧 narrative。
它借鉴 LongLive 的 context error-recycling，不是完整 SVI、DMD 或长时自生成 rollout；
不加载其成品权重，也不自动替换当前 Demo。公开配置包含同源开关对照、20步执行段和
严格恢复用法；新阶段的 GPU 资源表现、动作控制和15秒画质仍需实际验证。

## CPU 快速检查

Python 3.12；FFmpeg/FFprobe 用于媒体契约测试，不需要 GPU 或模型下载。

```bash
git clone https://github.com/HuanYn/InteractWorld.git
cd InteractWorld
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q tests \
  --ignore=tests/test_run_abot_long_followthrough.py \
  --ignore=tests/test_prepare_abot_first_results.py \
  --ignore=tests/test_export_public_repro.py
```

CI 执行同一 CPU 测试范围；没有完整模型前向、GPU 训练或画质验收。
缺少 FFmpeg 时安装发行版的 `ffmpeg` 包，否则相关媒体测试会跳过。

## GPU 端到端复现

以下为 Linux Bash 命令。参考路线使用单张 32GB RTX 5090、64GB 主存、
Python 3.12 和 PyTorch 2.8.0/CUDA 12.8；不是对其他显卡显存或耗时的承诺。
准备可写数据盘，建议至少 250GB 空闲；80GB 视频之外还需要基座、缓存和 checkpoint。
需要适配 CUDA 12.8 的 NVIDIA 驱动、FFmpeg/FFprobe、C/C++ 编译器和 Python 开发头文件；
因果 attention 的 Triton/Inductor JIT 会使用编译器。

### 1. 环境与本地路径

只修改第一行的数据盘目录。后续生成的配置、下载、缓存和输出都放在该目录。
仓库本身无需放在同一目录。

```bash
export INTERACTWORLD_ROOT=/absolute/path/on/data-disk/interactworld
mkdir -p "$INTERACTWORLD_ROOT"
python3.12 -m venv "$INTERACTWORLD_ROOT/venv"
source "$INTERACTWORLD_ROOT/venv/bin/activate"
export HF_HOME="$INTERACTWORLD_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$INTERACTWORLD_ROOT/cache/pip"
export XDG_CACHE_HOME="$INTERACTWORLD_ROOT/cache"
export TMPDIR="$INTERACTWORLD_ROOT/tmp"
export TORCHINDUCTOR_CACHE_DIR="$INTERACTWORLD_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$INTERACTWORLD_ROOT/cache/triton"
export CUDA_CACHE_PATH="$INTERACTWORLD_ROOT/cache/cuda"
mkdir -p "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$INTERACTWORLD_ROOT/logs"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt
```

这是非量化路线，不要求上游实时 Demo 的 SageAttention/LightX2V 栈。
FlashAttention 可选；若安装，必须匹配 Python、PyTorch、CUDA 和 C++ ABI。
未安装时 attention 有 SDPA 路径，但速度/显存可能不同，不能沿用原运行的资源数据。
直接依赖列表不等于完整环境锁；正式运行请保存自己的 `pip freeze`。

### 2. 官方基座、公开数据与 manifest

基座 revision：`921dbaf3f1674a56f47e83fb80a34bac8a8f203e`。
数据 revision：`49118ecb23a069abdab522b3cbd1f3d0588d040c`。
目录末级保留固定基座名称，以保持 checkpoint 来源校验。

```bash
export BASE="$INTERACTWORLD_ROOT/models/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
hf download Wan-AI/Wan2.2-TI2V-5B \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir "$BASE"
hf download acvlab/ABot-World-Explorer-500h metadata.jsonl --repo-type dataset \
  --revision 49118ecb23a069abdab522b3cbd1f3d0588d040c \
  --local-dir "$INTERACTWORLD_ROOT/data/index"
python scripts/fetch_abot_subset.py --project-root "$INTERACTWORLD_ROOT" \
  --metadata "$INTERACTWORLD_ROOT/data/index/metadata.jsonl" \
  --payload-root "$INTERACTWORLD_ROOT/data/ABot-World-Explorer-500h" \
  --state-dir "$INTERACTWORLD_ROOT/data/index/subset" \
  --revision 49118ecb23a069abdab522b3cbd1f3d0588d040c \
  --video-cap-bytes 80000000000 --required-output-frames 241 \
  --allow-third-person --execute
python scripts/build_abot_manifest.py \
  --metadata "$INTERACTWORLD_ROOT/data/index/subset/selected_metadata.jsonl" \
  --payload-root "$INTERACTWORLD_ROOT/data/ABot-World-Explorer-500h" \
  --output "$INTERACTWORLD_ROOT/data/manifests/train.jsonl" \
  --required-frames 241 --split-seed abot-v1 --allow-non-first-person
python scripts/configure_reproduction.py --data-root "$INTERACTWORLD_ROOT" \
  --output-dir "$INTERACTWORLD_ROOT/configs"
```

这一路线允许第三人称、排除 Minecraft；不要描述成严格第一人称数据。
下载器记录版本/文件哈希，manifest 固定 train/dev/test 划分。
实际合格数量以生成的收据为准，不能假设换机后一定得到某个固定样本数。

### 3. 串行缓存和训练

先检查自己拥有使用权的独立单卡机器；当前 GPU 门禁不支持共享多卡调度。
它要求无其他计算进程，并只允许有限的桌面显存占用。
`run_gpu` 在每次启动时取得本机 UUID 和新时间戳；不会停止其他人的进程。
所有训练 CLI 默认仅做 CPU 配置报告，必须显式 `--launch` 才会用 GPU。

```bash
set -euo pipefail
run_gpu() {
  local gpu_uuid
  gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
  python "$@" --launch --confirmed-gpu-index 0 \
    --confirmed-gpu-uuid "$gpu_uuid" \
    --confirmed-at-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --allocation-profile dedicated_local_single_gpu
}
cache_models=(--vae-path "$BASE/Wan2.2_VAE.pth"
  --t5-path "$BASE/models_t5_umt5-xxl-enc-bf16.pth"
  --tokenizer-path "$BASE/google/umt5-xxl")

# 49 RGB frames -> 13 latents; up to eight windows per episode.
run_gpu scripts/cache_abot_features.py \
  --manifest "$INTERACTWORLD_ROOT/data/manifests/train.jsonl" \
  --cache-root "$INTERACTWORLD_ROOT/data/features" \
  --max-windows-per-episode 8 --shard-size 2 --seed 42 --reuse-completed \
  "${cache_models[@]}" 2>&1 | tee "$INTERACTWORLD_ROOT/logs/cache49.log"

run_gpu train_action_teacher.py --config "$INTERACTWORLD_ROOT/configs/action.yaml" \
  2>&1 | tee "$INTERACTWORLD_ROOT/logs/action.log"
run_gpu train_causal_teacher_forcing.py --config "$INTERACTWORLD_ROOT/configs/causal.yaml" \
  2>&1 | tee "$INTERACTWORLD_ROOT/logs/causal.log"

# 241 RGB frames -> 61 latents; one continuous window per episode.
run_gpu scripts/cache_abot_long_features.py \
  --manifest "$INTERACTWORLD_ROOT/data/manifests/train.jsonl" \
  --cache-root "$INTERACTWORLD_ROOT/data/features" \
  --max-windows-per-episode 1 --shard-size 1 --seed 42 --reuse-completed \
  "${cache_models[@]}" 2>&1 | tee "$INTERACTWORLD_ROOT/logs/cache241.log"
run_gpu train_longforcing_lite.py --config "$INTERACTWORLD_ROOT/configs/longforcing.yaml" \
  2>&1 | tee "$INTERACTWORLD_ROOT/logs/longforcing.log"
```

配置为 action **1,075 steps** → causal **1,075 steps** → LongForcing-lite **80 steps**；
前两段 49 帧，第三段深度按 1/4/8/20 个块递增，4 步 student / 40 步 teacher，25% 短窗 replay。
这些是有限算力配置，不是完整基座预训练，1,075 steps 也不等于这里的数据完整一轮。
每 20 步保存可恢复状态，保留 best + last 2；best 按训练 loss 选，不是视觉最优。
中断后使用同一配置和该阶段自己的 `--resume /path/to/checkpoint.pt`，不要重新初始化覆盖已有实验。
大任务建议在 `screen`/`tmux` 内串行运行，并自行设定时限；此公开仓库不包含私人自动调度器。

可选静态文本修复：先复用已完成的 49 帧缓存，只新增文本特征。默认不带
`--launch` 的同一命令只核对输入并输出 CPU 计划；下面的 `run_gpu` 会实际编码：

```bash
run_gpu scripts/cache_scene_static_prompts.py --project-root "$INTERACTWORLD_ROOT" \
  --manifest "$INTERACTWORLD_ROOT/data/manifests/train.jsonl" \
  --feature-index "$INTERACTWORLD_ROOT/data/features/train.features.jsonl" \
  --base-model "$BASE" \
  --output "$INTERACTWORLD_ROOT/data/features/scene-static-prompts-r004.pt"
```

在独立的新 Action 配置中将 `data.prompt_cache_path` 设为该 `.pt` 绝对路径，
用 `--warm-start-from` 指向自己的已完成 Action checkpoint；按实际机器改写
R004 模板里的数据、模型和输出路径。不要对旧运行直接改配置后 `--resume`。
该模板保持非零动作尺度 0.03、Adapter 学习率 2e-5、LoRA 学习率 0，运行 780 新步。
后续新建 Causal 和 LongForcing 配置必须设置同一个 `data.prompt_cache_path` 和
`model.action_scale`，并指向这一分支对应的父 checkpoint；不可接回旧叙述/尺度分支。
Demo 素材准备会随实际训练配置选择静态文字并验证来源；只有未声明缓存的旧配置
才保持原叙述。不要覆盖旧素材目录，修复后的素材与视频使用新目录。

### 4. 真实输入与 15 秒输出

从 dev 划分准备首帧、文字和真实动作。未来参考帧只用于评测，不作为生成输入。
显式绑定第三段训练配置；`eval.yaml` 仅是几何/场景模板，不能代替实际训练来源：

```bash
python scripts/prepare_abot_demo_assets.py \
  --manifest "$INTERACTWORLD_ROOT/data/manifests/train.jsonl" \
  --output "$INTERACTWORLD_ROOT/eval/assets/dev3" \
  --template "$INTERACTWORLD_ROOT/configs/eval.yaml" \
  --training-config "$INTERACTWORLD_ROOT/configs/longforcing.yaml" \
  --stage longforcing --project-root "$INTERACTWORLD_ROOT"
```

执行生成前，在自己有权使用的机器上创建自己的授权说明记录；已有记录可直接复用，
下面的独占创建命令不会覆盖旧文件。说明记录不取代每次启动的实际 GPU 空闲检查。

```bash
python -c 'import json, os; from pathlib import Path; from datetime import datetime, timezone; p=Path(os.environ["INTERACTWORLD_ROOT"])/"local-authorization.json"; p.open("x", encoding="utf-8").write(json.dumps({"statement":"I authorize this reproduction on my own dedicated GPU and data directory.","recorded_at_utc":datetime.now(timezone.utc).isoformat()}, indent=2))'
run_gpu scripts/generate_abot_demo.py \
  --project-root "$INTERACTWORLD_ROOT" \
  --config "$INTERACTWORLD_ROOT/eval/assets/dev3/rollout15s_longforcing_week.yaml" \
  --output "$INTERACTWORLD_ROOT/eval/generated-longforcing" --scene-count 1 \
  --authorization-record "$INTERACTWORLD_ROOT/local-authorization.json" \
  2>&1 | tee "$INTERACTWORLD_ROOT/logs/generate.log"
```

可选 CPU 打包：`python scripts/package_abot_demo.py --project-root "$INTERACTWORLD_ROOT" --source "$INTERACTWORLD_ROOT/eval/generated-longforcing" --output "$INTERACTWORLD_ROOT/delivery/longforcing"`。

输出包括原始 `*.mp4`、顶部输入栏版本 `*.inputs.mp4`、播放页 `index.html` 和来源 `receipt.json`。
原始画面为 832×480、241 帧/16fps，首尾跨度 15 秒；输入栏另外增加 48 像素，
顶部保留真实首帧缩略图和 prompt；同步按键默认放在画面左下 WASD、右下方向箭头。
提示词铺满照片右侧可用宽度，右边只留8像素边距，不再给旧顶部按键预留空白。
箭头仅是 I/J/K/L 的显示映射（↑/←/↓/→），不改动作张量、顺序或模型条件。
左右按键 HUD 会覆盖对应的小块显示区域，画面不缩放、不裁剪、不补绘；原始视频单独保留。
`InputHeaderRenderer` 和 `annotate_video` 可用 `layout="inline"` 重现旧顶部按键布局。
画质应观看整个原视频再判断；输入栏与时长不是控制正确或长期稳定的证据。

## 来源与许可

基于 [AMAP CV Lab / ABot-World](https://github.com/amap-cvlab/ABot-World)，
使用 [Wan2.2](https://github.com/Wan-Video/Wan2.2) 模型代码与基座，
数据来自 [ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h)。
保留 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
数据与模型请遵守各自发布页的许可和使用条件；仓库代码许可不自动覆盖它们。
上游项目展示的实时性、质量和性能不属于本复现代码已经达到的结果。
