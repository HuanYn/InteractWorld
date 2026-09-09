# InterActWorld — ABot-inspired training reproduction

从官方 **Wan2.2-TI2V-5B 基座**训练动作 Adapter + LoRA、因果模型和 LongForcing-lite，
最后生成动作条件视频。不使用官方 ABot 成品模型替代自己的训练。

本仓库公开的是可复现的代码与配置，**不是已经达到稳定画质的产品**。
当前旧预览仍有人物形变、重复纹理和条纹；稳定展示效果尚未交付，也未达到实时交互。
LongForcing-lite 是本项目的少步 endpoint 蒸馏实现，**不是完整复现官方 LongForcing/DMD**。
代码运行成功、训练 loss 下降或视频达到 15 秒，都不等于画质和动作控制已经通过。

不包含权重、数据、生成素材、训练日志、服务器配置或操作方授权记录。

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
显示真实首帧缩略图、prompt 与同步按键，不修改或补绘模型生成画面。
画质应观看整个原视频再判断；输入栏与时长不是控制正确或长期稳定的证据。

## 来源与许可

基于 [AMAP CV Lab / ABot-World](https://github.com/amap-cvlab/ABot-World)，
使用 [Wan2.2](https://github.com/Wan-Video/Wan2.2) 模型代码与基座，
数据来自 [ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h)。
保留 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
数据与模型请遵守各自发布页的许可和使用条件；仓库代码许可不自动覆盖它们。
上游项目展示的实时性、质量和性能不属于本复现代码已经达到的结果。
