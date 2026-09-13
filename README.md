# InterActWorld

**基于 Wan2.2 的动作条件视频生成与世界模型探索原型。**

在官方 Wan2.2-TI2V-5B 预训练基座上训练动作 Adapter 和 LoRA，将首图、静态场景文本及按键时间线接入视频生成，并提供异步网页 Demo。复用 ABot 兼容的控制接口与部分训练思路，不使用其成品模型代替本项目训练，也不把上游实时性能当成本项目结果。

## Demo 视频与版本对照

### 当前最佳展示版：两组同输入、不同动作

以下使用固定的 **Action R005 step1040＋window6**，不为这四段重新训练。每组内固定同一首图、提示词、seed、checkpoint 和采样参数，只改变动作与视角控制；两组使用不同场景。顶部保留首图和提示词，左侧显示 WASD，右侧显示方向键。按键表示模型收到的控制输入，不等于模型已准确执行。

四段均于2026-09-13真实生成完成，完整241帧/16 FPS；没有重新训练、插帧或循环补时。配对内的图文、初始latent和噪声哈希一致，控制动作不同；详细绑定与实测值见[媒体清单](assets/demos/manifest.json)。

| 第一组：山地场景，同首图、同提示词 | 第一组：相同图文输入 |
| --- | --- |
| **前进＋左转视角：W＋←** | **后退＋右转视角：S＋→** |
| [![第一组：前进与左转视角输入，15秒轨迹预览](assets/demos/pair-a-forward-left-preview.gif)](assets/demos/pair-a-forward-left.mp4) | [![第一组：后退与右转视角输入，15秒轨迹预览](assets/demos/pair-a-backward-right-preview.gif)](assets/demos/pair-a-backward-right.mp4) |
| [完整 MP4](assets/demos/pair-a-forward-left.mp4) | [完整 MP4](assets/demos/pair-a-backward-right.mp4) |

| 第二组：草地、丘陵与树木，同首图、同提示词 | 第二组：相同图文输入 |
| --- | --- |
| **前进＋抬头：W＋↑** | **后退＋低头：S＋↓** |
| [![第二组：前进与抬头输入，15秒轨迹预览](assets/demos/pair-b-forward-up-preview.gif)](assets/demos/pair-b-forward-up.mp4) | [![第二组：后退与低头输入，15秒轨迹预览](assets/demos/pair-b-backward-down-preview.gif)](assets/demos/pair-b-backward-down.mp4) |
| [完整 MP4](assets/demos/pair-b-forward-up.mp4) | [完整 MP4](assets/demos/pair-b-backward-down.mp4) |

GIF 是压缩抽帧预览；点击查看保留原帧的完整 MP4，每段首尾帧跨度为15秒。两组是自定义控制的展示对比，没有对应的真实未来 GT，不报告其 RGB MSE，也不把历史样片分数移植到这些新视频上。固定 seed 有助于对照，但两对样片不能证明可靠控制或跨场景泛化。

动作时序为五个3秒周期：移动1秒 → 移动＋视角键0.5秒 → 移动1.5秒，共240行真实动作输入。GIF为4 FPS、320像素宽的抽帧预览，不代表推理速度。四段通过与网页相同的window6后端离线批处理生成，未修改原网页的默认场景。第一组首图是既有conditioning latent解码图，第二组使用既有原始RGB首图；配对内的输入协议保持一致。

| 本次样片 | 实际模型生成耗时 | 有限抽帧观察（0 / 7.5 / 15秒） |
| --- | --- | --- |
| 山地：W＋← | 130.60秒 | 人物和场景保留，末段细节与人形软化 |
| 山地：S＋→ | 129.55秒 | 末段人物难辨、画面偏地面，稳定性仍不足 |
| 草地：W＋↑ | 129.62秒 | 天空、树木、人物保留，末段肢体软化 |
| 草地：S＋↓ | 129.81秒 | 草木可辨，后段人物变形并有半透明感 |

四段峰值allocated显存均为12.57 GiB，每段监督器计时约137–138秒；以上模型生成耗时不含旧网页排队。这些样片展示真实差异，也暴露自定义动作下的质量限制，不作为“所有方向都稳定可控”的结论。

### 关键版本：改动与实际结果

README 只精选下列关键阶段，包含未成功的尝试；不再逐版铺满视频。这里的历史样片不全是同输入、同采样条件，不能作为公平模型排名。完整时间线及数值定义见[版本变化与指标](docs/version-history.md)。

| 关键阶段 | 改动与已有结果 | 生成预览（点击打开完整 MP4） |
| --- | --- | --- |
| **Action R004 step780** | 将动态叙事提示词改为静态场景文本。匹配基线的三档 flow MSE 均值下降约1.20%，但后段仍有半透明、重影；flow 指标不是成片 RGB 分数。 | [![R004780：静态文本阶段的15秒样片](assets/demos/action-r004-780-15s-preview.gif)](assets/demos/action-r004-780-15s.mp4) |
| **native97 resampled step120** | 训练窗增至97 RGB帧，并修复重复索引、噪声与时间步采样。单例 weighted RGB MSE：correct **.027365**，zero **.037809**，shuffled **.033990**；数值过线但人物仍偏细、边缘软化，未替换当前展示版。 | [![native97 resampled120：数值改善但仍有视觉问题](assets/demos/native97-resampled-120-preview.gif)](assets/demos/native97-resampled-120.mp4) |
| **因果 clean60 / recycling60** | 同一起点比较干净历史与预测误差回收。两臂均在后段丢失人物，未观察到续写质量收益；没有匹配的 zero / shuffled RGB 分数。保留配对负结果，不包装成有效改进。 | clean60：[![干净历史对照：15秒失败样片](assets/demos/causal-clean60-preview.gif)](assets/demos/causal-clean60.mp4)<br>recycling60：[![误差回收：15秒失败样片](assets/demos/causal-recycling60-preview.gif)](assets/demos/causal-recycling60.mp4) |

[完整13段历史视频归档](docs/demo-gallery.md) · [各版本改动与指标](docs/version-history.md) · [精确结果 JSON](docs/version-results.json) · [离线播放器](docs/demo-gallery.html)

历史归档包括 R003/R004/R005/R006/R007、长窗适配、因果回收及历史噪声样片；R003 是真实3秒诊断，其余为15秒轨迹。下载仓库后可打开离线播放器，无需 GPU 或网页推理服务。GitHub 若不直接播放 MP4，请使用 Raw/Download。

## 当前固定展示版

截至 2026-09-13，展示方案固定为 **Action R005 step1040 + window6**。后续因果、少步与历史扰动实验保留为研究分支，不自动替换这个版本。

| 项目 | 固定方案 |
| --- | --- |
| 权重 | 自训 Action Adapter + rank16 LoRA，step1040 |
| 输入 | RGB 首图、固定场景文本、240×8 二值动作、seed |
| 动作 | `W,A,S,D,I,J,K,L`；界面方向箭头对应 `I,J,K,L` |
| 推理 | 显式 `--inference-mode window6`；6+6+3 秒顺序续写 |
| 采样 | 每窗40步 flow Euler，shift5，BF16，无 CFG |
| 输出 | 832×480、241帧、16 FPS；带输入栏版本832×528 |
| 速度边界 | 已归档单次网页任务总处理约139.111秒；不是实时16 FPS |

文件时长为15.0625秒，首尾帧跨度15秒。已有单场景样例可用于异步演示，但仍有模糊、肢体软变形，**不宣称普遍稳定的长视频、可靠动作控制或实时交互**。该展示配置尚无完整同配置 zero/shuffled 对照，其他推理分区的数值结果不能转借为本版的控制结论。

完整参数与版本区别见 [固定展示说明](docs/demo-window6-v1.md) 和 [机器可读版本档案](docs/demo-window6-v1.json)。档案是版本说明，不是可直接执行的配置，更不附带模型权重。

## 系统与训练链路

```text
公开视频 + 对齐按键 + 静态场景文本
        ↓ manifest / VAE与文本缓存
冻结 Wan 基座，训练 Action Adapter + LoRA
        ↓ 可恢复 checkpoint
首图 + 文本 + 动作时间线
        ↓ window6：6秒 → 生成末帧重编码 → 6秒 → 同样续写3秒
异步任务队列 → 原视频 / 输入栏视频 / 来源收据
```

- 动作是真实数值条件，不只是拼到 prompt 中；每4帧按固定键序打包为32维控制。
- 训练与推理保存数据、配置、checkpoint 身份；严格 resume 与仅继承权重的新阶段明确区分。
- 展示版训练片段为49 RGB帧。6秒窗口是推理选择，不能说该 checkpoint 来自后续97帧训练。
- 后窗只使用前窗生成的浮点 RGB 末帧，不读取真实未来帧，也不循环或插帧凑视频时长。
- 网页固定首图与场景文本，可编辑动作时间线和 seed；顶部显示输入，左下 WASD、右下方向箭头显示模型收到的动作。
- 新网页任务调用自训模型，不以历史录像冒充新生成；没有模型或授权时直接拒绝生成。

## 快速开始：先做 CPU 检查

仓库提供最小源码、配置、测试及明确选定的自训生成样片：13段历史归档，另有上述四段配对展示。不包含模型权重、原始数据集视频、特征缓存、运行日志或私人部署/授权记录。克隆后可以运行 CPU 契约测试；**不能在缺少模型资产时一键生成上述视频**。

以下为 Linux Bash 示例。参考 CPU 环境为 Python3.12，依赖以仓库文件为准；虚拟环境、包缓存和后续训练资产应放在自己的可写数据盘。

```bash
git clone https://github.com/HuanYn/InteractWorld.git
cd InteractWorld
export INTERACTWORLD_ROOT=/absolute/path/on/data-disk/interactworld
mkdir -p "$INTERACTWORLD_ROOT"
export PIP_CACHE_DIR="$INTERACTWORLD_ROOT/cache/pip"
python3.12 -m venv "$INTERACTWORLD_ROOT/venv-cpu"
source "$INTERACTWORLD_ROOT/venv-cpu/bin/activate"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q tests/test_demo_window6_action.py tests/test_demo_euler_precision.py
```

[CI](.github/workflows/cpu-tests.yml)在 Linux 上执行更完整的 CPU 测试，包括媒体与服务契约；FFmpeg/FFprobe 用于媒体检查。CPU 测试使用小张量或测试替身，不代表真实 GPU 模型、画质或控制已验收。

## 准备固定版 Demo

操作方需要合法取得 Wan 基座/VAE、自训 checkpoint、原始训练 YAML、manifest/特征及静态文本缓存与收据，以及 RGB 首图和 `[240,8]` 二值 float32 动作文件。必须保持真实来源，不能伪造哈希或改写父训练配置来通过检查。

下面仅做 CPU 准备；路径替换为自己的数据盘。固定样片的 checkpoint SHA 在[版本档案](docs/demo-window6-v1.json)中。若用自己重新训练的模型，应填写其实际 SHA，并说明这不是同一个固定 checkpoint。

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m training.demo.prepare_action \
  --training-config /DATA_DISK/assets/original-action.yaml \
  --checkpoint /DATA_DISK/assets/step-0001040.pt \
  --checkpoint-sha256 EXACT_CHECKPOINT_SHA256 \
  --initial-frame /DATA_DISK/assets/initial.png \
  --episode-id ACTUAL_HELD_OUT_EPISODE_ID \
  --action-input /DATA_DISK/assets/actions.npy \
  --initial-origin decoded_condition_rgb --seed 42 \
  --inference-mode window6 \
  --output /DATA_DISK/demo/rollout-action.yaml
```

固定展示首图来自 conditioning latent 解码，故使用 `decoded_condition_rgb`；真正原始 RGB 图使用 `source_rgb`。不要混称二者，也不要将新 RGB 输入协议与冻结 latent/noise 的诊断协议合并比较。`window6` 必须显式指定；通用 CLI 默认仍是 `chunked`，本次不改变旧调用行为。

按照 [部署说明](training/demo/README.md)准备自己的 `deployment.json`，绑定上一步 rollout 配置。保持 `host: "127.0.0.1"`，将 `guard_command` 留空即可只预览界面：

```bash
CUDA_VISIBLE_DEVICES='' python scripts/serve_abot_demo.py \
  --deployment /DATA_DISK/demo/deployment.json
```

空 guard 会拒绝提交生成。启用真实 GPU 工作需要操作方提供有效授权、即时 GPU/空间/预算检查与全套模型资产；运行示例不是本项目或任何机器的使用许可。不要以恒真函数、伪造时间戳或历史授权绕过当前规则。

## 自行训练与研究分支

Action R005 的结构模板见 [R005配置](configs/train/action_teacher_5090_repair_r005.yaml)：action scale0.03、Adapter LR2e-5、LoRA LR2e-6、rank16、microbatch1×accum8、BF16。模板最初计划780步，固定展示选取同运行扩展后的step1040；使用原始配置与真实checkpoint记录，不为凑版本号改写来源。配置是历史实现参考，不保证重新训练得到相同样片。

基座/数据版本、缓存、训练 CLI 和历史三阶段流程见 [训练参考](docs/research-training-reference.md)。该参考不是固定展示版的一键流水线；尤其 **不需要为了运行 Action Demo 再训练因果或蒸馏模型**。

| 分支 | 当前公开定位 |
| --- | --- |
| Action + window6 | 本次固定的异步展示方案，质量与泛化边界见上文 |
| native97 / 因果 / context error recycling | 研究实现，未形成可替换展示版的长期质量结果 |
| LongForcing-lite | 曾完成80步训练但生成失败；简化少步终点蒸馏，不是完整官方方法或DMD |
| history-noise / clean 对照 | 各20步后15秒视觉失败；GT历史加噪不是完整自生成历史训练，不默认续训 |
| DMD / 新teacher-flow | 不属于本次固定展示交付，未作为成功方案发布 |

[历史误差回收说明](docs/context-error-recycling.md)、[双向辅助训练说明](docs/moba-inspired.md)用于理解实验入口。训练 loss 下降、进程结束、视频有15秒、模型准确响应动作，是不同层面的结论；失败实验不包装成已验证的改进。

## 代码导航

| 路径 | 内容 |
| --- | --- |
| [training/data](training/data) | 数据与动作对齐、缓存读取、采样 |
| [training/models](training/models) | 动作适配器与LoRA |
| [train_action_teacher.py](train_action_teacher.py) | Action训练、阶段迁移与恢复 |
| [training/demo](training/demo) | 输入契约、Action推理、异步服务与页面 |
| [scripts](scripts) | 数据、缓存、配置和媒体工具 |
| [tests](tests) | CPU契约与边界测试 |
| [REPRODUCTION.json](REPRODUCTION.json) | 脱敏源码导出身份与逐文件哈希 |

## 来源与许可

基于 [AMAP CV Lab / ABot-World](https://github.com/amap-cvlab/ABot-World)，使用 [Wan2.2](https://github.com/Wan-Video/Wan2.2) 模型代码与基座，数据来自 [ABot-World-Explorer-500h](https://huggingface.co/datasets/acvlab/ABot-World-Explorer-500h)。本项目主要工作是受资源约束的训练/适配实现、输入契约、恢复和部署链路及真实诊断，不将上游架构或性能作为原创成果。

保留 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型与数据还需遵守各自许可，代码许可不自动覆盖它们。公开样片仅限本项目明确选定的生成结果，不分发上游宣传视频、原始数据集视频、权重或其他私有实验资产。
