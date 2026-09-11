# BSAI ComfyUI vdn-minimax-h3 ｜ BSAI ComfyUI vdn-minimax-h3

**VDN-H3（Video DeltaNet on MiniMax H3）一站式节点套件 / All-in-one node suite for VDN-H3**

> 中英双语文档 / Bilingual documentation. 每个小节中英对照。

---

## 插件介绍 / Introduction

**VDN-H3** 是 MiniMax H3 的**混合注意力加速**方案：新增一条**帧级线性注意力分支**（高效）+ 保留 **softmax 分支**（维持视觉质量与一致性），配合 **default（50 步质量）** 与 **turbo（8 步 DMD2 蒸馏）** 两个小 LoRA 适配器，即插即用合并进骨干网络，实现"比播放还快"的视频生成（官方：8×B200 上 14.4 秒片段 8 步去噪仅 11.23 秒）。

**VDN-H3** is a **hybrid attention acceleration** for MiniMax H3: it adds a **frame-level linear attention branch** (efficient) while keeping the **softmax branch** (visual quality & consistency), combined with two tiny LoRA adapters — **default** (50-step quality) and **turbo** (8-step DMD2 distilled) — merged plug-and-play into the backbone. Result: video generation "faster than playback" (official: 11.23s for 8-step denoising of a 14.4s clip on 8×B200).

本插件把它封装为 **10 个 ComfyUI 节点**，覆盖加载、采样、加速、超分、人脸修复、去油、空 latent 全套流程。/ This plugin wraps it into **10 ComfyUI nodes** covering loading, sampling, acceleration, upscale, face restore, de-oiling and empty latent.

- 项目主页 / Homepage: https://openvdn.github.io/
- GitHub 代码 / Code: https://github.com/OpenVDN/vdn-minimax-h3
- HuggingFace 权重 / Weights: https://huggingface.co/OpenVDN/vdn-minimax-h3

---

## ⚠️ 依赖：已内嵌，用户零安装 / Zero-install: runtime embedded

本插件调用的 `vdn_h3` 运行时（Apache-2.0）**已直接内嵌在 `vendor/vdn_h3/` 目录随包分发**——用户**无需**单独安装 ComfyUI-VDN-H3 插件，覆盖安装本插件即可直接使用。

The `vdn_h3` runtime (Apache-2.0) is **embedded in this plugin's `vendor/vdn_h3/`** and ships with the package — you do **NOT** need to install ComfyUI-VDN-H3 separately.

- 加载顺序 / Load order：内嵌 `vendor/vdn_h3` →（已装用户）外部 ComfyUI-VDN-H3 →（极端兜底）自动 git clone。/ Embedded vendor → external ComfyUI-VDN-H3 (if installed) → auto git clone (last resort).
- 若你之前已装 ComfyUI-VDN-H3，互不冲突（同一份代码，优先使用内嵌版）。/ Coexists with an existing ComfyUI-VDN-H3 install.

---

## 节点总览 / Node Overview

| 节点 / Node | 能力 / Role |
|---|---|
| **BSAIVDNH3Loader** | 权重加载：基座 + VDN stage（linear_branch）+ default/turbo LoRA + 3 种画质模式 + fp8 / Weight loader with quality modes |
| **BSAIVDNH3DualLora** | 双 LoRA 混合加载（default 质量 + turbo 速度），去伪影后处理 / Dual-LoRA mixing |
| **BSAIVDNH3Timesteps** | VDN 精确时间步：8 步 turbo / 16 步质量 / 50 步质量 / 自定义阶梯 / Precise VDN sigmas |
| **BSAIVDNH3EulerSampler** | VDN Euler 采样器（video shift 12 / audio shift 3）/ VDN Euler sampler |
| **BSAIVDNH3Accel** | 加速整合：Block Cache / CacheDiT / TE-Speed / VSA 稀疏注意力 / Acceleration bundle |
| **BSAIVDNH3Upscale** | 画质超分放大 + 高清修复（桥接 BSAI-H3-upscale-4K 引擎）/ Upscale + HD fix |
| **BSAIVDNH3FaceRestore** | 远景小脸崩坏修复（GFPGAN / CodeFormer）/ Distant-face restore |
| **BSAIVDNH3FaceOil** | 脸部去油：高光抑制 + 保边平滑 / Face de-oiling |
| **BSAIVDNH3EmptyLatentVideo** | H3 视频专用 5D 空 latent [B,24,T/4,H/16,W/16] / H3 video empty latent |
| **BSAIVDNH3LoaderNative** | 一体化 Loader（兼容旧版：H3 主模型 + ApplyVDNH3 + ChunkFeedForward）/ Legacy all-in-one loader |

---

### 1. BSAIVDNH3Loader · 权重加载（基座 + 线性分支 + LoRA）

| 参数 / Parameter | 类型 | 默认 | 说明 / Description |
|---|---|---|---|
| `backbone` | 下拉 | 🔄 自动 | 基座 transformer 权重。**🔄 选项**=自动扫描 `models/diffusion_models` 中 ComfyUI 兼容的 MiniMax H3 bf16/fp16 基座（含 video/audio_patch_proj 检测点；官方 VDN h3-base 分片为 diffusers 布局，ComfyUI 无法直接加载，自动选用兼容基座）。其余为单文件（如 FastH3 4 步蒸馏）。 |
| `vdn_stage` | 下拉 | stage-dmd-step-250 | VDN stage 目录：`stage-dmd-step-250`=8 步 turbo（推荐）、`stage-b-step-2000`=50 步质量、`无(仅基座)`。目录放 `ComfyUI/models/vdn/`。 |
| `merge_linear_branch` | BOOLEAN | True | 合并 VDN linear_branch 混合注意力分支（需 FA4，无则自动软件回退）。 |
| `software_fallback` | BOOLEAN | True | 无 FA4 时启用软件回退（短卷积 + Softmax 门控混合注意力），画质接近原生。 |
| `quality_mode` | 枚举 | 🎨 画质优先 | **⚡ 速度优先**=8 步 + turbo LoRA（最快）；**🎨 画质优先**=16 步 + 弱 turbo + default（画质显著提升，推荐）；**💎 VDN纯质**=8 步 + 无 LoRA（依赖 linear_branch，近无损）。 |
| `merge_default_lora` | BOOLEAN | False | 合并 `stage/adapters/default` 的 50 步质量 LoRA。 |
| `merge_turbo_lora` | BOOLEAN | True | 合并 `stage/adapters/turbo` 的 8 步 DMD2 蒸馏 LoRA（速度 + 去伪影核心）。 |
| `lora_strength` | FLOAT | 1.0 | LoRA 合并强度（0.0-2.0）。 |
| `weight_dtype` | 枚举 | default | `default` / `fp8_e4m3fn` / `fp8_e4m3fn_fast` / `fp8_e5m2`（省显存）。 |

**输出 / Outputs:** `MODEL`, `info` (STRING 环境检测+加载报告), `stage_dir` (STRING)

---

### 2. BSAIVDNH3DualLora · 双 LoRA 混合（去伪影）

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `model` | MODEL | — | 上游模型。 |
| `lora_a` | 下拉 | — | 主 LoRA（如 VDN turbo 8 步蒸馏，速度 + 去伪影），自动识别 VDN/diffusers/ComfyUI 三种格式。 |
| `lora_a_strength` | FLOAT | 1.0 | LoRA A 强度。 |
| `lora_b` | 下拉 | — | 辅助 LoRA（如 VDN default 50 步质量，补偿细节、抑伪影）。 |
| `lora_b_strength` | FLOAT | 0.35 | 建议 0.2-0.5：轻叠加质量细节而不拖慢步数。 |
| `artifact_suppress` | FLOAT | 0.0 | 去伪影强度：>0 时对低步数典型伪影（条纹/振铃/结构粘连）做后处理抑制。 |

**输出 / Outputs:** `MODEL`, `info` (STRING)

---

### 3. BSAIVDNH3Timesteps · 精确时间步

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `model` | MODEL | — | 上游模型（用于 sigma 换算）。 |
| `steps` | 枚举 | 8 步 turbo | `8 步 turbo`=DMD2 蒸馏阶梯 [999,750,500,250]；`16 步质量`=密集高精度（画质优先推荐）；`50 步质量`=均匀高精度；`自定义`=下方 ladder。 |
| `ladder` | STRING | 999,750,500,250 | 自定义逗号分隔 timestep 阶梯（0-1000），末尾自动补 0。 |

**输出 / Outputs:** `SIGMAS` — 接入 `SamplerCustomAdvanced.sigmas`

---

### 4. BSAIVDNH3EulerSampler · VDN Euler 采样器

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `shift_video` | FLOAT | 12.0 | 视频流 flow shift（VDN 官方 12.0）。 |
| `shift_audio` | FLOAT | 3.0 | 音频流 flow shift（VDN 官方 3.0）。 |
| `schedule_mode` | 枚举 | auto | `auto`=检测 ModelSamplingAV；`native`=单调度；`legacy_dual`=音视频双调度。 |

**输出 / Outputs:** `SAMPLER` — 接入 `SamplerCustomAdvanced.sampler`，与 VDN Timesteps 搭配使用。

---

### 5. BSAIVDNH3Accel · 加速整合

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `model` | MODEL | — | 上游模型。 |
| `block_cache_enable` | BOOLEAN | True | Block Cache：跨步缓存中间 block 输出，跳过冗余计算（ComfyUI-CacheDiT 内核）。 |
| `block_cache_start` | FLOAT | 0.10 | 从该采样进度起启用（前期保持精确）。 |
| `block_cache_end` | FLOAT | 0.85 | 到该进度停用（收尾保持细节）。 |
| `cachedit_enable` | BOOLEAN | True | CacheDiT 扩散 Transformer 缓存：t 邻域步跳过注意力重算。 |
| `te_speed_enable` | BOOLEAN | True | TE-Speed 文本编码器快速路径。 |
| `vsa_enable` | BOOLEAN | True | VSA 视频稀疏注意力（FastH3 原生稀疏）：保留 top-k 视频块精确注意力。 |
| `vsa_keep_percent` | FLOAT | 20.0 | VSA 保留的精确注意力百分比（越大细节越好、越慢）。 |
| `detail_preserve` | FLOAT | 0.0 | 细节保护：>0 时对 cache/VSA 造成的细节损失做高频补偿。 |

**输出 / Outputs:** `MODEL`, `config_json` (STRING，供 BSAI H3 MotionFix 联动)

---

### 6. BSAIVDNH3Upscale · 超分放大 + 高清修复

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `images / 图像` | IMAGE | — | 输入图像批次。 |
| `scale / 放大倍数` | FLOAT | 4.0 | 放大倍数（1-8）。 |
| `model_name / 模型` | STRING | realesr-general-x4v3.pth | 超分模型（同 BSAI-H3-upscale-4K）。 |
| `tile_size / 分块大小` | INT | 0 | 0=自动。 |
| `tile_pad / 分块重叠` | INT | 16 | 分块重叠像素。 |
| `batch_frames / 批帧数` | INT | 4 | 批量帧数。 |
| `use_fp16 / 半精度` | BOOLEAN | True | 半精度推理。 |
| `use_compile / 编译加速` | BOOLEAN | True | torch.compile 加速。 |
| `temporal_strength / 时序强度` | FLOAT | 0.20 | 时序光流稳定强度。 |
| `detail_amount / 细节强度` | FLOAT | 0.50 | 细节锐化强度。 |
| `detail_radius / 细节半径` | FLOAT | 1.8 | 锐化半径。 |
| `softness / 柔和度` | FLOAT | 0.10 | 防振铃柔和。 |
| `detail_mode / 细节模式` | 枚举 | smart | `classic` / `smart`。 |

**输出 / Outputs:** `IMAGE`, `width`, `height`, `scale_used`, `info` — 桥接 BSAI-H3-upscale-4K 引擎；未安装时自动降级 Lanczos。

---

### 7. BSAIVDNH3FaceRestore · 远景小脸崩坏修复

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `images / 图像` | IMAGE | — | 输入图像批次。 |
| `face_restore / 人脸修复` | 枚举 | 小脸增强(CodeFormer) | `Off` / `GFPGANv1.4` / `CodeFormer` / `小脸增强(CodeFormer)`。 |
| `face_det_conf / 检测置信度` | FLOAT | 0.15 | YOLOv8-Face 检测阈值。 |
| `face_blend / 融合强度` | FLOAT | 0.70 | 修复结果融合。 |
| `face_fidelity / 保真度` | FLOAT | 0.60 | 保真度。 |

**输出 / Outputs:** `IMAGE`, `faces_detected` (INT), `info` — 修复 H3 全景/远景的小脸模糊变形（桥接 BSAI-H3-upscale-4K 人脸引擎）。

---

### 8. BSAIVDNH3FaceOil · 脸部去油

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `images / 图像` | IMAGE | — | 输入图像批次。 |
| `strength / 去油强度` | FLOAT | 0.60 | 高光抑制强度。0=关闭；0.5-0.7 自然去油。 |
| `bright_thr / 高光阈值` | FLOAT | 0.78 | 亮度高于此值视为高光（油光/过曝区）。 |
| `sat_thr / 低饱和阈值` | FLOAT | 0.32 | 饱和度低于此值且高亮视为油光。 |
| `smooth_radius / 平滑半径` | INT | 9 | 平滑半径（越大越柔和）。 |
| `face_only / 仅脸部` | BOOLEAN | False | True=OpenCV Haar 检测人脸后仅人脸去油；False=全图高光抑制。 |

**输出 / Outputs:** `IMAGE`, `info` — 高光抑制 + 高斯保边平滑，消除油腻反光同时保留皮肤纹理。

---

### 9. BSAIVDNH3EmptyLatentVideo · H3 视频 5D 空 latent

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `width` | INT | 1344 | 输出宽度（16 倍数）。 |
| `height` | INT | 768 | 输出高度（16 倍数）。 |
| `length` | INT | 124 | 输出帧数（时间下采样 4×，124≈5s@24fps）。 |
| `batch_size` | INT | 1 | 批次大小。 |

**输出 / Outputs:** `LATENT` — 生成 [B, 24, T/4, H/16, W/16] 5D latent，接入 `SamplerCustomAdvanced.latent_image`。

---

### 10. BSAIVDNH3LoaderNative · 一体化 Loader（兼容旧版工作流）

旧版一体化节点（兼容）：H3 主模型下拉框 + VDN stage/turbo/linear_branch/KV 压缩等全部参数 + 可选 ChunkFeedForward 分块前馈加速。旧示例工作流可直接加载使用。/ Legacy all-in-one loader kept for compatibility with older workflows.

---

## 权重安装 / Weight Installation

VDN stage 权重放到 `ComfyUI/models/vdn/`，**保持官方目录结构不变** / Place VDN stage weights in `ComfyUI/models/vdn/`, keeping the official structure:

```
ComfyUI/models/vdn/
└── stage-dmd-step-250/            # 8 步蒸馏（推荐）/ 8-step distilled (recommended)；stage-b-step-2000 为 50 步质量版
    ├── model_spec.json
    ├── linear_branch/
    │   ├── model.safetensors                    # 分支权重（bf16）/ branch weights (bf16)
    │   └── model_int8_convrot_comfyui.safetensors  # 预量化版（可选）/ pre-quantized (optional)
    └── adapters/
        ├── default/                             # 50 步质量 LoRA / 50-step quality LoRA
        └── turbo/                               # 8 步蒸馏 LoRA / 8-step distilled LoRA
```

> **判定标准 / Detection rule**：目录含 `linear_branch/` 且有 `.safetensors/.bin` 分支文件 → 识别为 VDN stage。
> **结构不完整时**（如只解压了 stage 根目录），下拉框会给出 `<⚠ xxx 目录存在但结构不完整>` 提示——不是模型丢失，是结构不完整。

下载方式（保持结构）/ Download (keeps structure):

```bash
# 需先安装 huggingface_hub / requires huggingface_hub
hf download OpenVDN/vdn-minimax-h3 stage-dmd-step-250 --local-dir ComfyUI/models/vdn
```

> 旧路径 `models/vdn_h3/` 已废弃但兼容；推荐统一使用 `models/vdn`。/ Legacy `models/vdn_h3/` still accepted; `models/vdn` recommended.

---

## H3 主模型 / H3 Backbone Model

`backbone` 下拉框扫描 `models/diffusion_models`，自动过滤 MiniMax H3 / minimax 关键字。**🔄 选项**自动选用 ComfyUI 兼容的 H3 bf16/fp16 基座（含 video_patch_proj/audio_patch_proj 检测点，优先全精度）。官方 VDN h3-base 分片为 diffusers 布局（proj_in/proj_out 合并结构），ComfyUI 无法直接加载，请用 🔄 自动选用或 ComfyUI 格式单文件（如 `minimax_h3_fl2va_bf16.safetensors`、FastH3 4 步蒸馏）。

The `backbone` dropdown scans `models/diffusion_models`. The 🔄 option auto-picks a ComfyUI-compatible H3 bf16/fp16 backbone (with video/audio_patch_proj checkpoints). Official VDN h3-base shards use the diffusers layout and cannot be loaded directly by ComfyUI.

---

## 环境要求 / Requirements

- **原生线性分支加速（推荐）**：PyTorch 2.13+ / FlashAttention-4（FlexAttention Flash backend）。加载器自动检测。/ Native linear-branch acceleration: PyTorch 2.13+ / FlashAttention-4; auto-detected.
- **软件回退**：无 FA4 时自动启用短卷积 + Softmax 门控混合注意力（`software_fallback`），画质接近原生、速度略慢。/ Software fallback when FA4 is unavailable (short-conv + gated mixing).
- **降级**：关闭回退时跳过 linear_branch，模型以 Dense 运行。/ Dense fallback when both are off.
- **ChunkFeedForward**：大模型可开分块前馈，降低 24GB 卡 OOM 概率。/ Chunked feed-forward reduces OOM on 24GB cards.
- VDN stage 模型遵循 **MiniMax H3 Community License Agreement**（适用领土排除欧盟/英国/韩国/美国）。/ VDN stage weights follow the MiniMax H3 Community License (EU/UK/KR/US excluded).

---

## 示例工作流 / Example Workflow

`example_workflows/` 下提供参考工作流，`Workflow → Open` 加载后把 `backbone` / `vdn_stage` 换成你本机实际模型即可。

A reference workflow is provided in `example_workflows/` — open it in ComfyUI, then set `backbone` / `vdn_stage` to your local models.

### `BSAI_VDN-H3_T2V_I2V_MultiRef.json` — 文生 / 图生 / 多参考生视频 v1.0

30 节点完整参考工作流，覆盖 VDN-H3 三种典型用法 / 30-node reference covering all three usage modes:

**节点链 / Node chain**:
```
BSAIVDNH3LoaderNative (基座+线性分支+8步turbo) ──model──> BasicGuider ──> SamplerCustomAdvanced ──> VAEDecode + VAEDecodeAudio ──> SaveVideo/CreateVideo
CLIPLoader ──> CLIP 条件
MiniMaxH3ImageToVideo (图生) / MiniMaxH3ReferenceToVideo (多参考) ──> latent
easy ifElse 分支切换（文生 vs 图生 vs 多参）
BSAI_H3_PromptTemplate (提示词模板) / LoadImage×3 (多参考图) / ResolutionSelector (分辨率)
```

**使用步骤 / How to use**:
1. 打开工作流，**BSAIVDNH3LoaderNative**：`unet_name` 选你的 H3 基座模型；`vdn_checkpoint` 选 `stage-dmd-step-250`（8 步 turbo）。
2. **BSAI_H3_PromptTemplate**：选择或填写提示词模板（支持 BSAI 提示词模板体系）。
3. **三种模式**（用 `easy ifElse` 切换）：
   - **文生视频**：直接用 H3 空 latent（EmptyMiniMaxH3LatentAV / BSAIVDNH3EmptyLatentVideo）；
   - **图生视频**：`MiniMaxH3ImageToVideo` 接首帧图（`LoadImage`）；
   - **多参考生视频**：`MiniMaxH3ReferenceToVideo` 接 1-9 张参考图（`LoadImage`×3 可扩展）。
4. **分辨率**：`ResolutionSelector` 选目标分辨率；或手动 `width/height`。
5. 点「运行」→ `SaveVideo`/`CreateVideo` 输出。

> 新套件推荐接法：用 `BSAIVDNH3Loader`（画质优先模式）+ `BSAIVDNH3Timesteps`（16 步质量）+ `BSAIVDNH3EulerSampler` 替换工作流中的旧 LoaderNative 链，画质更佳。/ For the new suite, swap in `BSAIVDNH3Loader` (quality mode) + `BSAIVDNH3Timesteps` (16-step) + `BSAIVDNH3EulerSampler` for better quality.

---

## 故障排查：模型下拉框不显示 / Troubleshooting: dropdown shows nothing

| 下拉框显示 / Dropdown text | 原因 / Cause | 解决 / Fix |
|---|---|---|
| `<未安装依赖：请先安装 ComfyUI-VDN-H3 插件>` | 内嵌 runtime 与外部插件均缺失，自动安装失败（无 git/离线）| 手动 `git clone https://github.com/OpenVDN/ComfyUI-VDN-H3` 到 `custom_nodes/` 后重启 |
| `<⚠ stage-xxx 目录存在但结构不完整：需放置 stage-xxx/linear_branch/model.safetensors>` | 模型目录结构不对（缺 `linear_branch/`）| 摆正结构：`models/vdn/stage-xxx/linear_branch/model.safetensors` |
| `<无VDN stage：models/vdn 下未找到含 linear_branch 的 stage 目录>` | models/vdn 下没有模型，或模型放错目录 | `hf download OpenVDN/vdn-minimax-h3 stage-dmd-step-250 --local-dir ComfyUI/models/vdn` |
| `🔄` 报"未找到 ComfyUI 兼容的 H3 基座" | diffusion_models 下没有 ComfyUI 格式 H3（含 video/audio_patch_proj）| 放入 ComfyUI 格式 H3 bf16/fp16 单文件（或手动选 FastH3 等） |
| 正常显示 `stage-dmd-step-250` 等 | ✅ 正常 / All good | 直接选用即可 / Just select it |

> 速查 / Quick check：**有文件却不显示 = ①依赖没装上 ②模型结构不完整 ③模型放错目录**。三种情况下拉框都会直接写明。/ The dropdown always states which.

---

## 版本历史 / Changelog

- **v2.3**：节点套件扩展为 10 节点——新增 **BSAIVDNH3Loader**（基座+stage+LoRA+3 画质模式）、**DualLora**（双 LoRA 去伪影）、**Timesteps**（8/16/50 步精确阶梯）、**EulerSampler**（v12/a3 双调度）、**Accel**（BlockCache/CacheDiT/TE/VSA 整合）、**Upscale/FaceRestore/FaceOil**（超分+人脸修复+去油）、**EmptyLatentVideo**（H3 5D 空 latent）；旧 LoaderNative 保留兼容。/ Node suite expanded to 10 nodes.
- **v2.2**：**vdn_h3 运行时内嵌**——`vendor/vdn_h3/` 随包分发（Apache-2.0），用户零安装。/ Runtime embedded — zero-install.
- **v2.1**：**依赖自动安装**——启动时自动检测并 clone 缺失的 ComfyUI-VDN-H3。/ Auto-install missing dependency.
- **v2.0**：重构为一体化 Native Loader（`BSAIVDNH3LoaderNative`）；自实现 VDN stage 扫描，结构不完整与依赖缺失均给出明确下拉提示。/ Refactored all-in-one native loader with clear dropdown diagnostics.

## License / 许可证

- 本插件代码 / This plugin code：Apache-2.0
- **VDN-H3 模型权重不随本仓库分发**，遵循 MiniMax H3 Community License Agreement。/ VDN-H3 weights are NOT distributed here; they follow the MiniMax H3 Community License.
