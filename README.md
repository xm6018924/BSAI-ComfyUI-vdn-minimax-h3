# BSAI ComfyUI vdn-minimax-h3

基于 **VDN-H3（Video DeltaNet on MiniMax H3）** 的一站式 ComfyUI 节点套件。

VDN-H3 是 MiniMax H3 的**混合注意力加速**方案：新增一条**帧级线性注意力分支**（高效）+ 保留 **softmax 分支**（维持视觉质量与一致性），配合 **default（50 步质量）** 与 **turbo（8 步 DMD2 蒸馏）** 两个小 LoRA 适配器，即插即用合并进骨干网络，实现"比播放还快"的视频生成（官方：8×B200 上 14.4 秒片段 8 步去噪仅 11.23 秒）。

- 项目主页: https://openvdn.github.io/
- GitHub 代码: https://github.com/OpenVDN/vdn-minimax-h3
- HuggingFace 权重: https://huggingface.co/OpenVDN/vdn-minimax-h3

---

## 功能一览

本插件把 VDN-H3 的推理 + 当前最新的一整套视频生成增强技术封装为 ComfyUI 节点：

| 节点 | 能力 |
|---|---|
| **BSAIVDNH3Loader** | VDN-H3 权重加载（h3-base 基座 + linear_branch 线性注意力分支 + turbo/default LoRA 合并），环境自动检测与降级 |
| **BSAIVDNH3DualLora** | 双 LoRA 混合加载（VDN default+turbo 或任意两 LoRA），低步数下补偿细节、**抑制蒸馏伪影**；可选伪影后处理 |
| **BSAIVDNH3Timesteps** | VDN 精确时间步：8 步 turbo（DMD 蒸馏阶梯）/ 50 步质量 / 自定义 |
| **BSAIVDNH3EulerSampler** | VDN Euler 采样器（video shift 12 / audio shift 3 双调度，自动检测 ModelSamplingAV） |
| **BSAIVDNH3Accel** | 最新加速技术整合：**Block Cache + CacheDiT + TE-Speed + VSA 视频稀疏注意力**，全部可开关 |
| **BSAIVDNH3Upscale** | **画质超分放大 + 高清修复**（桥接 BSAI-H3-upscale-4K 的 Real-ESRGAN/4K/DLSS 引擎，时序光流稳定 + 细节锐化 + 防振铃） |
| **BSAIVDNH3FaceRestore** | **远景小脸崩坏修复**（YOLOv8-Face 多尺度检测 + GFPGAN/CodeFormer 重建五官，小脸增强预设） |
| **BSAIVDNH3FaceOil** | **脸部去油**（高光+低饱和油光检测，保边 bilateral 平滑收敛，可选 Haar 人脸限定） |

## 权重安装

VDN-H3 权重（约 82 GB）结构与 HF 仓库一致，放到：

```
ComfyUI/models/vdn_h3/
├── h3-base/                  # MiniMax H3 基座：transformer、video/audio VAEs、schedulers (~72 GB)
├── stage-b-step-2000/        # VDN-H3-50 步：linear_branch/ + adapters/default/ LoRA (~4.3 GB)
└── stage-dmd-step-250/       # VDN-H3-8 步：上述 + adapters/turbo/ LoRA (~5.1 GB)
```

下载方式：

```bash
# 需先安装 huggingface_hub
hf download OpenVDN/vdn-minimax-h3 --local-dir ComfyUI/models/vdn_h3
```

> **h3-base 兼容性说明**：官方 `h3-base/` 是 diffusers 0.36 布局（`proj_in/proj_out` 合并结构），ComfyUI 的 `minimax_h3` 使用 `video_patch_proj/final_layer` 布局，**两者不通用、无法直接加载**。加载器 `backbone` 的 `🔄` 选项会自动扫描 `models/diffusion_models` 中 ComfyUI 兼容的 MiniMax H3 bf16/fp16 权重（如 `minimax_h3_fl2va_bf16.safetensors`、`10Eros_*_bf16*.safetensors`）并优先选择全精度版，等价于使用官方基座。官方分片保留在 `models/vdn_h3/h3-base/` 供 diffusers 侧工具使用。

> 基座 `backbone` 也可直接用 `models/diffusion_models` 里已有的任意 MiniMax H3 权重（如 FastH3 4 步蒸馏），VDN stage 目录照常提供线性分支 + LoRA。

## 环境要求

- **原生线性分支加速**（recommended）：PyTorch 2.13+ / FlashAttention-4（FlexAttention Flash backend）。加载器自动检测并打印环境报告。
- **降级模式**：无 FA4 时自动跳过 linear_branch 合并，模型仍以 Dense 运行，可叠加 **turbo LoRA 8 步 + VSA 稀疏注意力 + Block Cache/CacheDiT** 保持高速。
- **后处理增强**：超分/人脸修复桥接 `BSAI-H3-upscale-4K`；去油用 OpenCV（缺失时自动降级纯 torch 高光抑制）。

## 示例工作流

`example_workflows/vdn_h3_full.json` — 全功能参考流程：

```
VDN Loader → Dual LoRA(去伪影) → Accel(BlockCache/CacheDiT/TE/VSA)
   → H3 Conditioning(MiniMaxH3ImageToVideo) + Timesteps + Euler
   → SamplerCustomAdvanced → VAEDecode
   → Upscale(4K超分修复) → FaceRestore(小脸修复) → FaceOil(去油)
   → VHS_VideoCombine(保存mp4) + PreviewImage
```

在 ComfyUI 中 `Workflow → Open` 加载即可。使用前把 `LoadCLIP` / `VAELoader` / `BSAIVDNH3Loader.backbone` 换成你本机实际模型文件。

## 节点详解

### BSAIVDNH3Loader
- `backbone`：基座 transformer（models/diffusion_models）
- `vdn_stage`：stage-dmd-step-250（8 步，推荐）/ stage-b-step-2000（50 步）
- `merge_linear_branch`：合并 linear_branch 线性注意力分支（需 FA4，无则降级）
- `merge_default_lora` / `merge_turbo_lora`：合并 50 步质量 / 8 步蒸馏 LoRA
- `lora_strength`：LoRA 强度
- 输出 `MODEL` + `info`（环境/加载报告）+ `stage_dir`

### BSAIVDNH3DualLora（双 LoRA 去伪影）
- `lora_a` / `lora_a_strength`：主 LoRA（如 turbo 8 步，速度+去伪影核心）
- `lora_b` / `lora_b_strength`：辅助 LoRA（如 default 50 步质量，建议 0.2~0.5 轻叠细节）
- `artifact_suppress`：低步数典型伪影（条纹/振铃/结构粘连）后处理抑制强度，0=关闭
- LoRA 放 `models/loras`（ComfyUI 标准格式）；VDN 的 `adapter_model.safetensors` 会自动做 diffusers→ComfyUI 转换

### BSAIVDNH3Accel
- `block_cache_enable/start/end`：Block Cache 跨步缓存区间
- `cachedit_enable`：CacheDiT 邻域步缓存
- `te_speed_enable`：TE-Speed 文本编码快速路径
- `vsa_enable/keep_percent`：VSA 视频稀疏注意力（保留精确注意力百分比）
- `detail_preserve`：对 cache/VSA 造成的细节损失做高频补偿

### BSAIVDNH3Upscale
桥接 `BSAI-H3-upscale-4K` 引擎：Real-ESRGAN general-x4v3（默认）/ x4plus / 4K / DLSS5 等；时序光流稳定 + 细节锐化 + 柔和防振铃；未安装时自动降级 `F.interpolate` bicubic。

### BSAIVDNH3FaceRestore
远景/全景小脸崩坏修复：`小脸增强(CodeFormer)` 预设（低置信度 + 高融合 + 自动降保真），修复远处模糊变形五官。

### BSAIVDNH3FaceOil
- `strength`：去油强度（0.5~0.7 自然，过高抹平质感）
- `bright_thr` / `sat_thr`：高光判定（亮度高 + 彩度低 = 油光）
- `smooth_radius`：保边平滑半径
- `face_only`：仅 Haar 人脸区域去油

## 版本

- v1.0：首发。VDN-H3 加载 + 双 LoRA 去伪影 + 8/50 步时间步 + Euler 采样 + 加速整合 + 超分/人脸/去油 + 全功能示例工作流。

## License

- 本插件代码：Apache-2.0
- **VDN-H3 模型权重不随本仓库分发**，属于 MiniMax H3 的衍生，遵循 **MiniMax H3 Community License Agreement**（适用领土排除欧盟/英国/韩国/美国；使用与分发前请阅读 https://huggingface.co/OpenVDN/vdn-minimax-h3 附带的完整协议）。
