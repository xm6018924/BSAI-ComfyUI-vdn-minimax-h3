# BSAI ComfyUI vdn-minimax-h3

基于 **VDN-H3（Video DeltaNet on MiniMax H3）** 的一体化 ComfyUI 节点。

VDN-H3 是 MiniMax H3 的**混合注意力加速**方案：新增一条**帧级线性注意力分支**（高效）+ 保留 **softmax 分支**（维持视觉质量与一致性），配合 **default（50 步质量）** 与 **turbo（8 步 DMD2 蒸馏）** 两个小 LoRA 适配器，即插即用合并进骨干网络，实现"比播放还快"的视频生成（官方：8×B200 上 14.4 秒片段 8 步去噪仅 11.23 秒）。

- 项目主页: https://openvdn.github.io/
- GitHub 代码: https://github.com/OpenVDN/vdn-minimax-h3
- HuggingFace 权重: https://huggingface.co/OpenVDN/vdn-minimax-h3

---

## ⚠️ 依赖自动安装（无需手动操作）

本节点调用 **ComfyUI-VDN-H3** 插件提供的原生 `vdn_h3` 运行时。**从 v2.0 起，插件启动时会自动检测并 `git clone` 安装缺失的 ComfyUI-VDN-H3**——用户升级本插件后什么都不用做，首次启动会自动装好依赖（需联网，约 1-2 分钟，日志可见 `[BSAI VDN] ComfyUI-VDN-H3 自动安装成功`）。

- 若自动安装失败（无 git / 网络不通），下拉框会显示 `<未安装依赖>`，可手动补装：
  ```bash
  cd ComfyUI/custom_nodes
  git clone https://github.com/OpenVDN/ComfyUI-VDN-H3
  ```
- `vdn_h3` 无额外 pip 依赖，只使用 ComfyUI 自带的 torch/safetensors，clone 完成后即可直接使用。

## 节点

| 节点 | 能力 |
|---|---|
| **BSAIVDNH3LoaderNative** | 一体化权重加载：H3 主模型（models/diffusion_models）+ VDN stage（models/vdn）+ turbo/default 适配器 + 可选 ChunkFeedForward 分块前馈加速 |

## 权重安装

VDN stage 权重放到 `ComfyUI/models/vdn/`，**保持官方目录结构不变**：

```
ComfyUI/models/vdn/
└── stage-dmd-step-250/            # 8 步蒸馏（推荐）；stage-b-step-2000 为 50 步质量版
    ├── model_spec.json
    ├── linear_branch/
    │   ├── model.safetensors                    # 分支权重（bf16）
    │   └── model_int8_convrot_comfyui.safetensors  # 预量化版（可选）
    └── adapters/
        ├── default/                             # 50 步质量 LoRA
        └── turbo/                               # 8 步蒸馏 LoRA
```

> **判定标准**：节点把"目录下存在 `linear_branch/` 且其中有 `.safetensors/.bin` 分支文件"的目录识别为一个 VDN stage。
> **结构不完整时**（比如只解压了 stage 根目录、缺 `linear_branch/` 子目录），下拉框会给出 `<⚠ xxx 目录存在但结构不完整>` 提示，并说明正确放置方式——不是模型丢失，是结构不完整。

下载方式（保持结构）：

```bash
# 需先安装 huggingface_hub
hf download OpenVDN/vdn-minimax-h3 stage-dmd-step-250 --local-dir ComfyUI/models/vdn
```

> 旧版文档写的 `models/vdn_h3/` 路径已废弃；本插件同时兼容该旧路径，但推荐统一使用 `models/vdn`。

## H3 主模型

`unet_name` 下拉框扫描 `models/diffusion_models`，自动过滤 MiniMax H3 / minimax 关键字。需要 ComfyUI 兼容的单文件权重（如 `minimax_h3_fl2va_bf16.safetensors`、FastH3 4 步蒸馏等）。

## 环境要求

- **原生线性分支加速**（recommended）：PyTorch 2.13+ / FlashAttention-4（FlexAttention Flash backend）。加载器自动检测。
- **降级模式**：无 FA4 时自动跳过 linear_branch 合并，模型以 Dense 运行。
- VDN stage 模型遵循 **MiniMax H3 Community License Agreement**（适用领土排除欧盟/英国/韩国/美国），请阅读 https://huggingface.co/OpenVDN/vdn-minimax-h3 附带协议。

## 示例工作流

`example_workflows/` 下提供参考工作流，`Workflow → Open` 加载后把 `unet_name` / `vdn_checkpoint` 换成你本机实际模型即可。

## 版本

- v2.1：**依赖自动安装**——启动时自动检测并 clone 缺失的 ComfyUI-VDN-H3，用户升级插件即可全部自动装好。
- v2.0：重构为一体化 Native Loader（`BSAIVDNH3LoaderNative`）；自实现 VDN stage 扫描（不依赖 spec），兼容标准/嵌套目录结构，结构不完整与依赖缺失均给出明确下拉提示。

## License

- 本插件代码：Apache-2.0
- **VDN-H3 模型权重不随本仓库分发**，遵循 MiniMax H3 Community License Agreement。
