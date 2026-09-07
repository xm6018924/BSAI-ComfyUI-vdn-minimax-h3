"""BSAI VDN-H3 一体化Loader — 整合 H3主模型加载 + 原生VDN-H3(ApplyVDNH3Advanced) + ChunkFeedForward。

基于 ComfyUI-VDN-H3 插件的原生实现（vdn_h3 包），整合为单节点：
  - 顶部 H3 主模型下拉框（直接选择可用的 H3 基座模型）
  - VDN stage / turbo / linear_branch / KV压缩 等全部参数
  - 可选 ChunkFeedForward 分块前馈加速
"""

import logging
import os
import sys
import types

import folder_paths
import comfy.model_management
import comfy.sd

_log = logging.getLogger("comfy.bsai_vdn")

# ---------------------------------------------------------------------------
# 注入 ComfyUI-VDN-H3 插件路径，import 原生 vdn_h3 包
# ---------------------------------------------------------------------------
_VDN_PLUGIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ComfyUI-VDN-H3")
if _VDN_PLUGIN not in sys.path:
    sys.path.insert(0, _VDN_PLUGIN)

try:
    from vdn_h3.apply import apply_adapters
    from vdn_h3.hybrid import VDNState, apply_vdn
    from vdn_h3.branch import LinearBranch
    import vdn_h3.spec as spec
    from vdn_h3.nodes import _apply_vdn, _disable_comfy_compiler_on_broken_builds
    _VDN_NATIVE_AVAILABLE = True
except Exception as e:
    _log.warning(f"[BSAI VDN] 原生 vdn_h3 包不可用: {e}")
    _VDN_NATIVE_AVAILABLE = False


# ---------------------------------------------------------------------------
# ChunkFeedForward 实现（移植自 KJNodes MiniMaxChunkFeedForward）
# ---------------------------------------------------------------------------
def _minimax_mlp_chunked_forward(self, x, *args, **kwargs):
    """SwiGLU MLP 分块前向，降低峰值显存。"""
    chunks = getattr(self, "kj_num_chunks", 2)
    seq_threshold = getattr(self, "kj_seq_threshold", 4096)
    if x.shape[0] <= seq_threshold or chunks == 1:
        return self._original_forward(x, *args, **kwargs)
    out = []
    chunk_size = (x.shape[0] + chunks - 1) // chunks
    for i in range(0, x.shape[0], chunk_size):
        out.append(self._original_forward(x[i:i + chunk_size], *args, **kwargs))
    import torch
    return torch.cat(out, dim=0)


class _FFNChunkPatch:
    def __init__(self, num_chunks, seq_threshold):
        self.num_chunks = num_chunks
        self.seq_threshold = seq_threshold

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, "_original_forward"):
            obj._original_forward = obj.forward
        def wrapped_forward(self_module, *args, **kwargs):
            self_module.kj_num_chunks = self.num_chunks
            self_module.kj_seq_threshold = self.seq_threshold
            return _minimax_mlp_chunked_forward(self_module, *args, **kwargs)
        return types.MethodType(wrapped_forward, obj)


def _apply_chunk_ffn(model, chunks=2, seq_threshold=4096):
    """对 H3 模型的所有 block MLP 应用分块前向。"""
    if chunks <= 1:
        return model
    m = model.clone()
    dm = m.get_model_object("diffusion_model")
    blocks = getattr(dm, "blocks", None)
    if not blocks or not hasattr(blocks[0], "mlp") or not hasattr(blocks[0].mlp, "fc1"):
        _log.warning("[BSAI VDN] ChunkFeedForward: 模型不像 H3，跳过")
        return model
    for idx, block in enumerate(blocks):
        patched = _FFNChunkPatch(chunks, seq_threshold).__get__(block.mlp, block.mlp.__class__)
        m.add_object_patch(f"diffusion_model.blocks.{idx}.mlp.forward", patched)
    return m


# ---------------------------------------------------------------------------
# BSAI VDN-H3 一体化 Loader
# ---------------------------------------------------------------------------
class BSAIVDNH3LoaderNative:
    """BSAI VDN-H3 Loader · 权重加载（基座+线性分支）

    单节点整合：
      1. H3 主模型加载（UNETLoader）
      2. 原生 VDN-H3 应用（ApplyVDNH3Advanced：stage+turbo+linear_branch+KV压缩）
      3. 可选 ChunkFeedForward 分块前馈加速
    """

    @classmethod
    def INPUT_TYPES(cls):
        # H3 主模型列表（diffusion_models 目录）
        unet_names = folder_paths.get_filename_list("diffusion_models")
        h3_names = [n for n in unet_names if "h3" in n.lower() or "minimax" in n.lower()]
        if not h3_names:
            h3_names = unet_names

        # VDN checkpoint 列表
        try:
            vdn_names = spec.list_vdn_checkpoints() if _VDN_NATIVE_AVAILABLE else []
        except Exception:
            vdn_names = []

        return {"required": {
            "unet_name": (h3_names or ["<无H3模型>"], {
                "tooltip": "H3 主模型（diffusion_models 目录）"}),
            "vdn_checkpoint": (vdn_names or ["<无VDN stage>"], {
                "tooltip": "VDN stage 目录（models/vdn）"}),
            "apply_turbo_adapter": ("BOOLEAN", {"default": True}),
            "stage_b_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "turbo_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "lora_mode": (["bypass", "merge"], {"default": "merge"}),
            "branch_weights": (["auto", "stream", "cache_gpu"], {"default": "auto"}),
            "retain_buffers": (["auto", "on", "off"], {"default": "auto"}),
            "verbose": ("BOOLEAN", {"default": True}),
            "attention_backend": (["grouped", "flex"], {"default": "grouped"}),
            "window_radius": ("INT", {"default": 1, "min": 0, "max": 8}),
            "window_chunk": ("INT", {"default": 5, "min": 0, "max": 64}),
            "anchor_frames": (["both", "columns", "rows", "none"], {"default": "both"}),
            "text_state": ("BOOLEAN", {"default": True}),
            "linear_branch": ("BOOLEAN", {"default": True}),
            "fast_kernels": ("BOOLEAN", {"default": True}),
            "chunk_ffn": ("BOOLEAN", {"default": True, "tooltip": "启用 ChunkFeedForward 分块前馈加速"}),
            "chunks": ("INT", {"default": 2, "min": 1, "max": 64}),
            "seq_threshold": ("INT", {"default": 4096, "min": 256, "max": 262144, "step": 256}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "BSAI/VDN-H3"

    def load(self, unet_name, vdn_checkpoint, apply_turbo_adapter,
             stage_b_strength, turbo_strength, lora_mode, branch_weights,
             retain_buffers, verbose, attention_backend,
             window_radius, window_chunk, anchor_frames, text_state,
             linear_branch, fast_kernels, chunk_ffn, chunks, seq_threshold):

        if not _VDN_NATIVE_AVAILABLE:
            raise RuntimeError(
                "BSAI VDN-H3 Loader 需要 ComfyUI-VDN-H3 插件（vdn_h3 包）。"
                "请确认 custom_nodes/ComfyUI-VDN-H3 存在且可正常加载。")

        # 1. 加载 H3 主模型（必须传完整路径，load_diffusion_model不做路径解析）
        _log.info(f"[BSAI VDN] 加载主模型: {unet_name}")
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise RuntimeError(f"模型文件不存在: {unet_name} (diffusion_models目录)")
        model = comfy.sd.load_diffusion_model(unet_path)

        # 2. 应用原生 VDN-H3
        _log.info(f"[BSAI VDN] 应用 VDN stage: {vdn_checkpoint}")
        strength = {"default": stage_b_strength, "turbo": turbo_strength}
        cfg_overrides = {
            "radius": window_radius,
            "chunk": window_chunk,
            "anchor_frames": anchor_frames,
            "enable_text_state": text_state,
            "linear_enabled": linear_branch,
        }
        (model,) = _apply_vdn(
            model, vdn_checkpoint, strength, lora_mode, branch_weights,
            attention_backend, verbose,
            apply_turbo_adapter=apply_turbo_adapter,
            cfg_overrides=cfg_overrides,
            fast_kernels=fast_kernels,
            retain_buffers=retain_buffers,
        )

        # 3. 可选 ChunkFeedForward
        if chunk_ffn and chunks > 1:
            _log.info(f"[BSAI VDN] 应用 ChunkFeedForward: chunks={chunks}, threshold={seq_threshold}")
            model = _apply_chunk_ffn(model, chunks, seq_threshold)

        return (model,)


NODE_CLASS_MAPPINGS = {
    "BSAIVDNH3LoaderNative": BSAIVDNH3LoaderNative,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAIVDNH3LoaderNative": "BSAI VDN-H3 Loader · 权重加载（基座+线性分支)",
}
