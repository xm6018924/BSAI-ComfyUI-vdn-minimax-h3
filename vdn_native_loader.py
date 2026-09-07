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
# vdn_h3 运行时获取顺序：
#   1) 内嵌 vendor/vdn_h3（本插件自带，用户零安装，Apache-2.0 已随包分发）
#   2) 外部 ComfyUI-VDN-H3 插件（兼容已装用户）
#   3) 自动 git clone 兜底（以上都缺失时）
# ---------------------------------------------------------------------------
_PLUGIN_ROOT = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_PLUGIN_ROOT, "vendor")
_VENDOR_VDN = os.path.join(_VENDOR_DIR, "vdn_h3")
_VDN_PLUGIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ComfyUI-VDN-H3")


def _auto_install_vdn_h3():
    """自动安装缺失的 ComfyUI-VDN-H3 依赖插件（极端兜底：vendor 与外部均缺失时）。"""
    if os.path.isdir(_VDN_PLUGIN):
        return True
    try:
        import subprocess
        _log.info("[BSAI VDN] 内嵌 vdn_h3 缺失且未安装 ComfyUI-VDN-H3，正在自动安装...")
        proc = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/OpenVDN/ComfyUI-VDN-H3", _VDN_PLUGIN],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode == 0 and os.path.isdir(_VDN_PLUGIN):
            _log.info("[BSAI VDN] ComfyUI-VDN-H3 自动安装成功")
            return True
        _log.warning(f"[BSAI VDN] ComfyUI-VDN-H3 自动安装失败: {(proc.stderr or '')[:400]}")
    except Exception as e:
        _log.warning(f"[BSAI VDN] ComfyUI-VDN-H3 自动安装异常: {e}")
    return False


if os.path.isdir(_VENDOR_VDN):
    # 内嵌包优先：把 vendor 放到 sys.path 最前，import vdn_h3 命中自带包
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
elif _VDN_PLUGIN not in sys.path:
    # 兜底：外部插件
    _auto_install_vdn_h3()
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
# VDN stage 目录扫描（不依赖 vdn_h3.spec，自实现，兼容多种放置方式）
# ---------------------------------------------------------------------------
def _get_vdn_roots():
    """返回所有应扫描的 VDN 模型根目录：
      - 标准: ComfyUI/models/vdn（folder_paths 注册的 "vdn"）
      - 兼容旧 README 路径: ComfyUI/models/vdn_h3
    """
    roots = []
    try:
        if "vdn" not in folder_paths.folder_names_and_paths:
            for base in {os.path.dirname(p) for p in folder_paths.get_folder_paths("loras")}:
                folder_paths.add_model_folder_path("vdn", os.path.join(base, "vdn"))
        roots = list(folder_paths.get_folder_paths("vdn"))
    except Exception:
        roots = []
    try:
        legacy = os.path.join(folder_paths.base_path, "models", "vdn_h3")
        if os.path.isdir(legacy) and legacy not in roots:
            roots.append(legacy)
    except Exception:
        pass
    return roots


def _stage_has_branch(d):
    """stage 目录有效判定：linear_branch/ 子目录下存在 .safetensors/.bin 分支文件。"""
    lb = os.path.join(d, "linear_branch")
    if os.path.isdir(lb):
        try:
            for f in os.listdir(lb):
                if f.lower().endswith((".safetensors", ".bin")):
                    return True
        except OSError:
            return False
    return False


def _list_vdn_stages():
    """递归扫描 VDN 根目录，返回结构完整（含 linear_branch 分支文件）的 stage 相对名。
    兼容嵌套两层目录（vdn/stage-x/stage-x/...），自动取有效层去重。"""
    found = []
    for root in _get_vdn_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _files in os.walk(root):
            if os.path.basename(dirpath) == "linear_branch":
                continue  # 由父 stage 目录判定，避免把 linear_branch 本身当 stage
            rel = os.path.relpath(dirpath, root).replace("\\", "/")
            if rel == ".":
                continue
            if _stage_has_branch(dirpath):
                found.append(rel)
                dirnames[:] = []
    return sorted(set(found))


def _dir_contains_valid_stage(d):
    """目录本身或其任一子目录是否包含有效 stage（递归）。"""
    if _stage_has_branch(d):
        return True
    try:
        for name in os.listdir(d):
            sub = os.path.join(d, name)
            if os.path.isdir(sub) and _dir_contains_valid_stage(sub):
                return True
    except OSError:
        pass
    return False


def _detect_malformed_stages():
    """检测"模型在但显示不出来"的结构问题：根目录下有 stage-* 目录，
    但自身及子目录都不含 linear_branch 分支文件（结构不完整/为空）。"""
    hints = []
    for root in _get_vdn_roots():
        if not os.path.isdir(root):
            continue
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for name in names:
            d = os.path.join(root, name)
            if not os.path.isdir(d) or not name.startswith("stage-"):
                continue
            if _dir_contains_valid_stage(d):
                continue
            hints.append(
                f"<⚠ {name} 目录存在但结构不完整：需放置 {name}/linear_branch/model.safetensors "
                f"（官方结构，参见 README）>"
            )
    return hints


# ---------------------------------------------------------------------------
# ChunkFeedForward 实现（移植自 KJNodes MiniMaxChunkFeedForward）
# ---------------------------------------------------------------------------
def _minimax_mlp_chunked_forward(self, x, *args, **kwargs):
    """SwiGLU MLP 分块前向，降低峰值显存。"""
    chunks = getattr(self, "kj_num_chunks", 8)
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


def _apply_chunk_ffn(model, chunks=8, seq_threshold=4096):
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

        # VDN checkpoint 列表（自实现扫描，兼容 models/vdn 与旧 models/vdn_h3）
        if _VDN_NATIVE_AVAILABLE:
            try:
                vdn_names = _list_vdn_stages()
            except Exception:
                vdn_names = []
            # 结构不完整提示项（模型在但显示不出来的常见原因）
            try:
                vdn_names += _detect_malformed_stages()
            except Exception:
                pass
            vdn_names = vdn_names or ["<无VDN stage：models/vdn 下未找到含 linear_branch 的 stage 目录>"]
        else:
            vdn_names = ["<未安装依赖：请先安装 ComfyUI-VDN-H3 插件>"]

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
            "chunks": ("INT", {"default": 8, "min": 1, "max": 64}),
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
                "请先安装 custom_nodes/ComfyUI-VDN-H3（git clone "
                "https://github.com/OpenVDN/ComfyUI-VDN-H3），"
                "再重启 ComfyUI。模型目录有文件但下拉框显示不了，多半是此依赖缺失。")

        if str(vdn_checkpoint).startswith("<"):
            raise RuntimeError(
                f"请先修复 VDN 模型问题后再运行：{vdn_checkpoint}\n"
                "正确结构：ComfyUI/models/vdn/<stage名>/linear_branch/model.safetensors（或 "
                "model_int8_convrot_comfyui.safetensors）。")

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
