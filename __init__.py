"""BSAI-ComfyUI-vdn-minimax-h3 — VDN-H3 (Video DeltaNet on MiniMax H3) 节点套件。"""

# === BSAI H3 VAE 全量加载统一补丁（宿主 BSAI-ComfyUI-Sol-H3）===
try:
    import sys as _sys, os as _os
    _SOL = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "..", "BSAI-ComfyUI-Sol-H3")
    if _SOL not in _sys.path:
        _sys.path.insert(0, _SOL)
    import bsai_h3_vae_full as _h3v
    _h3v.apply_global_patch("BSAI-ComfyUI-vdn-minimax-h3")
except Exception:
    pass

import os
import logging

import folder_paths

# 注册 VDN-H3 权重目录（放 ComfyUI/models/vdn，结构与 HF 一致）
_VDN_DIR = os.path.join(folder_paths.models_dir, "vdn")
try:
    if not os.path.isdir(_VDN_DIR):
        os.makedirs(_VDN_DIR, exist_ok=True)
    folder_paths.add_model_folder_path("vdn", _VDN_DIR)
except Exception as e:
    logging.warning(f"[BSAI VDN-H3] 无法创建/注册 vdn 模型目录 {_VDN_DIR}: {e}")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 原生 VDN-H3 一体化 Loader（基于 ComfyUI-VDN-H3 插件）
try:
    from .vdn_native_loader import (
        NODE_CLASS_MAPPINGS as _NATIVE_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _NATIVE_DISPLAY,
    )
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_NATIVE_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_NATIVE_DISPLAY}
except Exception as e:
    logging.warning(f"[BSAI VDN-H3] 原生Loader加载失败: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
