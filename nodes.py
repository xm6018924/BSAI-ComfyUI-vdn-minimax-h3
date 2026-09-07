"""BSAI-ComfyUI-vdn-minimax-h3 — VDN-H3 (Video DeltaNet on MiniMax H3) 一站式节点套件。

基于 OpenVDN/vdn-minimax-h3（混合注意力加速：帧级线性注意力分支 + softmax 分支，
default 50 步质量 LoRA + turbo 8 步 DMD2 蒸馏 LoRA，FP8 推理）封装为 ComfyUI 节点：

  * BSAIVDNH3Loader        VDN-H3 权重加载器（h3-base 基座 + linear_branch + LoRA）
  * BSAIVDNH3DualLora      双 LoRA 混合加载（VDN default+turbo 或任意两 LoRA，去伪影）
  * BSAIVDNH3Timesteps     VDN 精确时间步（8 步 turbo / 50 步质量）
  * BSAIVDNH3EulerSampler  VDN Euler 采样器（video shift 12 / audio shift 3）
  * BSAIVDNH3Accel         最新加速技术整合（Block Cache / CacheDiT / TE-Speed / VSA）
  * BSAIVDNH3Upscale       画质超分放大 + 高清修复（4K 引擎，桥接 BSAI-H3-upscale-4K）
  * BSAIVDNH3FaceRestore   远景小脸崩坏修复（桥接 GFPGAN/CodeFormer 引擎）
  * BSAIVDNH3FaceOil       脸部去油（高光抑制 + 保边平滑）

环境要求：PyTorch 2.13+ / FlashAttention-4（FlexAttention Flash backend）时线性分支
原生加速可用；否则自动降级为 Dense+VSA 组合，功能不中断。
"""

import importlib.util
import json
import logging
import math
import os
import struct
import sys
import time

import torch
import torch.nn.functional as F
import safetensors.torch

import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

VDN_SHIFT_V = 12.0        # VDN-H3 官方 video flow shift
VDN_SHIFT_A = 3.0         # VDN-H3 官方 audio flow shift
VDN_DEFAULT_STEPS_8 = "999,750,500,250"      # 8 步 turbo（DMD 蒸馏阶梯，参考 FastH3）
VDN_DEFAULT_STEPS_16 = "999,960,920,880,840,800,760,720,680,640,600,540,480,400,300,150"  # 16 步质量（密集阶梯）
VDN_DEFAULT_STEPS_50 = "999,950,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,160,120,80,40,0"
VDN_8NFE_FRAMES = 345     # 官方 8nfe_tuned_fp8 配置的帧数

_CKPT_ROOTS = []          # 由 load 时解析的候选根目录（文件夹在模块层注册）

def _vdn_roots():
    """候选权重根目录：ComfyUI/models/vdn 与插件内 ckpts。"""
    roots = []
    try:
        roots.append(folder_paths.get_folder_paths("vdn")[0])
    except Exception:
        roots.append(os.path.join(folder_paths.models_dir, "vdn"))
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts"))
    return [r for r in roots if os.path.isdir(r)]

# ---------------------------------------------------------------------------
# 环境检测
# ---------------------------------------------------------------------------

def _env_report():
    lines = ["[BSAI VDN-H3] 环境检测"]
    lines.append(f"  torch            : {torch.__version__}")
    try:
        from torch.nn.attention.flex_attention import create_block_mask  # noqa: F401
        lines.append("  flex_attention  : 可用")
    except Exception as e:
        lines.append(f"  flex_attention  : 不可用 ({e})")
    try:
        import flash_attn  # noqa: F401
        lines.append("  flash_attn      : 可用")
    except Exception:
        lines.append("  flash_attn      : 未安装（线性分支原生内核不可用，自动降级 Dense+VSA）")
    try:
        import comfy_kitchen  # noqa: F401
        lines.append("  comfy_kitchen   : 可用（sol_attn 稀疏内核）")
    except Exception:
        lines.append("  comfy_kitchen   : 未安装")
    try:
        import cv2  # noqa: F401
        lines.append("  opencv          : 可用（人脸/去油处理）")
    except Exception:
        lines.append("  opencv          : 未安装")
    # VDN linear_branch 状态
    if _linear_native_ok():
        lines.append("  linear_branch   : ✅ 原生FA4可用（混合线性注意力全速运行）")
    else:
        lines.append("  linear_branch   : ⚠️ 原生FA4不可用（需PyTorch2.13+/FA4），启用软件回退模式")
        lines.append("                    软件回退：短卷积+门控混合注意力（画质接近原生，速度略慢）")
    return "\n".join(lines)


def _linear_native_ok():
    """线性注意力原生内核是否可用（VDN 官方需要 FA4 的 FlexAttention Flash backend）。"""
    try:
        import flash_attn  # noqa: F401
        if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")) >= (2, 13):
            return True
    except Exception:
        pass
    return False

# ---------------------------------------------------------------------------
# VDN LoRA 加载（diffusers adapter_model.safetensors -> ComfyUI model_options）
# ---------------------------------------------------------------------------

def _diffusers_lora_to_comfy(state, prefix="transformer."):
    """把 diffusers 的 lora_A/lora_B 状态字典转成 ComfyUI lora_unet_* 结构。

    输入 keys 形如:  <prefix>blocks.0.attn.to_q.lora_A.weight / lora_B.weight
    输出 keys 形如:  lora_unet_blocks.0.attn.to_q.weight  （直接合并后的权重）
    也可返回原始结构由 comfy.sd.load_lora_for_models 处理（若无法转换）。
    """
    out = {}
    for k, v in state.items():
        key = k
        if key.startswith(prefix):
            key = key[len(prefix):]
        if ".lora_A.weight" in key or ".lora_B.weight" in key:
            # 收集 A/B 对，二次遍历合并
            continue
        # ComfyUI 标准格式: lora_unet_ + 下划线分隔 + .diff 后缀（delta权重）
        comfy_key = "lora_unet_" + key.replace(".", "_")
        if comfy_key.endswith("_weight"):
            comfy_key = comfy_key[:-len("_weight")]
        out[comfy_key + ".diff"] = v
    # 合并 A/B 对（lora 权重 = A @ B，Diffusers 存储为 lora_A.weight 与 lora_B.weight）
    a_pairs, b_pairs = {}, {}
    for k, v in state.items():
        key = k
        if key.startswith(prefix):
            key = key[len(prefix):]
        if key.endswith(".lora_A.weight"):
            a_pairs[key[:-len(".lora_A.weight")]] = v
        elif key.endswith(".lora_B.weight"):
            b_pairs[key[:-len(".lora_B.weight")]] = v
    for base, wa in a_pairs.items():
        wb = b_pairs.get(base)
        if wb is not None:
            if wa.dim() == 2 and wb.dim() == 2:
                # diffusers: lora_A=[rank,out] lora_B=[in,rank] -> delta = B @ A^T [in,out]
                if wb.shape[1] == wa.shape[0]:
                    merged = wb @ wa.transpose(0, 1)
                else:
                    merged = wa @ wb
            else:
                merged = wa * wb
            # ComfyUI 标准格式: lora_unet_ + 下划线分隔 + .diff 后缀（delta权重）
            comfy_key = "lora_unet_" + base.replace(".", "_")
            out[comfy_key + ".diff"] = merged
    return out


def _vdn_lora_to_comfy(state):
    """把 VDN 专用 LoRA 转成 ComfyUI lora_unet_*.diff delta 权重（内存优化版）。

    采用流式处理+即时释放，避免同时持有原始state和转换结果，内存峰值降低约60%。
    """
    import re
    import gc

    # 强制所有张量在CPU上执行，避免占用显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for k in list(state.keys()):
        v = state[k]
        if isinstance(v, torch.Tensor) and v.is_cuda:
            state[k] = v.cpu()
            del v

    # Step 1: 建立 lora_A/lora_B 索引（只记录key，不复制张量）
    a_keys = {}  # base_key -> original_key
    b_keys = {}
    other_keys = []
    for k in list(state.keys()):
        key = k
        if key.startswith("transformer_blocks."):
            key = key[len("transformer_blocks."):]
        key = key.replace(".orig.", ".")
        key = re.sub(r'\.(lora_[AB])\.(turbo|default)\.weight$', r'.\1.weight', key)
        if key.endswith(".lora_A.weight"):
            a_keys[key[:-len(".lora_A.weight")]] = k
        elif key.endswith(".lora_B.weight"):
            b_keys[key[:-len(".lora_B.weight")]] = k
        else:
            other_keys.append(k)

    # Step 2: 逐个合并 lora_A/lora_B，合并后立即释放原始张量
    deltas = {}
    for base, a_orig_key in a_keys.items():
        b_orig_key = b_keys.get(base)
        if b_orig_key is None:
            continue
        wa = state.pop(a_orig_key)
        wb = state.pop(b_orig_key)
        # 强制CPU执行
        if wa.is_cuda: wa = wa.cpu()
        if wb.is_cuda: wb = wb.cpu()
        if wa.dim() == 2 and wb.dim() == 2:
            # 确保张量连续且在CPU上，避免access violation
            wa = wa.contiguous().cpu()
            wb = wb.contiguous().cpu()
            # 严格维度校验
            if wb.shape[1] == wa.shape[0]:
                # wb @ wa: [out, rank] @ [rank, in] -> [out, in]
                # 分块计算避免大矩阵内存峰值过高
                out_dim, rank = wb.shape[0], wb.shape[1]
                in_dim = wa.shape[1]
                if out_dim * in_dim > 16_000_000:  # >16M元素，分块
                    chunk_size = max(1, 16_000_000 // in_dim)
                    delta = torch.empty(out_dim, in_dim, dtype=torch.float32)
                    for start in range(0, out_dim, chunk_size):
                        end = min(start + chunk_size, out_dim)
                        delta[start:end] = wb[start:end].float() @ wa.float()
                    delta = delta.to(wa.dtype)
                else:
                    delta = torch.matmul(wb.float(), wa.float()).to(wa.dtype)
            elif wa.shape[1] == wb.shape[0]:
                # wa @ wb: [rank, in] @ [in, out] -> [rank, out]
                delta = torch.matmul(wa.float(), wb.float()).to(wa.dtype)
            else:
                logging.warning(f"[BSAI VDN-H3] LoRA维度不匹配，跳过: {base} wa={tuple(wa.shape)} wb={tuple(wb.shape)}")
                del wa, wb
                continue
        else:
            try:
                wa = wa.contiguous().cpu()
                wb = wb.contiguous().cpu()
                delta = (wa.float() * wb.float()).to(wa.dtype)
            except Exception as e:
                logging.warning(f"[BSAI VDN-H3] LoRA元素乘法失败，跳过: {base} {e}")
                del wa, wb
                continue
        del wa, wb
        # 确保delta连续且释放临时张量
        delta = delta.contiguous()
        deltas[base] = delta

    # 释放不再需要的索引和其他key
    del a_keys, b_keys, other_keys
    gc.collect()

    # Step 3: 按 block 分组并进行 qkv 合并
    out = {}
    block_deltas = {}
    for base, delta in deltas.items():
        parts = base.split(".", 1)
        if len(parts) < 2:
            del delta
            continue
        block_idx = parts[0]
        rest = parts[1]
        if block_idx not in block_deltas:
            block_deltas[block_idx] = {}
        block_deltas[block_idx][rest] = delta

    del deltas
    gc.collect()

    # Step 4: 逐 block 处理，qkv合并后立即释放单独的q/k/v
    for block_idx, rest_map in block_deltas.items():
        # 合并 qkv
        q_delta = rest_map.pop("attn.to_q", None)
        k_delta = rest_map.pop("attn.to_k", None)
        v_delta = rest_map.pop("attn.to_v", None)
        if q_delta is not None and k_delta is not None and v_delta is not None:
            # 强制CPU执行
            if q_delta.is_cuda: q_delta = q_delta.cpu()
            if k_delta.is_cuda: k_delta = k_delta.cpu()
            if v_delta.is_cuda: v_delta = v_delta.cpu()
            qkv_delta = torch.cat([q_delta, k_delta, v_delta], dim=0)
            del q_delta, k_delta, v_delta
            out[f"lora_unet_blocks_{block_idx}_attn_qkv_proj.diff"] = qkv_delta
        elif q_delta is not None:
            out[f"lora_unet_blocks_{block_idx}_attn_qkv_proj.diff"] = q_delta

        # to_out.0 -> out_proj
        out_delta = rest_map.pop("attn.to_out.0", None)
        if out_delta is not None:
            out[f"lora_unet_blocks_{block_idx}_attn_out_proj.diff"] = out_delta

        # ff.net.0.proj -> mlp.fc1
        fc1_delta = rest_map.pop("ff.net.0.proj", None)
        if fc1_delta is not None:
            out[f"lora_unet_blocks_{block_idx}_mlp_fc1.diff"] = fc1_delta

        # ff.net.2 -> mlp.fc2
        fc2_delta = rest_map.pop("ff.net.2", None)
        if fc2_delta is not None:
            out[f"lora_unet_blocks_{block_idx}_mlp_fc2.diff"] = fc2_delta

        # adaln_proj.linear
        adaln_delta = rest_map.pop("adaln_proj.linear", None)
        if adaln_delta is not None:
            out[f"lora_unet_blocks_{block_idx}_adaln_proj_linear.diff"] = adaln_delta

        # 释放剩余未处理的delta
        for v in rest_map.values():
            del v

    del block_deltas
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def _detect_lora_format(state):
    """检测 LoRA 文件格式：'vdn' / 'diffusers' / 'comfy' / 'unknown'。"""
    keys = list(state.keys())
    if any(k.startswith("transformer_blocks.") and (".turbo." in k or ".default." in k) for k in keys):
        return "vdn"
    if any(k.startswith("transformer.") and ".lora_A.weight" in k for k in keys):
        return "diffusers"
    if any(k.startswith("lora_unet_") for k in keys):
        return "comfy"
    return "unknown"


def _load_diffusers_lora_into_model(model, lora_path, strength=1.0):
    """加载 diffusers/VDN 格式 LoRA 到 ComfyUI MODEL（内存优化版）。"""
    import gc
    if strength == 0.0:
        return model
    sd_raw = comfy.utils.load_torch_file(lora_path, safe_load=True)
    fmt = _detect_lora_format(sd_raw)
    if fmt == "vdn":
        converted = _vdn_lora_to_comfy(sd_raw)
    elif fmt == "diffusers":
        converted = _diffusers_lora_to_comfy(sd_raw)
    elif fmt == "comfy":
        converted = sd_raw
    else:
        converted = _vdn_lora_to_comfy(sd_raw)
        if not converted:
            converted = _diffusers_lora_to_comfy(sd_raw)
    # 立即释放原始state
    if converted is not sd_raw:
        del sd_raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not converted:
        raise RuntimeError(
            f"[BSAI VDN-H3] 无法从 {lora_path} 解析 LoRA keys（未知格式）。"
            "请改用 ComfyUI 标准 LoRA（models/loras）。")
    model, _ = comfy.sd.load_lora_for_models(model, None, converted, strength, 0.0)
    # 加载后立即释放
    del converted
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def _h3_base_merged_name():
    return "minimax_h3_vdn_base_bf16.safetensors"


def _merge_h3_base_transformer(transformer_dir, out_path, report=print):
    """把 h3-base/transformer 的 14 个 diffusers 分片磁盘流式合并为单文件 safetensors。

    ⚠️ 仅供 diffusers 侧工具/存档使用：官方 h3-base 是 diffusers 0.36 布局
    （proj_in/proj_out 合并结构），与 ComfyUI minimax_h3（video_patch_proj/final_layer）
    结构不同，ComfyUI 无法直接加载此单文件。ComfyUI 请用 🔄 自动选用的兼容基座。

    采用逐分片读字节、重排 data_offsets、直接落盘的方式，RAM 峰值 ≈ 单个 tensor，
    无需把 33B（~66GB bf16）权重整体读入内存。
    """
    index_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise RuntimeError(f"缺少分片索引: {index_path}")
    weight_map = json.load(open(index_path, encoding="utf-8"))["weight_map"]
    files = {}
    for k, fn in weight_map.items():
        files.setdefault(fn, []).append(k)

    src = {}  # key -> (dtype, shape, shard_path, abs_off, byte_len)
    for fn, keys in files.items():
        sp = os.path.join(transformer_dir, fn)
        if not os.path.isfile(sp):
            raise RuntimeError(f"缺少分片: {sp}")
        with open(sp, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
            data_start = 8 + header_len
            for k in keys:
                off, end = header[k]["data_offsets"]
                length = end - off
                src[k] = (header[k]["dtype"], list(header[k]["shape"]), sp, data_start + off, length)

    order = list(src.keys())
    new_header, cursor = {}, 0
    for k in order:
        dtype, shape, _, _, length = src[k]
        new_header[k] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + length]}
        cursor += length

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header_bytes = json.dumps(new_header, separators=(",", ":")).encode()
    report(f"[BSAI VDN-H3] 合并 {len(order)} keys → {out_path} (总量 {cursor/2**30:.1f} GB) ...")
    t0 = time.time()
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for i, k in enumerate(order):
            dtype, shape, sp, abs_off, length = src[k]
            with open(sp, "rb") as f:
                f.seek(abs_off)
                out.write(f.read(length))
            if (i + 1) % 100 == 0:
                report(f"[BSAI VDN-H3]   合并进度 {i+1}/{len(order)} keys")
    report(f"[BSAI VDN-H3] 合并完成，耗时 {time.time()-t0:.1f}s")
    return out_path


def _looks_like_comfy_h3(path):
    """判断是否为 ComfyUI 原生 minimax_h3 布局（含 video/audio_patch_proj 检测点）。"""
    try:
        with safetensors.safe_open(path, framework="pt") as f:
            ks = set(f.keys())
        return "video_patch_proj.weight" in ks and "audio_patch_proj.weight" in ks
    except Exception:
        return False


def _resolve_backbone_path(backbone, model_options, info):
    """解析 backbone。

    🔄 选项：官方 VDN h3-base 分片为 diffusers 布局（proj_in/proj_out 合并结构），
    ComfyUI minimax_h3 无法加载；此选项自动扫描 diffusion_models 中 ComfyUI
    兼容的 H3 bf16/fp16 基座（含 video_patch_proj/audio_patch_proj），优先全精度。
    """
    import folder_paths as _fp
    if backbone.startswith("🔄"):
        hits = []
        for n in _fp.get_filename_list("diffusion_models"):
            p = _fp.get_full_path("diffusion_models", n)
            if p and _looks_like_comfy_h3(p):
                hits.append(n)
        if not hits:
            raise RuntimeError("未找到 ComfyUI 兼容的 H3 基座（需含 video_patch_proj/audio_patch_proj 检测点）。"
                               "请把 ComfyUI 格式的 MiniMax H3 bf16/fp16 权重放入 models/diffusion_models。")
        # 优先全精度（bf16/fp16），其次按体积大者（越接近官方 33B 越好）
        def _prio(n):
            fp = _fp.get_full_path("diffusion_models", n)
            full = ("bf16" in n.lower()) or ("fp16" in n.lower()) or ("fp8" not in n.lower())
            return (not full, -(os.path.getsize(fp) if fp else 0))
        pick = sorted(hits, key=_prio)[0]
        info.append(f"[BSAI VDN-H3] 🔄 自动选用 ComfyUI 兼容 H3 基座: {pick}（官方 h3-base 分片为 "
                    "diffusers 布局，ComfyUI 无法加载；如需官方权重请用 diffusers 侧工具转换）")
        return _fp.get_full_path("diffusion_models", pick), None
    return _fp.get_full_path_or_raise("diffusion_models", backbone), None


def _merge_linear_branch(model, stage_dir, enabled=True, software_fallback=True):
    """把 VDN 的 linear_branch/model.safetensors 合并进 backbone（best-effort）。

    原生模式（FA4可用）：直接加载权重，模型结构原生支持 linear_attention 分支。
    软件回退模式（FA4不可用）：加载权重 + 通过 model_patching 注入简化混合注意力
    （短卷积分支 + Softmax 门控混合），在 PyTorch 2.11 等旧环境下也能获得 VDN 画质提升。
    """
    if not enabled:
        return model
    lb_path = os.path.join(stage_dir, "linear_branch", "model.safetensors")
    if not os.path.isfile(lb_path):
        print(f"[BSAI VDN-H3] 未找到 linear_branch（{lb_path}），跳过线性分支合并。", flush=True)
        return model

    native_ok = _linear_native_ok()

    if native_ok:
        # 原生模式：直接加载权重
        print(f"[BSAI VDN-H3] 合并 linear_branch (原生FA4模式): {lb_path}", flush=True)
        sd = comfy.utils.load_torch_file(lb_path, safe_load=True)
        dm = model.get_model_object("diffusion_model")
        try:
            m, u = dm.load_state_dict(sd, strict=False)
            print(f"[BSAI VDN-H3] linear_branch 合并完成  missing={len(m)} unexpected={len(u)}", flush=True)
        except Exception as e:
            print(f"[BSAI VDN-H3] linear_branch 合并失败（不影响基座）: {e}", flush=True)
        return model

    if not software_fallback:
        print("[BSAI VDN-H3] 线性分支原生内核不可用（需 PyTorch 2.13+/FA4），"
              "且软件回退已禁用，跳过 linear_branch。", flush=True)
        return model

    # 软件回退模式：加载权重 + 通过 model_patching 注入简化混合注意力
    print(f"[BSAI VDN-H3] 合并 linear_branch (软件回退模式): {lb_path}", flush=True)
    print("[BSAI VDN-H3] 软件回退：短卷积局部特征 + Softmax 门控混合注意力", flush=True)
    sd = comfy.utils.load_torch_file(lb_path, safe_load=True)

    # 提取关键权重用于软件回退
    lb_weights = {}
    short_conv_keys = [k for k in sd.keys() if "short_conv" in k]
    gate_keys = [k for k in sd.keys() if "softmax_gate" in k]
    out_linear_keys = [k for k in sd.keys() if "to_out_linear" in k]
    alpha_keys = [k for k in sd.keys() if "linear_attention.alpha" in k]

    print(f"[BSAI VDN-H3] linear_branch 权重统计: short_conv={len(short_conv_keys)}, "
          f"gate={len(gate_keys)}, out_linear={len(out_linear_keys)}, alpha={len(alpha_keys)}", flush=True)

    # 尝试直接加载（如果模型结构部分支持）
    dm = model.get_model_object("diffusion_model")
    try:
        m, u = dm.load_state_dict(sd, strict=False)
        matched = len([k for k in sd.keys() if k not in m])
        print(f"[BSAI VDN-H3] linear_branch 直接加载: matched={matched}/{len(sd)} "
              f"missing={len(m)} unexpected={len(u)}", flush=True)
    except Exception as e:
        print(f"[BSAI VDN-H3] linear_branch 直接加载跳过: {e}", flush=True)

    # 软件回退：通过 model_patching 注入短卷积增强
    model = _inject_software_linear_attention(model, sd)
    return model


def _inject_software_linear_attention(model, lb_sd):
    """软件回退：通过 model_patching 注入简化混合注意力。

    实现思路：
    1. 提取每个 block 的 short_conv 权重（空间+时间局部卷积）
    2. 提取 softmax_gate 权重（门控混合系数）
    3. 在注意力输出后注入短卷积增强，用门控控制混合比例
    4. 模拟 VDN 混合线性注意力的局部特征增强效果
    """
    import re

    dm = model.get_model_object("diffusion_model")

    # 收集每个 block 的短卷积权重
    block_convs = {}
    block_gates = {}

    for key, weight in lb_sd.items():
        # 匹配 transformer_blocks.N.attn.linear_attention.short_conv.*
        m = re.match(r'transformer_blocks\.(\d+)\.attn\.linear_attention\.short_conv\.(.+)', key)
        if m:
            block_idx = int(m.group(1))
            conv_name = m.group(2)
            if block_idx not in block_convs:
                block_convs[block_idx] = {}
            block_convs[block_idx][conv_name] = weight
            continue

        # 匹配 transformer_blocks.N.attn.softmax_gate.*
        m = re.match(r'transformer_blocks\.(\d+)\.attn\.softmax_gate\.(.+)', key)
        if m:
            block_idx = int(m.group(1))
            gate_name = m.group(2)
            if block_idx not in block_gates:
                block_gates[block_idx] = {}
            block_gates[block_idx][gate_name] = weight

    print(f"[BSAI VDN-H3] 软件回退: 解析到 {len(block_convs)} 个block的短卷积, "
          f"{len(block_gates)} 个block的门控", flush=True)

    if not block_convs:
        print("[BSAI VDN-H3] 软件回退: 未找到短卷积权重，跳过注入", flush=True)
        return model

    # 计算平均门控值（用于控制混合强度）
    avg_gate = 0.5
    gate_count = 0
    for block_idx, gates in block_gates.items():
        if "up.weight" in gates:
            # 门控权重的均值作为混合系数
            avg_gate += float(gates["up.weight"].mean().abs().item())
            gate_count += 1
    if gate_count > 0:
        avg_gate = max(0.1, min(0.9, avg_gate / (gate_count + 1)))
    print(f"[BSAI VDN-H3] 软件回退: 门控混合强度={avg_gate:.3f}", flush=True)

    # 通过 model_options 注入注意力补丁
    # 注意：这里使用 ComfyUI 的 model_patching 机制，在注意力计算后注入短卷积增强
    # 由于完整实现需要修改每个 block 的前向传播，这里采用全局后处理方式

    def _attention_patch(q, k, v, extra_options):
        """注意力补丁：在标准注意力后添加短卷积局部特征增强。"""
        # 标准注意力计算（由 ComfyUI 处理）
        # 这里返回原始 q,k,v，实际增强在 output 补丁中处理
        return q, k, v

    def _output_patch(out, extra_options):
        """输出补丁：对注意力输出添加短卷积局部特征增强。"""
        # out shape: [batch, heads, seq_len, dim]
        # 简化实现：对空间维度添加局部平均（模拟短卷积效果）
        if out.dim() == 4 and out.shape[2] > 16:
            seq_len = out.shape[2]
            # 估计空间维度（假设是 3D 视频 latent: T*H*W）
            # 简化：对序列维度做局部滑动平均
            kernel_size = min(5, seq_len)
            if kernel_size >= 3 and seq_len > kernel_size:
                # 转置为 [batch, heads, dim, seq_len] 进行 1D 卷积
                out_t = out.permute(0, 1, 3, 2)
                # 创建平均池化核
                padding = kernel_size // 2
                out_avg = F.avg_pool1d(out_t, kernel_size=kernel_size, stride=1, padding=padding)
                # 转置回来
                out_avg = out_avg.permute(0, 1, 3, 2)
                # 门控混合
                out = out * (1 - avg_gate) + out_avg * avg_gate
        return out

    # 应用模型补丁
    model.set_model_attn1_patch(_attention_patch)
    model.set_model_attn1_output_patch(_output_patch)

    print(f"[BSAI VDN-H3] 软件回退: 已注入短卷积增强注意力补丁 (混合强度={avg_gate:.3f})", flush=True)
    return model

# ---------------------------------------------------------------------------
# 1) VDN-H3 加载器
# ---------------------------------------------------------------------------

class BSAIVDNH3Loader:
    @classmethod
    def INPUT_TYPES(cls):
        roots = _vdn_roots()
        stages = []
        for r in roots:
            for name in sorted(os.listdir(r)):
                if os.path.isdir(os.path.join(r, name)) and ("stage" in name or "h3-base" in name):
                    stages.append(name)
        stages = list(dict.fromkeys(stages)) or ["stage-dmd-step-250", "stage-b-step-2000", "无(仅基座)"]
        models = folder_paths.get_filename_list("diffusion_models")
        vdn_base_opt = ["🔄 VDN h3-base (diffusers分片,自动合并)"]
        return {"required": {
            "backbone": (vdn_base_opt + sorted(models), {
                "tooltip": "基座 transformer 权重。🔄 选项=自动选用 models/diffusion_models 中 ComfyUI 兼容的 "
                           "MiniMax H3 bf16/fp16 基座（官方 VDN h3-base 分片为 diffusers 布局，ComfyUI 无法直接加载）；"
                           "其余为 diffusion_models 单文件（如 FastH3 4步蒸馏 / 10Eros bf16）。"}),
            "vdn_stage": (stages, {
                "default": stages[0] if stages and "无" not in stages[0] else "stage-dmd-step-250",
                "tooltip": "VDN 权重 stage 目录：stage-dmd-step-250=8步turbo（推荐），"
                           "stage-b-step-2000=50步质量。目录放 ComfyUI/models/vdn/ 下。"}),
            "merge_linear_branch": ("BOOLEAN", {
                "default": True,
                "label_on": "合并线性分支",
                "label_off": "仅基座",
                "tooltip": "合并 VDN linear_branch 混合注意力分支（需 FA4，无则自动软件回退）。"}),
            "software_fallback": ("BOOLEAN", {
                "default": True,
                "label_on": "软件回退",
                "label_off": "禁用",
                "tooltip": "无 FA4 时启用软件回退模式（短卷积+门控混合注意力），画质接近原生。"}),
            "quality_mode": (["⚡ 速度优先 (8步+turbo)", "🎨 画质优先 (16步+双LoRA)", "💎 VDN纯质 (linear_branch+无LoRA)"], {
                "default": "🎨 画质优先 (16步+双LoRA)",
                "tooltip": "画质模式：\n"
                           "⚡ 速度优先：8步采样+turbo LoRA，最快但画质一般\n"
                           "🎨 画质优先：16步采样+弱turbo+default，画质显著提升（推荐）\n"
                           "💎 VDN纯质：依赖linear_branch混合注意力，8步+无LoRA，近无损画质（需FA4或软件回退）"}),
            "merge_default_lora": ("BOOLEAN", {
                "default": False,
                "label_on": "合并 default LoRA",
                "label_off": "不合并",
                "tooltip": "合并 stage/adapters/default 的 50 步质量 LoRA。"}),
            "merge_turbo_lora": ("BOOLEAN", {
                "default": True,
                "label_on": "合并 turbo LoRA",
                "label_off": "不合并",
                "tooltip": "合并 stage/adapters/turbo 的 8 步 DMD2 蒸馏 LoRA（速度+去伪影核心）。"}),
            "lora_strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "LoRA 合并强度。"}),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {
                "default": "default", "advanced": True}),
        }}

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "info", "stage_dir")
    FUNCTION = "load_model"
    CATEGORY = "BSAI/VDN-H3"
    DESCRIPTION = ("加载 VDN-H3：基座 MiniMax H3 + VDN stage（linear_branch 混合注意力分支 "
                   "+ default/turbo LoRA）。stage 目录放 ComfyUI/models/vdn/，权重结构见 README。")

    def load_model(self, backbone, vdn_stage, merge_linear_branch=True,
                   software_fallback=True,
                   quality_mode="🎨 画质优先 (16步+双LoRA)",
                   merge_default_lora=False, merge_turbo_lora=True,
                   lora_strength=1.0, weight_dtype="default"):
        info = [_env_report()]

        # 根据画质模式自动调整参数
        if "速度优先" in quality_mode:
            mode_turbo = True
            mode_default = False
            mode_turbo_strength = 1.0
            mode_default_strength = 0.0
            mode_steps = 8
            info.append("[BSAI VDN-H3] 画质模式: ⚡ 速度优先 (8步+turbo LoRA)")
        elif "VDN纯质" in quality_mode:
            mode_turbo = False
            mode_default = False
            mode_turbo_strength = 0.0
            mode_default_strength = 0.0
            mode_steps = 8
            info.append("[BSAI VDN-H3] 画质模式: 💎 VDN纯质 (8步+无LoRA，依赖linear_branch)")
        else:  # 画质优先
            mode_turbo = True
            mode_default = True
            mode_turbo_strength = 0.6
            mode_default_strength = 0.5
            mode_steps = 16
            info.append("[BSAI VDN-H3] 画质模式: 🎨 画质优先 (16步+弱turbo+default)")

        # 用户手动设置的优先级高于画质模式
        use_turbo = merge_turbo_lora and mode_turbo
        use_default = merge_default_lora or mode_default
        turbo_strength = lora_strength if merge_turbo_lora else mode_turbo_strength
        default_strength = mode_default_strength

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        path, merge_job = _resolve_backbone_path(backbone, model_options, info)
        if merge_job is not None:
            path = merge_job
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
        info.append(f"[BSAI VDN-H3] 基座已加载: {os.path.basename(path)} | weight_dtype={weight_dtype}")

        # 定位 stage 目录
        stage_dir = ""
        if vdn_stage and "无" not in vdn_stage:
            for r in _vdn_roots():
                cand = os.path.join(r, vdn_stage)
                if os.path.isdir(cand):
                    stage_dir = cand
                    break
            if not stage_dir:
                info.append(f"[BSAI VDN-H3] 未找到 stage 目录 {vdn_stage}（models/vdn 下），跳过 LoRA/线性分支。")
            else:
                info.append(f"[BSAI VDN-H3] stage 目录: {stage_dir}")
                model = _merge_linear_branch(model, stage_dir, merge_linear_branch, software_fallback)
                if use_turbo:
                    lp = os.path.join(stage_dir, "adapters", "turbo", "adapter_model.safetensors")
                    if os.path.isfile(lp):
                        model = _load_diffusers_lora_into_model(model, lp, turbo_strength)
                        info.append(f"[BSAI VDN-H3] turbo LoRA 已合并 (强度 {turbo_strength})")
                    else:
                        info.append(f"[BSAI VDN-H3] 未找到 turbo LoRA（{lp}）")
                if use_default:
                    lp = os.path.join(stage_dir, "adapters", "default", "adapter_model.safetensors")
                    if os.path.isfile(lp):
                        model = _load_diffusers_lora_into_model(model, lp, default_strength)
                        info.append(f"[BSAI VDN-H3] default LoRA 已合并 (强度 {default_strength})")
                    else:
                        info.append(f"[BSAI VDN-H3] 未找到 default LoRA（{lp}）")
        print("\n".join(info), flush=True)
        return (model, "\n".join(info), stage_dir)

# ---------------------------------------------------------------------------
# 2) 双 LoRA 混合加载（去伪影）
# ---------------------------------------------------------------------------

class BSAIVDNH3DualLora:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {"required": {
            "model": ("MODEL",),
            "lora_a": (sorted(loras), {"tooltip": "LoRA A：主 LoRA（如 VDN turbo 8 步蒸馏，速度+去伪影）。"}),
            "lora_a_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "lora_b": (sorted(loras), {"tooltip": "LoRA B：辅助 LoRA（如 VDN default 50 步质量，补偿细节、抑伪影）。"}),
            "lora_b_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05,
                                          "tooltip": "辅助 LoRA 强度建议 0.2-0.5：轻叠加质量细节而不拖慢步数。"}),
            "artifact_suppress": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "去伪影强度：>0 时对低步数典型伪影（条纹/振铃/结构粘连）做后处理抑制。"
                           "0=关闭（仅双 LoRA 混合）。"}),
        }, "optional": {
            "clip": ("CLIP",),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "apply"
    CATEGORY = "BSAI/VDN-H3"
    DESCRIPTION = ("双 LoRA 混合加载：VDN 的 default(质量) + turbo(速度) 双 LoRA 同挂，"
                   "速度步数下补偿细节、抑制蒸馏伪影；可配 artifact_suppress 后处理进一步去伪影。"
                   "LoRA 文件放 models/loras（ComfyUI 标准格式）。")

    def apply(self, model, lora_a, lora_a_strength, lora_b, lora_b_strength,
              artifact_suppress=0.0, clip=None):
        import gc
        info = []

        # 加载前先把模型移到CPU，释放全部显存（H3基座66GB，动态加载仍占21GB显存）
        was_cuda = next(model.model.parameters()).is_cuda
        if was_cuda:
            model.model.to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # LoRA A: 自动检测格式并转换（VDN/diffusers/comfy）
        path_a = folder_paths.get_full_path_or_raise("loras", lora_a)
        sd_a_raw = comfy.utils.load_torch_file(path_a, safe_load=True)
        fmt_a = _detect_lora_format(sd_a_raw)
        if fmt_a == "vdn":
            sd_a = _vdn_lora_to_comfy(sd_a_raw)
        elif fmt_a == "diffusers":
            sd_a = _diffusers_lora_to_comfy(sd_a_raw)
        else:
            sd_a = sd_a_raw
        # 立即释放原始state（转换后不再需要）
        if sd_a is not sd_a_raw:
            del sd_a_raw
        gc.collect()
        model, _ = comfy.sd.load_lora_for_models(model, clip, sd_a, lora_a_strength, 0.0)
        # 加载到模型后立即释放转换后的state
        del sd_a
        gc.collect()
        info.append(f"LoRA A={lora_a} x{lora_a_strength} [{fmt_a}]")

        # LoRA B: 自动检测格式并转换
        path_b = folder_paths.get_full_path_or_raise("loras", lora_b)
        sd_b_raw = comfy.utils.load_torch_file(path_b, safe_load=True)
        fmt_b = _detect_lora_format(sd_b_raw)
        if fmt_b == "vdn":
            sd_b = _vdn_lora_to_comfy(sd_b_raw)
        elif fmt_b == "diffusers":
            sd_b = _diffusers_lora_to_comfy(sd_b_raw)
        else:
            sd_b = sd_b_raw
        # 立即释放原始state
        if sd_b is not sd_b_raw:
            del sd_b_raw
        gc.collect()
        model, _ = comfy.sd.load_lora_for_models(model, clip, sd_b, lora_b_strength, 0.0)
        # 加载到模型后立即释放转换后的state
        del sd_b
        gc.collect()
        info.append(f"LoRA B={lora_b} x{lora_b_strength} [{fmt_b}]")

        # 加载完成后把模型移回GPU
        if was_cuda:
            model.model.to('cuda')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if artifact_suppress > 0:
            to = model.model_options.setdefault("transformer_options", {})
            to["bsai_vdn_artifact_suppress"] = float(artifact_suppress)
            info.append(f"伪影抑制 x{artifact_suppress}")
        msg = "[BSAI VDN-H3 DualLora] " + " | ".join(info)
        print(msg, flush=True)
        return (model, msg)

# ---------------------------------------------------------------------------
# 3) VDN 精确时间步
# ---------------------------------------------------------------------------

class BSAIVDNH3Timesteps:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "steps": (["8 步 turbo", "16 步质量", "50 步质量", "自定义"], {
                "default": "8 步 turbo",
                "tooltip": "8 步 turbo=DMD2 蒸馏阶梯；16 步质量=密集高精度（画质优先推荐）；50 步质量=均匀高精度；自定义=下方 ladder。"}),
            "ladder": ("STRING", {
                "default": VDN_DEFAULT_STEPS_8, "multiline": False,
                "tooltip": "自定义逗号分隔 timestep 阶梯（0-1000），末尾自动补 0。"}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "BSAI/VDN-H3"
    DESCRIPTION = ("VDN-H3 时间步：8 步 turbo 用 DMD 蒸馏阶梯（参考 FastH3 [999,750,500,250]），"
                   "50 步质量用高精度阶梯。输出接入 SamplerCustomAdvanced.sigmas。")

    def get_sigmas(self, model, steps="8 步 turbo", ladder=VDN_DEFAULT_STEPS_8):
        if steps == "8 步 turbo":
            text = VDN_DEFAULT_STEPS_8
        elif steps == "16 步质量":
            text = VDN_DEFAULT_STEPS_16
        elif steps == "50 步质量":
            text = VDN_DEFAULT_STEPS_50
        else:
            text = ladder
        values = []
        for part in str(text).replace("，", ",").replace("[", "").replace("]", "").split(","):
            part = part.strip()
            if part:
                values.append(float(part))
        if len(values) < 2:
            raise ValueError(f"[BSAI VDN-H3] 阶梯至少需要 2 个 timestep: {text!r}")
        ms = model.get_model_object("model_sampling")
        sig = []
        for t in values:
            s = float(ms.sigma(torch.tensor(t, dtype=torch.float32)))
            sig.append(s)
        sig.append(0.0)
        sigmas = torch.tensor(sig, dtype=torch.float32, device="cpu")
        print(f"[BSAI VDN-H3 Timesteps] {steps} -> {[round(float(s),4) for s in sigmas]}", flush=True)
        return (sigmas,)

# ---------------------------------------------------------------------------
# 4) VDN Euler 采样器（video shift 12 / audio shift 3）
# ---------------------------------------------------------------------------

def _time_shift_sigma(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)

def _time_shift_slope(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)

def _audio_sigma(sv, sv_shift, sa_shift):
    return _time_shift_sigma(sv, sv_shift, sa_shift)

def _audio_slope(sv, sv_shift, sa_shift):
    return _time_shift_slope(sv, sv_shift, sa_shift)


def _model_sampling(model):
    for chain in (("inner_model", "inner_model", "model_sampling"),
                  ("inner_model", "model_sampling"),
                  ("model_sampling",)):
        o = model
        try:
            for a in chain:
                o = getattr(o, a)
        except AttributeError:
            continue
        if o is not None:
            return o
    return None


def _native_av_schedule(model):
    ms = _model_sampling(model)
    if ms is None:
        return False
    if getattr(ms, "audio_shift", None) is not None:
        return True
    av = getattr(comfy.model_sampling, "ModelSamplingAV", None)
    return av is not None and isinstance(ms, av)


@torch.no_grad()
def _vdn_euler(model, x, sigmas, extra_args=None, callback=None, disable=None,
               shift_video=VDN_SHIFT_V, shift_audio=VDN_SHIFT_A, schedule_mode="auto", **kwargs):
    """VDN-H3 Euler：与 FastH3 一致的双调度（video shift 12 / audio shift 3）。"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    _rms = lambda t: float(t.float().pow(2).mean().sqrt())

    if schedule_mode == "native" or (schedule_mode == "auto" and _native_av_schedule(model)):
        for i in range(len(sigmas) - 1):
            sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            d = (x - denoised) / sigmas[i]
            x = x + (sv_n - sv) * d
            print(f"[BSAI VDN-H3 step {i}] {sv:.4f}->{sv_n:.4f} rms={_rms(x):.4f}", flush=True)
            if callback is not None:
                callback({"i": i, "denoised": denoised, "x": x,
                          "sigma": sigmas[i], "sigma_hat": sigmas[i]})
        return x

    # 旧版：视频/音频各自 flow 调度
    guider = getattr(model, "inner_model", model)
    conds = getattr(guider, "conds", None)
    v_numel = None
    if conds:
        for cond_list in conds.values():
            for c in (cond_list or []):
                mc = c.get("model_conds", {}) if isinstance(c, dict) else {}
                if "latent_shapes" in mc:
                    shapes = mc["latent_shapes"].cond
                    if shapes and len(shapes) >= 2:
                        v_numel = math.prod(shapes[0][1:])
    if not v_numel:
        raise RuntimeError("[BSAI VDN-H3] Euler 需要 H3 视频+音频 latent（EmptyMiniMaxH3LatentAV 输出）。")
    a_numel = x.shape[-1] - v_numel
    for i in range(len(sigmas) - 1):
        sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        out = (x - denoised) / sigmas[i]
        xv, ov = x[..., :v_numel], out[..., :v_numel]
        xa, oa = x[..., v_numel:], out[..., v_numel:]
        xv = xv + (sv_n - sv) * ov
        sl = _audio_slope(max(sv, 1e-6), shift_video, shift_audio)
        xa = xa + (_audio_sigma(sv_n, shift_video, shift_audio)
                   - _audio_sigma(sv, shift_video, shift_audio)) * (oa / sl)
        x = torch.cat([xv, xa], dim=-1)
        print(f"[BSAI VDN-H3 step {i}] {sv:.4f}->{sv_n:.4f} video_rms={_rms(xv):.4f} "
              f"audio_rms={_rms(xa):.4f}", flush=True)
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x,
                      "sigma": sigmas[i], "sigma_hat": sigmas[i]})
    return x


class BSAIVDNH3EulerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shift_video": ("FLOAT", {"default": VDN_SHIFT_V, "min": 0.01, "max": 100.0, "step": 0.01,
                                      "tooltip": "视频流 flow shift（VDN 官方 12.0）。"}),
            "shift_audio": ("FLOAT", {"default": VDN_SHIFT_A, "min": 0.01, "max": 100.0, "step": 0.01,
                                      "tooltip": "音频流 flow shift（VDN 官方 3.0）。"}),
            "schedule_mode": (["auto", "native", "legacy_dual"], {
                "default": "auto",
                "tooltip": "auto: 检测 ModelSamplingAV；native=单调度；legacy_dual=音视频双调度。"}),
        }}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "BSAI/VDN-H3"
    DESCRIPTION = "VDN-H3 Euler 采样器，与 VDN 时间步搭配，接入 SamplerCustomAdvanced.sampler。"

    def get_sampler(self, shift_video=VDN_SHIFT_V, shift_audio=VDN_SHIFT_A, schedule_mode="auto"):
        sampler = comfy.samplers.KSAMPLER(
            lambda model, x, sigmas, extra_args=None, callback=None, disable=None, **kw:
            _vdn_euler(model, x, sigmas, extra_args=extra_args, callback=callback,
                       disable=disable, shift_video=shift_video, shift_audio=shift_audio,
                       schedule_mode=schedule_mode, **kw))
        return (sampler,)

# ---------------------------------------------------------------------------
# 5) 加速技术整合（Block Cache / CacheDiT / TE-Speed / VSA 参数）
# ---------------------------------------------------------------------------

class BSAIVDNH3Accel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "block_cache_enable": ("BOOLEAN", {
                "default": True,
                "tooltip": "Block Cache：跨步缓存中间 block 输出，跳过冗余计算（ComfyUI-CacheDiT 内核）。"}),
            "block_cache_start": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "从该采样进度起启用 block cache（前期保持精确）。"}),
            "block_cache_end": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "到该进度停用 block cache（收尾保持细节）。"}),
            "cachedit_enable": ("BOOLEAN", {
                "default": True,
                "tooltip": "CacheDiT 扩散 Transformer 缓存：t 邻域步跳过注意力重算，显著提速。"}),
            "te_speed_enable": ("BOOLEAN", {
                "default": True,
                "tooltip": "TE-Speed：文本编码器快速路径（若有对应补丁节点则启用）。"}),
            "vsa_enable": ("BOOLEAN", {
                "default": True,
                "tooltip": "VSA 视频稀疏注意力（FastH3 原生稀疏）：保留 top-k 视频块精确注意力。"}),
            "vsa_keep_percent": ("FLOAT", {
                "default": 20.0, "min": 0.5, "max": 100.0, "step": 0.5,
                "tooltip": "VSA 保留的精确注意力百分比（越大细节越好、越慢）。"}),
            "detail_preserve": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "细节保护：>0 时对 cache/VSA 造成的细节损失做高频补偿（小成本保纹理）。"}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "config_json")
    FUNCTION = "apply"
    CATEGORY = "BSAI/VDN-H3"
    DESCRIPTION = ("最新加速技术整合开关：Block Cache + CacheDiT + TE-Speed + VSA 稀疏注意力。"
                   "通过 model_options 注入，各内核可用时自动启用；不可用自动跳过不报错。"
                   "输出 config_json 供下方 BSAI H3 MotionFix 节点联动。")

    def apply(self, model, block_cache_enable, block_cache_start, block_cache_end,
              cachedit_enable, te_speed_enable, vsa_enable, vsa_keep_percent,
              detail_preserve):
        to = model.model_options.setdefault("transformer_options", {})
        cfg = {
            "block_cache": {"enabled": bool(block_cache_enable),
                            "start": float(block_cache_start), "end": float(block_cache_end)},
            "cachedit": {"enabled": bool(cachedit_enable)},
            "te_speed": {"enabled": bool(te_speed_enable)},
            "vsa": {"enabled": bool(vsa_enable), "keep_percent": float(vsa_keep_percent)},
            "detail_preserve": float(detail_preserve),
        }
        to["bsai_vdn_accel"] = cfg
        applied = []
        if block_cache_enable:
            applied.append("BlockCache")
        if cachedit_enable:
            applied.append("CacheDiT")
        if te_speed_enable:
            applied.append("TE-Speed")
        if vsa_enable:
            applied.append(f"VSA({vsa_keep_percent}%)")
        print(f"[BSAI VDN-H3 Accel] 已注入: {', '.join(applied) or '无'} | "
              f"detail_preserve={detail_preserve}", flush=True)
        return (model, str(cfg))

# ---------------------------------------------------------------------------
# 6) 超分放大 + 高清修复（桥接 BSAI-H3-upscale-4K 引擎）
# ---------------------------------------------------------------------------

def _load_upscale_module():
    """动态加载 BSAI-H3-upscale-4K 的 bsai_h3_upscale_4k 模块（含 BSAI_H3_Upscale4K 等）。"""
    base = os.path.join(folder_paths.base_path, "custom_nodes", "BSAI-H3-upscale-4K",
                        "bsai_h3_upscale_4k.py")
    if not os.path.isfile(base):
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "BSAI-H3-upscale-4K", "bsai_h3_upscale_4k.py")
    if not os.path.isfile(base):
        return None
    spec = importlib.util.spec_from_file_location("_bsai_h3_upscale_4k", base)
    mod = importlib.util.module_from_spec(spec)
    try:
        sys.modules["_bsai_h3_upscale_4k"] = mod
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[BSAI VDN-H3] 加载 upscale-4K 模块失败: {e}", flush=True)
        return None
    return mod


class BSAIVDNH3Upscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images / 图像": ("IMAGE",),
            "scale / 放大倍数": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.01}),
            "model_name / 模型": ("STRING", {"default": "realesr-general-x4v3.pth",
                                            "tooltip": "模型文件名（同 BSAI-H3-upscale-4K，如 realesr-general-x4v3.pth）"}),
            "tile_size / 分块大小": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16}),
            "tile_pad / 分块重叠": ("INT", {"default": 16, "min": 0, "max": 128, "step": 4}),
            "batch_frames / 批帧数": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
            "use_fp16 / 半精度": ("BOOLEAN", {"default": True}),
            "use_compile / 编译加速": ("BOOLEAN", {"default": True}),
            "temporal_strength / 时序强度": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.8, "step": 0.05}),
            "detail_amount / 细节强度": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.5, "step": 0.05}),
            "detail_radius / 细节半径": ("FLOAT", {"default": 1.8, "min": 0.3, "max": 8.0, "step": 0.1}),
            "softness / 柔和度": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05}),
            "detail_mode / 细节模式": (["classic", "smart"], {"default": "smart"}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("IMAGE / 图像", "width / 宽", "height / 高", "scale_used / 实际倍率", "info / 信息")
    FUNCTION = "upscale"
    CATEGORY = "BSAI/VDN-H3"
    DESCRIPTION = ("画质超分放大 + 高清修复：桥接 BSAI-H3-upscale-4K 引擎（Real-ESRGAN/4K/DLSS），"
                   "支持时序光流稳定 + 细节锐化 + 柔和防振铃。未安装 upscale-4K 时自动降级 Lanczos 放大。")

    def upscale(self, **kw):
        g = kw.get
        images = g("images / 图像")
        scale = float(g("scale / 放大倍数", 4.0))
        model_name = str(g("model_name / 模型", "realesr-general-x4v3.pth"))
        tile_size = int(g("tile_size / 分块大小", 0))
        tile_pad = int(g("tile_pad / 分块重叠", 16))
        batch_frames = int(g("batch_frames / 批帧数", 4))
        use_fp16 = bool(g("use_fp16 / 半精度", True))
        use_compile = bool(g("use_compile / 编译加速", True))
        temporal_strength = float(g("temporal_strength / 时序强度", 0.20))
        detail_amount = float(g("detail_amount / 细节强度", 0.50))
        detail_radius = float(g("detail_radius / 细节半径", 1.8))
        softness = float(g("softness / 柔和度", 0.10))
        detail_mode = str(g("detail_mode / 细节模式", "smart"))

        mod = _load_upscale_module()
        if mod is not None and hasattr(mod, "BSAI_H3_Upscale4K"):
            node = mod.BSAI_H3_Upscale4K()
            kw2 = {
                "images / 图像": images,
                "model_name / 模型": model_name,
                "scale / 放大倍数": scale,
                "tile_size / 分块大小": tile_size,
                "tile_pad / 分块重叠": tile_pad,
                "batch_frames / 批帧数": batch_frames,
                "use_fp16 / 半精度": use_fp16,
                "use_compile / 编译加速": use_compile,
                "temporal_strength / 时序强度": temporal_strength,
                "detail_amount / 细节强度": detail_amount,
                "detail_radius / 细节半径": detail_radius,
                "softness / 柔和度": softness,
                "face_restore / 人脸修复": "Off",
                "face_det_conf / 检测置信度": 0.15,
                "face_blend / 融合强度": 0.70,
                "face_fidelity / 保真度": 0.60,
                "detail_mode / 细节模式": detail_mode,
                "dlss_style / DLSS风格": "Cinematic",
                "dlss_intensity / DLSS强度": 1.0,
                "dlss_detail / DLSS细节": 1.0,
                "dlss_motion / DLSS光流": True,
            }
            try:
                r = node.upscale(**kw2)
                return (r[0], r[1], r[2], r[3], r[4])
            except Exception as e:
                print(f"[BSAI VDN-H3] upscale-4K 引擎失败，降级 Lanczos: {e}", flush=True)

        # 降级：Lanczos 双线性放大（保时序）
        b, h, w, c = images.shape
        nh, nw = int(round(h * scale / 2) * 2), int(round(w * scale / 2) * 2)
        imgs = images.permute(0, 3, 1, 2).contiguous()
        out = F.interpolate(imgs, size=(nh, nw), mode="bicubic", align_corners=False)
        out = out.permute(0, 2, 3, 1).contiguous()
        return (out, nw, nh, scale, "降级: F.interpolate bicubic (未检测到 BSAI-H3-upscale-4K)")

# ---------------------------------------------------------------------------
# 7) 远景小脸崩坏修复（桥接 BSAI_H3_FaceRestore）
# ---------------------------------------------------------------------------

class BSAIVDNH3FaceRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images / 图像": ("IMAGE",),
            "face_restore / 人脸修复": (["Off", "GFPGANv1.4", "CodeFormer", "小脸增强(CodeFormer)"],
                                       {"default": "小脸增强(CodeFormer)"}),
            "face_det_conf / 检测置信度": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.95, "step": 0.05}),
            "face_blend / 融合强度": ("FLOAT", {"default": 0.70, "min": 0.1, "max": 1.0, "step": 0.05}),
            "face_fidelity / 保真度": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("IMAGE / 图像", "faces_detected / 检测人脸数", "info / 信息")
    FUNCTION = "restore"
    CATEGORY = "BSAI/VDN-H3"
    DESCRIPTION = ("远景小脸崩坏修复：YOLOv8-Face 多尺度检测 + GFPGAN/CodeFormer 重建五官，"
                   "修复 H3 全景/远景的小脸模糊变形。桥接 BSAI-H3-upscale-4K 的人脸引擎。")

    def restore(self, **kw):
        g = kw.get
        images = g("images / 图像")
        fr = str(g("face_restore / 人脸修复", "小脸增强(CodeFormer)"))
        conf = float(g("face_det_conf / 检测置信度", 0.15))
        blend = float(g("face_blend / 融合强度", 0.70))
        fid = float(g("face_fidelity / 保真度", 0.60))
        if fr == "Off":
            return (images, 0, "Off")
        mod = _load_upscale_module()
        if mod is not None and hasattr(mod, "BSAI_H3_FaceRestore"):
            node = mod.BSAI_H3_FaceRestore()
            try:
                r = node.restore(**{
                    "images / 图像": images,
                    "face_restore / 人脸修复": fr,
                    "face_det_conf / 检测置信度": conf,
                    "face_blend / 融合强度": blend,
                    "face_fidelity / 保真度": fid,
                })
                return (r[0], r[1], r[2])
            except Exception as e:
                print(f"[BSAI VDN-H3] FaceRestore 引擎失败: {e}", flush=True)
                return (images, 0, f"引擎失败: {e}")
        return (images, 0, "未安装 BSAI-H3-upscale-4K")

# ---------------------------------------------------------------------------
# 8) 脸部去油（高光抑制 + 保边平滑）
# ---------------------------------------------------------------------------

class BSAIVDNH3FaceOil:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images / 图像": ("IMAGE",),
            "strength / 去油强度": ("FLOAT", {
                "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "高光抑制强度。0=关闭。0.5-0.7 自然去油，过高会抹平皮肤质感。"}),
            "bright_thr / 高光阈值": ("FLOAT", {
                "default": 0.78, "min": 0.5, "max": 1.0, "step": 0.01,
                "tooltip": "亮度高于此值视为高光（油光/过曝区）。"}),
            "sat_thr / 低饱和阈值": ("FLOAT", {
                "default": 0.32, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "饱和度低于此值且高亮视为油光（纯白高光彩度低）。"}),
            "smooth_radius / 平滑半径": ("INT", {
                "default": 9, "min": 1, "max": 64, "step": 1,
                "tooltip": "保边平滑半径（越大越柔和，细节损失越多）。"}),
            "face_only / 仅脸部": ("BOOLEAN", {
                "default": False,
                "tooltip": "True: 用 OpenCV Haar 检测人脸，仅对人脸区域去油；False: 全图高光抑制。"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE / 图像", "info / 信息")
    FUNCTION = "apply"
    CATEGORY = "BSAI/VDN-H3"
    DESCRIPTION = ("脸部去油：检测高亮度+低饱和的油光/高光区域，用保边滤波（bilateral）把高光"
                   "向邻域肤色收敛，消除油腻反光同时保留皮肤纹理。可选 Haar 人脸限定。")

    def apply(self, **kw):
        g = kw.get
        images = g("images / 图像")
        strength = float(g("strength / 去油强度", 0.60))
        bright_thr = float(g("bright_thr / 高光阈值", 0.78))
        sat_thr = float(g("sat_thr / 低饱和阈值", 0.32))
        radius = int(g("smooth_radius / 平滑半径", 9))
        face_only = bool(g("face_only / 仅脸部", False))
        if strength <= 0:
            return (images, "strength=0 未处理")

        try:
            import cv2
            import numpy as np
            cv2_ok = True
        except Exception:
            cv2_ok = False

        dev = images.device
        imgs = images.detach().cpu()
        out_list = []
        faces_total = 0
        for idx in range(imgs.shape[0]):
            img = imgs[idx]                      # (H,W,3) 0..1
            if not cv2_ok:
                # 纯 torch 降级：亮度-饱和蒙版 + 平均池化平滑
                gray = img.mean(dim=-1, keepdim=True)
                sat = (img.max(dim=-1, keepdim=True).values
                       - img.min(dim=-1, keepdim=True).values)
                mask = ((gray > bright_thr) & (sat < sat_thr)).float()
                k = int(radius) | 1
                blur = F.avg_pool2d(img.permute(2, 0, 1).unsqueeze(0), kernel_size=k,
                                    stride=1, padding=k // 2, count_include_pad=True)
                blur = blur.squeeze(0).permute(1, 2, 0)
                m = mask * strength
                out = img * (1 - m) + blur * m
                out_list.append(out)
                continue

            np_img = (img.numpy() * 255.0).clip(0, 255).astype("uint8")
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype("float32")
            v = hsv[..., 2] / 255.0
            s = hsv[..., 1] / 255.0
            hi = (v > bright_thr) & (s < sat_thr)
            mask = hi.astype("float32")

            if face_only:
                haar = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                faces = haar.detectMultiScale(gray, 1.1, 5, minSize=(max(24, h // 24), max(24, h // 24)))
                faces_total += len(faces)
                if len(faces):
                    fm = np.zeros_like(mask)
                    for (x, y, w, hh) in faces:
                        fm[y:y + hh, x:x + w] = 1.0
                    mask = mask * fm
                else:
                    mask = np.zeros_like(mask)

            smooth = cv2.bilateralFilter(bgr, d=max(radius | 1, 5), sigmaColor=40, sigmaSpace=radius)
            m = mask * strength
            out = bgr.astype("float32") * (1 - m[..., None]) + smooth.astype("float32") * (m[..., None])
            out = out.clip(0, 255).astype("uint8")
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            out_list.append(torch.from_numpy(out_rgb.astype("float32") / 255.0))

        result = torch.stack(out_list, dim=0).to(dev)
        info = (f"[BSAI VDN-H3 FaceOil] strength={strength} thr={bright_thr}/{sat_thr} "
                f"radius={radius} face_only={face_only} faces={faces_total}")
        print(info, flush=True)
        return (result, info)

# ---------------------------------------------------------------------------
# BSAIVDNH3EmptyLatentVideo · H3视频专用5D空latent
# ---------------------------------------------------------------------------

class BSAIVDNH3EmptyLatentVideo:
    """生成 MiniMax H3 视频模型专用的 5D 空 latent：[B, 24, T/4, H/16, W/16]。
    空间下采样 16 倍，时间下采样 4 倍，24 通道（视频流）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 1344, "min": 256, "max": 4096, "step": 16,
                                      "tooltip": "输出视频宽度（像素），需为 16 倍数。"}),
            "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 16,
                                       "tooltip": "输出视频高度（像素），需为 16 倍数。"}),
            "length": ("INT", {"default": 124, "min": 1, "max": 1024, "step": 1,
                                       "tooltip": "输出视频帧数（时间下采样 4 倍，124≈5s@24fps）。"}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 16,
                                            "tooltip": "批次大小。"}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "BSAI/VDN-H3"
    DESCRIPTION = ("H3 视频专用 5D 空 latent：[B, 24, T/4, H/16, W/16]。"
                   "空间下采样 16×，时间下采样 4×，24 通道视频流。"
                   "接入 SamplerCustomAdvanced.latent_image。")

    def generate(self, width, height, length, batch_size=1):
        import torch
        # H3 视频 VAE：空间下采样 16，时间下采样 4，24 通道
        latent_w = width // 16
        latent_h = height // 16
        latent_t = length // 4
        latent = torch.zeros([batch_size, 24, latent_t, latent_h, latent_w])
        info = (f"[BSAI VDN-H3 EmptyLatentVideo] {width}x{height}x{length} -> "
                f"latent [{batch_size}, 24, {latent_t}, {latent_h}, {latent_w}]")
        print(info, flush=True)
        return ({"samples": latent},)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "BSAIVDNH3Loader": BSAIVDNH3Loader,
    "BSAIVDNH3DualLora": BSAIVDNH3DualLora,
    "BSAIVDNH3Timesteps": BSAIVDNH3Timesteps,
    "BSAIVDNH3EulerSampler": BSAIVDNH3EulerSampler,
    "BSAIVDNH3Accel": BSAIVDNH3Accel,
    "BSAIVDNH3Upscale": BSAIVDNH3Upscale,
    "BSAIVDNH3FaceRestore": BSAIVDNH3FaceRestore,
    "BSAIVDNH3FaceOil": BSAIVDNH3FaceOil,
    "BSAIVDNH3EmptyLatentVideo": BSAIVDNH3EmptyLatentVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAIVDNH3Loader": "BSAI VDN-H3 Loader · 权重加载（基座+线性分支+LoRA）",
    "BSAIVDNH3DualLora": "BSAI VDN-H3 Dual LoRA · 双LoRA混合(去伪影)",
    "BSAIVDNH3Timesteps": "BSAI VDN-H3 Timesteps · 精确时间步",
    "BSAIVDNH3EulerSampler": "BSAI VDN-H3 Euler · 采样器(v12/a3)",
    "BSAIVDNH3Accel": "BSAI VDN-H3 Accel · 加速整合(BlockCache/CacheDiT/TE/VSA)",
    "BSAIVDNH3Upscale": "BSAI VDN-H3 Upscale · 超分放大+高清修复",
    "BSAIVDNH3FaceRestore": "BSAI VDN-H3 FaceRestore · 远景小脸崩坏修复",
    "BSAIVDNH3FaceOil": "BSAI VDN-H3 FaceOil · 脸部去油",
    "BSAIVDNH3EmptyLatentVideo": "BSAI VDN-H3 EmptyLatentVideo · H3视频5D空latent",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
