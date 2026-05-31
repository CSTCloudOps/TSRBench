# -*- coding: utf-8 -*-

import os
import json
import logging
import inspect
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models.ts2vec import TS2Vec

# ===== Optional foundation-model deps =====
try:
    from transformers import TimesFmModelForPrediction
except Exception:
    TimesFmModelForPrediction = None

try:
    from transformers import AutoModel
except Exception:
    AutoModel = None

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
except Exception:
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoModelForSeq2SeqLM = None

try:
    from chronos import Chronos2Pipeline
except Exception:
    Chronos2Pipeline = None

try:
    from tabpfn import TabPFNUnsupervisedModel
except Exception:
    TabPFNUnsupervisedModel = None
try:
    from tabpfn import TabPFNClassifier
except Exception:
    TabPFNClassifier = None

logger = logging.getLogger(__name__)

def _is_cuda_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        ("out of memory" in msg)
        or ("cuda oom" in msg)
        or ("not enough memory" in msg)
        or ("defaultcpuallocator" in msg)
        or ("std::bad_alloc" in msg)
    )

def maybe_wrap_data_parallel(module: nn.Module, device_ids: List[int], name: str) -> nn.Module:
    if module is None or not torch.cuda.is_available():
        return module
    if not device_ids or len(device_ids) <= 1:
        return module
    if isinstance(module, nn.DataParallel):
        return module
    primary_device = torch.device(f"cuda:{device_ids[0]}")
    module = module.to(primary_device)
    logger.info("%s 启用 DataParallel, GPUs=%s", name, device_ids)
    return nn.DataParallel(module, device_ids=device_ids)

def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.DataParallel) else module

def _strip_prefix_in_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    plen = len(prefix)
    return {k[plen:] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

def _load_ts2vec_ckpt_flexible(model: TS2Vec, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        raise TypeError(f"TS2Vec checkpoint格式不支持: {type(state_dict)}")

    target = unwrap_module(model.net)
    candidates = [state_dict]
    candidates.append(_strip_prefix_in_state_dict(state_dict, "module."))
    candidates.append(_strip_prefix_in_state_dict(state_dict, "module.module."))

    last_error = None
    for i, sd in enumerate(candidates):
        try:
            target.load_state_dict(sd, strict=True)
            if i > 0:
                logger.info("TS2Vec checkpoint已自动修正前缀后加载成功（策略 #%d）", i + 1)
            return
        except Exception as e:
            last_error = e
    raise RuntimeError(f"TS2Vec checkpoint加载失败: {last_error}")

def _infer_cost_max_train_length_from_ckpt(ckpt_path: str, device: str = "cpu") -> Optional[int]:
    """
    Infer CoST encoder length from checkpoint tensor shape.
    CoST's BandedFourierLayer uses total_freqs = (length // 2) + 1,
    and with num_bands=1, sfd.0.weight first dim == total_freqs.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        return None
    key = "sfd.0.weight"
    if key not in state_dict:
        return None
    num_freqs = int(state_dict[key].shape[0])
    if num_freqs <= 0:
        return None
    # Choose the even representative length. (L//2)+1 == num_freqs
    return 2 * (num_freqs - 1)

def _resolve_hf_model_dir(model_dir: str) -> str:
    """
    Resolve actual HF snapshot folder that contains config.json.
    Handles layouts like:
      <dir>/config.json
      <dir>/snapshots/<hash>/config.json
      <dir>/<repo_name>/config.json
    """
    if not model_dir:
        return model_dir
    if os.path.isfile(os.path.join(model_dir, "config.json")):
        return model_dir
    if not os.path.isdir(model_dir):
        return model_dir

    candidates: List[Tuple[int, str]] = []
    root_depth = model_dir.rstrip(os.sep).count(os.sep)
    for root, _, files in os.walk(model_dir):
        if "config.json" not in files:
            continue
        depth = root.rstrip(os.sep).count(os.sep) - root_depth
        if depth <= 12:
            candidates.append((depth, root))
    if not candidates:
        return model_dir
    candidates.sort(key=lambda x: x[0])
    resolved = candidates[0][1]
    if resolved != model_dir:
        logger.info("检测到嵌套HF目录，自动使用: %s -> %s", model_dir, resolved)
    return resolved

def precompute_ts2vec_embeddings(dataset, ts2vec_ckpt_path, device='cuda:0', device_ids: Optional[List[int]] = None, use_dp: bool = True):
    """
    TS2Vec B-route: pre-encode all series.
    Returns: Dict[(class_id, i), np.ndarray]
    """

    # 1) 构造 padding 后的训练数组，顺便拿 T_max
    all_series = []
    keys = []
    for class_id, samples in dataset.items():
        for i, x in enumerate(samples):
            all_series.append(x)
            keys.append((class_id, i))

    max_len = max(len(x) for x in all_series)

    padded = []
    for x in all_series:
        pad = np.zeros(max_len, dtype=np.float32)
        pad[:len(x)] = x
        padded.append(pad)

    data = np.stack(padded)[..., None]  # (N, T_max, 1)

    # 2) load TS2Vec
    model = TS2Vec(
        input_dims=1,
        output_dims=320,
        device=device,
        batch_size=64,
        max_train_length=max_len,
    )
    _load_ts2vec_ckpt_flexible(model, ts2vec_ckpt_path, device=device)
    if hasattr(model, "net") and use_dp:
        model.net = maybe_wrap_data_parallel(model.net, device_ids or [], "TS2Vec")
    model.net.eval()

    # 3) encode
    try:
        with torch.no_grad():
            embeddings = model.encode(
                data,
                encoding_window='full_series'
            )  # (N, H)
    except RuntimeError as e:
        msg = str(e).lower()
        if isinstance(model.net, nn.DataParallel) and ("busy or unavailable" in msg or "devicesunavailable" in msg):
            logger.warning("TS2Vec 多卡推理失败，自动降级为单卡重试: %s", e)
            model.net = unwrap_module(model.net)
            model.net = model.net.to(device)
            with torch.no_grad():
                embeddings = model.encode(
                    data,
                    encoding_window='full_series'
                )
        else:
            raise

    # 4) build bank
    bank = {}
    for k, emb in zip(keys, embeddings):
        bank[k] = emb

    return bank

def _cost_align_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    T = x.shape[0]
    if T == target_len:
        return x
    if T > target_len:
       
        return x[-target_len:]   
    pad = np.full((target_len - T,), np.nan, dtype=np.float32) 

    return np.concatenate([pad, x], axis=0)

def precompute_cost_embeddings(dataset, cost_model, device='cuda:0', batch_size=64):
    cost_model.net.eval()
    target_len = unwrap_module(cost_model.net).sfd[0].length

    keys, series = [], []
    for class_id, samples in dataset.items():
        for i, x in enumerate(samples):
            keys.append((class_id, i))
            series.append(_cost_align_to_length(x, target_len).astype(np.float32))

    data = np.stack(series, axis=0)[..., None]  # (N, T, 1)

    try:
        emb = cost_model.encode(
            data,
            mode='forecasting',
            encoding_window='full_series',
            batch_size=batch_size
        )
    except RuntimeError as e:
        if isinstance(cost_model.net, nn.DataParallel):
            logger.warning("CoST 多卡预编码失败，自动降级单卡重试: %s", e)
            cost_model.net = unwrap_module(cost_model.net)
            cost_model.net = cost_model.net.to(device)
            emb = cost_model.encode(
                data,
                mode='forecasting',
                encoding_window='full_series',
                batch_size=batch_size
            )
        else:
            raise

    if torch.is_tensor(emb):
        emb = emb.detach().cpu().numpy()
    if emb.ndim == 3:
        emb = emb[:, 0, :]
    emb = emb.astype(np.float32, copy=False)

    logger.debug("CoST embedding shape: %s", emb.shape)

    return {k: emb[j] for j, k in enumerate(keys)}

def precompute_timesfm_embeddings(
    dataset: Dict[int, List[np.ndarray]],
    timesfm_model_dir: str,
    device: str = "cuda:0",
    batch_size: int = 64,
    freq_id: int = 0,
    pooling: str = "last",   # "mean" or "last"
) -> Dict[tuple, np.ndarray]:

    
    #  返回: {(class_id, idx): embedding(np.float32)}
    
    if TimesFmModelForPrediction is None:
        raise ImportError("TimesFM需要 transformers (包含 TimesFM 模型)。")

    # TimesFM transformers doc示例用 TimesFmModelForPrediction.from_pretrained。:contentReference[oaicite:3]{index=3}
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and str(device).startswith("cuda")) else torch.float32
    model = TimesFmModelForPrediction.from_pretrained(
        timesfm_model_dir,
        dtype=torch_dtype,
        attn_implementation="sdpa",
        device_map="auto" if str(device).startswith("cuda") else None,
        local_files_only=True,
    )
    model.eval()

    ctx_len = getattr(model.config, "context_length", None)

    keys = []
    series_list = []
    for class_id, samples in dataset.items():
        for i, x in enumerate(samples):
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            if ctx_len is not None and len(x) > ctx_len:
                x = x[-ctx_len:]  # 截断到最大context
            keys.append((class_id, i))
            series_list.append(x)

    bank = {}
    for start in range(0, len(series_list), batch_size):
        chunk = series_list[start:start + batch_size]
        lens = [len(c) for c in chunk]
        past_values = [torch.tensor(c, dtype=torch_dtype, device=model.device) for c in chunk]
        freq = torch.tensor([freq_id] * len(past_values), dtype=torch.long, device=model.device)

        with torch.no_grad():
            out = model(
                past_values=past_values,
                freq=freq,
                return_dict=True,
                output_hidden_states=False
            )
            h = out.last_hidden_state  # (B, L, H)  文档说明有 last_hidden_state :contentReference[oaicite:4]{index=4}

        # mean pooling / last pooling
        maxL = h.shape[1]
        lens_t = torch.tensor(lens, device=h.device)
        if pooling == "last":
            idx = (lens_t - 1).clamp(min=0)
            emb = h[torch.arange(h.size(0), device=h.device), idx]  # (B,H)
        else:
            mask = (torch.arange(maxL, device=h.device)[None, :] < lens_t[:, None]).to(h.dtype)  # (B,L)
            emb = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1e-6).unsqueeze(-1)

        emb = emb.float().cpu().numpy()
        for j, e in enumerate(emb):
            bank[keys[start + j]] = e.astype(np.float32, copy=False)

    return bank

def precompute_chronos2_embeddings(
    dataset: Dict[int, List[np.ndarray]],
    chronos2_model_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    pooling: str = "mean",   # "mean" or "last"
) -> Dict[tuple, np.ndarray]:
    if Chronos2Pipeline is None:
        raise ImportError("Chronos-2需要 chronos-forecasting>=2.1.0 (含 Chronos2Pipeline.embed)。")

    # Prefer a single explicit device; if CUDA is temporarily unavailable, fall back to CPU.
    preferred_device = str(device) if str(device).startswith("cuda") else "cpu"
    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            chronos2_model_dir,
            device_map=preferred_device,
            local_files_only=True,
        )
    except Exception as e:
        if preferred_device.startswith("cuda"):
            logger.warning("Chronos-2 加载到 %s 失败，自动回退CPU: %s", preferred_device, e)
            pipeline = Chronos2Pipeline.from_pretrained(
                chronos2_model_dir,
                device_map="cpu",
                local_files_only=True,
            )
        else:
            raise

    keys = []
    series_list = []
    for class_id, samples in dataset.items():
        for i, x in enumerate(samples):
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            keys.append((class_id, i))
            series_list.append(torch.tensor(x, dtype=torch.float32))

    bank = {}

    def _to_padded_batch(embs_list: list, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        embs_list: List[Tensor(L_i, H)] or List[np.ndarray(L_i, H)] or List[List[...]]
        device: 张量创建的目标设备（传模型设备即可），默认用第一个张量的设备
        return:
        padded: Tensor(B, Lmax, H) - 与device一致
        mask:   Tensor(B, Lmax)  (1 for valid, 0 for pad) - 与device一致
        """
        # 转成 tensor list
        t_list = []
        for e in embs_list:
            if isinstance(e, torch.Tensor):
                t = e
            else:
                t = torch.tensor(e)
            # 期望形状 (L, H)
            if t.dim() == 3 and t.size(0) == 1:
                t = t.squeeze(0)  # (1,L,H)->(L,H)
            if t.dim() != 2:
                raise ValueError(f"Chronos2 embed 单条输出维度不符合预期: {t.shape}")
            t_list.append(t)

        lengths = torch.tensor([t.size(0) for t in t_list], dtype=torch.long, device=device)
        Lmax = int(lengths.max().item())
        H = int(t_list[0].size(1))
        # 关键修改1：基于指定device创建padded和mask，从根源统一设备
        padded = torch.zeros((len(t_list), Lmax, H), dtype=t_list[0].dtype, device=device)
        mask = torch.zeros((len(t_list), Lmax), dtype=torch.float32, device=device)
        for i, t in enumerate(t_list):
            L = t.size(0)
            padded[i, :L, :] = t.to(device)  # 确保输入张量也迁移到目标设备
            mask[i, :L] = 1.0
        return padded, mask

    for start in range(0, len(series_list), batch_size):
        chunk = series_list[start:start + batch_size]

        with torch.no_grad():
            out = pipeline.embed(chunk)
            embs = out[0] if isinstance(out, (tuple, list)) else out

        # ---- 关键兼容：embs 可能是 Tensor(B,L,H) 或 list[Tensor(L_i,H)] ----
        if isinstance(embs, torch.Tensor):
            # (B,L,H) 直接 pooling
            if pooling == "last":
                emb = embs[:, -1, :]
            else:
                emb = embs.mean(dim=1)
        elif isinstance(embs, list):
            # 关键修改2：获取模型的目标设备，传给_to_padded_batch
            model_device = next(pipeline.model.parameters()).device if hasattr(pipeline, "model") else torch.device("cpu")
            # 直接创建GPU设备的padded和mask，无需后期迁移
            padded, mask = _to_padded_batch(embs, device=model_device)
            
            if pooling == "last":
                lengths = mask.sum(dim=1).long().clamp(min=1)
                idx = lengths - 1
                emb = padded[torch.arange(padded.size(0)), idx, :]
            else:
                denom = mask.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)
                # 所有张量（padded/mask/denom）已在同一设备，直接运算无报错
                emb = (padded * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            raise TypeError(f"Chronos2 embed 返回了未知类型: {type(embs)}")

        emb = emb.float().cpu().numpy()
        for j, e in enumerate(emb):
            bank[keys[start + j]] = e.astype(np.float32, copy=False)

    return bank

def _pool_hidden_states(hidden: torch.Tensor, lengths: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
    max_len = hidden.shape[1]
    lengths = lengths.to(hidden.device).clamp(min=1, max=max_len)
    mask = (torch.arange(max_len, device=hidden.device)[None, :] < lengths[:, None]).to(hidden.dtype)
    if pooling == "last":
        idx = (lengths - 1).clamp(min=0, max=max_len - 1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp(min=1e-6).unsqueeze(-1)

def _fallback_series_embedding(x2: torch.Tensor, lengths: torch.Tensor, dim: int = 320) -> torch.Tensor:
    """
    Deterministic fallback embedding from raw series when model hidden states are unavailable.
    """
    out = torch.zeros((x2.shape[0], dim), dtype=torch.float32, device=x2.device)
    for i in range(x2.shape[0]):
        L = int(max(1, lengths[i].item()))
        s = x2[i, :L].to(torch.float32)
        s = (s - s.mean()) / (s.std(unbiased=False) + 1e-6)
        if L == 1:
            out[i] = s[0]
        else:
            rs = F.interpolate(
                s.view(1, 1, L),
                size=dim,
                mode="linear",
                align_corners=False,
            ).view(-1)
            out[i] = rs
    return out

def _extract_hidden_tensor(out: Any) -> Optional[torch.Tensor]:
    if out is None:
        return None
    if torch.is_tensor(out):
        if out.dim() == 2:
            return out.unsqueeze(1)
        if out.dim() == 3:
            return out
        return None
    if isinstance(out, np.ndarray):
        t = torch.from_numpy(out)
        if t.dim() == 2:
            return t.unsqueeze(1)
        if t.dim() == 3:
            return t
        return None
    if isinstance(out, (tuple, list)):
        for item in out:
            if torch.is_tensor(item):
                if item.dim() == 2:
                    return item.unsqueeze(1)
                if item.dim() == 3:
                    return item
            if isinstance(item, np.ndarray):
                t = torch.from_numpy(item)
                if t.dim() == 2:
                    return t.unsqueeze(1)
                if t.dim() == 3:
                    return t
        return None

    for attr in ("last_hidden_state", "encoder_last_hidden_state", "embeddings", "representations", "forecast", "predictions"):
        val = getattr(out, attr, None)
        if torch.is_tensor(val):
            if val.dim() == 2:
                return val.unsqueeze(1)
            if val.dim() == 3:
                return val
        if isinstance(val, np.ndarray):
            t = torch.from_numpy(val)
            if t.dim() == 2:
                return t.unsqueeze(1)
            if t.dim() == 3:
                return t

    hs = getattr(out, "hidden_states", None)
    if hs is not None:
        if torch.is_tensor(hs):
            if hs.dim() == 2:
                return hs.unsqueeze(1)
            if hs.dim() == 3:
                return hs
        elif isinstance(hs, (tuple, list)) and len(hs) > 0:
            last = hs[-1]
            if torch.is_tensor(last):
                if last.dim() == 2:
                    return last.unsqueeze(1)
                if last.dim() == 3:
                    return last

    # Last resort for CausalLM outputs.
    logits = getattr(out, "logits", None)
    if torch.is_tensor(logits):
        if logits.dim() == 2:
            return logits.unsqueeze(1)
        if logits.dim() == 3:
            return logits
    return None

def _extract_tensor_deep(obj: Any) -> Optional[torch.Tensor]:
    """
    Deeply search common container outputs and pick a usable tensor.
    """
    found: List[torch.Tensor] = []

    def _walk(x: Any) -> None:
        if x is None:
            return
        if torch.is_tensor(x):
            found.append(x)
            return
        if isinstance(x, np.ndarray):
            try:
                found.append(torch.from_numpy(x))
            except Exception:
                pass
            return
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
            return
        if isinstance(x, (list, tuple)):
            for v in x:
                _walk(v)
            return
        for attr in (
            "last_hidden_state", "encoder_last_hidden_state", "hidden_states", "logits",
            "representations", "embeddings", "forecast", "forecasts", "prediction", "predictions",
            "output", "outputs",
        ):
            if hasattr(x, attr):
                _walk(getattr(x, attr))

    _walk(obj)
    if not found:
        return None
    # Prefer 3D hidden states, then 2D features.
    ranked = sorted(found, key=lambda t: (t.dim() == 3, t.dim() == 2, t.numel()), reverse=True)
    t = ranked[0]
    if t.dim() == 2:
        return t.unsqueeze(1)
    if t.dim() == 3:
        return t
    return None

def _try_forward_with_kwargs(model: nn.Module, kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
    try:
        out = model(**kwargs)
    except Exception:
        return None
    hidden = _extract_hidden_tensor(out)
    if hidden is not None:
        return hidden
    return _extract_tensor_deep(out)

def _iter_model_chain(model: nn.Module) -> List[nn.Module]:
    """
    Prefer core decoder module first (model.model), then wrapper module.
    """
    modules: List[nn.Module] = []
    for m in (getattr(model, "model", None), model):
        if m is None:
            continue
        if all(m is not existed for existed in modules):
            modules.append(m)
    return modules

def _infer_module_dtype(module: nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except Exception:
        return fallback

def _infer_input_embed_dtype(module: nn.Module) -> Optional[torch.dtype]:
    """
    Best-effort detection of the first projection layer dtype that consumes input_ids.
    """
    candidate_paths = [
        ("embed_layer", "emb"),
        ("embed_layer", "emb_layer"),
        ("embed_layer", "gate_layer"),
        ("model", "embed_layer", "emb"),
        ("model", "embed_layer", "emb_layer"),
        ("model", "embed_layer", "gate_layer"),
    ]
    for path in candidate_paths:
        obj: Any = module
        ok = True
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                ok = False
                break
        if not ok:
            continue
        w = getattr(obj, "weight", None)
        if torch.is_tensor(w):
            return w.dtype
    getter = getattr(module, "get_input_embeddings", None)
    if callable(getter):
        try:
            emb = getter()
            w = getattr(emb, "weight", None)
            if torch.is_tensor(w):
                return w.dtype
        except Exception:
            pass
    return None

def _candidate_input_dtypes(module: nn.Module, x: torch.Tensor) -> List[torch.dtype]:
    dtypes: List[torch.dtype] = []

    def _add(dt: Optional[torch.dtype]) -> None:
        if isinstance(dt, torch.dtype) and dt not in dtypes:
            dtypes.append(dt)

    _add(_infer_input_embed_dtype(module))
    _add(_infer_module_dtype(module, fallback=x.dtype))
    _add(x.dtype)
    _add(torch.float32)
    if x.device.type == "cuda":
        _add(torch.bfloat16)
        _add(torch.float16)
    return dtypes

def _detect_ts_model_family(model: nn.Module, model_tag: str = "") -> str:
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower().strip()
    tag = str(model_tag).lower()
    if model_type in ("timer",):
        return "timer"
    if model_type in ("sundial",):
        return "sundial"
    if model_type in ("time_moe", "time-moe"):
        return "time_moe"
    if "timer" in tag:
        return "timer"
    if "sundial" in tag:
        return "sundial"
    if "time-moe" in tag or "timemoe" in tag:
        return "time_moe"
    return ""

def _forward_timer_or_sundial_strict(
    model: nn.Module,
    x2: torch.Tensor,
    model_tag: str,
) -> torch.Tensor:
    """
    Strict adapter for Timer/Sundial:
    - feed raw float time series through input_ids
    - disable cache
    - do NOT pass sample-level attention_mask by default
      (these models patchify input first; raw mask length mismatches token length)
    """
    cfg = getattr(model, "config", None)
    token_len = int(getattr(cfg, "input_token_len", 1) or 1)
    token_len = max(token_len, 1)

    # Optional token-level mask for a second try.
    n_tok = int(x2.shape[1] // token_len)
    tok_mask = None
    if n_tok > 0:
        tok_mask = torch.ones((x2.shape[0], n_tok), dtype=torch.long, device=x2.device)

    last_err: Optional[Exception] = None
    last_ctx: str = ""
    for mod in _iter_model_chain(model):
        for dt in _candidate_input_dtypes(mod, x2):
            x_in = x2.to(dtype=dt)
            base_kwargs: Dict[str, Any] = {
                "input_ids": x_in,
                "output_hidden_states": True,
                "return_dict": True,
                "use_cache": False,
            }
            try_kwargs = [base_kwargs]
            if tok_mask is not None:
                try_kwargs.append({**base_kwargs, "attention_mask": tok_mask})
            for kwargs in try_kwargs:
                try:
                    out = mod(**kwargs)
                    hidden = _extract_hidden_tensor(out)
                    if hidden is None:
                        hidden = _extract_tensor_deep(out)
                    if hidden is not None:
                        return hidden
                except Exception as e:
                    last_err = e
                    last_ctx = (
                        f"module={type(mod).__name__}, dtype={dt}, "
                        f"has_attention_mask={'attention_mask' in kwargs}"
                    )
                    continue

    raise RuntimeError(f"{model_tag} 严格专用适配失败: {last_err} | {last_ctx}")

def _forward_timemoe_strict(
    model: nn.Module,
    x2: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Strict adapter for Time-MoE:
    - feed 3D input_ids [B, L, 1] to avoid in-place unsqueeze side effects
    - disable cache to avoid model-side cache path warnings/failures
    """
    last_err: Optional[Exception] = None
    last_ctx: str = ""
    for mod in _iter_model_chain(model):
        param_dtype = _infer_module_dtype(mod, fallback=x2.dtype)
        if param_dtype in (torch.float32, torch.float16, torch.bfloat16):
            dtypes = [param_dtype]
        else:
            dtypes = _candidate_input_dtypes(mod, x2)
        for dt in dtypes:
            x_base = x2.to(dtype=dt)
            x3 = x_base.unsqueeze(-1).contiguous()
            for kwargs in (
                {
                    "input_ids": x3,
                    "attention_mask": mask,
                    "output_hidden_states": True,
                    "return_dict": True,
                    "use_cache": False,
                },
                {
                    "input_ids": x3,
                    "output_hidden_states": True,
                    "return_dict": True,
                    "use_cache": False,
                },
                {
                    "input_ids": x_base,
                    "attention_mask": mask,
                    "output_hidden_states": True,
                    "return_dict": True,
                    "use_cache": False,
                },
            ):
                try:
                    out = mod(**kwargs)
                    hidden = _extract_hidden_tensor(out)
                    if hidden is None:
                        hidden = _extract_tensor_deep(out)
                    if hidden is not None:
                        return hidden
                except Exception as e:
                    last_err = e
                    last_ctx = (
                        f"module={type(mod).__name__}, dtype={dt}, param_dtype={param_dtype}, "
                        f"has_attention_mask={'attention_mask' in kwargs}, "
                        f"input_rank={int(kwargs['input_ids'].dim())}"
                    )
                    continue

    raise RuntimeError(f"Time-MoE 严格专用适配失败: {last_err} | {last_ctx}")

def _forward_strict_special_timeseries_model(
    model: nn.Module,
    model_tag: str,
    x2: torch.Tensor,
    mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    family = _detect_ts_model_family(model, model_tag=model_tag)
    if family == "timer":
        return _forward_timer_or_sundial_strict(model, x2, model_tag="Timer")
    if family == "sundial":
        return _forward_timer_or_sundial_strict(model, x2, model_tag="Sundial")
    if family == "time_moe":
        return _forward_timemoe_strict(model, x2, mask)
    return None

def _forward_special_timeseries_model(
    model: nn.Module,
    x2: torch.Tensor,
    x3: torch.Tensor,
    mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    def _try_module(m: nn.Module, kw: Dict[str, Any]) -> Optional[torch.Tensor]:
        kw = dict(kw)
        kw.setdefault("return_dict", True)
        kw.setdefault("output_hidden_states", True)
        kw.setdefault("use_cache", False)
        hidden = _try_forward_with_kwargs(m, kw)
        if hidden is not None:
            return hidden
        # retry without extra flags for custom remote-code models
        for k in ("return_dict", "output_hidden_states", "use_cache"):
            kw.pop(k, None)
        return _try_forward_with_kwargs(m, kw)

    model_core = getattr(model, "model", None)

    def _series_to_token_ids(values_2d: torch.Tensor, attn_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
        bsz, seqlen = values_2d.shape
        out = torch.zeros((bsz, seqlen), dtype=torch.long, device=values_2d.device)
        vmax = max(8, int(vocab_size) - 1)
        for i in range(bsz):
            valid = attn_mask[i] > 0
            if not torch.any(valid):
                continue
            v = values_2d[i, valid]
            vmin = torch.min(v)
            vrange = torch.max(v) - vmin
            if float(vrange) < 1e-8:
                out[i, valid] = 1
            else:
                norm = (v - vmin) / vrange
                out[i, valid] = (norm * (vmax - 1)).long().clamp(min=1, max=vmax)
        return out

    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 4096))
    token_ids = _series_to_token_ids(x2, mask, vocab_size=vocab_size)
    zeros_mark = torch.zeros((x2.shape[0], x2.shape[1], 4), dtype=x2.dtype, device=x2.device)
    freq = torch.zeros((x2.shape[0],), dtype=torch.long, device=x2.device)

    value_map = {
        "x_enc": x3,
        "x": x3,
        "x_in": x3,
        "past_values": x2,
        "inputs": x2,
        "input_values": x2,
        "series": x2,
        # Timer/Sundial/Time-MoE signatures annotate input_ids as float tensor.
        "input_ids": x2,
        "token_ids": token_ids,
        "attention_mask": mask,
        "mask": mask,
        "input_mask": mask,
        "observed_mask": mask,
        "past_observed_mask": mask,
        "x_mark_enc": zeros_mark,
        "x_mark_dec": zeros_mark,
        "x_dec": x3,
        "decoder_input": x3,
        "freq": freq,
        "loss_masks": mask.to(dtype=x2.dtype),
        "mask_y": mask.to(dtype=x2.dtype),
        "max_output_length": 1,
        "max_horizon_length": 1,
        "num_samples": 1,
        "revin": False,
    }

    def _auto_call_by_signature(mod: nn.Module, method_name: str) -> Optional[torch.Tensor]:
        fn = getattr(mod, method_name, None)
        if not callable(fn):
            return None
        try:
            sig = inspect.signature(fn)
        except Exception:
            return None
        kwargs: Dict[str, Any] = {}
        for pname, p in sig.parameters.items():
            if pname in ("self",):
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL,):
                continue
            if pname in value_map:
                kwargs[pname] = value_map[pname]
            elif p.default is inspect._empty and p.kind != inspect.Parameter.VAR_KEYWORD:
                return None
        if "return_dict" in sig.parameters and "return_dict" not in kwargs:
            kwargs["return_dict"] = True
        if "output_hidden_states" in sig.parameters and "output_hidden_states" not in kwargs:
            kwargs["output_hidden_states"] = True
        if "use_cache" in sig.parameters and "use_cache" not in kwargs:
            kwargs["use_cache"] = False
        return _try_forward_with_kwargs(mod, kwargs)

    def _try_named_methods(mod: nn.Module) -> Optional[torch.Tensor]:
        for mname in (
            "embed", "embedding", "get_embeddings", "extract_features", "features",
            "representation", "represent", "forward_encoder", "encode_series",
            "timeseries_encode", "encode_time_series", "predict", "forecast",
            "inference", "infer", "forward_features", "get_representation",
        ):
            hidden = _auto_call_by_signature(mod, mname)
            if hidden is not None:
                return hidden
            fn = getattr(mod, mname, None)
            if not callable(fn):
                continue
            for kwargs in (
                {"x_enc": x3, "attention_mask": mask},
                {"x_enc": x3},
                {"past_values": x2, "attention_mask": mask},
                {"past_values": x2},
                {"inputs": x2, "attention_mask": mask},
                {"inputs": x2},
                {"x": x3, "mask": mask},
                {"x": x3},
                {"x": x2, "mask": mask},
                {"x": x2},
                {"input_ids": x2, "attention_mask": mask},
                {"input_ids": token_ids, "attention_mask": mask},
            ):
                try:
                    out = fn(**kwargs)
                except Exception:
                    continue
                hidden = _extract_hidden_tensor(out)
                if hidden is None:
                    hidden = _extract_tensor_deep(out)
                if hidden is not None:
                    return hidden
        return None

    # 1) Try model.encode style APIs first.
    encode_fn = getattr(model, "encode", None)
    if callable(encode_fn):
        for kwargs in (
            {"x_enc": x3, "attention_mask": mask},
            {"x_enc": x3},
            {"past_values": x2, "attention_mask": mask},
            {"past_values": x2},
            {"inputs": x2, "attention_mask": mask},
            {"inputs": x2},
            {"x": x3, "mask": mask},
            {"x": x3},
            {"x": x2, "mask": mask},
            {"x": x2},
        ):
            try:
                out = encode_fn(**kwargs)
                hidden = _extract_hidden_tensor(out)
                if hidden is None:
                    hidden = _extract_tensor_deep(out)
                if hidden is not None:
                    return hidden
            except Exception:
                continue
    if model_core is not None:
        encode_fn = getattr(model_core, "encode", None)
        if callable(encode_fn):
            for kwargs in (
                {"x_enc": x3, "attention_mask": mask},
                {"x_enc": x3},
                {"past_values": x2, "attention_mask": mask},
                {"past_values": x2},
                {"inputs": x2, "attention_mask": mask},
                {"inputs": x2},
                {"x": x3, "mask": mask},
                {"x": x3},
                {"x": x2, "mask": mask},
                {"x": x2},
            ):
                try:
                    out = encode_fn(**kwargs)
                    hidden = _extract_hidden_tensor(out)
                    if hidden is None:
                        hidden = _extract_tensor_deep(out)
                    if hidden is not None:
                        return hidden
                except Exception:
                    continue

    # 1.5) Try common embedding/extract APIs.
    hidden = _try_named_methods(model)
    if hidden is not None:
        return hidden
    if model_core is not None:
        hidden = _try_named_methods(model_core)
        if hidden is not None:
            return hidden

    # 1.75) Signature-driven direct forward call.
    hidden = _auto_call_by_signature(model, "forward")
    if hidden is not None:
        return hidden
    if model_core is not None:
        hidden = _auto_call_by_signature(model_core, "forward")
        if hidden is not None:
            return hidden

    # 1.9) Build inputs_embeds from embedding layer if available.
    def _build_inputs_embeds(mod: nn.Module, tok: torch.Tensor, vals: torch.Tensor, msk: torch.Tensor) -> Optional[torch.Tensor]:
        def _call_callable_by_sig(fn: Any) -> Optional[torch.Tensor]:
            if not callable(fn):
                return None
            try:
                sig = inspect.signature(fn)
            except Exception:
                sig = None
            if sig is None:
                for inp in (tok, vals):
                    try:
                        out = fn(inp)
                        hidden = _extract_hidden_tensor(out)
                        if hidden is None:
                            hidden = _extract_tensor_deep(out)
                        if hidden is not None:
                            return hidden
                    except Exception:
                        continue
                return None
            kwargs: Dict[str, Any] = {}
            for pname, p in sig.parameters.items():
                if pname == "self":
                    continue
                if p.kind == inspect.Parameter.VAR_POSITIONAL:
                    continue
                if pname in value_map:
                    kwargs[pname] = value_map[pname]
                elif p.default is inspect._empty and p.kind != inspect.Parameter.VAR_KEYWORD:
                    return None
            if kwargs:
                try:
                    out = fn(**kwargs)
                    hidden = _extract_hidden_tensor(out)
                    if hidden is None:
                        hidden = _extract_tensor_deep(out)
                    if hidden is not None:
                        return hidden
                except Exception:
                    pass
            for inp in (tok, vals):
                try:
                    out = fn(inp)
                    hidden = _extract_hidden_tensor(out)
                    if hidden is None:
                        hidden = _extract_tensor_deep(out)
                    if hidden is not None:
                        return hidden
                except Exception:
                    continue
            return None

        emb_layer = None
        if hasattr(mod, "get_input_embeddings") and callable(getattr(mod, "get_input_embeddings")):
            try:
                emb_layer = mod.get_input_embeddings()
            except Exception:
                emb_layer = None
        if emb_layer is None and hasattr(mod, "_input_embed_layer"):
            emb_layer = getattr(mod, "_input_embed_layer")
        if emb_layer is None:
            return None
        hidden = _call_callable_by_sig(emb_layer)
        if hidden is not None:
            return hidden.to(vals.device)
        return None

    for target in (model, model_core):
        if target is None:
            continue
        in_emb = _build_inputs_embeds(target, token_ids, x2, mask)
        if in_emb is None:
            continue
        for item in (
            {"inputs_embeds": in_emb, "attention_mask": mask, "use_cache": False},
            {"inputs_embeds": in_emb, "attention_mask": mask},
            {"inputs_embeds": in_emb},
        ):
            hidden = _try_module(target, item)
            if hidden is not None:
                return hidden
        # Last fallback: use embedding-layer output directly.
        return in_emb

    # 2) Try forward with common TS signatures for Timer/Sundial/Time-MoE.
    base_candidates = [
        {"x_enc": x3, "attention_mask": mask},
        {"x_enc": x3, "input_mask": mask},
        {"x_enc": x3, "past_observed_mask": mask},
        {"x_enc": x3},
        {"past_values": x2, "attention_mask": mask},
        {"past_values": x2, "past_observed_mask": mask},
        {"past_values": x2, "observed_mask": mask},
        {"past_values": x2},
        {"inputs": x2, "attention_mask": mask},
        {"inputs": x2},
        {"x": x3, "mask": mask},
        {"x": x3},
        {"x": x2, "mask": mask},
        {"x": x2},
        {"input_ids": x2, "attention_mask": mask},
        {"input_ids": x2, "attention_mask": mask, "use_cache": False},
        {"input_ids": token_ids, "attention_mask": mask},
        {"input_ids": token_ids, "attention_mask": mask, "use_cache": False},
        {"series": x2, "observed_mask": mask},
        {"x_enc": x3, "x_mark_enc": zeros_mark, "x_dec": x3, "x_mark_dec": zeros_mark},
        {"x_enc": x3, "x_mark_enc": zeros_mark, "freq": freq},
    ]

    for item in base_candidates:
        # Keep only args that forward likely accepts when signature is explicit.
        try:
            sig = inspect.signature(model.forward)
            params = sig.parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not has_var_kw:
                item = {k: v for k, v in item.items() if k in params}
        except Exception:
            pass
        if not item:
            continue
        hidden = _try_module(model, item)
        if hidden is not None:
            return hidden
        if model_core is not None:
            hidden = _try_module(model_core, item)
            if hidden is not None:
                return hidden

    # 3) Informer-like signature fallback.
    informer_kwargs = {
        "x_enc": x3,
        "x_mark_enc": zeros_mark,
        "x_dec": x3,
        "x_mark_dec": zeros_mark,
    }
    hidden = _try_module(model, informer_kwargs)
    if hidden is not None:
        return hidden
    if model_core is not None:
        hidden = _try_module(model_core, informer_kwargs)
        if hidden is not None:
            return hidden

    return None

def precompute_causallm_special_embeddings(
    dataset: Dict[int, List[np.ndarray]],
    model_dir: str,
    model_tag: str,
    device: str = "cuda:0",
    batch_size: int = 64,
    pooling: str = "mean",
    allow_proxy_fallback: bool = False,
) -> Dict[tuple, np.ndarray]:
    if AutoModelForCausalLM is None and AutoModel is None:
        raise ImportError("需要 transformers 来加载该baseline。")

    model_dir = _resolve_hf_model_dir(model_dir)
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and str(device).startswith("cuda")) else torch.float32
    tag_lower = str(model_tag).lower()
    if tag_lower in ("sundial",):
        # These models perform internal ops that upcast to float32; use fp32 for strict, stable inference.
        torch_dtype = torch.float32

    loaders = [("AutoModelForCausalLM", AutoModelForCausalLM), ("AutoModel", AutoModel)]
    model = None
    last_err = None
    for loader_name, loader in loaders:
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True,
                dtype=torch_dtype,
            )
            logger.info("%s 使用 %s 加载成功", model_tag, loader_name)
            break
        except Exception as e:
            last_err = e
    if model is None:
        raise RuntimeError(f"{model_tag} 加载失败: {last_err}")

    model = model.to(device if str(device).startswith("cuda") else "cpu")
    if torch_dtype == torch.float32:
        model = model.float()
    model.eval()
    strict_family = _detect_ts_model_family(model, model_tag=model_tag)
    max_context_len = 0
    if strict_family == "time_moe":
        # Guard CPU/GPU memory for very long sequences. Keep the latest context,
        # consistent with autoregressive-style backbones.
        max_context_len = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0)
        if max_context_len <= 0:
            max_context_len = 4096
        # CPU runs are much more memory-constrained for attention models.
        if not (str(device).startswith("cuda") and torch.cuda.is_available()):
            max_context_len = min(max_context_len, 1024)

    keys = []
    series_list = []
    truncated_count = 0
    for class_id, samples in dataset.items():
        for i, x in enumerate(samples):
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
            if max_context_len > 0 and arr.shape[0] > max_context_len:
                arr = arr[-max_context_len:]
                truncated_count += 1
            keys.append((class_id, i))
            series_list.append(arr)
    if truncated_count > 0:
        logger.info(
            "%s sequence truncation applied: %d/%d series clipped to max_context_len=%d",
            model_tag,
            truncated_count,
            len(series_list),
            max_context_len,
        )

    bank = {}
    use_series_fallback = False
    adapter_debug_logged = False
    fallback_batches = 0
    total_batches = (len(series_list) + batch_size - 1) // batch_size if batch_size > 0 else 0

    def _encode_batch_to_emb(
        batch_x2: torch.Tensor,
        batch_mask: torch.Tensor,
        batch_lengths: torch.Tensor,
    ) -> np.ndarray:
        hidden_local = _forward_strict_special_timeseries_model(
            model=model,
            model_tag=model_tag,
            x2=batch_x2,
            mask=batch_mask,
        )
        if hidden_local is None:
            raise RuntimeError(f"{model_tag} 严格专用适配失败：模型未返回 hidden states。")
        pool_lengths_local = batch_lengths
        if strict_family in ("timer", "sundial"):
            token_len = int(getattr(getattr(model, "config", None), "input_token_len", 1) or 1)
            token_len = max(token_len, 1)
            pool_lengths_local = torch.div(batch_lengths + token_len - 1, token_len, rounding_mode="floor")
        pool_lengths_local = pool_lengths_local.clamp(min=1, max=int(hidden_local.shape[1]))
        return _pool_hidden_states(hidden_local, pool_lengths_local, pooling=pooling).float().cpu().numpy()
    for start in range(0, len(series_list), batch_size):
        chunk = series_list[start:start + batch_size]
        lengths = torch.tensor([len(c) for c in chunk], dtype=torch.long, device=model.device)
        max_len = int(lengths.max().item())
        x2 = torch.zeros((len(chunk), max_len), dtype=torch.float32, device=model.device)
        mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=model.device)
        for i, c in enumerate(chunk):
            t = torch.tensor(c, dtype=torch.float32, device=model.device)
            x2[i, : t.shape[0]] = t
            mask[i, : t.shape[0]] = 1
        x3 = x2.unsqueeze(-1)

        with torch.no_grad():
            if use_series_fallback:
                emb = _fallback_series_embedding(x2, lengths, dim=320).cpu().numpy()
                for j, e in enumerate(emb):
                    bank[keys[start + j]] = e.astype(np.float32, copy=False)
                fallback_batches += 1
                continue

            strict_err: Optional[Exception] = None
            hidden = None
            if strict_family in ("timer", "sundial", "time_moe"):
                try:
                    hidden = _forward_strict_special_timeseries_model(
                        model=model,
                        model_tag=model_tag,
                        x2=x2,
                        mask=mask,
                    )
                except Exception as e:
                    strict_err = e
                    # Time-MoE can OOM on long sequences; try micro-batching before failing.
                    if strict_family == "time_moe" and _is_cuda_oom(e):
                        torch.cuda.empty_cache()
                        micro_bs = max(1, min(len(chunk), batch_size // 2 if batch_size > 1 else 1))
                        success = False
                        while micro_bs >= 1:
                            try:
                                for sub in range(0, len(chunk), micro_bs):
                                    sub_x2 = x2[sub:sub + micro_bs]
                                    sub_mask = mask[sub:sub + micro_bs]
                                    sub_lengths = lengths[sub:sub + micro_bs]
                                    sub_emb = _encode_batch_to_emb(sub_x2, sub_mask, sub_lengths)
                                    for j, e_emb in enumerate(sub_emb):
                                        bank[keys[start + sub + j]] = e_emb.astype(np.float32, copy=False)
                                success = True
                                break
                            except Exception as e2:
                                if _is_cuda_oom(e2) and micro_bs > 1:
                                    torch.cuda.empty_cache()
                                    micro_bs = max(1, micro_bs // 2)
                                    continue
                                strict_err = e2
                                break
                        if success:
                            continue

            if hidden is None and strict_family not in ("timer", "sundial", "time_moe"):
                hidden = _forward_special_timeseries_model(model, x2, x3, mask)

            if hidden is None:
                if (strict_err is not None) and (not adapter_debug_logged):
                    logger.warning("%s 严格专用适配失败: %s", model_tag, strict_err)
                if not adapter_debug_logged:
                    method_hints = sorted(
                        [
                            n for n in dir(model)
                            if any(k in n.lower() for k in ("encode", "embed", "forecast", "predict", "infer", "feature"))
                        ]
                    )[:40]
                    try:
                        fwd_sig = str(inspect.signature(model.forward))
                    except Exception:
                        fwd_sig = "N/A"
                    core = getattr(model, "model", None)
                    if core is not None:
                        try:
                            core_sig = str(inspect.signature(core.forward))
                        except Exception:
                            core_sig = "N/A"
                    else:
                        core_sig = "N/A"
                    logger.warning("%s 可调用候选方法(截断): %s", model_tag, method_hints)
                    logger.warning("%s forward签名: model.forward%s | model.model.forward%s", model_tag, fwd_sig, core_sig)
                    adapter_debug_logged = True
                if not allow_proxy_fallback:
                    if strict_err is not None:
                        raise RuntimeError(f"{model_tag} 专用适配失败（严格模式）: {strict_err}")
                    raise RuntimeError(f"{model_tag} 专用适配失败（严格模式）：模型未返回可用hidden states。")
                logger.warning("%s 专用适配失败，回退到基于原序列的确定性embedding。", model_tag)
                use_series_fallback = True
                emb = _fallback_series_embedding(x2, lengths, dim=320).cpu().numpy()
                for j, e in enumerate(emb):
                    bank[keys[start + j]] = e.astype(np.float32, copy=False)
                fallback_batches += 1
                continue
            pool_lengths = lengths
            if strict_family in ("timer", "sundial"):
                token_len = int(getattr(getattr(model, "config", None), "input_token_len", 1) or 1)
                token_len = max(token_len, 1)
                # Patch-based models: valid token count is ceil(valid_points / token_len).
                pool_lengths = torch.div(lengths + token_len - 1, token_len, rounding_mode="floor")
            # Guard against rank mismatch between raw-point lengths and hidden token length.
            pool_lengths = pool_lengths.clamp(min=1, max=int(hidden.shape[1]))

            emb = _pool_hidden_states(hidden, pool_lengths, pooling=pooling).float().cpu().numpy()
            for j, e in enumerate(emb):
                bank[keys[start + j]] = e.astype(np.float32, copy=False)
    if fallback_batches > 0:
        logger.info(
            "%s 预编码采用原序列fallback: %d/%d batches（模型适配未成功）",
            model_tag, fallback_batches, total_batches
        )
    return bank

def precompute_hf_generic_embeddings(
    dataset: Dict[int, List[np.ndarray]],
    model_dir: str,
    model_tag: str,
    device: str = "cuda:0",
    batch_size: int = 64,
    pooling: str = "mean",
) -> Dict[tuple, np.ndarray]:
    """
    Generic local-HF embedding precompute for baseline models with custom remote code.
    """
    if AutoModel is None:
        raise ImportError("需要 transformers 的 AutoModel 来加载该baseline。")

    model_dir = _resolve_hf_model_dir(model_dir)
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_json = json.load(f)
            if not cfg_json.get("model_type"):
                if str(model_tag).lower() == "moirai":
                    raise RuntimeError(
                        "Moirai 当前目录不是 Transformers 可直读格式（config.json 缺少 model_type）。"
                        "请使用 Moirai/Uni2TS 官方推理接口做专用适配，或提供带 model_type 的HF导出目录。"
                    )
                raise RuntimeError(
                    f"{model_tag} 的 config.json 缺少 model_type，无法用 AutoModel 严格加载。"
                )
        except RuntimeError:
            raise
        except Exception:
            # 解析失败时继续走后续加载逻辑，保留原始报错。
            pass
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and str(device).startswith("cuda")) else torch.float32

    # Some repos only register AutoModelForCausalLM/AutoModelForSeq2SeqLM via auto_map.
    # We detect and choose the best loader instead of hard-coding AutoModel only.
    model = None
    auto_map = {}
    if AutoConfig is not None:
        try:
            cfg = AutoConfig.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
            auto_map = getattr(cfg, "auto_map", {}) or {}
        except Exception:
            auto_map = {}

    loader_order = []
    if "AutoModel" in auto_map:
        loader_order.append(("AutoModel", AutoModel))
    if "AutoModelForCausalLM" in auto_map and AutoModelForCausalLM is not None:
        loader_order.append(("AutoModelForCausalLM", AutoModelForCausalLM))
    if "AutoModelForSeq2SeqLM" in auto_map and AutoModelForSeq2SeqLM is not None:
        loader_order.append(("AutoModelForSeq2SeqLM", AutoModelForSeq2SeqLM))
    if not loader_order:
        loader_order = [
            ("AutoModel", AutoModel),
            ("AutoModelForCausalLM", AutoModelForCausalLM),
            ("AutoModelForSeq2SeqLM", AutoModelForSeq2SeqLM),
        ]

    last_err = None
    for loader_name, loader in loader_order:
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True,
                dtype=torch_dtype,
            )
            logger.info("%s 使用 %s 加载成功", model_tag, loader_name)
            break
        except Exception as e:
            last_err = e
            continue
    if model is None:
        raise RuntimeError(f"{model_tag} 加载失败: {last_err}")
    model = model.to(device if str(device).startswith("cuda") else "cpu")
    model.eval()

    keys = []
    series_list = []
    for class_id, samples in dataset.items():
        for i, x in enumerate(samples):
            keys.append((class_id, i))
            series_list.append(np.asarray(x, dtype=np.float32).reshape(-1))

    bank = {}
    for start in range(0, len(series_list), batch_size):
        chunk = series_list[start:start + batch_size]
        lengths = torch.tensor([len(c) for c in chunk], dtype=torch.long, device=model.device)
        max_len = int(lengths.max().item())
        x = torch.zeros((len(chunk), max_len), dtype=torch.float32, device=model.device)
        mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=model.device)
        for i, c in enumerate(chunk):
            t = torch.tensor(c, dtype=torch.float32, device=model.device)
            x[i, : t.shape[0]] = t
            mask[i, : t.shape[0]] = 1

        with torch.no_grad():
            out = None
            for kwargs in (
                {"past_values": x, "attention_mask": mask, "return_dict": True},
                {"x_enc": x, "attention_mask": mask, "return_dict": True},
                {"inputs": x, "attention_mask": mask, "return_dict": True},
            ):
                try:
                    out = model(**kwargs)
                    break
                except Exception:
                    out = None
            if out is None:
                raise RuntimeError(f"{model_tag} 不支持通用前向接口，请在 precompute_hf_generic_embeddings 中补充专用适配。")

            hidden = None
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                hidden = out.last_hidden_state
            elif hasattr(out, "encoder_last_hidden_state") and out.encoder_last_hidden_state is not None:
                hidden = out.encoder_last_hidden_state
            elif hasattr(out, "hidden_states") and out.hidden_states is not None:
                hs = out.hidden_states
                if torch.is_tensor(hs):
                    hidden = hs
                elif isinstance(hs, (tuple, list)) and len(hs) > 0:
                    hidden = hs[-1]
            if hidden is None:
                raise RuntimeError(f"{model_tag} 输出中未找到 hidden states。")

            pool_lengths = lengths.clamp(min=1, max=int(hidden.shape[1]))
            emb = _pool_hidden_states(hidden, pool_lengths, pooling=pooling).float().cpu().numpy()
            for j, e in enumerate(emb):
                bank[keys[start + j]] = e.astype(np.float32, copy=False)

    return bank

def precompute_tabpfn_embeddings(
    dataset: Dict[int, List[np.ndarray]],
    device: str = "cpu",
    allow_proxy_fallback: bool = False,
) -> Dict[tuple, np.ndarray]:
    """
    TabPFN baseline embedding.
    Priority:
      1) TabPFNUnsupervisedModel.encode (if available)
      2) TabPFNClassifier fallback with pseudo labels + predict_proba embedding
    """
    if TabPFNUnsupervisedModel is None and TabPFNClassifier is None:
        raise ImportError("需要安装 tabpfn 才能启用 TabPFN baseline。")

    keys = []
    series_list = []
    max_len = 0
    for class_id, samples in dataset.items():
        for i, x in enumerate(samples):
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
            keys.append((class_id, i))
            series_list.append(arr)
            max_len = max(max_len, arr.shape[0])

    X = np.zeros((len(series_list), max_len), dtype=np.float32)
    for i, arr in enumerate(series_list):
        X[i, : arr.shape[0]] = arr

    if TabPFNUnsupervisedModel is not None:
        tabpfn_device = "cuda" if str(device).startswith("cuda") else "cpu"
        model = TabPFNUnsupervisedModel(device=tabpfn_device)
        with torch.no_grad():
            emb = model.encode(X)
        if torch.is_tensor(emb):
            emb = emb.detach().cpu().numpy()
        emb = np.asarray(emb, dtype=np.float32)
        return {k: emb[j] for j, k in enumerate(keys)}

    # Fallback: no unsupervised API in current tabpfn release.
    if not allow_proxy_fallback:
        raise RuntimeError(
            "TabPFN 当前版本仅提供有监督接口；严格模式下不启用伪标签/代理embedding。"
            "如需近似baseline，请显式设置 --allow_proxy_embedding_fallback true。"
        )
    logger.info("TabPFN 未提供无监督API，启用 TabPFNClassifier 伪标签fallback。")
    X2 = X
    # Keep dimension moderate for stability/speed.
    if X2.shape[1] > 512 and X2.shape[0] > 4:
        n_comp = max(8, min(512, X2.shape[0] - 1, X2.shape[1]))
        X2 = PCA(n_components=n_comp, random_state=42).fit_transform(X2)
    X2 = StandardScaler().fit_transform(X2).astype(np.float32, copy=False)

    n_clusters = max(2, min(8, X2.shape[0] // 32 if X2.shape[0] >= 64 else 2))
    pseudo_y = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X2)

    cache_dir = os.path.expanduser("~/.cache/tabpfn")
    has_local_ckpt = False
    if os.path.isdir(cache_dir):
        for fn in os.listdir(cache_dir):
            if fn.endswith(".ckpt"):
                has_local_ckpt = True
                break
    if not has_local_ckpt:
        if not allow_proxy_fallback:
            raise RuntimeError("TabPFN 本地未检测到ckpt，严格模式下不使用PCA替代。")
        logger.warning("TabPFN 本地未检测到ckpt，跳过在线下载并回退到本地PCA embedding。")
        pca_dim = max(2, min(64, X2.shape[1], X2.shape[0] - 1 if X2.shape[0] > 1 else 1))
        emb = PCA(n_components=pca_dim, random_state=42).fit_transform(X2) if pca_dim >= 2 else X2
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[:, None]
        return {k: emb[j] for j, k in enumerate(keys)}

    clf = None
    last_err = None
    tabpfn_device = "cuda" if str(device).startswith("cuda") else "cpu"
    for kwargs in (
        {"device": tabpfn_device, "ignore_pretraining_limits": True},
        {"device": tabpfn_device},
        {"ignore_pretraining_limits": True},
        {},
    ):
        try:
            clf = TabPFNClassifier(**kwargs)
            break
        except Exception as e:
            last_err = e
    if clf is None:
        raise RuntimeError(f"TabPFNClassifier 初始化失败: {last_err}")

    try:
        clf.fit(X2, pseudo_y)
        if hasattr(clf, "predict_proba"):
            emb = clf.predict_proba(X2)
        elif hasattr(clf, "transform"):
            emb = clf.transform(X2)
        elif hasattr(clf, "decision_function"):
            emb = clf.decision_function(X2)
        else:
            raise RuntimeError("TabPFNClassifier 不支持可用于embedding的输出接口。")
    except Exception as e:
        if not allow_proxy_fallback:
            raise RuntimeError(f"TabPFNClassifier 下载/训练失败（严格模式不回退）: {e}")
        logger.warning("TabPFNClassifier 下载/训练失败，回退到本地PCA embedding（非TabPFN权重）: %s", e)
        pca_dim = max(2, min(64, X2.shape[1], X2.shape[0] - 1 if X2.shape[0] > 1 else 1))
        if pca_dim >= 2:
            emb = PCA(n_components=pca_dim, random_state=42).fit_transform(X2)
        else:
            emb = X2

    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[:, None]
    return {k: emb[j] for j, k in enumerate(keys)}
