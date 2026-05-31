from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from methods import embedding_methods as em

logger = logging.getLogger(__name__)
Key = Tuple[int, int]


@dataclass(frozen=True)
class EmbeddingConfig:
    device: str = "cpu"
    cuda_devices: str = ""
    ts2vec_ckpt: str | None = None
    cost_ckpt: str | None = None
    timesfm_dir: str | None = None
    chronos2_dir: str | None = None
    timer_dir: str | None = None
    sundial_dir: str | None = None
    moirai_dir: str | None = None
    timemoe_dir: str | None = None
    use_tabpfn: bool = False
    fm_pooling: str = "mean"
    batch_size: int = 64
    allow_proxy_embedding_fallback: bool = True


def _exists(path: str | None) -> bool:
    if not path:
        return False
    item = Path(path)
    if not item.exists():
        return False
    if item.is_dir():
        return any(item.iterdir())
    return True


def _parse_cuda_devices(spec: str) -> list[int]:
    out = []
    for item in str(spec or "").split(","):
        item = item.strip()
        if item:
            try:
                out.append(int(item))
            except ValueError:
                pass
    return out


def build_embedding_banks(
    classes: Dict[int, list[np.ndarray]],
    config: EmbeddingConfig,
) -> Dict[str, Dict[Key, np.ndarray]]:
    banks: Dict[str, Dict[Key, np.ndarray]] = {}
    cuda_ids = _parse_cuda_devices(config.cuda_devices)

    if _exists(config.ts2vec_ckpt):
        try:
            banks["TS2Vec"] = em.precompute_ts2vec_embeddings(
                classes,
                config.ts2vec_ckpt,
                device=config.device,
                device_ids=cuda_ids,
                use_dp=bool(cuda_ids),
            )
        except Exception as exc:
            logger.warning("Skip TS2Vec embeddings: %s", exc)

    if _exists(config.cost_ckpt):
        try:
            from models.CoST.cost import CoST

            t_max = max(len(s) for items in classes.values() for s in items)
            inferred_len = em._infer_cost_max_train_length_from_ckpt(config.cost_ckpt, device="cpu")
            model = CoST(
                input_dims=1,
                kernels=[1, 2, 4, 8, 16, 32, 64, 128],
                alpha=0.0005,
                max_train_length=int(inferred_len or t_max),
                output_dims=320,
                hidden_dims=64,
                depth=10,
                device=config.device,
                batch_size=64,
                lr=0.001,
            )
            model.load(config.cost_ckpt)
            banks["CoST"] = em.precompute_cost_embeddings(classes, model, device=config.device)
        except Exception as exc:
            logger.warning("Skip CoST embeddings: %s", exc)

    if _exists(config.timesfm_dir):
        try:
            banks["TimesFM"] = em.precompute_timesfm_embeddings(
                classes,
                config.timesfm_dir,
                device=config.device,
                batch_size=config.batch_size,
                pooling=config.fm_pooling,
            )
        except Exception as exc:
            logger.warning("Skip TimesFM embeddings: %s", exc)

    if _exists(config.chronos2_dir):
        try:
            banks["Chronos2"] = em.precompute_chronos2_embeddings(
                classes,
                config.chronos2_dir,
                device=config.device,
                batch_size=config.batch_size,
                pooling=config.fm_pooling,
            )
        except Exception as exc:
            logger.warning("Skip Chronos2 embeddings: %s", exc)

    special_models = [
        ("Timer", config.timer_dir, "Timer"),
        ("Sundial", config.sundial_dir, "Sundial"),
        ("TimeMoE", config.timemoe_dir, "Time-MoE"),
    ]
    for bank_name, model_dir, model_tag in special_models:
        if _exists(model_dir):
            try:
                banks[bank_name] = em.precompute_causallm_special_embeddings(
                    classes,
                    model_dir,
                    model_tag=model_tag,
                    device=config.device,
                    batch_size=config.batch_size,
                    pooling=config.fm_pooling,
                    allow_proxy_fallback=config.allow_proxy_embedding_fallback,
                )
            except Exception as exc:
                logger.warning("Skip %s embeddings: %s", bank_name, exc)

    if _exists(config.moirai_dir):
        try:
            banks["Moirai"] = em.precompute_hf_generic_embeddings(
                classes,
                config.moirai_dir,
                model_tag="Moirai",
                device=config.device,
                batch_size=config.batch_size,
                pooling=config.fm_pooling,
            )
        except Exception as exc:
            logger.warning("Skip Moirai embeddings: %s", exc)

    if config.use_tabpfn:
        try:
            banks["TabPFN"] = em.precompute_tabpfn_embeddings(
                classes,
                device=config.device,
                allow_proxy_fallback=config.allow_proxy_embedding_fallback,
            )
        except Exception as exc:
            logger.warning("Skip TabPFN embeddings: %s", exc)

    return banks
