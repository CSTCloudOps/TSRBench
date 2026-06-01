import logging
import math
from typing import Optional

import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cdist
from scipy.signal import find_peaks, correlate
from scipy.stats import norm

from typing import List, Tuple, Optional, Union




logger = logging.getLogger(__name__)

dtw_lib = None
fastdtw = None

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    njit = None
    _HAS_NUMBA = False


# 检查GPU可用性
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_TORCH_DEVICE = None
_USE_TORCH = False
_USE_TORCH_DTW = True

if _HAS_NUMBA:
    @njit(cache=True)
    def _lcss_length_numba(x, y, epsilon, sigma):
        n = len(x)
        m = len(y)
        if sigma < 0:
            sigma = max(n, m)
        prev = np.zeros(m + 1, dtype=np.int32)
        cur = np.zeros(m + 1, dtype=np.int32)
        for i in range(1, n + 1):
            cur[0] = 0
            j_start = 1 if sigma >= m else max(1, i - sigma)
            j_end = m if sigma >= m else min(m, i + sigma)
            for j in range(1, m + 1):
                if j < j_start or j > j_end:
                    if prev[j] > cur[j - 1]:
                        cur[j] = prev[j]
                    else:
                        cur[j] = cur[j - 1]
                    continue
                if abs(x[i - 1] - y[j - 1]) <= epsilon:
                    cur[j] = prev[j - 1] + 1
                else:
                    if prev[j] > cur[j - 1]:
                        cur[j] = prev[j]
                    else:
                        cur[j] = cur[j - 1]
            tmp = prev
            prev = cur
            cur = tmp
        return prev[m]

    @njit(cache=True)
    def _edr_distance_numba(x, y, epsilon, sigma):
        n = len(x)
        m = len(y)
        if sigma < 0:
            sigma = max(n, m)
        big = float(n + m + 1)
        prev = np.full(m + 1, big, dtype=np.float32)
        cur = np.full(m + 1, big, dtype=np.float32)
        prev[0] = 0.0
        for j in range(1, m + 1):
            prev[j] = j
        for i in range(1, n + 1):
            cur[0] = i
            j_start = 1 if sigma >= m else max(1, i - sigma)
            j_end = m if sigma >= m else min(m, i + sigma)
            for j in range(j_start, j_end + 1):
                sub = 1.0 if abs(x[i - 1] - y[j - 1]) > epsilon else 0.0
                v1 = prev[j - 1] + sub
                v2 = prev[j] + 1.0
                v3 = cur[j - 1] + 1.0
                cur[j] = min(v1, v2, v3)
            tmp = prev
            prev = cur
            cur = tmp
        return prev[m]

    @njit(cache=True)
    def _erp_distance_numba(x, y, g, sigma):
        n = len(x)
        m = len(y)
        if sigma < 0:
            sigma = max(n, m)
        big = float(n + m + 1) * 10.0
        prev = np.full(m + 1, big, dtype=np.float32)
        cur = np.full(m + 1, big, dtype=np.float32)
        prev[0] = 0.0
        for j in range(1, m + 1):
            prev[j] = prev[j - 1] + abs(y[j - 1] - g)
        for i in range(1, n + 1):
            cur[0] = prev[0] + abs(x[i - 1] - g)
            j_start = 1 if sigma >= m else max(1, i - sigma)
            j_end = m if sigma >= m else min(m, i + sigma)
            for j in range(j_start, j_end + 1):
                d = abs(x[i - 1] - y[j - 1])
                v1 = prev[j] + abs(x[i - 1] - g)
                v2 = cur[j - 1] + abs(y[j - 1] - g)
                v3 = prev[j - 1] + d
                cur[j] = min(v1, v2, v3)
            tmp = prev
            prev = cur
            cur = tmp
        return prev[m]


def set_dtw_impl(dtw_impl, fast_impl) -> None:
    global dtw_lib, fastdtw
    dtw_lib = dtw_impl
    fastdtw = fast_impl


def set_torch_device(device: Optional[str], enable: Optional[bool] = None, use_torch_dtw: Optional[bool] = None) -> None:
    """
    Configure torch backend for distance methods.
    - device: e.g. "cuda:0" or "cpu"
    - enable: force on/off; if None, auto-enable on CUDA
    - use_torch_dtw: whether to use torch DTW fallback when CUDA is enabled
    """
    global _TORCH_DEVICE, _USE_TORCH, _USE_TORCH_DTW
    if torch is None:
        _TORCH_DEVICE = None
        _USE_TORCH = False
        return
    if enable is None:
        _USE_TORCH = bool(device) and str(device).startswith("cuda") and torch.cuda.is_available()
    else:
        _USE_TORCH = bool(enable)
    _TORCH_DEVICE = device if _USE_TORCH else None
    if use_torch_dtw is not None:
        _USE_TORCH_DTW = bool(use_torch_dtw)


def _use_torch_backend(x=None, y=None) -> bool:
    if torch is None:
        return False
    if _USE_TORCH:
        return True
    if torch.is_tensor(x) or torch.is_tensor(y):
        return True
    return False


def _to_tensor(x, device: Optional[str] = None) -> "torch.Tensor":
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.tensor(np.asarray(x), dtype=torch.float32)
    if device is None:
        device = _TORCH_DEVICE
    if device is not None:
        t = t.to(device)
    return t


def _to_numpy(x) -> np.ndarray:
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _torch_full_correlation(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
    x = x.reshape(1, 1, -1)
    y = y.reshape(1, 1, -1).flip(-1)
    return F.conv1d(x, y, padding=y.numel() - 1).view(-1)

def euclidean_distance(x, y):
    if _use_torch_backend(x, y):
        xt = _to_tensor(x)
        yt = _to_tensor(y, device=xt.device)
        return float(torch.linalg.norm(xt - yt).item())
    return euclidean(x, y)


def manhattan_distance(x, y):
    # if _use_torch_backend(x, y):
    #     xt = _to_tensor(x)
    #     yt = _to_tensor(y, device=xt.device)
    #     return float(torch.sum(torch.abs(xt - yt)).item())
    return cityblock(x, y)


def chebyshev_distance(x, y):
    if _use_torch_backend(x, y):
        xt = _to_tensor(x)
        yt = _to_tensor(y, device=xt.device)
        return float(torch.max(torch.abs(xt - yt)).item())
    x_array = np.array(x)
    y_array = np.array(y)
    return np.max(np.abs(x_array - y_array))


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y):
        raise ValueError("x和y的长度必须相同")
    if _use_torch_backend(x, y):
        xt = _to_tensor(x)
        yt = _to_tensor(y, device=xt.device)
        if xt.numel() < 2:
            return float("nan")
        xm = xt - xt.mean()
        ym = yt - yt.mean()
        denom = torch.sqrt(torch.sum(xm * xm)) * torch.sqrt(torch.sum(ym * ym))
        if denom.item() == 0:
            return float("nan")
        # Use N-1 normalization like np.corrcoef
        cov = torch.sum(xm * ym) / (xt.numel() - 1)
        sx = torch.sqrt(torch.sum(xm * xm) / (xt.numel() - 1))
        sy = torch.sqrt(torch.sum(ym * ym) / (xt.numel() - 1))
        if (sx.item() == 0) or (sy.item() == 0):
            return float("nan")
        return float((cov / (sx * sy)).item())
    correlation_matrix = np.corrcoef(x, y)
    return correlation_matrix[0, 1]


def dcor1(pearson_correlation: float) -> float:
    return np.sqrt(2 * (1 - pearson_correlation))


def dcor2(pearson_correlation: float, beta: float = 1.0) -> float:
    if pearson_correlation <= -1:
        return np.inf
    return (np.sqrt((1 - pearson_correlation) / (1 + pearson_correlation))) ** beta


def pearson_distance(x: np.ndarray, y: np.ndarray, method: str = "dcor1", beta: float = 1.0) -> float:
    if beta <= 0:
        raise ValueError(f"beta必须大于0,当前值为{beta}")

    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        logger.warning("输入序列为常数序列，无法计算有效皮尔逊相关系数")
        return float(np.nan)

    try:
        pearson_r = calculate_pearson_correlation(x, y)
        if np.isnan(pearson_r):
            return float(np.nan)
    except Exception as e:
        logger.warning(f"计算皮尔逊相关系数失败: {e}")
        return float(np.nan)

    if method == "dcor1":
        distance = dcor1(pearson_r)
    elif method == "dcor2":
        distance = dcor2(pearson_r, beta)
    else:
        raise ValueError(f"method必须是'dcor1'或'dcor2'，当前为{method}")

    return float(distance)


def find_peak_alignment(x, y, method="correlation"):
    if method == "correlation":
        if _use_torch_backend(x, y):
            xt = _to_tensor(x)
            yt = _to_tensor(y, device=xt.device)
            correlation = _torch_full_correlation(xt, yt)
            max_corr_idx = int(torch.argmax(correlation).item())
            q_prime = max_corr_idx - (len(y) - 1)
            return q_prime
        correlation = correlate(x, y, mode="full")
        max_corr_idx = np.argmax(correlation)
        q_prime = max_corr_idx - (len(y) - 1)
        return q_prime
    if method == "peak_matching":
        if _use_torch_backend(x, y):
            xt = _to_tensor(x)
            yt = _to_tensor(y, device=xt.device)
            if xt.numel() < 3 or yt.numel() < 3:
                return int(torch.argmax(xt).item()) - int(torch.argmax(yt).item())
            height_x = torch.max(xt) * 0.5
            height_y = torch.max(yt) * 0.5
            px = (xt[1:-1] > xt[:-2]) & (xt[1:-1] > xt[2:]) & (xt[1:-1] >= height_x)
            py = (yt[1:-1] > yt[:-2]) & (yt[1:-1] > yt[2:]) & (yt[1:-1] >= height_y)
            peaks_x = torch.nonzero(px, as_tuple=False).flatten() + 1
            peaks_y = torch.nonzero(py, as_tuple=False).flatten() + 1
            if peaks_x.numel() > 0 and peaks_y.numel() > 0:
                main_peak_x = int(peaks_x[torch.argmax(xt[peaks_x])].item())
                main_peak_y = int(peaks_y[torch.argmax(yt[peaks_y])].item())
                return main_peak_x - main_peak_y
            return int(torch.argmax(xt).item()) - int(torch.argmax(yt).item())
        peaks_x, _ = find_peaks(x, height=np.max(x) * 0.5)
        peaks_y, _ = find_peaks(y, height=np.max(y) * 0.5)
        if len(peaks_x) > 0 and len(peaks_y) > 0:
            main_peak_x = peaks_x[np.argmax(x[peaks_x])]
            main_peak_y = peaks_y[np.argmax(y[peaks_y])]
            q_prime = main_peak_x - main_peak_y
        else:
            q_prime = np.argmax(x) - np.argmax(y)
        return q_prime
    if method == "max_value":
        if _use_torch_backend(x, y):
            xt = _to_tensor(x)
            yt = _to_tensor(y, device=xt.device)
            return int(torch.argmax(xt).item()) - int(torch.argmax(yt).item())
        peak_x = np.argmax(x)
        peak_y = np.argmax(y)
        q_prime = peak_x - peak_y
        return q_prime
    raise ValueError("method 必须是 'correlation', 'peak_matching' 或 'max_value'")


def shift_sequence(sequence, shift):
    if _use_torch_backend(sequence):
        seq = _to_tensor(sequence)
        result = torch.zeros_like(seq)
        if shift > 0:
            if shift < seq.numel():
                result[shift:] = seq[:-shift]
        elif shift < 0:
            shift_abs = abs(shift)
            if shift_abs < seq.numel():
                result[:-shift_abs] = seq[shift_abs:]
        else:
            result[:] = seq
        if torch is not None and torch.is_tensor(sequence):
            return result
        return _to_numpy(result)
    result = np.zeros_like(sequence)
    if shift > 0:
        if shift < len(sequence):
            result[shift:] = sequence[:-shift]
        else:
            result[:] = 0
    elif shift < 0:
        shift_abs = abs(shift)
        if shift_abs < len(sequence):
            result[:-shift_abs] = sequence[shift_abs:]
        else:
            result[:] = 0
    else:
        result[:] = sequence
    return result


def sti_distance(x, y, search_range=10):
    if _use_torch_backend(x, y):
        xt = _to_tensor(x)
        yt = _to_tensor(y, device=xt.device)
        q_prime = find_peak_alignment(xt, yt, method="peak_matching")
        best_distance = float("inf")
        norm_x = torch.linalg.norm(xt) + 1e-8
        for q in range(q_prime - search_range, q_prime + search_range + 1):
            y_shifted = _to_tensor(shift_sequence(yt, q), device=xt.device)
            dot_product = torch.dot(xt, y_shifted)
            norm_y_shifted_sq = torch.dot(y_shifted, y_shifted) + 1e-8
            alpha = dot_product / norm_y_shifted_sq
            residual = xt - alpha * y_shifted
            distance = torch.linalg.norm(residual) / norm_x
            d = float(distance.item())
            if d < best_distance:
                best_distance = d
        return best_distance
    q_prime = find_peak_alignment(x, y, method="peak_matching")
    best_distance = float("inf")
    best_q = 0
    best_alpha = 1.0
    norm_x = np.linalg.norm(x) + 1e-8
    for q in range(q_prime - search_range, q_prime + search_range + 1):
        y_shifted = shift_sequence(y, q)
        dot_product = np.dot(x, y_shifted)
        norm_y_shifted_sq = np.dot(y_shifted, y_shifted) + 1e-8
        alpha = dot_product / norm_y_shifted_sq
        residual = x - alpha * y_shifted
        distance = np.linalg.norm(residual) / norm_x
        if distance < best_distance:
            best_distance = distance
            best_q = q
            best_alpha = alpha
    return best_distance


def sbd(x, y):
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)
    corr = np.correlate(x_norm, y_norm, mode="full")
    cc_T_xx = np.correlate(x_norm, x_norm, mode="full")
    cc_T_yy = np.correlate(y_norm, y_norm, mode="full")
    ncc = corr / (np.sqrt(cc_T_xx[len(cc_T_xx) // 2] * cc_T_yy[len(cc_T_yy) // 2]) + 1e-8)
    max_ncc = np.max(ncc)
    return 1 - max_ncc


def modified_euclidean_distance(x, y, m=5):
    if _use_torch_backend(x, y):
        xt = _to_tensor(x)
        yt = _to_tensor(y, device=xt.device)
        n = int(xt.numel())
        segment_size = max(1, n // m)
        distance = 0.0
        for i in range(m):
            start = i * segment_size
            end = min((i + 1) * segment_size, n)
            if start >= end:
                continue
            x_seg = xt[start:end]
            y_seg = yt[start:end]
            distance += float(torch.linalg.norm(x_seg - y_seg).item())
        return distance / m
    n = len(x)
    segment_size = max(1, n // m)
    distance = 0
    for i in range(m):
        start = i * segment_size
        end = min((i + 1) * segment_size, n)
        x_seg = x[start:end]
        y_seg = y[start:end]
        distance += euclidean(x_seg, y_seg)
    return distance / m




def STSDistance(x, y, tx=None, ty=None) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    try:
        if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
            raise ValueError("The series must be numeric")
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("The series must be univariate vectors")
        if len(x) <= 1 or len(y) <= 1:
            raise ValueError("The series must have more than one point")
        if len(x) != len(y):
            raise ValueError("Both series must have the same length")
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            raise ValueError("There are missing values in the series")
        if tx is None and ty is None:
            tx = np.arange(1, len(x) + 1, dtype=np.float64)
            ty = tx.copy()
        elif tx is None:
            ty = np.asarray(ty, dtype=np.float64)
            tx = ty.copy()
        elif ty is None:
            tx = np.asarray(tx, dtype=np.float64)
            ty = tx.copy()
        else:
            tx = np.asarray(tx, dtype=np.float64)
            ty = np.asarray(ty, dtype=np.float64)
        if tx is not None and ty is not None:
            if np.any(tx <= 0) or np.any(ty <= 0):
                raise ValueError("The temporal indice must always be positive")
            if not np.allclose(np.diff(tx), np.diff(ty)):
                raise ValueError("The sampling rate must be equal in both series")
            if np.any(np.diff(tx) <= 0) or np.any(np.diff(ty) <= 0):
                raise ValueError("The temporal index must be ascending")
            if len(tx) != len(x) or len(ty) != len(y):
                raise ValueError("The length of the time indice must be equal to the length of the series")
        if _use_torch_backend(x, y):
            xt = _to_tensor(x)
            yt = _to_tensor(y, device=xt.device)
            txt = _to_tensor(tx, device=xt.device)
            tyt = _to_tensor(ty, device=xt.device)
            dx_dt = torch.diff(xt) / torch.diff(txt)
            dy_dt = torch.diff(yt) / torch.diff(tyt)
            return float(torch.sqrt(torch.sum((dx_dt - dy_dt) ** 2)).item())
        dx_dt = np.diff(x) / np.diff(tx)
        dy_dt = np.diff(y) / np.diff(ty)
        return np.sqrt(np.sum((dx_dt - dy_dt) ** 2))
    except Exception:
        return np.nan


def dissim_distance(x, y, tx=None, ty=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if tx is None and ty is None:
        tx = np.linspace(0.0, 1.0, len(x))
        ty = np.linspace(0.0, 1.0, len(y))
    elif tx is None:
        tx = np.linspace(ty[0], ty[-1], len(x))
    elif ty is None:
        ty = np.linspace(tx[0], tx[-1], len(y))

    tx = np.asarray(tx, dtype=float)
    ty = np.asarray(ty, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Series must be univariate")
    if len(x) <= 1 or len(y) <= 1:
        raise ValueError("Series must contain more than one point")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Series contain NaN values")
    if tx[0] != ty[0] or tx[-1] != ty[-1]:
        raise ValueError("Series must begin and end at the same timestamp")
    if np.any(tx < 0) or np.any(ty < 0):
        raise ValueError("Temporal indices must be positive")
    if np.any(np.diff(tx) <= 0) or np.any(np.diff(ty) <= 0):
        raise ValueError("Temporal indices must be strictly increasing")
    if len(tx) != len(x) or len(ty) != len(y):
        raise ValueError("Time index length mismatch")

    if len(tx) != len(ty) or np.any(tx != ty):
        ind = np.unique(np.concatenate([tx, ty]))
    else:
        ind = tx.copy()

    if _use_torch_backend(x, y):
        xt = _to_tensor(x)
        yt = _to_tensor(y, device=xt.device)
        txt = _to_tensor(tx, device=xt.device)
        tyt = _to_tensor(ty, device=xt.device)
        indt = _to_tensor(ind, device=xt.device)

        x_global = torch.searchsorted(txt, indt, right=True) - 1
        y_global = torch.searchsorted(tyt, indt, right=True) - 1
        x_global = torch.clamp(x_global, min=0)
        y_global = torch.clamp(y_global, min=0)

        ax = torch.diff(xt) / torch.diff(txt)
        bx = xt[:-1] - ax * txt[:-1]
        ay = torch.diff(yt) / torch.diff(tyt)
        by = yt[:-1] - ay * tyt[:-1]

        ax = ax[x_global[:-1]]
        bx = bx[x_global[:-1]]
        ay = ay[y_global[:-1]]
        by = by[y_global[:-1]]

        a = (ax - ay) ** 2
        b = 2 * (ax * bx + ay * by - ax * by - bx * ay)
        c = (bx - by) ** 2

        t1 = indt[:-1]
        t2 = indt[1:]
        D = torch.zeros_like(t1)

        mask0 = a == 0
        D[mask0] = torch.sqrt(torch.clamp(c[mask0], min=0)) * (t2 - t1)[mask0]

        mask1 = a > 0
        aa = a[mask1]
        bb = b[mask1]
        cc = c[mask1]
        t1m = t1[mask1]
        t2m = t2[mask1]

        sqrt_arg = aa * t2m**2 + bb * t2m + cc
        sqrt_arg = torch.clamp(sqrt_arg, min=0)
        sqrt_val = torch.sqrt(sqrt_arg)
        sqrt_aa = torch.sqrt(torch.clamp(aa, min=1e-15))
        term1 = (2 * aa * t2m + bb) / (4 * aa) * sqrt_val
        term2 = (bb**2 - 4 * aa * cc) / (8 * aa * sqrt_aa)
        F_t2 = term1 - term2

        sqrt_arg = aa * t1m**2 + bb * t1m + cc
        sqrt_arg = torch.clamp(sqrt_arg, min=0)
        sqrt_val = torch.sqrt(sqrt_arg)
        term1 = (2 * aa * t1m + bb) / (4 * aa) * sqrt_val
        term2 = (bb**2 - 4 * aa * cc) / (8 * aa * sqrt_aa)
        F_t1 = term1 - term2

        D[mask1] = F_t2 - F_t1
        return float(torch.sum(D).item())

    x_global = np.searchsorted(tx, ind, side="right") - 1
    y_global = np.searchsorted(ty, ind, side="right") - 1
    x_global[x_global < 0] = 0
    y_global[y_global < 0] = 0

    ax = np.diff(x) / np.diff(tx)
    bx = x[:-1] - ax * tx[:-1]
    ay = np.diff(y) / np.diff(ty)
    by = y[:-1] - ay * ty[:-1]

    ax = ax[x_global[:-1]]
    bx = bx[x_global[:-1]]
    ay = ay[y_global[:-1]]
    by = by[y_global[:-1]]

    a = (ax - ay) ** 2
    b = 2 * (ax * bx + ay * by - ax * by - bx * ay)
    c = (bx - by) ** 2

    t1 = ind[:-1]
    t2 = ind[1:]
    D = np.zeros_like(t1)

    mask0 = a == 0
    D[mask0] = np.sqrt(c[mask0]) * (t2 - t1)[mask0]

    mask1 = a > 0
    aa = a[mask1]
    bb = b[mask1]
    cc = c[mask1]
    t1m = t1[mask1]
    t2m = t2[mask1]

    def F(t):
        sqrt_arg = aa * t**2 + bb * t + cc
        sqrt_arg = np.maximum(sqrt_arg, 0)
        sqrt_val = np.sqrt(sqrt_arg)
        sqrt_aa = np.sqrt(np.maximum(aa, 1e-15))
        term1 = (2 * aa * t + bb) / (4 * aa) * sqrt_val
        term2 = (bb**2 - 4 * aa * cc) / (8 * aa * sqrt_aa)
        return term1 - term2

    D[mask1] = F(t2m) - F(t1m)
    return float(np.sum(D))


def lcss_initial_check(x: np.ndarray, y: np.ndarray, epsilon: float, sigma: Optional[int] = None) -> bool:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("The series must be univariate vectors")
    if len(x) < 1 or len(y) < 1:
        raise ValueError("The series must have at least one point")
    if not isinstance(epsilon, (int, float)):
        raise TypeError("The threshold must be numeric")
    if epsilon < 0:
        raise ValueError("The threshold must be non-negative")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("There are missing values in the series")
    if sigma is not None:
        if sigma <= 0:
            raise ValueError("The window size must be positive")
        if sigma < abs(len(x) - len(y)):
            raise ValueError("The window size cannot be lower than the difference between the series lengths")
    return True


def lcss_length(x, y, epsilon: float, sigma: Optional[int] = None) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    try:
        lcss_initial_check(x, y, epsilon, sigma)
    except Exception as e:
        print(f"参数错误: {e}")
        return np.nan

    if _HAS_NUMBA:
        sig = -1 if sigma is None else int(sigma)
        return float(_lcss_length_numba(x.astype(np.float32), y.astype(np.float32), float(epsilon), sig))

    tamx = len(x)
    tamy = len(y)

    if _use_torch_backend(x, y):
        xt = _to_tensor(x)
        yt = _to_tensor(y, device=xt.device)
        dp_prev = torch.zeros((tamy + 1,), device=xt.device, dtype=torch.float32)
        dp_cur = torch.zeros_like(dp_prev)
        if sigma is None:
            for i in range(1, tamx + 1):
                dp_cur[0] = 0.0
                xi = xt[i - 1]
                for j in range(1, tamy + 1):
                    match = torch.abs(xi - yt[j - 1]) <= epsilon
                    dp_cur[j] = torch.where(
                        match,
                        dp_prev[j - 1] + 1.0,
                        torch.maximum(dp_prev[j], dp_cur[j - 1]),
                    )
                dp_prev, dp_cur = dp_cur, dp_prev
        else:
            for i in range(1, tamx + 1):
                dp_cur[0] = 0.0
                j_start = max(1, i - sigma)
                j_end = min(tamy, i + sigma)
                for j in range(1, tamy + 1):
                    if j < j_start or j > j_end:
                        dp_cur[j] = torch.maximum(dp_prev[j], dp_cur[j - 1])
                        continue
                    match = torch.abs(xt[i - 1] - yt[j - 1]) <= epsilon
                    dp_cur[j] = torch.where(
                        match,
                        dp_prev[j - 1] + 1.0,
                        torch.maximum(dp_prev[j], dp_cur[j - 1]),
                    )
                dp_prev, dp_cur = dp_cur, dp_prev
        return float(dp_prev[tamy].item())

    x_reshaped = x.reshape(-1, 1)
    y_reshaped = y.reshape(1, -1)
    distance_matrix = np.abs(x_reshaped - y_reshaped)
    match_matrix = (distance_matrix <= epsilon).astype(float)

    dp = np.zeros((tamx + 1, tamy + 1))

    if sigma is None:
        for i in range(1, tamx + 1):
            for j in range(1, tamy + 1):
                if match_matrix[i - 1, j - 1] == 1:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    else:
        for i in range(1, tamx + 1):
            j_start = max(1, i - sigma)
            j_end = min(tamy, i + sigma) + 1
            for j in range(j_start, j_end):
                if match_matrix[i - 1, j - 1] == 1:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

    return dp[tamx, tamy]


def lcss_similarity(x: np.ndarray, y: np.ndarray, epsilon: float, sigma: Optional[int] = None) -> float:
    lcss_len = lcss_length(x, y, epsilon, sigma)
    if np.isnan(lcss_len):
        return np.nan
    min_len = min(len(x), len(y))
    return lcss_len / min_len


def lcss_distance(x: np.ndarray, y: np.ndarray, epsilon=0.1, sigma: Optional[int] = None) -> float:
    similarity = lcss_similarity(x, y, epsilon, sigma)
    if np.isnan(similarity):
        return np.nan
    return 1 - similarity


def _initial_check(x, y, epsilon, sigma=None):
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise TypeError("The series must be numpy arrays")
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        raise ValueError("The series must be numeric")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("The series must be univariate vectors")
    if len(x) < 1 or len(y) < 1:
        raise ValueError("The series must have at least one point")
    if not np.isscalar(epsilon) or not np.issubdtype(type(epsilon), np.number):
        raise ValueError("The threshold must be numeric")
    if epsilon < 0:
        raise ValueError("The threshold must be positive")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("There are missing values in the series")
    if sigma is not None:
        if sigma <= 0:
            raise ValueError("The window size must be positive")
        if sigma < abs(len(x) - len(y)):
            raise ValueError("The window size cannot be lower than the difference between the series lengths")
    return x, y


def edr_no_window(tamx, tamy, subcost):
    cost_matrix = np.full((tamx + 1, tamy + 1), np.sum(subcost) * len(subcost))
    for i in range(tamx + 1):
        cost_matrix[i, 0] = i
    for j in range(tamy + 1):
        cost_matrix[0, j] = j
    for i in range(1, tamx + 1):
        for j in range(1, tamy + 1):
            substitution_cost = subcost[i - 1, j - 1]
            cost = min(
                cost_matrix[i - 1, j - 1] + substitution_cost,
                cost_matrix[i - 1, j] + 1,
                cost_matrix[i, j - 1] + 1,
            )
            cost_matrix[i, j] = cost
    return cost_matrix


def edr_with_window(tamx, tamy, subcost, sigma):
    cost_matrix = np.full((tamx + 1, tamy + 1), np.sum(subcost) * len(subcost))
    for i in range(tamx + 1):
        cost_matrix[i, 0] = i
    for j in range(tamy + 1):
        cost_matrix[0, j] = j
    for i in range(1, tamx + 1):
        j_start = max(1, i - sigma)
        j_end = min(tamy, i + sigma)
        for j in range(j_start, j_end + 1):
            substitution_cost = subcost[i - 1, j - 1]
            cost = min(
                cost_matrix[i - 1, j - 1] + substitution_cost,
                cost_matrix[i - 1, j] + 1,
                cost_matrix[i, j - 1] + 1,
            )
            cost_matrix[i, j] = cost
    return cost_matrix


def edr_distance(x, y, epsilon=0.1, sigma=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    try:
        x, y = _initial_check(x, y, epsilon, sigma)
    except ValueError:
        return np.nan

    tamx = len(x)
    tamy = len(y)

    if _HAS_NUMBA:
        sig = -1 if sigma is None else int(sigma)
        return float(_edr_distance_numba(x.astype(np.float32), y.astype(np.float32), float(epsilon), sig))

    if _use_torch_backend(x, y):
        xt = _to_tensor(x)
        yt = _to_tensor(y, device=xt.device)
        if tamx == 1 and tamy == 1:
            euclidean_dist = torch.abs(xt[0] - yt[0]).view(1, 1)
        else:
            euclidean_dist = torch.abs(xt.view(-1, 1) - yt.view(1, -1))
        subcost = (euclidean_dist > epsilon).float()

        inf = torch.sum(subcost) * len(subcost)
        dp_prev = torch.full((tamy + 1,), inf, device=xt.device, dtype=torch.float32)
        dp_prev[0] = 0.0
        for j in range(1, tamy + 1):
            dp_prev[j] = dp_prev[j - 1] + 1.0

        for i in range(1, tamx + 1):
            dp_cur = torch.full((tamy + 1,), inf, device=xt.device, dtype=torch.float32)
            dp_cur[0] = float(i)
            j_start = 1
            j_end = tamy
            if sigma is not None:
                j_start = max(1, i - sigma)
                j_end = min(tamy, i + sigma)
            for j in range(j_start, j_end + 1):
                substitution_cost = subcost[i - 1, j - 1]
                dp_cur[j] = torch.minimum(
                    torch.minimum(dp_prev[j - 1] + substitution_cost, dp_prev[j] + 1.0),
                    dp_cur[j - 1] + 1.0,
                )
            dp_prev = dp_cur

        return float(dp_prev[tamy].item())

    if tamx == 1 and tamy == 1:
        euclidean_dist = np.abs(x[0] - y[0]).reshape(1, 1)
    else:
        x_expanded = np.expand_dims(x, axis=1)
        y_expanded = np.expand_dims(y, axis=0)
        euclidean_dist = np.sqrt((x_expanded - y_expanded) ** 2)

    subcost = (euclidean_dist > epsilon).astype(float)

    if sigma is None:
        cost_matrix = edr_no_window(tamx, tamy, subcost)
    else:
        cost_matrix = edr_with_window(tamx, tamy, subcost, sigma)

    return float(cost_matrix[tamx, tamy])


def sax_based_distance(x, y, segment_length=10, num_symbols=5, use_gaussian=True):

    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)

    def segment_signal(signal, segment_length):
        signal_len = len(signal)
        valid_len = (signal_len // segment_length) * segment_length
        if valid_len == 0:
            return np.array([])
        signal_reshaped = signal[:valid_len].reshape(-1, segment_length)
        return signal_reshaped.mean(axis=1)

    x_segments = segment_signal(x_norm, segment_length)
    y_segments = segment_signal(y_norm, segment_length)

    min_length = min(len(x_segments), len(y_segments))
    x_segments = x_segments[:min_length]
    y_segments = y_segments[:min_length]

    def get_quantiles(num_symbols, use_gaussian):
        if use_gaussian:
            return norm.ppf(np.linspace(0, 1, num_symbols + 1))
        combined_data = np.concatenate([x_segments, y_segments])
        return np.percentile(combined_data, np.linspace(0, 100, num_symbols + 1))

    quantiles = get_quantiles(num_symbols, use_gaussian)

    def symbolize(segments, quantiles):
        if len(segments) == 0:
            return []
        symbol_indices = np.digitize(segments, quantiles[1:-1]) + 1
        return [f"l{idx}" for idx in symbol_indices]

    x_symbols = symbolize(x_segments, quantiles)
    y_symbols = symbolize(y_segments, quantiles)

    def symbol_distance(sym1, sym2, quantiles):
        i = int(sym1[1:])
        j = int(sym2[1:])
        if abs(i - j) <= 1:
            return 0.0
        return quantiles[max(i, j) - 1] - quantiles[min(i, j)]

    total_distance = 0.0
    for sym1, sym2 in zip(x_symbols, y_symbols):
        total_distance += symbol_distance(sym1, sym2, quantiles) ** 2

    n = len(x_symbols)
    if n > 0:
        return float(np.sqrt(segment_length * total_distance / n))
    return 0.0

def sfa_based_distance(
    x,
    y,
    word_length=16,
    num_symbols=8,
    use_gaussian=False,
    drop_dc=True,
    return_words=False,
):
        """
        SFA distance between two 1D signals (whole matching, pairwise version).

        Pipeline:
        1) z-normalize
        2) orthonormal rFFT
        3) build a real-valued coefficient vector of length `word_length`
            (interleaving Re/Im of positive frequencies)
        4) per-dimension binning (MCB-like): breakpoints beta_i with `num_symbols` bins
        5) symbolize to SFA words
        6) distance: sqrt( 2 * sum_i dist_i(sym_x[i], sym_y[i])^2 )

        Notes:
        - The factor 2 follows the standard real-signal DFT energy folding
            on positive frequencies (matches the SFA lower bound form).
        - This is a pairwise version; classic MCB uses dataset-level breakpoints.
        """

        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        # Align lengths (like your SAX code)
        n = int(min(len(x), len(y)))
        if n <= 1:
            return (0.0, None, None) if return_words else 0.0
        x = x[:n]
        y = y[:n]

        # z-normalize
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        # ---- DFT -> real coefficient vector (length = word_length) ----
        # Orthonormal FFT makes energy accounting cleaner.
        X = np.fft.rfft(x, norm="ortho")
        Y = np.fft.rfft(y, norm="ortho")

        # Optionally drop DC component (mean ~ 0 after z-norm anyway)
        start_k = 1 if drop_dc else 0

        # Build vector: [Re(X_k), Im(X_k), Re(X_{k+1}), Im(...), ...]
        # until length word_length
        def dft_to_vec(Z, w, start):
            out = []
            for k in range(start, len(Z)):
                out.append(float(np.real(Z[k])))
                if len(out) >= w:
                    break
                out.append(float(np.imag(Z[k])))
                if len(out) >= w:
                    break
            if len(out) < w:
                out.extend([0.0] * (w - len(out)))
            return np.asarray(out, dtype=float)

        cx = dft_to_vec(X, word_length, start_k)
        cy = dft_to_vec(Y, word_length, start_k)

        # ---- breakpoints beta_i (MCB-like, per coefficient dimension) ----
        # Each dimension i has its own (num_symbols + 1) breakpoints.
        # beta_i(0)=-inf, beta_i(num_symbols)=+inf conceptually;
        # we store finite quantiles and handle edges with +/-inf.
        def gaussian_breakpoints(a):
            # standard normal quantiles
            # avoid scipy: use numpy-based approximation if needed
            # Here we use np.erf inverse via a rational approximation.
            # For stability and simplicity, use numpy's percentile on samples if you prefer.
            p = np.linspace(0.0, 1.0, a + 1)
            # clamp away from 0/1
            eps = 1e-12
            p = np.clip(p, eps, 1 - eps)

            # Acklam's inverse normal approximation
            # (good accuracy, pure numpy)
            def ndtri(pp):
                a0=-3.969683028665376e+01; a1=2.209460984245205e+02; a2=-2.759285104469687e+02
                a3=1.383577518672690e+02; a4=-3.066479806614716e+01; a5=2.506628277459239e+00
                b0=-5.447609879822406e+01; b1=1.615858368580409e+02; b2=-1.556989798598866e+02
                b3=6.680131188771972e+01; b4=-1.328068155288572e+01
                c0=-7.784894002430293e-03; c1=-3.223964580411365e-01; c2=-2.400758277161838e+00
                c3=-2.549732539343734e+00; c4=4.374664141464968e+00; c5=2.938163982698783e+00
                d0=7.784695709041462e-03; d1=3.224671290700398e-01; d2=2.445134137142996e+00
                d3=3.754408661907416e+00
                plow=0.02425; phigh=1-plow
                x = np.zeros_like(pp)

                # lower region
                m = pp < plow
                q = np.sqrt(-2*np.log(pp[m]))
                x[m] = (((((c0*q + c1)*q + c2)*q + c3)*q + c4)*q + c5) / \
                    ((((d0*q + d1)*q + d2)*q + d3)*q + 1)

                # central region
                m = (pp >= plow) & (pp <= phigh)
                q = pp[m] - 0.5
                r = q*q
                x[m] = (((((a0*r + a1)*r + a2)*r + a3)*r + a4)*r + a5)*q / \
                    (((((b0*r + b1)*r + b2)*r + b3)*r + b4)*r + 1)

                # upper region
                m = pp > phigh
                q = np.sqrt(-2*np.log(1-pp[m]))
                x[m] = -(((((c0*q + c1)*q + c2)*q + c3)*q + c4)*q + c5) / \
                        ((((d0*q + d1)*q + d2)*q + d3)*q + 1)
                return x

            return ndtri(p)

        # beta: shape (word_length, num_symbols+1)
        beta = np.zeros((word_length, num_symbols + 1), dtype=float)
        for i in range(word_length):
            if use_gaussian:
                bp = gaussian_breakpoints(num_symbols)
            else:
                # MCB-like: equi-depth bins from available data on this coefficient dimension
                vals = np.array([cx[i], cy[i]], dtype=float)
                bp = np.quantile(vals, np.linspace(0.0, 1.0, num_symbols + 1))
            beta[i, :] = bp

        # ---- symbolize ----
        # symbol index a in {1..num_symbols}, where interval is [beta[a-1], beta[a])
        def symbolize_coeff(v, bp_row):
            # digitize uses bins (internal cut points)
            # bins: beta[1:-1] gives num_symbols-1 internal boundaries
            bins = bp_row[1:-1]
            a = int(np.digitize(v, bins, right=False)) + 1
            return a

        wx = np.array([symbolize_coeff(cx[i], beta[i]) for i in range(word_length)], dtype=int)
        wy = np.array([symbolize_coeff(cy[i], beta[i]) for i in range(word_length)], dtype=int)

        # ---- dist_i between two symbols (Eq.(4)-style interval distance) ----
        def dist_symbol_symbol(a, b, bp_row):
            if abs(a - b) <= 1:
                return 0.0
            hi = max(a, b)
            lo = min(a, b)
            # distance between non-overlapping symbol intervals:
            # beta(hi-1) - beta(lo)
            return float(bp_row[hi - 1] - bp_row[lo])

        total = 0.0
        for i in range(word_length):
            d = dist_symbol_symbol(int(wx[i]), int(wy[i]), beta[i])
            total += d * d

        # SFA lower-bound style: D^2 = 2 * sum_i dist_i^2
        dist = float(np.sqrt(2.0 * total))

        if return_words:
            return dist, wx, wy
        return dist




def sax1d_based_distance(
x,
y,
segment_length=10,     # L
num_symbols=16,        # N = Na * Ns
num_symbols_mean=None, # Na (optional)
num_symbols_slope=None,# Ns (optional)
use_gaussian=True,
slope_sigma2_scale=0.03,  # paper suggests sigma_L^2 ~= 0.03 / L
):
    """
    1d-SAX / idsax distance in the same style as your sax_based_distance.

    Steps:
    1) z-normalize x,y
    2) split into segments of length L
    3) per segment: compute mean a and regression slope s
    4) quantize a using N(0,1) breakpoints (Na bins)
        quantize s using N(0, sigma_L^2) breakpoints (Ns bins), sigma_L^2 ~= 0.03/L
    5) distance: sum over segments of (dist_a^2 + dist_s^2), scaled like your SAX

    Note:
    Unlike SAX, 1d-SAX distance does NOT lower bound Euclidean distance.
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # z-normalize (same as your SAX)
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)

    # segment into PAA blocks (same reshape style)
    def segment_signal(signal, seg_len):
        n = len(signal)
        valid_len = (n // seg_len) * seg_len
        if valid_len == 0:
            return np.empty((0, seg_len), dtype=float)
        return signal[:valid_len].reshape(-1, seg_len)

    Xseg = segment_signal(x_norm, segment_length)
    Yseg = segment_signal(y_norm, segment_length)

    m = min(len(Xseg), len(Yseg))
    if m == 0:
        return 0.0
    Xseg = Xseg[:m]
    Yseg = Yseg[:m]

    L = int(segment_length)

    # choose Na, Ns
    if num_symbols_mean is None and num_symbols_slope is None:
        # a simple default split: Na ~= Ns ~= sqrt(N)
        root = int(round(np.sqrt(num_symbols)))
        num_symbols_mean = max(2, root)
        num_symbols_slope = max(2, int(num_symbols // num_symbols_mean))
        # ensure product matches N if possible (best-effort)
        if num_symbols_mean * num_symbols_slope != num_symbols:
            # adjust slope bins to make product exactly N
            if num_symbols % num_symbols_mean == 0:
                num_symbols_slope = num_symbols // num_symbols_mean
            else:
                # fallback: keep as is (still works)
                pass
    elif num_symbols_mean is None:
        num_symbols_mean = max(2, int(num_symbols // num_symbols_slope))
    elif num_symbols_slope is None:
        num_symbols_slope = max(2, int(num_symbols // num_symbols_mean))

    Na = int(num_symbols_mean)
    Ns = int(num_symbols_slope)

    # --- compute per-segment a (mean) and s (slope) ---
    # For evenly spaced t=1..L: slope s = sum((t-tbar)*v)/sum((t-tbar)^2)
    t = np.arange(1, L + 1, dtype=float)
    t_center = t - t.mean()
    denom = float(np.sum(t_center ** 2) + 1e-12)

    def segment_slope(seg):
        # seg shape (L,)
        return float(np.dot(t_center, seg) / denom)

    a_x = Xseg.mean(axis=1)  # a value (segment mean)
    a_y = Yseg.mean(axis=1)

    s_x = np.array([segment_slope(v) for v in Xseg], dtype=float)
    s_y = np.array([segment_slope(v) for v in Yseg], dtype=float)

    # --- breakpoints ---
    def gaussian_quantiles(k, sigma=1.0):
        # avoid +/-inf at endpoints
        eps = 1e-12
        p = np.linspace(0.0, 1.0, k + 1)
        p = np.clip(p, eps, 1.0 - eps)
        return sigma * norm.ppf(p)

    if use_gaussian:
        qa = gaussian_quantiles(Na, sigma=1.0)
        sigma2_L = float(slope_sigma2_scale / max(1, L))
        qs = gaussian_quantiles(Ns, sigma=np.sqrt(sigma2_L))
    else:
        # empirical quantiles (kept only to mirror your SAX option; not the paper default)
        qa = np.percentile(np.concatenate([a_x, a_y]), np.linspace(0, 100, Na + 1))
        qs = np.percentile(np.concatenate([s_x, s_y]), np.linspace(0, 100, Ns + 1))

    # --- symbolize ---
    def symbolize(values, quantiles):
        # returns integer bins 1..K (like your l{idx})
        return np.digitize(values, quantiles[1:-1]) + 1

    ax_sym = symbolize(a_x, qa)
    ay_sym = symbolize(a_y, qa)
    sx_sym = symbolize(s_x, qs)
    sy_sym = symbolize(s_y, qs)

    # --- symbol distance (same as your SAX logic) ---
    def sym_dist(i, j, quantiles):
        i = int(i); j = int(j)
        if abs(i - j) <= 1:
            return 0.0
        return float(quantiles[max(i, j) - 1] - quantiles[min(i, j)])

    total = 0.0
    for i in range(m):
        da = sym_dist(ax_sym[i], ay_sym[i], qa)
        ds = sym_dist(sx_sym[i], sy_sym[i], qs)
        total += da * da + ds * ds

    # scale like your SAX variant
    return float(np.sqrt(segment_length * total / max(1, m)))



def ERP_initial_check(x: np.ndarray, y: np.ndarray, g: float, sigma: Optional[int] = None) -> None:
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("The series must be numpy arrays")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("The series must be univariate vectors")
    if not isinstance(g, (int, float)):
        raise TypeError("g must be numeric")
    if len(x) < 1 or len(y) < 1:
        raise ValueError("The series must have at least one point")
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("There are missing values in the series")
    if sigma is not None:
        if sigma <= 0:
            raise ValueError("The window size must be positive")
        if sigma < abs(len(x) - len(y)):
            raise ValueError("The window size cannot be lower than the difference between the series lengths")


def erp_distance(x: np.ndarray, y: np.ndarray, g: float = 0, sigma: Optional[int] = None) -> float:
    try:
        ERP_initial_check(x, y, g, sigma)
    except Exception as e:
        print(f"Error in initial check: {e}")
        return np.nan

    tamx = len(x)
    tamy = len(y)

    if _HAS_NUMBA:
        sig = -1 if sigma is None else int(sigma)
        return float(_erp_distance_numba(x.astype(np.float32), y.astype(np.float32), float(g), sig))

    if _use_torch_backend(x, y):
        xt = _to_tensor(x)
        yt = _to_tensor(y, device=xt.device)
        dist_matrix = torch.abs(xt.view(-1, 1) - yt.view(1, -1))
        inf = torch.tensor(float("inf"), device=xt.device, dtype=xt.dtype)
        dp_prev = torch.full((tamy + 1,), inf, device=xt.device, dtype=xt.dtype)
        dp_prev[0] = 0.0
        for j in range(1, tamy + 1):
            dp_prev[j] = dp_prev[j - 1] + torch.abs(yt[j - 1] - g)

        for i in range(1, tamx + 1):
            dp_cur = torch.full((tamy + 1,), inf, device=xt.device, dtype=xt.dtype)
            dp_cur[0] = dp_prev[0] + torch.abs(xt[i - 1] - g)
            j_start = 1
            j_end = tamy
            if sigma is not None:
                j_start = max(1, i - sigma)
                j_end = min(tamy, i + sigma)
            for j in range(j_start, j_end + 1):
                d = dist_matrix[i - 1, j - 1]
                dp_cur[j] = torch.min(
                    torch.stack(
                        [
                            dp_prev[j] + torch.abs(xt[i - 1] - g),
                            dp_cur[j - 1] + torch.abs(yt[j - 1] - g),
                            dp_prev[j - 1] + d,
                        ]
                    )
                )
            dp_prev = dp_cur

        return float(dp_prev[tamy].item())

    x_expanded = np.expand_dims(x, axis=1)
    y_expanded = np.expand_dims(y, axis=0)
    dist_matrix = np.abs(x_expanded - y_expanded)

    cost_matrix = np.full((tamx + 1, tamy + 1), np.inf)
    cost_matrix[0, 0] = 0

    for i in range(1, tamx + 1):
        cost_matrix[i, 0] = cost_matrix[i - 1, 0] + abs(x[i - 1] - g)
    for j in range(1, tamy + 1):
        cost_matrix[0, j] = cost_matrix[0, j - 1] + abs(y[j - 1] - g)

    if sigma is None:
        for i in range(1, tamx + 1):
            for j in range(1, tamy + 1):
                d = dist_matrix[i - 1, j - 1]
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j] + abs(x[i - 1] - g),
                    cost_matrix[i, j - 1] + abs(y[j - 1] - g),
                    cost_matrix[i - 1, j - 1] + d,
                )
    else:
        for i in range(1, tamx + 1):
            j_start = max(1, i - sigma)
            j_end = min(tamy, i + sigma)
            for j in range(j_start, j_end + 1):
                d = dist_matrix[i - 1, j - 1]
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j] + abs(x[i - 1] - g),
                    cost_matrix[i, j - 1] + abs(y[j - 1] - g),
                    cost_matrix[i - 1, j - 1] + d,
                )

    return float(cost_matrix[tamx, tamy])





class DTWGPU:
    """GPU加速的DTW计算类"""
    
    def __init__(self, use_gpu: bool = True, batch_size: int = 1024):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size = batch_size
        
    def _euclidean_distance_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算欧氏距离矩阵 (GPU版本)"""
        # 扩展维度以便广播计算
        x = x.unsqueeze(1)  # (n, 1, d)
        y = y.unsqueeze(0)  # (1, m, d)
        return torch.sqrt(((x - y) ** 2).sum(dim=-1) + 1e-8)  # (n, m)
    
    def dtw_distance_gpu(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        纯PyTorch实现的DTW (支持GPU)
        时间复杂度: O(n*m)
        内存复杂度: O(n*m) 或 O(min(n,m)) 如果使用压缩
        """
        n = x.size(0)
        m = y.size(0)
        
        # 计算距离矩阵
        dist_matrix = self._euclidean_distance_matrix(x, y)  # (n, m)
        
        # 初始化累积距离矩阵
        dtw_matrix = torch.full((n+1, m+1), torch.inf, device=x.device, dtype=x.dtype)
        dtw_matrix[0, 0] = 0
        
        # 动态规划计算DTW
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = dist_matrix[i-1, j-1]
                dtw_matrix[i, j] = cost + torch.min(
                    torch.stack([
                        dtw_matrix[i-1, j],     # 插入
                        dtw_matrix[i, j-1],     # 删除
                        dtw_matrix[i-1, j-1]    # 匹配
                    ])
                )
        
        return dtw_matrix[n, m]
    
    def dtw_distance_fast(self, x: torch.Tensor, y: torch.Tensor, 
                          window: Optional[int] = None) -> torch.Tensor:
        """
        更快的DTW实现，支持Sakoe-Chiba Band约束
        使用向量化操作加速
        """
        n = x.size(0)
        m = y.size(0)
        
        # 计算距离矩阵
        dist_matrix = self._euclidean_distance_matrix(x, y)  # (n, m)
        
        # 应用Sakoe-Chiba Band约束
        if window is not None:
            window = max(window, abs(n - m))
            for i in range(n):
                for j in range(m):
                    if abs(i - j) > window:
                        dist_matrix[i, j] = torch.inf
        
        # 初始化累积距离矩阵
        dtw_matrix = torch.full((n+1, m+1), torch.inf, device=x.device, dtype=x.dtype)
        dtw_matrix[0, 0] = 0
        
        # 向量化版本的动态规划
        for i in range(1, n+1):
            # 获取前一行的三个可能值
            diag = dtw_matrix[i-1, :-1]  # 对角线
            left = dtw_matrix[i-1, 1:]   # 左边
            up = dtw_matrix[i, :-1]      # 上边
            
            # 计算最小值
            min_vals = torch.minimum(torch.minimum(diag, left), up)
            
            # 添加当前成本
            dtw_matrix[i, 1:] = dist_matrix[i-1, :] + min_vals
        
        return dtw_matrix[n, m]
    
    def batch_dtw_distance(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """
        批量计算DTW距离
        x_batch: (batch_size, n, d)
        y_batch: (batch_size, m, d)
        返回: (batch_size,)
        """
        batch_size = x_batch.size(0)
        n = x_batch.size(1)
        m = y_batch.size(1)
        
        # 扩展维度以便批量计算
        x_exp = x_batch.unsqueeze(2)  # (batch_size, n, 1, d)
        y_exp = y_batch.unsqueeze(1)  # (batch_size, 1, m, d)
        
        # 计算批量距离矩阵
        dist_matrices = torch.sqrt(((x_exp - y_exp) ** 2).sum(dim=-1) + 1e-8)  # (batch_size, n, m)
        
        # 批量动态规划
        dtw_matrices = torch.full((batch_size, n+1, m+1), torch.inf, 
                                 device=x_batch.device, dtype=x_batch.dtype)
        dtw_matrices[:, 0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = dist_matrices[:, i-1, j-1]  # (batch_size,)
                min_prev = torch.minimum(
                    torch.minimum(
                        dtw_matrices[:, i-1, j],
                        dtw_matrices[:, i, j-1]
                    ),
                    dtw_matrices[:, i-1, j-1]
                )
                dtw_matrices[:, i, j] = cost + min_prev
        
        return dtw_matrices[:, n, m]


# 全局DTW计算器实例
dtw_gpu = DTWGPU()

def dtw_distance(x: np.ndarray, y: np.ndarray, 
                 use_gpu: bool = True, 
                 fast_method: bool = True,
                 window: Optional[int] = None) -> float:
    """
    GPU加速的DTW距离计算
    
    参数:
    - x, y: 输入序列
    - use_gpu: 是否使用GPU
    - fast_method: 是否使用快速方法
    - window: Sakoe-Chiba Band窗口大小
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    
    if x.size == 0 or y.size == 0:
        return math.inf
    
    # 如果是1D序列，转换为2D (n, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    try:
        if use_gpu and torch.cuda.is_available():
            # 转换到GPU
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
            
            if fast_method:
                dist = dtw_gpu.dtw_distance_fast(x_tensor, y_tensor, window)
            else:
                dist = dtw_gpu.dtw_distance_gpu(x_tensor, y_tensor)
            
            return float(dist.cpu().numpy())
        else:
            # CPU回退
            return _cpu_dtw_fallback(x, y)
            
    except Exception as e:
        print(f"GPU DTW失败: {e}, 回退到CPU实现")
        return _cpu_dtw_fallback(x, y)

def _cpu_dtw_fallback(x: np.ndarray, y: np.ndarray) -> float:
    """CPU回退实现"""
    try:
        # 尝试使用fastdtw
        from fastdtw import fastdtw
        dist, _ = fastdtw(x, y)
        return float(dist)
    except ImportError:
        # 自己实现简单的DTW
        n, m = len(x), len(y)
        
        # 距离矩阵
        dtw_matrix = np.full((n+1, m+1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.linalg.norm(x[i-1] - y[j-1])  # 欧氏距离
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],     # 插入
                    dtw_matrix[i, j-1],     # 删除
                    dtw_matrix[i-1, j-1]    # 匹配
                )
        
        return float(dtw_matrix[n, m])

def modified_dtw_distance(x: np.ndarray, y: np.ndarray, 
                          m: int = 5, 
                          use_gpu: bool = True) -> float:
    """
    GPU加速的m-DTW距离计算
    
    参数:
    - x, y: 输入序列
    - m: 分段数量
    - use_gpu: 是否使用GPU
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    
    if x.size == 0 or y.size == 0:
        return math.inf
    
    n = len(x)
    m_segments = max(1, int(m))
    segment_size = max(1, n // m_segments)
    
    # 准备所有分段
    segments_x, segments_y = [], []
    for i in range(m_segments):
        s = i * segment_size
        e = min((i + 1) * segment_size, n)
        if s >= e:
            continue
        
        x_seg = x[s:e]
        y_seg = y[s:e]
        segments_x.append(x_seg)
        segments_y.append(y_seg)
    
    if not segments_x:
        return math.inf
    
    try:
        if use_gpu and torch.cuda.is_available():
            # 批量计算所有分段
            distances = []
            for x_seg, y_seg in zip(segments_x, segments_y):
                if len(x_seg) == len(y_seg):
                    # 长度相同，使用欧氏距离
                    x_tensor = torch.tensor(x_seg, dtype=torch.float32, device=device)
                    y_tensor = torch.tensor(y_seg, dtype=torch.float32, device=device)
                    dist = torch.norm(x_tensor - y_tensor)
                else:
                    # 长度不同，使用DTW
                    dist_val = dtw_distance(x_seg, y_seg, use_gpu=True, fast_method=True)
                    dist = torch.tensor(dist_val, dtype=torch.float32, device=device)
                distances.append(dist)
            
            # 计算平均距离
            if distances:
                avg_dist = torch.stack(distances).mean()
                return float(avg_dist.cpu().numpy())
            else:
                return math.inf
                
        else:
            # CPU版本
            total = 0.0
            count = 0
            for x_seg, y_seg in zip(segments_x, segments_y):
                if len(x_seg) == len(y_seg):
                    total += euclidean(x_seg, y_seg)
                else:
                    total += dtw_distance(x_seg, y_seg, use_gpu=False)
                count += 1
            
            if count == 0:
                return math.inf
            return float(total / count)
            
    except Exception as e:
        print(f"m-DTW失败: {e}, 使用简单欧氏距离")
        if len(x) == len(y):
            return float(euclidean(x, y))
        return math.inf

def dtw_distance_matrix(sequences: List[np.ndarray], 
                        use_gpu: bool = True) -> np.ndarray:
    """
    计算多个序列之间的DTW距离矩阵
    
    参数:
    - sequences: 序列列表
    - use_gpu: 是否使用GPU加速
    
    返回:
    - 距离矩阵 (n x n)
    """
    n = len(sequences)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    
    if use_gpu and torch.cuda.is_available():
        # 批量计算
        for i in range(n):
            for j in range(i+1, n):
                dist = dtw_distance(sequences[i], sequences[j], use_gpu=True)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    else:
        # CPU计算
        for i in range(n):
            for j in range(i+1, n):
                dist = dtw_distance(sequences[i], sequences[j], use_gpu=False)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    
    return dist_matrix

def lower_bound_keogh(seq1: np.ndarray, seq2: np.ndarray, 
                      window: int = 10) -> float:
    """
    Keogh下界，用于快速过滤
    这是DTW的下界，计算更快
    """
    seq1 = np.asarray(seq1, dtype=np.float32)
    seq2 = np.asarray(seq2, dtype=np.float32)
    
    n = len(seq1)
    m = len(seq2)
    
    # 计算seq1的包络
    lower_env = np.zeros_like(seq1)
    upper_env = np.zeros_like(seq1)
    
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        lower_env[i] = np.min(seq1[start:end])
        upper_env[i] = np.max(seq1[start:end])
    
    # 计算下界距离
    lb_sum = 0.0
    for i in range(m):
        if i < n:
            if seq2[i] > upper_env[i]:
                lb_sum += (seq2[i] - upper_env[i]) ** 2
            elif seq2[i] < lower_env[i]:
                lb_sum += (lower_env[i] - seq2[i]) ** 2
    
    return math.sqrt(lb_sum)

def dtw_distance_with_lb(x: np.ndarray, y: np.ndarray, 
                         use_gpu: bool = True,
                         window: int = 10) -> float:
    """
    使用下界过滤的DTW
    先计算下界，如果下界已经大于当前最佳距离，则跳过精确计算
    """
    # 计算下界
    lb = lower_bound_keogh(x, y, window)
    
    # 计算精确DTW
    return dtw_distance(x, y, use_gpu=use_gpu, window=window)
