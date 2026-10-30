"""评价指标。"""

from __future__ import annotations

import numpy as np


def corr2(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-15:
        return 0.0
    return float(np.dot(a, b) / denom)


def normalize01(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)


def ssim_simple(x, y) -> float:
    """轻量 SSIM；有 skimage 时优先用它。"""
    try:
        from skimage.metrics import structural_similarity

        return float(
            structural_similarity(
                x, y, data_range=max(x.max() - x.min(), y.max() - y.min(), 1e-12)
            )
        )
    except Exception:
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        c1, c2 = 1e-4, 9e-4
        mx, my = x.mean(), y.mean()
        sx, sy = x.std(), y.std()
        sxy = np.mean((x - mx) * (y - my))
        return float(
            ((2 * mx * my + c1) * (2 * sxy + c2))
            / ((mx**2 + my**2 + c1) * (sx**2 + sy**2 + c2) + 1e-12)
        )


def evaluate_dpc(x_rec, gt) -> dict:
    x_rec = np.asarray(x_rec, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    metrics = {
        "ssim_raw": ssim_simple(x_rec, gt),
        "ssim_norm": ssim_simple(normalize01(x_rec), normalize01(gt)),
        "corr": corr2(x_rec, gt),
    }

    gx = np.gradient(x_rec)
    gg = np.gradient(gt)
    mag_x = np.hypot(gx[0], gx[1])
    mag_g = np.hypot(gg[0], gg[1])
    metrics["grad_ssim"] = ssim_simple(normalize01(mag_x), normalize01(mag_g))
    return metrics
