"""Tikhonov 三方向 DPC 重建。"""

from __future__ import annotations

import numpy as np


def prepare_dpc_contrast(images, mode: str = "mean_norm"):
    """
    从三张强度图得到差分对比度。

    mode:
      - 'mean_norm': I/mean(I) - 1（实验校准脚本用法）
      - 'cdpc': (2*Ia - Ib - Ic) / sum（tv_main 最终用法）
      - 'bg_cdpc': 背景校正后再做 2Ia-Ib-Ic（无再除 I_sum）
    """
    i1, i2, i3 = [np.asarray(x, dtype=np.float64) for x in images]

    if mode == "mean_norm":
        d1 = i1 / (np.mean(i1) + 1e-12) - 1.0
        d2 = i2 / (np.mean(i2) + 1e-12) - 1.0
        d3 = i3 / (np.mean(i3) + 1e-12) - 1.0
        return d1, d2, d3

    if mode == "cdpc":
        i_sum = i1 + i2 + i3 + 1e-12
        d1 = (2 * i1 - i2 - i3) / i_sum
        d2 = (2 * i2 - i1 - i3) / i_sum
        d3 = (2 * i3 - i1 - i2) / i_sum
        return d1, d2, d3

    if mode == "bg_cdpc":
        raise ValueError("bg_cdpc 需要传入 (imgs, bgs)，请用 prepare_bg_cdpc()")

    raise ValueError(f"未知 mode: {mode}")


def prepare_bg_cdpc(imgs, bgs):
    """背景校正 + 全局归一化去零频 + 2Ia-Ib-Ic（tv_main 中间逻辑）。"""
    ratios = [np.asarray(im, dtype=np.float64) / (np.asarray(bg, dtype=np.float64) + 1e-12) for im, bg in zip(imgs, bgs)]
    m = np.mean([np.mean(r) for r in ratios])
    processed = [(r / (m + 1e-12)) - 1.0 for r in ratios]
    d1 = 2 * processed[0] - processed[1] - processed[2]
    d2 = 2 * processed[1] - processed[0] - processed[2]
    d3 = 2 * processed[2] - processed[0] - processed[1]
    return d1, d2, d3


def tikhonov_dpc(diff_imgs, phis, alpha: float = 5e-3, negate: bool = True) -> np.ndarray:
    """
    经典频域 Tikhonov DPC。

    phi 为 calc_transfer 输出；内部使用 T = 2*conj(phi)。
    """
    diffs = [np.asarray(d, dtype=np.float64) for d in diff_imgs]
    phis = [np.asarray(p, dtype=np.complex128) for p in phis]

    freqs = [np.fft.fftshift(np.fft.fft2(d)) for d in diffs]
    transfers = [2.0 * np.conj(p) for p in phis]

    num = sum(t * f for t, f in zip(transfers, freqs))
    den = sum(np.abs(t) ** 2 for t in transfers) + alpha
    phi_rec = np.real(np.fft.ifft2(np.fft.ifftshift(num / den)))
    if negate:
        phi_rec = -phi_rec
    phi_rec[~np.isfinite(phi_rec)] = 0.0
    return phi_rec
