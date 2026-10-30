"""无真值一致性评价。"""

from __future__ import annotations

import numpy as np

from .evaluate import corr2


def calc_dpc_consistency(phi_rec_x, s1, s2, s3, phi1, phi2, phi3) -> float:
    """MSE 重投影误差（越小越好）。"""
    phi = np.fft.fftshift(np.fft.fft2(phi_rec_x))
    e = 0.0
    for s_rec, ptf in ((s1, phi1), (s2, phi2), (s3, phi3)):
        t = 2.0 * np.conj(ptf)
        d_pred = np.real(np.fft.ifft2(np.fft.ifftshift(t * phi)))
        mean_s = np.mean(s_rec)
        s_pred = d_pred * mean_s + mean_s
        e += np.mean((s_pred - s_rec) ** 2)
    return float(e)


def calc_dpc_consistency_corr(phi_rec_x, s1, s2, s3, phi1, phi2, phi3) -> float:
    """
    差分图相关性均值 C_mean（越大越好）。
    对应 MATLAB calc_dpc_consistency_corr 最终返回 J = C_mean。
    """
    phi = np.fft.fftshift(np.fft.fft2(phi_rec_x))
    corrs = []
    for s_rec, ptf in ((s1, phi1), (s2, phi2), (s3, phi3)):
        d_rec = s_rec / (np.mean(s_rec) + 1e-12) - 1.0
        t = 2.0 * np.conj(ptf)
        d_pred = np.real(np.fft.ifft2(np.fft.ifftshift(t * phi)))
        corrs.append(corr2(d_rec, d_pred))
    return float(np.mean(corrs))
