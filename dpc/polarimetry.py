"""偏振通道混合 / 反演（马吕斯定律）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def build_mixing_matrix(
    cam_angles=(0.0, 45.0, 90.0, 135.0),
    src_angles=(0.0, 120.0, 240.0),
) -> np.ndarray:
    """A[c,s] = cos^2(cam_c - src_s)，形状 (4, 3)。"""
    cam = np.asarray(cam_angles, dtype=np.float64).reshape(-1, 1)
    src = np.asarray(src_angles, dtype=np.float64).reshape(1, -1)
    return np.cos(np.deg2rad(cam - src)) ** 2


def unmix_illumination(
    i_cam4: np.ndarray,
    x_deg: float,
    src_base=(0.0, 120.0, 240.0),
    cam_angles=(0.0, 45.0, 90.0, 135.0),
    lambda_ls: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    四通道相机强度 -> 三方向照明。

    Parameters
    ----------
    i_cam4 : (4, N) 行顺序 I0, I45, I90, I135
    x_deg : 旋转角（度）
    """
    src_angles = np.asarray(src_base, dtype=np.float64) + float(x_deg)
    a_x = build_mixing_matrix(cam_angles, src_angles)
    ata = a_x.T @ a_x
    if lambda_ls != 0:
        ata = ata + lambda_ls * np.eye(3)
    pinv_a = np.linalg.solve(ata, a_x.T)
    s_rec = pinv_a @ i_cam4
    return s_rec[0], s_rec[1], s_rec[2]


def load_polcam_4ch(filepath, crop: int | None = None):
    """
    读取四通道偏振 TIF。

    MATLAB 帧约定（1-based）：
        frame1 -> I90, frame2 -> I135, frame3 -> I45, frame4 -> I0
    """
    stack = tifffile.imread(str(filepath))
    stack = np.asarray(stack, dtype=np.float64)

    if stack.ndim == 2:
        raise ValueError("需要多页 TIFF（至少 4 帧）")

    # (pages, H, W) 或偶尔 (H, W, pages)
    if stack.shape[-1] == 4 and stack.shape[0] > 4:
        pages = [stack[:, :, i] for i in range(4)]
    else:
        if stack.shape[0] < 4:
            raise ValueError(f"期望至少 4 页，得到 {stack.shape}")
        pages = [stack[i] for i in range(4)]

    i90, i135, i45, i0 = pages[0], pages[1], pages[2], pages[3]

    if crop is not None:
        h_full, w_full = i0.shape
        if h_full > crop or w_full > crop:
            ch = (h_full - crop) // 2
            cw = (w_full - crop) // 2
            sl = (slice(ch, ch + crop), slice(cw, cw + crop))
            i0, i45, i90, i135 = i0[sl], i45[sl], i90[sl], i135[sl]

    h, w = i0.shape
    i_cam4 = np.vstack([i0.ravel(), i45.ravel(), i90.ravel(), i135.ravel()])
    return {
        "I0": i0,
        "I45": i45,
        "I90": i90,
        "I135": i135,
        "I_cam4": i_cam4,
        "shape": (h, w),
        "path": Path(filepath),
    }


def reshape_sources(s1_flat, s2_flat, s3_flat, shape):
    h, w = shape
    return (
        s1_flat.reshape(h, w),
        s2_flat.reshape(h, w),
        s3_flat.reshape(h, w),
    )
