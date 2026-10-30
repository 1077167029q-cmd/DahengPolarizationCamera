"""多页 float32 TIFF 读写（对应 MATLAB saveFloatTiff / readFloatTiff）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def save_float_tiff(filepath, arr) -> None:
    """保存 float32 图像。2D -> 单页；3D (H,W,C) -> 多页堆栈。"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(arr, dtype=np.float32)

    if data.ndim == 2:
        tifffile.imwrite(str(filepath), data, photometric="minisblack")
        return

    if data.ndim != 3:
        raise ValueError(f"期望 2D 或 3D 数组，得到 shape={data.shape}")

    # MATLAB: ff(:,:,i) 按第 3 维写页；这里写成 (pages, H, W)
    stack = np.moveaxis(data, -1, 0)
    tifffile.imwrite(
        str(filepath),
        stack,
        photometric="minisblack",
        metadata=None,
    )


def read_float_tiff(filepath) -> np.ndarray:
    """读取 float TIFF。单页返回 (H,W)；多页返回 (H,W,C)。"""
    data = tifffile.imread(str(filepath))
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 2:
        return data
    if data.ndim == 3:
        # tifffile 多为 (pages, H, W) -> (H, W, C)
        if data.shape[0] <= 64 and data.shape[0] < min(data.shape[1], data.shape[2]):
            return np.moveaxis(data, 0, -1)
        return data
    raise ValueError(f"不支持的 TIFF 维度: {data.shape}")
