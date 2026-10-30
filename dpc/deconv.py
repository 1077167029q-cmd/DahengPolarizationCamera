"""ADMM / HQS TV 反卷积与去倾斜平面。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_tiff import save_float_tiff


def remove_tilt_plane(x: np.ndarray) -> np.ndarray:
    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # MATLAB meshgrid(1:W,1:H) -> xx 列坐标 1..W, yy 行坐标 1..H
    A = np.column_stack([xx.ravel() + 1, yy.ravel() + 1, np.ones(h * w)])
    coeff, _, _, _ = np.linalg.lstsq(A, x.ravel(), rcond=None)
    plane = (A @ coeff).reshape(h, w)
    return x - plane


def _pad_fft_kernel(kernel: np.ndarray, shape) -> np.ndarray:
    """对应 MATLAB fft2(k, M, N) + fftshift。"""
    return np.fft.fftshift(np.fft.fft2(kernel, s=shape))


def admm_tv_deconvolution(
    diff_img1,
    diff_img2,
    diff_img3,
    phi_trans_1,
    phi_trans_2,
    phi_trans_3,
    lambda_: float,
    rho: float,
    max_iter: int,
    save_dir: str | Path | None = None,
    show: bool = False,
) -> np.ndarray:
    d1 = np.asarray(diff_img1, dtype=np.float64)
    d2 = np.asarray(diff_img2, dtype=np.float64)
    d3 = np.asarray(diff_img3, dtype=np.float64)
    h1 = np.asarray(phi_trans_1, dtype=np.complex128)
    h2 = np.asarray(phi_trans_2, dtype=np.complex128)
    h3 = np.asarray(phi_trans_3, dtype=np.complex128)

    f1 = np.fft.fftshift(np.fft.fft2(d1))
    f2 = np.fft.fftshift(np.fft.fft2(d2))
    f3 = np.fft.fftshift(np.fft.fft2(d3))

    d_x = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float64)
    d_y = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float64)
    shape = d1.shape
    fdx = _pad_fft_kernel(d_x, shape)
    fdy = _pad_fft_kernel(d_y, shape)

    x = d1.copy()
    zx = np.zeros_like(x)
    zy = np.zeros_like(x)
    p_x = np.zeros_like(x)
    p_y = np.zeros_like(x)

    den = np.abs(h1) ** 2 + np.abs(h2) ** 2 + np.abs(h3) ** 2 + rho * (np.abs(fdx) ** 2 + np.abs(fdy) ** 2) + np.finfo(float).eps

    for k in range(1, max_iter + 1):
        fzx = np.fft.fftshift(np.fft.fft2(zx))
        fzy = np.fft.fftshift(np.fft.fft2(zy))
        pfx = np.fft.fftshift(np.fft.fft2(p_x))
        pfy = np.fft.fftshift(np.fft.fft2(p_y))

        num = (
            np.conj(h1) * f1
            + np.conj(h2) * f2
            + np.conj(h3) * f3
            + rho * (np.conj(fdx) * (fzx - pfx) + np.conj(fdy) * (fzy - pfy))
        )
        x_freq = num / den
        x = np.real(np.fft.ifft2(np.fft.ifftshift(x_freq)))

        dx = np.real(np.fft.ifft2(np.fft.ifftshift(fdx * x_freq)))
        dy = np.real(np.fft.ifft2(np.fft.ifftshift(fdy * x_freq)))
        thresh = lambda_ / rho
        zx = np.sign(dx + p_x) * np.maximum(np.abs(dx + p_x) - thresh, 0)
        zy = np.sign(dy + p_y) * np.maximum(np.abs(dy + p_y) - thresh, 0)

        p_x = p_x + (dx - zx)
        p_y = p_y + (dy - zy)

        if show:
            import matplotlib.pyplot as plt

            plt.figure(7, figsize=(12, 6))
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(x, cmap="gray")
            plt.title(f"x at {k}")
            plt.subplot(1, 2, 2)
            plt.imshow(np.sqrt(zx**2 + zy**2), cmap="gray")
            plt.title(f"|z| at {k}")
            plt.pause(0.01)

        if k % 5 == 0 or k == max_iter:
            print(f"ADMM iter {k}/{max_iter}")

    x_est = -remove_tilt_plane(-x)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_float_tiff(save_dir / "phase_admm.tif", x_est)
        save_float_tiff(save_dir / "diff_x_admm.tif", zx)
        save_float_tiff(save_dir / "diff_y_admm.tif", zy)

    return x_est


def hqs_tv_deconvolution(
    diff_img1,
    diff_img2,
    diff_img3,
    phi_trans_1,
    phi_trans_2,
    phi_trans_3,
    lambda_: float,
    rho: float,
    max_iter: int,
    save_dir: str | Path | None = None,
    show: bool = False,
) -> np.ndarray:
    d1 = np.asarray(diff_img1, dtype=np.float64)
    d2 = np.asarray(diff_img2, dtype=np.float64)
    d3 = np.asarray(diff_img3, dtype=np.float64)
    h1 = np.asarray(phi_trans_1, dtype=np.complex128)
    h2 = np.asarray(phi_trans_2, dtype=np.complex128)
    h3 = np.asarray(phi_trans_3, dtype=np.complex128)

    f1 = np.fft.fftshift(np.fft.fft2(d1))
    f2 = np.fft.fftshift(np.fft.fft2(d2))
    f3 = np.fft.fftshift(np.fft.fft2(d3))

    d_x = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float64)
    d_y = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float64)
    shape = d1.shape
    fdx = _pad_fft_kernel(d_x, shape)
    fdy = _pad_fft_kernel(d_y, shape)

    x = d1.copy()
    fx = np.fft.fftshift(np.fft.fft2(x))
    z_x = np.fft.ifft2(np.fft.ifftshift(fdx * fx))
    z_y = np.fft.ifft2(np.fft.ifftshift(fdy * fx))

    denominator = (
        (1 - rho) * (np.abs(h1) ** 2 + np.abs(h2) ** 2 + np.abs(h3) ** 2)
        + rho * (np.abs(fdx) ** 2 + np.abs(fdy) ** 2)
        + np.finfo(float).eps
    )

    z = None
    for it in range(1, max_iter + 1):
        fzx = np.fft.fftshift(np.fft.fft2(z_x))
        fzy = np.fft.fftshift(np.fft.fft2(z_y))
        numerator = (1 - rho) * (
            np.conj(h1) * f1 + np.conj(h2) * f2 + np.conj(h3) * f3
        ) + rho * (np.conj(fdx) * fzx + np.conj(fdy) * fzy)
        x_freq = numerator / denominator
        x = np.real(np.fft.ifft2(np.fft.ifftshift(x_freq)))

        dx = np.fft.ifft2(np.fft.ifftshift(x_freq * fdx))
        dy = np.fft.ifft2(np.fft.ifftshift(x_freq * fdy))
        thresh = lambda_ / rho
        z_x = np.sign(dx) * np.maximum(np.abs(dx) - thresh, 0)
        z_y = np.sign(dy) * np.maximum(np.abs(dy) - thresh, 0)
        z = np.sqrt(np.abs(z_x) ** 2 + np.abs(z_y) ** 2)

        if show:
            import matplotlib.pyplot as plt

            plt.figure(7, figsize=(12, 6))
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(x, cmap="gray")
            plt.title(f"x at {it}")
            plt.subplot(1, 2, 2)
            plt.imshow(np.real(z), cmap="gray")
            plt.title(f"z at {it}")
            plt.pause(0.01)

        if it % 5 == 0 or it == max_iter:
            print(f"HQS iter {it}/{max_iter}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_float_tiff(save_dir / "phase_hqs.tif", x)
        save_float_tiff(save_dir / "diff_x_hqs.tif", np.real(z_x))
        save_float_tiff(save_dir / "diff_y_hqs.tif", np.real(z_y))
        if z is not None:
            save_float_tiff(save_dir / "z_hqs.tif", np.real(z))

    return x
