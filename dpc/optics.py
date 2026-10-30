"""光瞳、扇区光源、传递函数与正向模型（对应 旋转偏移/calc_*.m）。"""

from __future__ import annotations

import numpy as np


def calc_pupil(length: int, na_radius: float) -> np.ndarray:
    """圆形光瞳掩模。MATLAB: xx = x-(length/2)-1, x=1:length。"""
    y, x = np.mgrid[0:length, 0:length]
    xx = x - length / 2.0
    yy = y - length / 2.0
    return (np.sqrt(xx**2 + yy**2) <= na_radius).astype(np.float64)


def calc_sector(length: int, r_inner: float, r_outer: float, direction) -> np.ndarray:
    """120° 环形扇形光源。direction 为起始角（度），可为数字或字符串。"""
    start_angle = float(direction)
    sweep_angle = 120.0
    end_angle = start_angle + sweep_angle

    # MATLAB: meshgrid(1:len), xc=(len+1)/2
    y, x = np.mgrid[1 : length + 1, 1 : length + 1]
    xc = (length + 1) / 2.0
    xx = x - xc
    yy = y - xc
    dist = np.sqrt(xx**2 + yy**2)
    theta = np.degrees(np.arctan2(yy, xx))
    theta = np.where(theta < 0, theta + 360.0, theta)

    mask_radius = (dist >= r_inner) & (dist <= r_outer)
    if end_angle <= 360:
        mask_angle = (theta >= start_angle) & (theta < end_angle)
    else:
        mask_angle = (theta >= start_angle) | (theta < (end_angle % 360.0))

    return (mask_radius & mask_angle).astype(np.float64)


def calc_quadrant_ring(
    length: int,
    ring_radius_inner: float,
    ring_radius_outer: float,
    direction: str,
    theta_deg: float = 0.0,
) -> np.ndarray:
    """四个 90° 扇区环形光源，可整体旋转 theta_deg。"""
    y, x = np.mgrid[0:length, 0:length]
    xx = x - length / 2.0
    yy = y - length / 2.0
    distance = np.sqrt(xx**2 + yy**2)
    source = (distance <= ring_radius_outer) & (distance >= ring_radius_inner)

    theta = np.deg2rad(theta_deg)
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)

    if direction == "right_up":
        source = source & (xr > 0) & (yr < 0)
    elif direction == "right_down":
        source = source & (xr > 0) & (yr > 0)
    elif direction == "left_up":
        source = source & (xr < 0) & (yr < 0)
    elif direction == "left_down":
        source = source & (xr < 0) & (yr > 0)
    else:
        raise ValueError("direction 必须是 right_up/right_down/left_up/left_down")

    return source.astype(np.float64)


def _shift_pupil(length: int, na_radius: float, shift_x: float, shift_y: float) -> np.ndarray:
    y, x = np.mgrid[0:length, 0:length]
    xx = x - length / 2.0 - shift_x
    yy = y - length / 2.0 - shift_y
    return (np.sqrt(xx**2 + yy**2) <= na_radius).astype(np.float64)


def calc_single_transfer(length: int, na_radius: float, shift_x: float, shift_y: float) -> np.ndarray:
    return 1j * (
        _shift_pupil(length, na_radius, shift_x, shift_y)
        - _shift_pupil(length, na_radius, -shift_x, -shift_y)
    )


def _source_shifts(source: np.ndarray):
    """提取非零光源点的频域位移（与 MATLAB 1-based 中心一致）。"""
    source = np.asarray(source, dtype=np.float64)
    height, width = source.shape
    ys, xs = np.nonzero(source)
    # MATLAB: shift = i - (width/2+1), i 为 1-based -> py: i_py - width/2
    shift_x = xs.astype(np.float64) - width / 2.0
    shift_y = ys.astype(np.float64) - height / 2.0
    return shift_x, shift_y, ys, xs


def calc_transfer(source: np.ndarray, na_radius: float, verbose: bool = False) -> np.ndarray:
    """弱物体相位传递函数（无离焦）。对光源非零点平均。"""
    source = np.asarray(source, dtype=np.float64)
    height = source.shape[0]
    shift_x, shift_y, _, _ = _source_shifts(source)
    num = len(shift_x)
    if num == 0:
        return np.zeros((height, height), dtype=np.complex128)

    trans = np.zeros((height, height), dtype=np.complex128)
    report_every = max(num // 100, 1)
    for ii in range(num):
        trans += calc_single_transfer(height, na_radius, shift_x[ii], shift_y[ii])
        if verbose and (ii + 1) % report_every == 0:
            print(f"transfer: {ii + 1}/{num} ({100.0 * (ii + 1) / num:.1f}%)")
    return trans / num


def _defocus_pupil(length: int, na_radius: float, defocus: float) -> np.ndarray:
    y, x = np.mgrid[0:length, 0:length]
    xx = x - length / 2.0
    yy = y - length / 2.0
    distance = np.sqrt(xx**2 + yy**2)
    rho = distance / (na_radius + 1e-12)
    return (distance <= na_radius) * np.exp(1j * 0.5 * defocus * (rho**2))


def _shift_defocus_pupil(
    length: int, na_radius: float, shift_x: float, shift_y: float, defocus: float
) -> np.ndarray:
    y, x = np.mgrid[0:length, 0:length]
    xx = x - length / 2.0 - shift_x
    yy = y - length / 2.0 - shift_y
    distance = np.sqrt(xx**2 + yy**2)
    rho = distance / (na_radius + 1e-12)
    return (distance <= na_radius) * np.exp(1j * 0.5 * defocus * (rho**2))


def calc_single_transfer_defocus(
    length: int,
    na_radius: float,
    shift_x: float,
    shift_y: float,
    defocus: float,
    p_val: complex,
) -> np.ndarray:
    p_minus = _shift_defocus_pupil(length, na_radius, shift_x, shift_y, defocus)
    p_plus = _shift_defocus_pupil(length, na_radius, -shift_x, -shift_y, defocus)
    return 1j * (np.conj(p_val) * p_plus - p_val * np.conj(p_minus))


def calc_transfer_defocus(
    source: np.ndarray, na_radius: float, defocus: float, verbose: bool = False
) -> np.ndarray:
    """含离焦参数的相位传递函数（DPC_search）。"""
    source = np.asarray(source, dtype=np.float64)
    height = source.shape[0]
    shift_x, shift_y, ys, xs = _source_shifts(source)
    num = len(shift_x)
    if num == 0:
        return np.zeros((height, height), dtype=np.complex128)

    p_base = _defocus_pupil(height, na_radius, defocus)
    trans = np.zeros((height, height), dtype=np.complex128)
    report_every = max(num // 100, 1)
    for ii in range(num):
        p_val = p_base[ys[ii], xs[ii]]
        trans += calc_single_transfer_defocus(
            height, na_radius, shift_x[ii], shift_y[ii], defocus, p_val
        )
        if verbose and (ii + 1) % report_every == 0:
            print(f"transfer_defocus: {ii + 1}/{num}")
    return trans / num


def calc_single_forward(data: np.ndarray, pupil: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    freq_data = np.fft.fftshift(np.fft.fft2(data))
    # MATLAB circshift(freq, [shift_y, shift_x])
    shift_data = np.roll(freq_data, shift=(int(round(shift_y)), int(round(shift_x))), axis=(0, 1))
    return np.fft.ifft2(np.fft.ifftshift(shift_data * pupil))


def calc_forward(
    data: np.ndarray, source: np.ndarray, pupil: np.ndarray, length: int, verbose: bool = False
) -> np.ndarray:
    """部分相干正向成像强度。"""
    shift_x, shift_y, _, _ = _source_shifts(source)
    num = len(shift_x)
    if num == 0:
        return np.zeros((length, length), dtype=np.float64)

    image = np.zeros((length, length), dtype=np.float64)
    report_every = max(num // 100, 1)
    for ii in range(num):
        temp = calc_single_forward(data, pupil, shift_x[ii], shift_y[ii])
        image += np.abs(temp) ** 2
        if verbose and (ii + 1) % report_every == 0:
            print(f"forward: {ii + 1}/{num} ({100.0 * (ii + 1) / num:.1f}%)")
    return image / num
