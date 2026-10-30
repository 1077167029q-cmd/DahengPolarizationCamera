# -*- coding: utf-8 -*-
"""
DPC Reconstruction Pipeline for Daheng Polarization Camera.
Integrates RAW conversion, rotation calibration, and TV reconstruction (ADMM/HQS/Tikhonov).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np

# Force matplotlib to use non-interactive Agg backend for thread safety
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PySide6.QtCore import QThread, Signal

# Import from local dpc package
from dpc.io_tiff import read_float_tiff, save_float_tiff
from dpc.optics import calc_sector, calc_transfer
from dpc.polarimetry import load_polcam_4ch, reshape_sources, unmix_illumination
from dpc.consistency import calc_dpc_consistency_corr
from dpc.dpc_recon import prepare_dpc_contrast, tikhonov_dpc
from dpc.deconv import admm_tv_deconvolution, hqs_tv_deconvolution


def raw_to_polcam_tiff_data(raw_path: Path, width: int, height: int) -> np.ndarray:
    """
    Reads Mono10/uint16 RAW (column-major representation, matching MATLAB fread)
    and splits it into a 4-channel polarization stack.
    """
    data = np.fromfile(raw_path, dtype=np.uint16)
    expected = width * height
    if data.size != expected:
        # Fallback to uint8 if size matches mono8 format
        if data.size * 2 == expected:
            data = np.fromfile(raw_path, dtype=np.uint8)
        else:
            raise ValueError(f"RAW pixel count {data.size} does not match {width}x{height}={expected}")

    # Column-major representation to Row-major
    image = data.reshape((width, height), order="F").T.astype(np.float32)

    rows, cols = image.shape
    if rows % 2 or cols % 2:
        raise ValueError("Image dimensions must be even for 2x2 demosaicing")

    output_stack = np.zeros((rows // 2, cols // 2, 4), dtype=np.float32)
    output_stack[:, :, 0] = image[0::2, 0::2]  # Top-Left (90 deg)
    output_stack[:, :, 1] = image[0::2, 1::2]  # Top-Right (135 deg)
    output_stack[:, :, 2] = image[1::2, 0::2]  # Bottom-Left (45 deg)
    output_stack[:, :, 3] = image[1::2, 1::2]  # Bottom-Right (0 deg)

    return output_stack


def run_calibrate_rotation(
    input_tif: Path,
    save_root: Path,
    length: int = 256,
    na_radius: float = 103,
    ring_inner: float = 0,
    ring_outer: float | None = None,
    alpha_dpc: float = 5e-3,
    x_list=None,
    src_base=(0.0, 120.0, 240.0),
    progress_callback=None
) -> tuple[float, np.ndarray]:
    """
    Estimates the rotation angle using DPC consistency check.
    """
    if ring_outer is None:
        ring_outer = na_radius
    if x_list is None:
        x_list = np.arange(0, 65, 5, dtype=float)

    save_root.mkdir(parents=True, exist_ok=True)
    data = load_polcam_4ch(input_tif, crop=length)
    h, w = data["shape"]
    i_cam4 = data["I_cam4"]

    c_list = []
    total_angles = len(x_list)
    for idx, x in enumerate(x_list):
        if progress_callback:
            progress_callback(int(10 + 60 * (idx / total_angles)), f"正在校准旋转角: {x:.1f}° ({idx+1}/{total_angles})")
        
        folder_x = save_root / f"x_{x:05.2f}"
        folder_x.mkdir(parents=True, exist_ok=True)

        s1f, s2f, s3f = unmix_illumination(i_cam4, x, src_base=src_base)
        s1, s2, s3 = reshape_sources(s1f, s2f, s3f, (h, w))
        save_float_tiff(folder_x / "S_rec.tif", np.stack([s1, s2, s3], axis=-1))

        # 检查是否已存在对应角度且尺寸匹配的 6 个传输函数文件 (phi1~3_amp/phase)
        phi_files_exist = True
        phis = []
        for i in range(1, 4):
            f_amp = folder_x / f"phi{i}_amp_x{x:.2f}.tif"
            f_phase = folder_x / f"phi{i}_phase_x{x:.2f}.tif"
            if not (f_amp.exists() and f_phase.exists()):
                phi_files_exist = False
                break

        if phi_files_exist:
            try:
                loaded_phis = _load_phi_from_dir(folder_x, x)
                if len(loaded_phis) == 3 and loaded_phis[0].shape == (h, w):
                    phis = loaded_phis
                else:
                    phi_files_exist = False
            except Exception:
                phi_files_exist = False

        if not phi_files_exist:
            phis = []
            for i, base in enumerate(src_base, start=1):
                direction = base + x
                src_mask = calc_sector(length, ring_inner, ring_outer, direction)
                phi = calc_transfer(src_mask, na_radius, verbose=False)
                phis.append(phi)
                save_float_tiff(folder_x / f"phi{i}_amp_x{x:.2f}.tif", np.abs(phi))
                save_float_tiff(folder_x / f"phi{i}_phase_x{x:.2f}.tif", np.angle(phi))

        d1, d2, d3 = prepare_dpc_contrast([s1, s2, s3], mode="mean_norm")
        phi_rec = tikhonov_dpc([d1, d2, d3], phis, alpha=alpha_dpc, negate=True)
        save_float_tiff(folder_x / f"phi_rec_x{x:.2f}.tif", phi_rec)

        c = calc_dpc_consistency_corr(phi_rec, s1, s2, s3, phis[0], phis[1], phis[2])
        c_list.append(c)

    c_list = np.asarray(c_list)
    max_idx = int(np.argmax(c_list))
    x_est = float(x_list[max_idx])

    # Save curve plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(x_list) > 1:
        ax.plot(x_list, c_list, "r-", lw=2)
        ax.axvline(x_est, color="g", ls=":", label=f"x_est={x_est:.2f}°")
    else:
        ax.plot(x_list, c_list, "ro", label=f"x_est={x_est:.2f}°")
    ax.set_xlabel("Rotation angle x (deg)")
    ax.set_ylabel("C_dpc(x)")
    ax.set_title("Rotation Calibration Curve")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_root / "calibration_curve.png", dpi=150)
    plt.close(fig)

    np.savez(save_root / "calibration_result.npz", x_list=x_list, c_list=c_list, x_est=x_est)
    return x_est, c_list


def _load_phi_from_dir(phi_dir: Path, x: float):
    phis = []
    for i in range(1, 4):
        amp = read_float_tiff(phi_dir / f"phi{i}_amp_x{x:.2f}.tif")
        phase = read_float_tiff(phi_dir / f"phi{i}_phase_x{x:.2f}.tif")
        if amp.ndim == 3:
            amp = amp[:, :, 0]
        if phase.ndim == 3:
            phase = phase[:, :, 0]
        phis.append(amp * np.exp(1j * phase))
    return phis


def run_tv_reconstruction(
    s_rec_tif: Path,
    phi_dir: Path,
    x_angle: float,
    output_dir: Path,
    method: str = "admm",
    lambda_: float = 0.03,
    rho: float = 0.6,
    max_iter: int = 30,
    alpha_tikhonov: float = 5e-3,
    mode: str = "cdpc",
    progress_callback=None
) -> np.ndarray:
    """
    Runs TV regularization phase reconstruction (ADMM / HQS / Tikhonov).
    """
    if progress_callback:
        progress_callback(80, "加载校准数据并预处理对比度...")

    # Load 3-channel stack
    stack = read_float_tiff(s_rec_tif)
    if stack.ndim == 2:
        raise ValueError("s_rec.tif must be a 3-page TIFF")
    imgs = [stack[:, :, i] for i in range(3)]

    # Prepare contrasts
    diffs = prepare_dpc_contrast(imgs, mode=mode)

    # Load transfers
    phis = _load_phi_from_dir(phi_dir, x_angle)

    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(85, f"执行 {method.upper()} 重建算法...")

    if method == "admm":
        x_est = admm_tv_deconvolution(
            *diffs, *phis, lambda_, rho, max_iter, save_dir=output_dir, show=False
        )
        out_name = "phase_admm.tif"
    elif method == "hqs":
        x_est = hqs_tv_deconvolution(
            *diffs, *phis, lambda_, rho, max_iter, save_dir=output_dir, show=False
        )
        out_name = "phase_hqs.tif"
    elif method == "tikhonov":
        x_est = tikhonov_dpc(diffs, phis, alpha=alpha_tikhonov, negate=True)
        out_name = "phase_tikhonov.tif"
        save_float_tiff(output_dir / out_name, x_est)
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")

    if progress_callback:
        progress_callback(100, "重建完成")

    return x_est


class DPCReconThread(QThread):
    """
    Asynchronous DPC Calibration and Reconstruction Thread.
    """
    progress = Signal(int, str)  # percentage, status_msg
    finished = Signal(bool, str, dict)  # success, message, result_dict

    def __init__(
        self,
        mode: str,  # 'live' or 'file'
        save_dir: str | Path,
        last_images: list[np.ndarray] | None = None,
        file_path: str | Path | None = None,
        raw_width: int = 2448,
        raw_height: int = 2048,
        length: int = 256,
        na_radius: float = 103,
        ring_inner: float = 0,
        ring_outer: float = 103,
        alpha_dpc: float = 5e-3,
        x_min: float = 0.0,
        x_max: float = 60.0,
        x_step: float = 5.0,
        tv_method: str = "admm",
        tv_lambda: float = 0.03,
        tv_rho: float = 0.6,
        tv_max_iter: int = 30,
        parent=None
    ):
        super().__init__(parent)
        self.mode = mode
        self.save_dir = Path(save_dir)
        self.last_images = last_images
        self.file_path = Path(file_path) if file_path else None
        self.raw_width = raw_width
        self.raw_height = raw_height
        self.length = length
        self.na_radius = na_radius
        self.ring_inner = ring_inner
        self.ring_outer = ring_outer
        self.alpha_dpc = alpha_dpc
        self.x_min = x_min
        self.x_max = x_max
        self.x_step = x_step
        self.tv_method = tv_method
        self.tv_lambda = tv_lambda
        self.tv_rho = tv_rho
        self.tv_max_iter = tv_max_iter

    def run(self):
        try:
            self.progress.emit(2, "初始化输出目录...")
            self.save_dir.mkdir(parents=True, exist_ok=True)
            input_tif_path = self.save_dir / "1_4.tif"

            # Step 1: Get 1_4.tif (Convert RAW or write Stack)
            if self.mode == 'live':
                self.progress.emit(5, "生成四通道 TIFF (1_4.tif)...")
                if not self.last_images or len(self.last_images) < 4:
                    raise ValueError("无有效的采集帧")
                # Stack order: I90, I135, I45, I0
                # self.last_images has [img_90, img_45, img_135, img_0]
                # We need to map them to the correct order:
                # 01_raw_to_tiff.py stacks them exactly as they are sliced:
                # [img_90, img_45, img_135, img_0] which matches the expected stack.
                stack = np.stack(self.last_images, axis=-1).astype(np.float32)
                save_float_tiff(input_tif_path, stack)
            else:
                if not self.file_path or not self.file_path.exists():
                    raise ValueError(f"文件不存在: {self.file_path}")
                
                suffix = self.file_path.suffix.lower()
                if suffix == '.raw':
                    self.progress.emit(5, "解析 RAW 文件并保存为 1_4.tif...")
                    stack = raw_to_polcam_tiff_data(self.file_path, self.raw_width, self.raw_height)
                    save_float_tiff(input_tif_path, stack)
                elif suffix in ['.tif', '.tiff']:
                    self.progress.emit(5, "复制四通道 TIFF (1_4.tif)...")
                    # Simply copy or read/write to target directory
                    stack = read_float_tiff(self.file_path)
                    if stack.ndim != 3 or stack.shape[-1] < 4:
                        raise ValueError("输入 TIFF 必须为至少4页/4通道的偏振 TIFF 堆栈")
                    save_float_tiff(input_tif_path, stack[:, :, :4])
                else:
                    raise ValueError(f"不支持的文件格式: {suffix}")

            # Step 2: Run Calibration
            self.progress.emit(10, "启动旋转校准...")
            x_list = np.arange(self.x_min, self.x_max + 0.5 * self.x_step, self.x_step, dtype=float)
            results_scan_dir = self.save_dir / "results_x_scan"
            
            x_est, c_list = run_calibrate_rotation(
                input_tif=input_tif_path,
                save_root=results_scan_dir,
                length=self.length,
                na_radius=self.na_radius,
                ring_inner=self.ring_inner,
                ring_outer=self.ring_outer,
                alpha_dpc=self.alpha_dpc,
                x_list=x_list,
                progress_callback=self.progress.emit
            )

            # Step 3: Run TV Reconstruction
            self.progress.emit(75, f"计算最佳旋转角: {x_est:.2f}°，启动 TV 重建...")
            s_rec_path = results_scan_dir / f"x_{x_est:05.2f}" / "S_rec.tif"
            phi_dir_path = results_scan_dir / f"x_{x_est:05.2f}"
            tv_recon_dir = self.save_dir / "tv_recon_result"
            
            phase_rec = run_tv_reconstruction(
                s_rec_tif=s_rec_path,
                phi_dir=phi_dir_path,
                x_angle=x_est,
                output_dir=tv_recon_dir,
                method=self.tv_method,
                lambda_=self.tv_lambda,
                rho=self.tv_rho,
                max_iter=self.tv_max_iter,
                alpha_tikhonov=self.alpha_dpc,
                progress_callback=self.progress.emit
            )

            # Save the final metadata results
            result_info = {
                "x_est": x_est,
                "x_list": x_list.tolist(),
                "c_list": c_list.tolist(),
                "save_dir": str(self.save_dir),
                "results_scan_dir": str(results_scan_dir),
                "tv_recon_dir": str(tv_recon_dir),
                "phase_path": str(tv_recon_dir / ("phase_admm.tif" if self.tv_method == "admm" else "phase_hqs.tif" if self.tv_method == "hqs" else "phase_tikhonov.tif")),
                "curve_path": str(results_scan_dir / "calibration_curve.png"),
                "s_rec_path": str(results_scan_dir / f"x_{x_est:05.2f}" / "S_rec.tif"),
            }
            
            self.finished.emit(True, "DPC 偏振重建流程顺利执行完成！", result_info)

        except Exception as err:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(err), {})
