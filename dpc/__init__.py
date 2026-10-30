"""偏振 DPC：MATLAB 工作流的 Python 移植。"""

from .io_tiff import read_float_tiff, save_float_tiff
from .optics import (
    calc_pupil,
    calc_sector,
    calc_quadrant_ring,
    calc_transfer,
    calc_transfer_defocus,
    calc_forward,
)
from .polarimetry import build_mixing_matrix, unmix_illumination, load_polcam_4ch
from .dpc_recon import tikhonov_dpc, prepare_dpc_contrast
from .consistency import calc_dpc_consistency, calc_dpc_consistency_corr
from .deconv import admm_tv_deconvolution, hqs_tv_deconvolution, remove_tilt_plane
from .evaluate import evaluate_dpc, normalize01, corr2, ssim_simple

__all__ = [
    "read_float_tiff",
    "save_float_tiff",
    "calc_pupil",
    "calc_sector",
    "calc_quadrant_ring",
    "calc_transfer",
    "calc_transfer_defocus",
    "calc_forward",
    "build_mixing_matrix",
    "unmix_illumination",
    "load_polcam_4ch",
    "tikhonov_dpc",
    "prepare_dpc_contrast",
    "calc_dpc_consistency",
    "calc_dpc_consistency_corr",
    "admm_tv_deconvolution",
    "hqs_tv_deconvolution",
    "remove_tilt_plane",
    "evaluate_dpc",
    "normalize01",
    "corr2",
    "ssim_simple",
]
