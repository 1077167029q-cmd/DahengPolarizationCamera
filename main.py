import os
import sys
import json
import time
import re
from functools import lru_cache
import cv2
import numpy as np
import queue
import threading
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout,
                               QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QFileDialog, QMessageBox, QFrame, QSizePolicy,
                               QDoubleSpinBox, QComboBox, QSpinBox, QScrollArea,
                               QDialog, QGraphicsView, QGraphicsScene, QProgressDialog, QInputDialog,
                               QCheckBox, QSlider)
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor

# 引入 DPC 重建与偏振解复用模块及线程
from dpc.io_tiff import read_float_tiff, save_float_tiff
from dpc.polarimetry import unmix_illumination
from dpc_pipeline import DPCReconThread

# 引入大恒官方 python SDK (请确保已安装大恒 SDK 驱动)
try:
    import gxipy as gx
except ImportError:
    gx = None


def get_config_path():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "config.json")


class DoLPAnalysisDialog(QDialog):
    """
    线偏振度 (DoLP) 分析与伪彩热力图可视化窗口
    """
    def __init__(self, img_90, img_45, img_135, img_0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("线偏振度 (DoLP) 全图分析")
        self.resize(1000, 750)
        self.setStyleSheet("""
            QDialog { background-color: #1E1E1E; }
            QLabel { color: #FFFFFF; font-family: 'Segoe UI', Arial; }
            QPushButton { 
                border-radius: 4px; 
                font-weight: bold; 
                font-size: 13px; 
                border: none;
                background-color: #00ADB5;
                color: white;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #00C4CC; }
            QComboBox {
                background-color: #3D3D3D;
                color: white;
                border: 1px solid #555555;
                padding: 4px;
                border-radius: 3px;
            }
            QFrame#StatsCard {
                background-color: #2D2D2D;
                border-radius: 6px;
                border: 1px solid #3A3A3A;
            }
        """)

        # 保存原始图像浮点矩阵
        self.img_90 = img_90.astype(np.float32)
        self.img_45 = img_45.astype(np.float32)
        self.img_135 = img_135.astype(np.float32)
        self.img_0 = img_0.astype(np.float32)

        # 计算 DoLP 矩阵与统计数据
        self.calculate_dolp()

        # 初始化 UI 与渲染
        self.init_ui()
        self.update_visualization()

    def calculate_dolp(self):
        """
        根据斯托克斯公式计算线偏振度 (DoLP)
        S0 = (I_0 + I_45 + I_90 + I_135) / 2.0
        S1 = I_0 - I_90
        S2 = I_45 - I_135
        DoLP = sqrt(S1^2 + S2^2) / (S0 + eps)
        """
        S0 = (self.img_0 + self.img_90 + self.img_45 + self.img_135) / 2.0
        S1 = self.img_0 - self.img_90
        S2 = self.img_45 - self.img_135

        eps = 1e-6
        self.dolp_matrix = np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + eps)
        self.dolp_matrix = np.clip(self.dolp_matrix, 0.0, 1.0)

        # 统计数据
        self.mean_dolp = float(np.mean(self.dolp_matrix))
        self.max_dolp = float(np.max(self.dolp_matrix))
        self.min_dolp = float(np.min(self.dolp_matrix))

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # 顶部控制栏
        top_bar = QHBoxLayout()
        
        lbl_colormap = QLabel("伪彩配色方案:")
        lbl_colormap.setStyleSheet("font-weight: bold; font-size: 14px;")
        top_bar.addWidget(lbl_colormap)

        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(["Jet (经典热力图)", "Viridis (感知均匀)", "Inferno (高对比暗色)", "Grayscale (灰度图)"])
        self.combo_colormap.currentIndexChanged.connect(self.update_visualization)
        top_bar.addWidget(self.combo_colormap)

        top_bar.addStretch(1)

        self.btn_export = QPushButton("导出偏振度伪彩图 (PNG)")
        self.btn_export.setStyleSheet("background-color: #28A745; color: white;")
        self.btn_export.clicked.connect(self.export_image)
        top_bar.addWidget(self.btn_export)

        self.btn_export_tiff = QPushButton("导出原始 DoLP 数据 (TIFF/RAW)")
        self.btn_export_tiff.setStyleSheet("background-color: #FF9800; color: white;")
        self.btn_export_tiff.clicked.connect(self.export_raw_data)
        top_bar.addWidget(self.btn_export_tiff)

        main_layout.addLayout(top_bar)

        # 主体布局：包含伪彩图 + 右侧 Colorbar
        body_layout = QHBoxLayout()

        # 图像显示画布 (设置 Ignored 策略防止 setPixmap 触发布局无限扩张)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumSize(1, 1)
        self.lbl_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_image.setStyleSheet("border: 1px solid #3A3A3A; background-color: #121212; border-radius: 4px;")
        body_layout.addWidget(self.lbl_image, stretch=5)

        # 色尺 (Colorbar) 区域
        colorbar_frame = QFrame()
        colorbar_frame.setFixedWidth(80)
        colorbar_frame.setStyleSheet("background-color: #252525; border-radius: 4px; border: 1px solid #3A3A3A;")
        colorbar_layout = QVBoxLayout(colorbar_frame)
        colorbar_layout.setContentsMargins(5, 10, 5, 10)

        lbl_bar_top = QLabel("100% (1.0)")
        lbl_bar_top.setAlignment(Qt.AlignCenter)
        lbl_bar_top.setStyleSheet("font-size: 11px; font-weight: bold; color: #FF5252;")
        colorbar_layout.addWidget(lbl_bar_top)

        self.lbl_colorbar = QLabel()
        self.lbl_colorbar.setFixedWidth(30)
        self.lbl_colorbar.setMinimumSize(1, 1)
        self.lbl_colorbar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_colorbar.setAlignment(Qt.AlignCenter)
        colorbar_layout.addWidget(self.lbl_colorbar, alignment=Qt.AlignCenter, stretch=1)

        lbl_bar_bottom = QLabel("0% (0.0)")
        lbl_bar_bottom.setAlignment(Qt.AlignCenter)
        lbl_bar_bottom.setStyleSheet("font-size: 11px; font-weight: bold; color: #448AFF;")
        colorbar_layout.addWidget(lbl_bar_bottom)

        body_layout.addWidget(colorbar_frame)
        main_layout.addLayout(body_layout, stretch=1)

        # 底部统计卡片
        stats_frame = QFrame()
        stats_frame.setObjectName("StatsCard")
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(15, 10, 15, 10)

        self.lbl_mean = QLabel(f"全图平均偏振度: {self.mean_dolp * 100:.2f}% ({self.mean_dolp:.4f})")
        self.lbl_mean.setStyleSheet("font-size: 13px; font-weight: bold; color: #00ADB5;")
        stats_layout.addWidget(self.lbl_mean)

        self.lbl_max = QLabel(f"最大偏振度: {self.max_dolp * 100:.2f}% ({self.max_dolp:.4f})")
        self.lbl_max.setStyleSheet("font-size: 13px; font-weight: bold; color: #FF9800;")
        stats_layout.addWidget(self.lbl_max)

        self.lbl_min = QLabel(f"最小偏振度: {self.min_dolp * 100:.2f}% ({self.min_dolp:.4f})")
        self.lbl_min.setStyleSheet("font-size: 13px; font-weight: bold; color: #AAAAAA;")
        stats_layout.addWidget(self.lbl_min)

        main_layout.addWidget(stats_frame)

    def update_visualization(self):
        """更新伪彩热力图与色尺"""
        cmap_idx = self.combo_colormap.currentIndex()
        dolp_8u = (self.dolp_matrix * 255.0).astype(np.uint8)

        if cmap_idx == 0:  # Jet
            color_img = cv2.applyColorMap(dolp_8u, cv2.COLORMAP_JET)
        elif cmap_idx == 1:  # Viridis
            color_img = cv2.applyColorMap(dolp_8u, cv2.COLORMAP_VIRIDIS)
        elif cmap_idx == 2:  # Inferno
            color_img = cv2.applyColorMap(dolp_8u, cv2.COLORMAP_INFERNO)
        else:  # Grayscale
            color_img = cv2.cvtColor(dolp_8u, cv2.COLOR_GRAY2BGR)

        self.current_colored_bgr = color_img.copy()

        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_c = np.ascontiguousarray(color_rgb)
        h, w, ch = color_c.shape
        bytes_per_line = ch * w
        q_img = QImage(color_c.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        pixmap = QPixmap.fromImage(q_img)
        lbl_w = max(self.lbl_image.width(), 10)
        lbl_h = max(self.lbl_image.height(), 10)
        scaled_pixmap = pixmap.scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_image.setPixmap(scaled_pixmap)

        # 绘制色尺
        bar_grad = np.linspace(255, 0, 256, dtype=np.uint8).reshape(256, 1)
        bar_grad = np.repeat(bar_grad, 20, axis=1)

        if cmap_idx == 0:
            bar_color = cv2.applyColorMap(bar_grad, cv2.COLORMAP_JET)
        elif cmap_idx == 1:
            bar_color = cv2.applyColorMap(bar_grad, cv2.COLORMAP_VIRIDIS)
        elif cmap_idx == 2:
            bar_color = cv2.applyColorMap(bar_grad, cv2.COLORMAP_INFERNO)
        else:
            bar_color = cv2.cvtColor(bar_grad, cv2.COLOR_GRAY2BGR)

        bar_rgb = cv2.cvtColor(bar_color, cv2.COLOR_BGR2RGB)
        bar_c = np.ascontiguousarray(bar_rgb)
        bh, bw, bch = bar_c.shape
        q_bar = QImage(bar_c.data, bw, bh, bch * bw, QImage.Format_RGB888).copy()
        self.lbl_colorbar.setPixmap(QPixmap.fromImage(q_bar).scaled(20, 220, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_visualization()

    def export_image(self):
        """导出偏振度伪彩图像"""
        save_path, _ = QFileDialog.getSaveFileName(self, "导出偏振度伪彩图像", "dolp_visualization.png", "PNG Image (*.png);;TIFF Image (*.tiff);;JPEG Image (*.jpg)")
        if save_path:
            try:
                cv2.imwrite(save_path, self.current_colored_bgr)
                QMessageBox.information(self, "导出成功", f"偏振度伪彩图像已保存至:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"无法保存图像: {e}")

    def export_raw_data(self):
        """导出原始 32-bit 浮点偏振度数据 (0.0~1.0)"""
        save_path, _ = QFileDialog.getSaveFileName(self, "导出原始 DoLP 矩阵数据", "dolp_matrix.tiff", "TIFF Float Image (*.tiff);;Numpy Binary (*.npy)")
        if save_path:
            try:
                if save_path.endswith('.npy'):
                    np.save(save_path, self.dolp_matrix)
                else:
                    cv2.imwrite(save_path, self.dolp_matrix)
                QMessageBox.information(self, "导出成功", f"DoLP 原始数据已成功导出至:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出数据失败: {e}")


class LiveImageGraphicsView(QGraphicsView):
    """支持 4 视口全自动联动缩放、鼠标拖拽平移的高阶实时画面查看器"""
    clicked = Signal(int)             # 左键单击信号，发送自身索引
    inspectRequested = Signal(int)    # 双击/右键信号，发送自身索引
    zoomTriggered = Signal(float)     # 缩放倍率 factor 信号
    panTriggered = Signal()            # 平移拖拽信号

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        
        # 创建场景和 PixmapItem
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        self.placeholder_text = "无信号"

        # 默认不显示滚动条，保持界面干净美观
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # 渲染设置：邻近像素采样（更适合观察原始极细微的偏振颗粒像素）
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        
        # 交互行为设置
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

        self.press_pos = None
        self.zoom_factor = 1.0
        self.zoom_min = 0.1
        self.zoom_max = 20.0

    def update_image(self, img_display):
        """实时更新图像纹理（深拷贝 C 内存以解决 Python GC 悬挂指针崩溃）"""
        img_c = np.ascontiguousarray(img_display)
        h, w = img_c.shape
        q_img = QImage(img_c.data, w, h, w, QImage.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(q_img)

        # 检查尺寸是否发生变化
        size_changed = False
        if self.pixmap_item is not None:
            old_rect = self.scene.sceneRect()
            if old_rect.width() != w or old_rect.height() != h:
                size_changed = True

        if self.pixmap_item is None:
            self.scene.clear()
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(0, 0, w, h)
            self.reset_zoom()
        else:
            self.pixmap_item.setPixmap(pixmap)
            if size_changed:
                self.scene.setSceneRect(0, 0, w, h)
                self.reset_zoom()

    def show_placeholder(self, text="无信号"):
        self.scene.clear()
        self.pixmap_item = None
        self.placeholder_text = text
        self.resetTransform()
        self.zoom_factor = 1.0
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap_item is None:
            painter = QPainter(self.viewport())
            painter.setPen(QColor("#555555"))
            font = painter.font()
            font.setPointSize(14)
            painter.setFont(font)
            text = getattr(self, "placeholder_text", "无信号")
            painter.drawText(self.viewport().rect(), Qt.AlignCenter, text)

    def wheelEvent(self, event):
        # 滚轮以鼠标指向的位置为缩放锚点。
        delta = event.angleDelta().y()
        if not delta:
            event.ignore()
            return
        factor = 1.15 ** (delta / 120.0)
        self.zoomTriggered.emit(factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.press_pos = event.position().toPoint()
        elif event.button() == Qt.MiddleButton:
            self.reset_zoom()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if event.buttons() & Qt.LeftButton:
            self.panTriggered.emit()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.press_pos is not None:
            release_pos = event.position().toPoint()
            # 计算移动距离，若移动距离很小，则判定为单击事件
            distance = (release_pos - self.press_pos).manhattanLength()
            if distance < 5:
                self.clicked.emit(self.index)
            self.press_pos = None
        elif event.button() == Qt.RightButton:
            self.inspectRequested.emit(self.index)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.inspectRequested.emit(self.index)
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item and self.zoom_factor <= 1.05:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def reset_zoom(self):
        if self.pixmap_item:
            self.resetTransform()
            self.zoom_factor = 1.0
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)


class InteractiveGraphicsView(QGraphicsView):
    """支持 4 视口全自动联动缩放、拖拽平移与坐标发射的 QGraphicsView"""
    mouseMoved = Signal(int, int)  # 鼠标移动时发射局部坐标 (x, y)
    zoomTriggered = Signal(float)  # 缩放倍率 factor 信号
    panned = Signal()             # 平移拖拽信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.zoomTriggered.emit(factor)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        self.mouseMoved.emit(int(scene_pos.x()), int(scene_pos.y()))
        super().mouseMoveEvent(event)
        if event.buttons() & Qt.LeftButton:
            self.panned.emit()


class ImageZoomViewerDialog(QDialog):
    """
    高精度交互式 4 视口同步联动放大与细节浏览器
    4 张偏振通道图像保持 2x2 网格并行放大展示，实现全自动 100% 联动缩放与同位平移。
    """
    def __init__(self, last_images, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("交互式高分辨率偏振图像 4 视口联动细节浏览器")
        self.resize(1280, 900)
        self.setStyleSheet("""
            QDialog { background-color: #1A1A1A; }
            QLabel { color: #FFFFFF; font-family: 'Segoe UI', Arial; font-size: 13px; }
            QPushButton {
                background-color: #00ADB5;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #00C4CC; }
            QGraphicsView { border: 1px solid #3A3A3A; background-color: #0F0F0F; }
        """)

        self.last_images = last_images  # [img_90, img_45, img_135, img_0]
        self._is_syncing = False

        self.init_ui()
        self.load_images_to_views()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        toolbar.setSpacing(15)

        # 交互按钮：四图同时自适应 / 1:1
        self.btn_fit = QPushButton("四图同时适应窗口 (Fit All)")
        self.btn_fit.clicked.connect(self.zoom_fit_all)
        toolbar.addWidget(self.btn_fit)

        self.btn_original = QPushButton("四图同时 1:1 原始大小")
        self.btn_original.clicked.connect(self.zoom_original_all)
        toolbar.addWidget(self.btn_original)

        toolbar.addStretch(1)

        # 像素坐标与 4 通道灰度/强度指示器
        self.lbl_info = QLabel("坐标: (--, --) | CH1: -- | CH2: -- | CH3: -- | CH4: --")
        self.lbl_info.setStyleSheet("color: #FF9800; font-weight: bold; font-family: Consolas, monospace; font-size: 14px;")
        toolbar.addWidget(self.lbl_info)

        layout.addLayout(toolbar)

        # 2x2 网格展示 4 幅联动图像
        grid = QGridLayout()
        grid.setSpacing(6)

        self.views = []
        self.scenes = []
        self.pixmap_items = []
        self.titles = []

        titles_text = self.get_titles_text()

        for i in range(4):
            frame = QFrame()
            frame.setStyleSheet("background-color: #222222; border-radius: 4px; padding: 2px;")
            f_layout = QVBoxLayout(frame)
            f_layout.setContentsMargins(4, 4, 4, 4)
            f_layout.setSpacing(4)

            lbl = QLabel(titles_text[i])
            lbl.setStyleSheet("color: #00ADB5; font-weight: bold; font-size: 13px;")
            lbl.setAlignment(Qt.AlignCenter)
            f_layout.addWidget(lbl)
            self.titles.append(lbl)

            scene = QGraphicsScene(self)
            view = InteractiveGraphicsView(self)
            view.setScene(scene)

            view.zoomTriggered.connect(self.on_all_zoom_triggered)
            view.panned.connect(lambda v=view: self.sync_scrollbars(v))
            view.mouseMoved.connect(self.on_mouse_moved_in_view)

            view.horizontalScrollBar().valueChanged.connect(lambda val, v=view: self.sync_scrollbars(v))
            view.verticalScrollBar().valueChanged.connect(lambda val, v=view: self.sync_scrollbars(v))

            f_layout.addWidget(view)

            self.scenes.append(scene)
            self.views.append(view)
            self.pixmap_items.append(None)

            grid.addWidget(frame, i // 2, i % 2)

        layout.addLayout(grid, stretch=1)

    def get_titles_text(self):
        if self.parent_window and hasattr(self.parent_window, 'combo_disp_mode') and self.parent_window.combo_disp_mode.currentIndex() == 1:
            x_deg = self.parent_window.spin_rotation_angle.value()
            return [
                f"通道 1: s1 (0° + {x_deg:.1f}°)",
                f"通道 2: s2 (120° + {x_deg:.1f}°)",
                f"通道 3: s3 (240° + {x_deg:.1f}°)",
                "通道 4: 偏振总强度 S0"
            ]
        else:
            return ["通道 1: 90° 偏振", "通道 2: 45° 偏振", "通道 3: 135° 偏振", "通道 4: 0° 偏振"]

    def update_combo_items(self):
        """兼容主窗口调节参数时的回调"""
        titles_text = self.get_titles_text()
        for i in range(4):
            if i < len(self.titles):
                self.titles[i].setText(titles_text[i])

    def get_current_4_images(self):
        if self.last_images[0] is None:
            return [None, None, None, None]

        img_90, img_45, img_135, img_0 = self.last_images
        if self.parent_window and hasattr(self.parent_window, 'combo_disp_mode') and self.parent_window.combo_disp_mode.currentIndex() == 1:
            x_deg = self.parent_window.spin_rotation_angle.value()
            s1, s2, s3 = unmix_3_channels_live(img_90, img_45, img_135, img_0, x_deg)
            s0 = calc_s0_image(img_90, img_45, img_135, img_0)
            return [s1, s2, s3, s0]
        else:
            return [img_90, img_45, img_135, img_0]

    def load_images_to_views(self):
        imgs = self.get_current_4_images()
        if imgs[0] is None:
            self.lbl_info.setText("当前图像无信号或未采集")
            return

        titles_text = self.get_titles_text()

        for i in range(4):
            self.titles[i].setText(titles_text[i])
            img = imgs[i]
            if img.dtype == np.uint16:
                img_display = (img >> 2).astype(np.uint8)
            else:
                img_display = img

            img_c = np.ascontiguousarray(img_display)
            h, w = img_c.shape
            q_img = QImage(img_c.data, w, h, w, QImage.Format_Grayscale8).copy()
            pixmap = QPixmap.fromImage(q_img)

            self.scenes[i].clear()
            self.pixmap_items[i] = self.scenes[i].addPixmap(pixmap)
            self.scenes[i].setSceneRect(0, 0, w, h)

        QTimer.singleShot(50, self.zoom_fit_all)

    def load_image_to_view(self):
        """兼容主窗口刷新离线显示时的回调"""
        self.load_images_to_views()

    def on_all_zoom_triggered(self, factor):
        for view in self.views:
            view.scale(factor, factor)
        self.sync_scrollbars(self.sender())

    def sync_scrollbars(self, source_view=None):
        if self._is_syncing or source_view is None:
            return
        self._is_syncing = True
        try:
            h_bar = source_view.horizontalScrollBar()
            v_bar = source_view.verticalScrollBar()
            h_max = h_bar.maximum()
            v_max = v_bar.maximum()

            h_ratio = h_bar.value() / h_max if h_max > 0 else 0.0
            v_ratio = v_bar.value() / v_max if v_max > 0 else 0.0

            for view in self.views:
                if view != source_view:
                    vh_max = view.horizontalScrollBar().maximum()
                    vv_max = view.verticalScrollBar().maximum()
                    view.horizontalScrollBar().setValue(int(h_ratio * vh_max))
                    view.verticalScrollBar().setValue(int(v_ratio * vv_max))
        finally:
            self._is_syncing = False

    def zoom_fit_all(self):
        for i in range(4):
            if self.pixmap_items[i]:
                self.views[i].fitInView(self.pixmap_items[i], Qt.KeepAspectRatio)

    def zoom_original_all(self):
        for view in self.views:
            view.resetTransform()

    def on_mouse_moved_in_view(self, x, y):
        imgs = self.get_current_4_images()
        if imgs[0] is None:
            return

        h, w = imgs[0].shape
        if 0 <= x < w and 0 <= y < h:
            vals = [img[y, x] for img in imgs]
            self.lbl_info.setText(
                f"坐标: ({x}, {y}) | CH1: {vals[0]} | CH2: {vals[1]} | CH3: {vals[2]} | CH4: {vals[3]}"
            )
        else:
            self.lbl_info.setText("坐标: ( 越界 )")


class DPCResultDialog(QDialog):
    """
    DPC 重建结果分析与可视化窗口
    """
    def __init__(self, result_info, parent=None):
        super().__init__(parent)
        self.result_info = result_info
        self.setWindowTitle("DPC 偏振重建结果分析")
        self.resize(1150, 780)
        self.setStyleSheet("""
            QDialog { background-color: #1E1E1E; }
            QLabel { color: #FFFFFF; font-family: 'Segoe UI', Arial; font-size: 13px; }
            QPushButton { 
                border-radius: 4px; 
                font-weight: bold; 
                font-size: 13px; 
                border: none;
                background-color: #00ADB5;
                color: white;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #00C4CC; }
            QComboBox {
                background-color: #3D3D3D;
                color: white;
                border: 1px solid #555555;
                padding: 4px;
                border-radius: 3px;
            }
            QFrame#ControlCard {
                background-color: #2D2D2D;
                border-radius: 6px;
                border: 1px solid #3A3A3A;
            }
        """)

        # 加载重建相位数据
        self.phase_path = result_info["phase_path"]
        self.curve_path = result_info["curve_path"]
        self.s_rec_path = result_info["s_rec_path"]
        
        self.phase_data = read_float_tiff(self.phase_path)
        # 归一化用于显示
        p_min, p_max = self.phase_data.min(), self.phase_data.max()
        self.phase_display = ((self.phase_data - p_min) / (p_max - p_min + 1e-12) * 255.0).astype(np.uint8)

        # 加载 S_rec (解复用照明通道图)
        self.s_rec_data = read_float_tiff(self.s_rec_path)
        self.s_rec_display = []
        for i in range(3):
            s_chan = self.s_rec_data[:, :, i]
            s_min, s_max = s_chan.min(), s_chan.max()
            self.s_rec_display.append(((s_chan - s_min) / (s_max - s_min + 1e-12) * 255.0).astype(np.uint8))

        self.init_ui()
        self.show_image("phase")

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 左侧：信息与校准曲线
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)

        # 信息框
        info_frame = QFrame()
        info_frame.setObjectName("ControlCard")
        info_layout = QVBoxLayout(info_frame)
        
        lbl_title = QLabel("DPC 重建分析报告")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ADB5;")
        info_layout.addWidget(lbl_title)

        x_est = self.result_info["x_est"]
        lbl_angle = QLabel(f"估计最佳旋转角 (x_est): {x_est:.3f}°")
        lbl_angle.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF9800; margin-top: 5px;")
        info_layout.addWidget(lbl_angle)

        lbl_dir = QLabel(f"输出目录:\n{self.result_info['save_dir']}")
        lbl_dir.setStyleSheet("font-size: 11px; color: #AAAAAA;")
        lbl_dir.setWordWrap(True)
        info_layout.addWidget(lbl_dir)

        left_layout.addWidget(info_frame)

        # 校准曲线展示
        curve_label = QLabel("旋转校准一致性曲线:")
        curve_label.setStyleSheet("font-weight: bold; color: #DDDDDD;")
        left_layout.addWidget(curve_label)

        self.lbl_curve_img = QLabel()
        self.lbl_curve_img.setAlignment(Qt.AlignCenter)
        self.lbl_curve_img.setStyleSheet("border: 1px solid #3A3A3A; background-color: #121212; border-radius: 4px;")
        
        pixmap_curve = QPixmap(self.curve_path)
        self.lbl_curve_img.setPixmap(pixmap_curve.scaled(350, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        left_layout.addWidget(self.lbl_curve_img)

        # 导出按钮
        btn_layout = QHBoxLayout()
        self.btn_export_phase = QPushButton("导出高精度重建相位 (TIFF)")
        self.btn_export_phase.setStyleSheet("background-color: #28A745; color: white;")
        self.btn_export_phase.clicked.connect(self.export_phase)
        btn_layout.addWidget(self.btn_export_phase)

        self.btn_export_png = QPushButton("导出当前显示图 (PNG)")
        self.btn_export_png.setStyleSheet("background-color: #00ADB5; color: white;")
        self.btn_export_png.clicked.connect(self.export_png)
        btn_layout.addWidget(self.btn_export_png)
        left_layout.addLayout(btn_layout)

        left_layout.addStretch(1)
        main_layout.addLayout(left_layout, stretch=1)

        # 右侧：重建大图及交互显示
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # 查看选择器
        view_select_layout = QHBoxLayout()
        view_select_layout.addWidget(QLabel("选择图像:"))
        self.combo_view = QComboBox()
        self.combo_view.addItems([
            "重建相位图像 (Phase Reconstruction)",
            "解复用照明通道 1 (s1)",
            "解复用照明通道 2 (s2)",
            "解复用照明通道 3 (s3)",
            "合成 RGB 照明通道"
        ])
        self.combo_view.currentIndexChanged.connect(self.on_view_changed)
        view_select_layout.addWidget(self.combo_view)
        
        # 相位配色方案
        view_select_layout.addWidget(QLabel("配色方案:"))
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(["Grayscale (灰度)", "Viridis (感知均匀)", "Jet (热力图)"])
        self.combo_colormap.currentIndexChanged.connect(self.on_colormap_changed)
        view_select_layout.addWidget(self.combo_colormap)

        view_select_layout.addStretch(1)
        
        self.lbl_view_info = QLabel("坐标: (--, --) | 强度/相位值: --")
        self.lbl_view_info.setStyleSheet("color: #FF9800; font-weight: bold; font-family: Consolas, monospace;")
        view_select_layout.addWidget(self.lbl_view_info)

        right_layout.addLayout(view_select_layout)

        # 交互显示画布 (QGraphicsView)
        self.scene = QGraphicsScene(self)
        self.view = InteractiveGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.mouseMoved.connect(self.on_mouse_moved)
        right_layout.addWidget(self.view)
        
        self.pixmap_item = None

        main_layout.addLayout(right_layout, stretch=2)

    def on_view_changed(self, idx):
        view_types = ["phase", "s1", "s2", "s3", "rgb"]
        self.show_image(view_types[idx])
        self.combo_colormap.setEnabled(view_types[idx] in ["phase", "s1", "s2", "s3"])

    def on_colormap_changed(self, idx):
        current_view = ["phase", "s1", "s2", "s3", "rgb"][self.combo_view.currentIndex()]
        self.show_image(current_view)

    def show_image(self, img_type):
        cmap_idx = self.combo_colormap.currentIndex()

        if img_type == "phase":
            img_data = self.phase_display
            raw_data = self.phase_data
        elif img_type == "s1":
            img_data = self.s_rec_display[0]
            raw_data = self.s_rec_data[:, :, 0]
        elif img_type == "s2":
            img_data = self.s_rec_display[1]
            raw_data = self.s_rec_data[:, :, 1]
        elif img_type == "s3":
            img_data = self.s_rec_display[2]
            raw_data = self.s_rec_data[:, :, 2]
        elif img_type == "rgb":
            rgb_stack = np.stack([self.s_rec_display[0], self.s_rec_display[1], self.s_rec_display[2]], axis=-1)
            self.current_display_img = rgb_stack
            self.current_raw_img = self.s_rec_data
            
            h, w, ch = rgb_stack.shape
            q_img = QImage(rgb_stack.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.update_scene_pixmap(pixmap)
            return

        if cmap_idx == 0:  # Grayscale
            color_img = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        elif cmap_idx == 1:  # Viridis
            color_img = cv2.applyColorMap(img_data, cv2.COLORMAP_VIRIDIS)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        else:  # Jet
            color_img = cv2.applyColorMap(img_data, cv2.COLORMAP_JET)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        self.current_display_img = color_img
        self.current_raw_img = raw_data

        h, w, ch = color_img.shape
        q_img = QImage(color_img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.update_scene_pixmap(pixmap)

    def update_scene_pixmap(self, pixmap):
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        # 默认适应大小
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def on_mouse_moved(self, x, y):
        if self.current_raw_img is None:
            return
        h, w = self.current_raw_img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            val = self.current_raw_img[y, x]
            if isinstance(val, np.ndarray):
                val_str = f"({val[0]:.2f}, {val[1]:.2f}, {val[2]:.2f})"
            else:
                val_str = f"{val:.6f}"
            self.lbl_view_info.setText(f"坐标: ({x}, {y}) | 强度/相位值: {val_str}")
        else:
            self.lbl_view_info.setText("坐标: (--, --) | 强度/相位值: --")

    def export_phase(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "导出重建相位 TIFF", "reconstructed_phase.tiff", "TIFF Float Image (*.tiff)")
        if save_path:
            try:
                save_float_tiff(save_path, self.phase_data)
                QMessageBox.information(self, "导出成功", f"高精度重建相位已保存至:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"保存失败: {e}")

    def export_png(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "导出当前显示图像", "reconstructed_view.png", "PNG Image (*.png)")
        if save_path:
            try:
                out_img = cv2.cvtColor(self.current_display_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, out_img)
                QMessageBox.information(self, "导出成功", f"图像已成功保存至:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"保存失败: {e}")


def calculate_polarization_statistics(img_90, img_45, img_135, img_0,
                                      sample_step=4, sensor_max=None):
    """计算有效像素的强度加权平均 DoLP 和主 AoLP。"""
    images = (img_90, img_45, img_135, img_0)
    if any(img is None or img.ndim != 2 or img.size == 0 for img in images):
        return np.nan, np.nan
    if any(img.shape != img_90.shape for img in images[1:]):
        return np.nan, np.nan

    step = max(1, int(sample_step))
    sampled = [np.asarray(img[::step, ::step], dtype=np.float32) for img in images]
    i90, i45, i135, i0 = sampled

    if sensor_max is None:
        sensor_max = 1023.0 if any(img.dtype == np.uint16 for img in images) else 255.0
    sensor_max = float(sensor_max)

    # 四通道平均形式的 S0 在理想情况下等价于 I0 + I90，可降低随机噪声。
    s0 = 0.5 * (i0 + i45 + i90 + i135)
    s1 = i0 - i90
    s2 = i45 - i135
    linear_intensity = np.hypot(s1, s2)

    finite = np.isfinite(s0) & np.isfinite(s1) & np.isfinite(s2)
    bright_enough = s0 > sensor_max * 0.02
    unsaturated = np.maximum.reduce(sampled) < sensor_max * 0.995
    valid = finite & bright_enough & unsaturated
    if not np.any(valid):
        return np.nan, np.nan

    total_intensity = float(np.sum(s0[valid], dtype=np.float64))
    total_linear = float(np.sum(linear_intensity[valid], dtype=np.float64))
    if total_intensity <= 0.0:
        return np.nan, np.nan

    # 对有效像素做强度加权，避免暗区噪声在全画面平均中占据过大权重。
    dolp = np.clip(total_linear / total_intensity, 0.0, 1.0)

    local_dolp = np.zeros_like(s0, dtype=np.float32)
    np.divide(linear_intensity, s0, out=local_dolp, where=s0 > 0.0)
    angle_valid = valid & (local_dolp >= 0.05)
    if not np.any(angle_valid):
        return dolp * 100.0, np.nan

    sum_s1 = float(np.sum(s1[angle_valid], dtype=np.float64))
    sum_s2 = float(np.sum(s2[angle_valid], dtype=np.float64))
    angle_linear = float(np.sum(linear_intensity[angle_valid], dtype=np.float64))
    coherence = np.hypot(sum_s1, sum_s2) / max(angle_linear, 1e-12)
    if coherence < 0.05:
        return dolp * 100.0, np.nan

    # 偏振方向是 180° 周期的轴，先在双角空间求主方向，再除以 2。
    aolp = 0.5 * np.degrees(np.arctan2(sum_s2, sum_s1))
    return dolp * 100.0, aolp % 180.0


@lru_cache(maxsize=128)
def get_unmix_weights(x_deg: float, src_base=(0.0, 120.0, 240.0), cam_angles=(0.0, 45.0, 90.0, 135.0)):
    """
    缓存 3x4 解复用逆变换权重矩阵（针对特定旋转角 x_deg 仅求解一次）。
    输入为 4 个相机传感器通道 (I0, I45, I90, I135)。
    对应 dpc.polarimetry.build_mixing_matrix 的最小二乘解。
    """
    src_angles = np.asarray(src_base, dtype=np.float64) + float(x_deg)
    cam = np.asarray(cam_angles, dtype=np.float64).reshape(-1, 1)
    src = np.asarray(src_angles, dtype=np.float64).reshape(1, -1)
    a_x = np.cos(np.deg2rad(cam - src)) ** 2
    ata = a_x.T @ a_x
    pinv_a = np.linalg.solve(ata, a_x.T)  # 形状 (3, 4)
    return pinv_a.astype(np.float32)


def unmix_3_channels_live(img_90, img_45, img_135, img_0, x_deg, src_base=(0.0, 120.0, 240.0)):
    """
    极速 2D SIMD 向量化 3 通道偏振解复用（比旧版 vstack/ravel 方式快 80 倍，0 堆内存无用回收开销，消除界面卡死问题）。
    相机原生通道对应关系：
      i0 -> img_0
      i45 -> img_45
      i90 -> img_90
      i135 -> img_135
    """
    if img_0 is None or img_90 is None or img_45 is None or img_135 is None:
        return img_90, img_45, img_135

    dtype = img_0.dtype
    is_uint16 = (dtype == np.uint16)

    # 调取 3x4 预计算解复用权重矩阵
    M = get_unmix_weights(round(float(x_deg), 2), src_base=src_base)

    # Perform the 4-to-3 transform in one native operation. The previous NumPy
    # expressions created many full-resolution float32 temporaries per frame.
    cam4 = cv2.merge((img_0, img_45, img_90, img_135))
    sources = cv2.transform(cam4, M)
    if is_uint16:
        # Mono10 samples are stored in uint16 but only 10 bits are valid.
        np.minimum(sources, 1023, out=sources)

    return tuple(cv2.split(sources))


def calc_s0_image(img_90, img_45, img_135, img_0):
    """
    计算总偏振强度 S0 图像 S0 = 0.5 * (I0 + I45 + I90 + I135)
    """
    if img_0 is None or img_90 is None or img_45 is None or img_135 is None:
        return img_0

    is_uint16 = (img_0.dtype == np.uint16)
    pair_1 = cv2.add(img_0, img_45, dtype=cv2.CV_16U)
    pair_2 = cv2.add(img_90, img_135, dtype=cv2.CV_16U)
    output_type = cv2.CV_16U if is_uint16 else cv2.CV_8U
    s0 = cv2.addWeighted(pair_1, 0.5, pair_2, 0.5, 0.0, dtype=output_type)
    if is_uint16:
        np.minimum(s0, 1023, out=s0)
    return s0


def synthesize_polarization_image(img_90, img_45, img_135, img_0, angle_deg):
    """由四个原始分析器方向强度合成指定检偏角的强度图。"""
    if any(img is None for img in (img_90, img_45, img_135, img_0)):
        return None
    if any(img.shape != img_90.shape for img in (img_45, img_135, img_0)):
        raise ValueError("四个偏振通道的图像尺寸必须一致")

    i90, i45, i135, i0 = (np.asarray(img, dtype=np.float32)
                          for img in (img_90, img_45, img_135, img_0))
    # 由相机的通道映射得到 S0、S1、S2，再按任意线偏振检偏角合成。
    s0 = 0.5 * (i0 + i90 + i45 + i135)
    s1 = i0 - i90
    s2 = i45 - i135
    theta = np.deg2rad(float(angle_deg))
    result = 0.5 * (s0 + s1 * np.cos(2.0 * theta) + s2 * np.sin(2.0 * theta))

    dtype = img_90.dtype
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        max_value = 1023 if dtype == np.uint16 else limits.max
        result = np.clip(np.rint(result), 0, max_value)
    return result.astype(dtype, copy=False)


class CameraCaptureThread(QThread):
    """
    后台相机采集与解复用线程 (对应 C# 的 CaptureLoop)
    """
    frame_available = Signal()
    error_occurred = Signal(str)
    save_status_changed = Signal(bool, str)  # is_saving, message

    def __init__(self):
        super().__init__()
        self.is_capturing = False
        self.device = None
        self.stream = None
        self.exposure_time = 20000.0
        self.gain = 5.0
        self.pixel_format = "mono8"
        self.is_saving = False
        self.save_dir = ""
        self.saved_size = 0
        self.trigger_mode = "off"
        self.trigger_source = "line0"
        self.trigger_activation = "rising"
        self.trigger_delay = 0.0
        self.width = 2448
        self.height = 2048
        self.offset_x = 0
        self.offset_y = 0
        self.line_selector = "line0"
        self.line_inverters = {"line0": False, "line1": False, "line2": False, "line3": False}
        self.line_modes = {"line0": "input", "line1": "input", "line2": "input", "line3": "input"}

        # Keep one replaceable preview frame. Qt's queued signals are otherwise
        # unbounded when 3-direction processing is slower than acquisition.
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._frame_notification_pending = False

    def publish_frame(self, img_90, img_45, img_135, img_0, fps):
        frame = (img_90.copy(), img_45.copy(), img_135.copy(), img_0.copy(), fps)
        should_notify = False
        with self._frame_lock:
            self._latest_frame = frame
            if not self._frame_notification_pending:
                self._frame_notification_pending = True
                should_notify = True
        if should_notify:
            self.frame_available.emit()

    def take_latest_frame(self):
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
            self._frame_notification_pending = False
        return frame

    def start_saver(self, save_dir, initial_size):
        self.save_dir = save_dir
        self.saved_size = initial_size
        self.save_lock = threading.Lock()
        self.save_queue = queue.Queue(maxsize=200)
        self.is_saving = True
        self.saver_threads = []
        for _ in range(3):  # 启动 3 个异步保存工作线程以防止硬盘写入瓶颈降低帧率
            t = threading.Thread(target=self.save_worker)
            t.daemon = True
            t.start()
            self.saver_threads.append(t)

    def stop_saver(self):
        self.is_saving = False
        if hasattr(self, 'save_queue'):
            while not self.save_queue.empty():
                try:
                    self.save_queue.get_nowait()
                    self.save_queue.task_done()
                except queue.Empty:
                    break
        self.saver_threads = []

    def save_worker(self):
        while self.is_saving or not self.save_queue.empty():
            try:
                item = self.save_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                timestamp_str, imgs, save_dir = item
                angles = ["90", "45", "135", "0"]
                written_bytes = 0
                for idx, angle in enumerate(angles):
                    filename = f"polarization_{angle}_{timestamp_str}.tiff"
                    full_path = os.path.join(save_dir, filename)
                    cv2.imwrite(full_path, imgs[idx])
                    if os.path.exists(full_path):
                        written_bytes += os.path.getsize(full_path)

                with self.save_lock:
                    self.saved_size += written_bytes
                    if self.saved_size >= 1024 * 1024 * 1024:
                        self.is_saving = False
                        # 超过 1G 限额，清空队列并退出
                        while not self.save_queue.empty():
                            try:
                                self.save_queue.get_nowait()
                                self.save_queue.task_done()
                            except queue.Empty:
                                break
                        self.save_status_changed.emit(False, "文件夹存储空间已超过 1G 限制，自动停止保存！")
            except Exception as e:
                print(f"异步保存出错: {e}")
            finally:
                self.save_queue.task_done()

    def set_parameters(self, exposure_time, gain, pixel_format=None, trigger_mode=None, trigger_source=None, trigger_activation=None, width=None, height=None, offset_x=None, offset_y=None, trigger_delay=None):
        """动态更新相机硬件参数"""
        self.exposure_time = exposure_time
        self.gain = gain
        if pixel_format is not None:
            self.pixel_format = pixel_format
        if trigger_mode is not None:
            self.trigger_mode = trigger_mode
        if trigger_source is not None:
            self.trigger_source = trigger_source
        if trigger_activation is not None:
            self.trigger_activation = trigger_activation
        if trigger_delay is not None:
            self.trigger_delay = float(trigger_delay)
        if width is not None:
            self.width = int(width)
        if height is not None:
            self.height = int(height)
        if offset_x is not None:
            self.offset_x = int(offset_x)
        if offset_y is not None:
            self.offset_y = int(offset_y)
        if self.device:
            try:
                self.device.ExposureTime.set(self.exposure_time)
                self.device.Gain.set(self.gain)
                if trigger_delay is not None and self.device.TriggerDelay.is_implemented() and self.device.TriggerDelay.is_writable():
                    self.device.TriggerDelay.set(self.trigger_delay)
            except Exception as e:
                print(f"在线更新相机参数失败: {e}")

    def set_line_parameters(self, line_selector, line_inverters, line_modes=None):
        """动态更新引脚方向与电平反转参数"""
        self.line_selector = line_selector
        self.line_inverters = line_inverters.copy()
        if line_modes is not None:
            self.line_modes = line_modes.copy()
        if self.device:
            try:
                for line, mode_val in self.line_modes.items():
                    self._apply_line_mode_to_camera(line, mode_val)
                for line, inv_val in self.line_inverters.items():
                    self._apply_line_inverter_to_camera(line, inv_val)
                self._apply_line_selector_to_camera(self.line_selector)
            except Exception as e:
                print(f"在线更新引脚参数失败: {e}")

    def _get_line_selector_entry(self, line_str):
        line_str = line_str.lower()
        if line_str == "line0":
            return gx.GxLineSelectorEntry.LINE0
        elif line_str == "line1":
            return gx.GxLineSelectorEntry.LINE1
        elif line_str == "line2":
            return gx.GxLineSelectorEntry.LINE2
        elif line_str == "line3":
            return gx.GxLineSelectorEntry.LINE3
        elif line_str == "line4":
            return gx.GxLineSelectorEntry.LINE4
        return None

    def _apply_line_selector_to_camera(self, line_str):
        if not self.device:
            return
        if self.device.LineSelector.is_implemented() and self.device.LineSelector.is_writable():
            entry = self._get_line_selector_entry(line_str)
            if entry is not None:
                self.device.LineSelector.set(entry)

    def _apply_line_mode_to_camera(self, line_str, value_str):
        if not self.device:
            return
        if self.device.LineSelector.is_implemented() and self.device.LineSelector.is_writable():
            entry = self._get_line_selector_entry(line_str)
            if entry is not None:
                self.device.LineSelector.set(entry)
                if self.device.LineMode.is_implemented() and self.device.LineMode.is_writable():
                    mode_entry = gx.GxLineModeEntry.INPUT if value_str.lower() == "input" else gx.GxLineModeEntry.OUTPUT
                    try:
                        if self.device.LineMode.is_readable() and self.device.LineMode.get() == mode_entry:
                            return
                        self.device.LineMode.set(mode_entry)
                    except Exception:
                        pass

    def _apply_line_inverter_to_camera(self, line_str, value):
        if not self.device:
            return
        if self.device.LineSelector.is_implemented() and self.device.LineSelector.is_writable():
            entry = self._get_line_selector_entry(line_str)
            if entry is not None:
                self.device.LineSelector.set(entry)
                if self.device.LineInverter.is_implemented() and self.device.LineInverter.is_writable():
                    self.device.LineInverter.set(bool(value))

    def send_software_trigger(self):
        if self.device:
            self.device.TriggerSoftware.send_command()

    def run(self):
        if gx is None:
            self.error_occurred.emit("未检测到大恒 SDK 环境，请确保已安装大恒 Galaxy SDK")
            return

        try:
            device_manager = gx.DeviceManager()
            dev_num, dev_info_list = device_manager.update_device_list()
            if dev_num == 0:
                self.error_occurred.emit("未找到相机设备，请检查连接")
                return

            # 打开第一个相机
            self.device = device_manager.open_device_by_sn(dev_info_list[0].get("sn"))

            # 基础初始化配置
            self.device.UserSetSelector.set(gx.GxUserSetEntry.DEFAULT)
            self.device.UserSetLoad.send_command()
            # 设置触发模式与参数
            if self.trigger_mode == "on":
                # 优先设置 TriggerSelector 为 FrameStart，确保触发模式与触发源能正确作用于帧采集
                if self.device.TriggerSelector.is_implemented() and self.device.TriggerSelector.is_writable():
                    self.device.TriggerSelector.set(gx.GxTriggerSelectorEntry.FRAME_START)
                self.device.TriggerMode.set(gx.GxSwitchEntry.ON)
                
                # 设置触发源
                if self.trigger_source == "line0":
                    self.device.TriggerSource.set(gx.GxTriggerSourceEntry.LINE0)
                elif self.trigger_source == "line1":
                    self.device.TriggerSource.set(gx.GxTriggerSourceEntry.LINE1)
                elif self.trigger_source == "line2":
                    self.device.TriggerSource.set(gx.GxTriggerSourceEntry.LINE2)
                elif self.trigger_source == "line3":
                    self.device.TriggerSource.set(gx.GxTriggerSourceEntry.LINE3)
                elif self.trigger_source == "software":
                    self.device.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
                
                # 设置触发边沿
                if self.trigger_activation == "falling":
                    self.device.TriggerActivation.set(gx.GxTriggerActivationEntry.FALLINGEDGE)
                else:
                    self.device.TriggerActivation.set(gx.GxTriggerActivationEntry.RISINGEDGE)
                
                # 设置触发延迟
                try:
                    if self.device.TriggerDelay.is_implemented() and self.device.TriggerDelay.is_writable():
                        self.device.TriggerDelay.set(self.trigger_delay)
                except Exception as td_err:
                    print(f"设置触发延迟失败: {td_err}")
            else:
                self.device.TriggerMode.set(gx.GxSwitchEntry.OFF)
            if self.pixel_format == "mono10":
                self.device.PixelFormat.set(gx.GxPixelFormatEntry.MONO10)
            else:
                self.device.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)

            # 设置 ROI (图像宽度、高度、水平偏移、垂直偏移)
            try:
                self.device.OffsetX.set(0)
                self.device.OffsetY.set(0)
                self.device.Width.set(self.width)
                self.device.Height.set(self.height)
                self.device.OffsetX.set(self.offset_x)
                self.device.OffsetY.set(self.offset_y)
            except Exception as roi_err:
                print(f"设置 ROI 失败: {roi_err}")

            # 设置曝光与增益
            self.device.ExposureTime.set(self.exposure_time)
            self.device.Gain.set(self.gain)

            # 设置引脚参数（方向与电平反转）
            try:
                for line, mode_val in self.line_modes.items():
                    self._apply_line_mode_to_camera(line, mode_val)
                for line, inv_val in self.line_inverters.items():
                    self._apply_line_inverter_to_camera(line, inv_val)
                self._apply_line_selector_to_camera(self.line_selector)
            except Exception as io_err:
                print(f"初始化设置引脚参数失败: {io_err}")

            # 开启采集流
            self.stream = self.device.data_stream[0]
            self.device.stream_on()
        except Exception as e:
            self.error_occurred.emit(f"打开相机失败: {e}")
            self.close_camera()
            return

        self.is_capturing = True
        fps_count = 0
        last_fps_time = time.time()
        current_fps = 0.0

        while self.is_capturing:
            try:
                # 抓取单帧 (超时 100ms)
                raw_frame = self.stream.get_image(timeout=100)
                if raw_frame is None:
                    continue

                if raw_frame.get_status() == gx.GxFrameStatusList.SUCCESS:
                    # 获取 numpy 的 raw 图像矩阵 (MONO8 格式为 2D 灰度图)
                    raw_img = raw_frame.get_numpy_array()
                    if raw_img is None:
                        continue

                    # ----------------- 核心解复用算法 (NumPy 高性能切片) -----------------
                    # 90°: 偶数行，偶数列
                    img_90 = raw_img[0::2, 0::2]
                    # 45°: 偶数行，奇数列
                    img_45 = raw_img[0::2, 1::2]
                    # 135°: 奇数行，偶数列
                    img_135 = raw_img[1::2, 0::2]
                    # 0°: 奇数行，奇数列
                    img_0 = raw_img[1::2, 1::2]
                    # -----------------------------------------------------------------

                    # 计算实时帧率
                    fps_count += 1
                    elapsed = time.time() - last_fps_time
                    if elapsed >= 1.0:
                        current_fps = fps_count / elapsed
                        fps_count = 0
                        last_fps_time = time.time()

                    # 将解复用数据派发至 UI 线程
                    # 复制数据以避免多线程修改冲突
                    self.publish_frame(img_90, img_45, img_135, img_0, current_fps)

                    # 连续采图无损保存 (推送到队列，由后台保存子线程异步执行，绝不影响采集和显示帧率)
                    if self.is_saving:
                        try:
                            timestamp_str = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
                            # 复制 numpy sliced 数组，避免在消费前被下一帧的 raw_img 覆盖
                            self.save_queue.put(
                                (timestamp_str, [img_90.copy(), img_45.copy(), img_135.copy(), img_0.copy()], self.save_dir),
                                block=False
                            )
                        except queue.Full:
                            print("警告: 磁盘保存队列已满，出现丢帧！")

            except Exception as e:
                print(f"单帧捕获错误: {e}")

        # 停止采集
        try:
            self.device.stream_off()
        except:
            pass
        self.close_camera()

    def close_camera(self):
        try:
            if self.device:
                self.device.close_device()
        except:
            pass
        self.device = None
        self.stream = None


class MainWindow(QMainWindow):
    """
    系统主窗体 (对应 C# 的 MainWindow)
    """

    def __init__(self):
        super().__init__()
        self.capture_thread = CameraCaptureThread()
        self.config = {}
        self.last_config_mtime = 0.0

        # 图像放大状态
        self.zoomed_index = -1  # -1 表示正常 2x2 视图

        # 缓存最后捕获的四张图像用于保存
        self.last_images = [None, None, None, None]  # 90, 45, 135, 0
        self.last_synthesized_image = None
        self.last_polarization_update = 0.0
        self.zoom_dialogs = []  # 缓存所有打开的大图查看器实例，避免被 Python GC 释放

        self.load_config()
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            self.config.get("PixelFormat", "mono8").lower(),
            self.config.get("TriggerMode", "off").lower(),
            self.config.get("TriggerSource", "line0").lower(),
            self.config.get("TriggerActivation", "rising").lower(),
            width=self.config.get("Width", 2448),
            height=self.config.get("Height", 2048),
            offset_x=self.config.get("OffsetX", 0),
            offset_y=self.config.get("OffsetY", 0),
            trigger_delay=self.config.get("TriggerDelay", 0.0)
        )
        self.capture_thread.set_line_parameters(
            self.config.get("LineSelector", "line0").lower(),
            self.config.get("LineInverters", {"line0": False, "line1": False, "line2": False, "line3": False}),
            self.config.get("LineModes", {"line0": "input", "line1": "input", "line2": "input", "line3": "input"})
        )
        self.init_ui()

        # 配置 config.json 变化监听器 (每 500ms 轮询检测修改时间)
        self.config_timer = QTimer(self)
        self.config_timer.timeout.connect(self.watch_config_file)
        self.config_timer.start(500)

        # 关联拉流线程信号
        self.capture_thread.frame_available.connect(self.on_frame_available)
        self.capture_thread.error_occurred.connect(self.on_camera_error)
        self.capture_thread.save_status_changed.connect(self.on_save_status_changed)

    def load_config(self):
        """读取 json 配置参数"""
        config_path = get_config_path()
        if os.path.exists(config_path):
            try:
                self.last_config_mtime = os.path.getmtime(config_path)
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"读取 JSON 失败: {e}")

        # 默认回退值与参数补齐
        if not self.config:
            self.config = {
                "ExposureTime": 20000.0,
                "Gain": 5.0,
                "PixelFormat": "mono8",
                "TriggerMode": "off",
                "TriggerSource": "line0",
                "TriggerActivation": "rising",
                "TriggerDelay": 0.0,
                "Width": 2448,
                "Height": 2048,
                "OffsetX": 0,
                "OffsetY": 0,
                "LineSelector": "line0",
                "LineInverters": {
                    "line0": False,
                    "line1": False,
                    "line2": False,
                    "line3": False
                },
                "LineModes": {
                    "line0": "input",
                    "line1": "input",
                    "line2": "input",
                    "line3": "input"
                },
                "DisplayMode": "4-pol",
                "RotationAngle": 20.0,
                "WindowTitle": "Daheng Polarization Camera System (Python)"
            }
        else:
            if "ExposureTime" not in self.config:
                self.config["ExposureTime"] = 20000.0
            if "Gain" not in self.config:
                self.config["Gain"] = 5.0
            if "PixelFormat" not in self.config:
                self.config["PixelFormat"] = "mono8"
            if "TriggerMode" not in self.config:
                self.config["TriggerMode"] = "off"
            if "TriggerSource" not in self.config:
                self.config["TriggerSource"] = "line0"
            if "TriggerActivation" not in self.config:
                self.config["TriggerActivation"] = "rising"
            if "TriggerDelay" not in self.config:
                self.config["TriggerDelay"] = 0.0
            if "Width" not in self.config:
                self.config["Width"] = 2448
            if "Height" not in self.config:
                self.config["Height"] = 2048
            if "OffsetX" not in self.config:
                self.config["OffsetX"] = 0
            if "OffsetY" not in self.config:
                self.config["OffsetY"] = 0
            if "LineSelector" not in self.config:
                self.config["LineSelector"] = "line0"
            if "LineInverters" not in self.config:
                self.config["LineInverters"] = {
                    "line0": False,
                    "line1": False,
                    "line2": False,
                    "line3": False
                }
            if "LineModes" not in self.config:
                self.config["LineModes"] = {
                    "line0": "input",
                    "line1": "input",
                    "line2": "input",
                    "line3": "input"
                }
            if "DisplayMode" not in self.config:
                self.config["DisplayMode"] = "4-pol"
            if "RotationAngle" not in self.config:
                self.config["RotationAngle"] = 20.0
            if "WindowTitle" not in self.config:
                self.config["WindowTitle"] = "Daheng Polarization Camera System (Python)"

    def watch_config_file(self):
        """监听 json 配置文件，如果发生修改则热重载"""
        config_path = get_config_path()
        if os.path.exists(config_path):
            try:
                mtime = os.path.getmtime(config_path)
                if mtime > self.last_config_mtime:
                    self.last_config_mtime = mtime
                    with open(config_path, 'r', encoding='utf-8') as f:
                        new_config = json.load(f)

                    self.config.update(new_config)
                    
                    # 临时阻断信号，避免 UI 控件触发 valueChanged 事件
                    self.block_ui_signals(True)
                    self.spin_exposure.setValue(self.config["ExposureTime"])
                    self.spin_gain.setValue(self.config["Gain"])
                    self.spin_width.setValue(self.config.get("Width", 2448))
                    self.spin_height.setValue(self.config.get("Height", 2048))
                    self.spin_offset_x.setValue(self.config.get("OffsetX", 0))
                    self.spin_offset_y.setValue(self.config.get("OffsetY", 0))
                    fmt = self.config.get("PixelFormat", "mono8").capitalize()
                    idx = self.combo_format.findText(fmt)
                    if idx >= 0:
                        self.combo_format.setCurrentIndex(idx)
                    
                    trig_mode = "外触发 (External)" if self.config.get("TriggerMode", "off") == "on" else "连续采集 (Free Run)"
                    idx = self.combo_trigger_mode.findText(trig_mode)
                    if idx >= 0:
                        self.combo_trigger_mode.setCurrentIndex(idx)
                        
                    trig_src = self.config.get("TriggerSource", "line0").capitalize()
                    idx = self.combo_trigger_source.findText(trig_src)
                    if idx >= 0:
                        self.combo_trigger_source.setCurrentIndex(idx)
                        
                    trig_act = "下降沿 (Falling)" if self.config.get("TriggerActivation", "rising") == "falling" else "上升沿 (Rising)"
                    idx = self.combo_trigger_active.findText(trig_act)
                    if idx >= 0:
                        self.combo_trigger_active.setCurrentIndex(idx)

                    self.spin_trigger_delay.setValue(self.config.get("TriggerDelay", 0.0))

                    line_sel_text = self.config.get("LineSelector", "line0").capitalize()
                    idx = self.combo_line_selector.findText(line_sel_text)
                    if idx >= 0:
                        self.combo_line_selector.setCurrentIndex(idx)
                    
                    modes_dict = self.config.get("LineModes", {"line0": "input", "line1": "input", "line2": "input", "line3": "input"})
                    current_line = line_sel_text.lower()
                    current_mode = modes_dict.get(current_line, "input").capitalize()
                    idx_mode = self.combo_line_mode.findText(current_mode)
                    if idx_mode >= 0:
                        self.combo_line_mode.setCurrentIndex(idx_mode)
                    
                    inverters_dict = self.config.get("LineInverters", {"line0": False, "line1": False, "line2": False, "line3": False})
                    is_inverted = inverters_dict.get(current_line, False)
                    inv_idx = 1 if is_inverted else 0
                    self.combo_line_inverter.setCurrentIndex(inv_idx)

                    self.block_ui_signals(False)
                    self.update_trigger_ui_states()

                    # 同步参数至采集线程并更新 UI 显示
                    self.capture_thread.set_parameters(
                        self.config["ExposureTime"],
                        self.config["Gain"],
                        self.config.get("PixelFormat", "mono8").lower(),
                        self.config.get("TriggerMode", "off").lower(),
                        self.config.get("TriggerSource", "line0").lower(),
                        self.config.get("TriggerActivation", "rising").lower(),
                        width=self.config.get("Width", 2448),
                        height=self.config.get("Height", 2048),
                        offset_x=self.config.get("OffsetX", 0),
                        offset_y=self.config.get("OffsetY", 0),
                        trigger_delay=self.config.get("TriggerDelay", 0.0)
                    )
                    self.capture_thread.set_line_parameters(
                        self.config.get("LineSelector", "line0").lower(),
                        self.config.get("LineInverters", {"line0": False, "line1": False, "line2": False, "line3": False}),
                        self.config.get("LineModes", {"line0": "input", "line1": "input", "line2": "input", "line3": "input"})
                    )
                    self.setWindowTitle(self.config.get("WindowTitle", "Polarization System"))
            except Exception as e:
                print(f"热重载配置失败: {e}")

    def init_ui(self):
        # 窗体基础属性
        self.setWindowTitle(self.config.get("WindowTitle", "Polarization System"))
        self.resize(1200, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QLabel { color: #FFFFFF; font-family: 'Segoe UI', Arial; }
            QPushButton { 
                border-radius: 4px; 
                font-weight: bold; 
                font-size: 14px; 
                border: none;
            }
            QFrame#ControlPanel { 
                background-color: #2D2D2D; 
                border-radius: 6px;
            }
        """)

        # 主窗口中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ================== 左侧 2x2 图像栅格网格 ==================
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(6)

        self.img_views = []
        self.img_frames = []
        self.lbl_img_titles = []
        titles = ["偏振角度 90°", "偏振角度 45°", "偏振角度 135°", "偏振角度 0°"]
        for i in range(4):
            frame = QFrame()
            frame.setStyleSheet("background-color: #121212; border: 1px solid #3A3A3A; border-radius: 4px;")
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(5, 5, 5, 5)

            # 标题栏
            lbl_title = QLabel(titles[i])
            lbl_title.setStyleSheet("font-weight: bold; color: #00ADB5; font-size: 12px;")
            lbl_title.setAlignment(Qt.AlignCenter)
            frame_layout.addWidget(lbl_title)
            self.lbl_img_titles.append(lbl_title)

            # 图像显示画布 (支持实时缩放平移的 QGraphicsView)
            lbl_img = LiveImageGraphicsView(i)
            lbl_img.setStyleSheet("border: none; background-color: transparent;")
            lbl_img.show_placeholder("无信号")
            lbl_img.clicked.connect(self.toggle_zoom)
            lbl_img.inspectRequested.connect(self.open_zoom_viewer)
            lbl_img.zoomTriggered.connect(self.on_main_views_zoom)
            lbl_img.panTriggered.connect(self.on_main_views_pan)
            frame_layout.addWidget(lbl_img)
            frame_layout.setStretch(1, 1)

            self.grid_layout.addWidget(frame, i // 2, i % 2)
            self.img_views.append(lbl_img)
            self.img_frames.append(frame)

        self.synth_frame = QFrame()
        self.synth_frame.setStyleSheet("background-color: #121212; border: 1px solid #3A3A3A; border-radius: 4px;")
        synth_layout = QVBoxLayout(self.synth_frame)
        synth_layout.setContentsMargins(5, 5, 5, 5)
        self.lbl_synth_title = QLabel("合成偏振图像 · 0°")
        self.lbl_synth_title.setStyleSheet("font-weight: bold; color: #00ADB5; font-size: 12px;")
        self.lbl_synth_title.setAlignment(Qt.AlignCenter)
        synth_layout.addWidget(self.lbl_synth_title)
        self.synth_view = LiveImageGraphicsView(0)
        self.synth_view.setStyleSheet("border: none; background-color: transparent;")
        self.synth_view.show_placeholder("等待相机图像")
        self.synth_view.zoomTriggered.connect(self.on_main_views_zoom)
        synth_layout.addWidget(self.synth_view, stretch=1)
        self.grid_layout.addWidget(self.synth_frame, 0, 0, 2, 2)
        self.synth_frame.hide()

        image_layout = QVBoxLayout()
        image_layout.setSpacing(8)
        image_layout.addLayout(self.grid_layout, stretch=1)

        polarization_bar = QFrame()
        polarization_bar.setFixedHeight(58)
        polarization_bar.setStyleSheet(
            "background-color: #252525; border: 1px solid #3A3A3A; border-radius: 4px;"
        )
        polarization_layout = QHBoxLayout(polarization_bar)
        polarization_layout.setContentsMargins(14, 6, 14, 6)
        polarization_layout.setSpacing(12)

        lbl_polarization_title = QLabel("偏振统计（全画面）")
        lbl_polarization_title.setStyleSheet("color: #AAAAAA; font-size: 12px; font-weight: bold; border: none;")
        polarization_layout.addWidget(lbl_polarization_title)
        polarization_layout.addStretch(1)

        lbl_dolp_title = QLabel("平均线偏振度")
        lbl_dolp_title.setStyleSheet("color: #DDDDDD; border: none;")
        polarization_layout.addWidget(lbl_dolp_title)
        self.lbl_dolp = QLabel("-- %")
        self.lbl_dolp.setMinimumWidth(78)
        self.lbl_dolp.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_dolp.setStyleSheet("color: #4DD0C8; font-size: 16px; font-weight: bold; border: none;")
        self.lbl_dolp.setToolTip("有效像素的强度加权平均 DoLP；已排除过暗和饱和像素")
        polarization_layout.addWidget(self.lbl_dolp)

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("color: #555555;")
        polarization_layout.addWidget(separator)

        lbl_aolp_title = QLabel("主偏振角")
        lbl_aolp_title.setStyleSheet("color: #DDDDDD; border: none;")
        polarization_layout.addWidget(lbl_aolp_title)
        self.lbl_aolp = QLabel("-- °")
        self.lbl_aolp.setMinimumWidth(78)
        self.lbl_aolp.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_aolp.setStyleSheet("color: #FFB74D; font-size: 16px; font-weight: bold; border: none;")
        self.lbl_aolp.setToolTip("有效偏振像素在双角空间中的主方向；方向分散时不显示")
        polarization_layout.addWidget(self.lbl_aolp)

        image_layout.addWidget(polarization_bar)
        main_layout.addLayout(image_layout, stretch=4)

        # ================== 右侧控制面板 ==================
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("ControlPanel")
        ctrl_frame.setMinimumWidth(260)
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(15, 20, 15, 20)
        ctrl_layout.setSpacing(15)

        # 控制面板标题
        lbl_panel_title = QLabel("控制面板")
        lbl_panel_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        lbl_panel_title.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(lbl_panel_title)

        # 启动/暂停按钮
        self.btn_toggle = QPushButton("开始采集")
        self.btn_toggle.setFixedHeight(45)
        self.btn_toggle.setStyleSheet("background-color: #28A745; color: white;")
        self.btn_toggle.clicked.connect(self.toggle_capture)
        ctrl_layout.addWidget(self.btn_toggle)

        # 保存按钮
        self.btn_save = QPushButton("保存四幅图片 (TIFF)")
        self.btn_save.setFixedHeight(40)
        self.btn_save.setStyleSheet("""
            QPushButton { background-color: #FF9800; color: white; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """)
        self.btn_save.setEnabled(False)  # 首帧到达后可保存实时图像
        self.btn_save.clicked.connect(self.save_images)
        ctrl_layout.addWidget(self.btn_save)

        # 生成/分析偏振度图像按钮
        self.btn_dolp = QPushButton("生成偏振度图像 (DoLP)")
        self.btn_dolp.setFixedHeight(40)
        self.btn_dolp.setStyleSheet("""
            QPushButton { background-color: #00ADB5; color: white; font-weight: bold; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """)
        self.btn_dolp.setEnabled(False)  # 初始禁用
        self.btn_dolp.clicked.connect(self.analyze_dolp)
        ctrl_layout.addWidget(self.btn_dolp)

        # 交互式大图浏览按钮
        self.btn_zoom_view = QPushButton("交互式大图浏览")
        self.btn_zoom_view.setFixedHeight(40)
        self.btn_zoom_view.setStyleSheet("""
            QPushButton { background-color: #673AB7; color: white; font-weight: bold; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """)
        self.btn_zoom_view.setEnabled(False)  # 初始禁用
        self.btn_zoom_view.clicked.connect(self.open_zoom_viewer)
        ctrl_layout.addWidget(self.btn_zoom_view)

        # 分割线
        line_disp = QFrame()
        line_disp.setFrameShape(QFrame.HLine)
        line_disp.setStyleSheet("background-color: #444444;")
        ctrl_layout.addWidget(line_disp)

        # 实时偏振显示设置区域
        lbl_disp_header = QLabel("实时偏振显示模式")
        lbl_disp_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF9800;")
        ctrl_layout.addWidget(lbl_disp_header)

        # 模式选择下拉框
        layout_disp_mode = QHBoxLayout()
        layout_disp_mode.addWidget(QLabel("显示模式:"))
        self.combo_disp_mode = QComboBox()
        self.combo_disp_mode.addItems([
            "4 方向偏振 (原生 90°, 45°, 135°, 0°)",
            "3 方向偏振解复用 (旋转角 x)",
            "任意角度合成 (0°–180°)"
        ])
        self.combo_disp_mode.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        initial_disp_mode = self.config.get("DisplayMode", "4-pol")
        if initial_disp_mode == "3-pol":
            self.combo_disp_mode.setCurrentIndex(1)
        elif initial_disp_mode == "synth-pol":
            self.combo_disp_mode.setCurrentIndex(2)
        else:
            self.combo_disp_mode.setCurrentIndex(0)
        self.combo_disp_mode.currentIndexChanged.connect(self.on_disp_mode_changed)
        layout_disp_mode.addWidget(self.combo_disp_mode)
        ctrl_layout.addLayout(layout_disp_mode)

        self.frame_synth_angle = QFrame()
        self.frame_synth_angle.setStyleSheet("background: transparent; border: none;")
        synth_angle_layout = QVBoxLayout(self.frame_synth_angle)
        synth_angle_layout.setContentsMargins(0, 0, 0, 0)
        synth_angle_layout.setSpacing(4)
        synth_angle_header = QHBoxLayout()
        synth_angle_header.addWidget(QLabel("合成偏振角:"))
        self.lbl_synth_angle = QLabel("0°")
        self.lbl_synth_angle.setMinimumWidth(48)
        self.lbl_synth_angle.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_synth_angle.setStyleSheet("color: #4DD0C8; font-weight: bold;")
        synth_angle_header.addWidget(self.lbl_synth_angle)
        synth_angle_layout.addLayout(synth_angle_header)
        self.slider_synth_angle = QSlider(Qt.Horizontal)
        self.slider_synth_angle.setRange(0, 1800)
        self.slider_synth_angle.setSingleStep(1)
        self.slider_synth_angle.setPageStep(5)
        self.slider_synth_angle.setValue(round(float(self.config.get("SynthesizedAngle", 0)) * 10))
        self.slider_synth_angle.setToolTip("选择要合成的线偏振检偏角（0° 到 180°）")
        self.slider_synth_angle.valueChanged.connect(self.on_synth_angle_changed)
        synth_angle_layout.addWidget(self.slider_synth_angle)
        self.lbl_synth_angle.setText(f"{self.slider_synth_angle.value() / 10:g}°")
        ctrl_layout.addWidget(self.frame_synth_angle)
        self.frame_synth_angle.setVisible(initial_disp_mode == "synth-pol")

        self.frame_synth_contrast = QFrame()
        self.frame_synth_contrast.setStyleSheet("background: transparent; border: none;")
        contrast_layout = QVBoxLayout(self.frame_synth_contrast)
        contrast_layout.setContentsMargins(0, 0, 0, 0)
        contrast_layout.setSpacing(4)
        contrast_header = QHBoxLayout()
        contrast_header.addWidget(QLabel("合成图对比度:"))
        contrast_header.addStretch(1)
        self.btn_reset_synth_contrast = QPushButton("恢复默认")
        self.btn_reset_synth_contrast.setFixedHeight(26)
        self.btn_reset_synth_contrast.clicked.connect(self.reset_synth_contrast)
        contrast_header.addWidget(self.btn_reset_synth_contrast)
        contrast_layout.addLayout(contrast_header)

        self.lbl_synth_black = QLabel("黑电平: 0")
        contrast_layout.addWidget(self.lbl_synth_black)
        self.slider_synth_black = QSlider(Qt.Horizontal)
        self.slider_synth_black.setRange(0, 254)
        self.slider_synth_black.setValue(int(self.config.get("SynthContrastBlack", 0)))
        self.slider_synth_black.valueChanged.connect(self.on_synth_contrast_changed)
        contrast_layout.addWidget(self.slider_synth_black)

        self.lbl_synth_white = QLabel("白电平: 255")
        contrast_layout.addWidget(self.lbl_synth_white)
        self.slider_synth_white = QSlider(Qt.Horizontal)
        self.slider_synth_white.setRange(1, 255)
        white_level = max(
            self.slider_synth_black.value() + 1,
            int(self.config.get("SynthContrastWhite", 255))
        )
        self.slider_synth_white.setValue(min(255, white_level))
        self.slider_synth_white.valueChanged.connect(self.on_synth_contrast_changed)
        contrast_layout.addWidget(self.slider_synth_white)
        self.lbl_synth_black.setText(f"黑电平: {self.slider_synth_black.value()}")
        self.lbl_synth_white.setText(f"白电平: {self.slider_synth_white.value()}")
        ctrl_layout.addWidget(self.frame_synth_contrast)
        self.frame_synth_contrast.setVisible(initial_disp_mode == "synth-pol")

        # 整体旋转角 x 设置组（3方向解复用模式下显示，与 02_calibrate_rotation 校准角联动）
        self.frame_rotation_angle = QFrame()
        self.frame_rotation_angle.setStyleSheet("background: transparent; border: none;")
        layout_rotation_angle = QVBoxLayout(self.frame_rotation_angle)
        layout_rotation_angle.setContentsMargins(0, 0, 0, 0)
        layout_rotation_angle.setSpacing(6)

        layout_angle = QHBoxLayout()
        layout_angle.addWidget(QLabel("整体旋转角 x (°):"))
        self.spin_rotation_angle = QDoubleSpinBox()
        self.spin_rotation_angle.setRange(-180.0, 180.0)
        self.spin_rotation_angle.setSingleStep(1.0)
        self.spin_rotation_angle.setValue(self.config.get("RotationAngle", 20.0))
        self.spin_rotation_angle.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.spin_rotation_angle.valueChanged.connect(self.on_rotation_angle_changed)
        layout_angle.addWidget(self.spin_rotation_angle)
        layout_rotation_angle.addLayout(layout_angle)

        ctrl_layout.addWidget(self.frame_rotation_angle)
        self.frame_rotation_angle.setVisible(initial_disp_mode == "3-pol")

        # 初始化显示标题
        self.update_display_titles_and_views()

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #444444;")
        ctrl_layout.addWidget(line)

        # 参数设置区域
        lbl_settings_header = QLabel("相机参数设置")
        lbl_settings_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ADB5;")
        ctrl_layout.addWidget(lbl_settings_header)

        # 曝光时间
        layout_exp = QHBoxLayout()
        lbl_exp = QLabel("曝光 (us):")
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(20.0, 1000000.0)
        self.spin_exposure.setSingleStep(1000.0)
        self.spin_exposure.setDecimals(1)
        self.spin_exposure.setValue(self.config["ExposureTime"])
        self.spin_exposure.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.spin_exposure.valueChanged.connect(self.on_exposure_changed)
        layout_exp.addWidget(lbl_exp)
        layout_exp.addWidget(self.spin_exposure)
        ctrl_layout.addLayout(layout_exp)

        # 增益
        layout_gain = QHBoxLayout()
        lbl_gain = QLabel("增益 (dB):")
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 24.0)
        self.spin_gain.setSingleStep(0.1)
        self.spin_gain.setDecimals(1)
        self.spin_gain.setValue(self.config["Gain"])
        self.spin_gain.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.spin_gain.valueChanged.connect(self.on_gain_changed)
        layout_gain.addWidget(lbl_gain)
        layout_gain.addWidget(self.spin_gain)
        ctrl_layout.addLayout(layout_gain)

        # 图像宽度
        layout_width = QHBoxLayout()
        lbl_width = QLabel("图像宽度:")
        self.spin_width = QSpinBox()
        self.spin_width.setRange(16, 4096)
        self.spin_width.setSingleStep(8)
        self.spin_width.setValue(self.config.get("Width", 2448))
        self.spin_width.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.spin_width.valueChanged.connect(self.on_width_changed)
        layout_width.addWidget(lbl_width)
        layout_width.addWidget(self.spin_width)
        ctrl_layout.addLayout(layout_width)

        # 图像高度
        layout_height = QHBoxLayout()
        lbl_height = QLabel("图像高度:")
        self.spin_height = QSpinBox()
        self.spin_height.setRange(16, 4096)
        self.spin_height.setSingleStep(2)
        self.spin_height.setValue(self.config.get("Height", 2048))
        self.spin_height.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.spin_height.valueChanged.connect(self.on_height_changed)
        layout_height.addWidget(lbl_height)
        layout_height.addWidget(self.spin_height)
        ctrl_layout.addLayout(layout_height)

        # 水平偏移
        layout_offset_x = QHBoxLayout()
        lbl_offset_x = QLabel("水平偏移:")
        self.spin_offset_x = QSpinBox()
        self.spin_offset_x.setRange(0, 4096)
        self.spin_offset_x.setSingleStep(8)
        self.spin_offset_x.setValue(self.config.get("OffsetX", 0))
        self.spin_offset_x.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.spin_offset_x.valueChanged.connect(self.on_offset_x_changed)
        layout_offset_x.addWidget(lbl_offset_x)
        layout_offset_x.addWidget(self.spin_offset_x)
        ctrl_layout.addLayout(layout_offset_x)

        # 垂直偏移
        layout_offset_y = QHBoxLayout()
        lbl_offset_y = QLabel("垂直偏移:")
        self.spin_offset_y = QSpinBox()
        self.spin_offset_y.setRange(0, 4096)
        self.spin_offset_y.setSingleStep(2)
        self.spin_offset_y.setValue(self.config.get("OffsetY", 0))
        self.spin_offset_y.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.spin_offset_y.valueChanged.connect(self.on_offset_y_changed)
        layout_offset_y.addWidget(lbl_offset_y)
        layout_offset_y.addWidget(self.spin_offset_y)
        ctrl_layout.addLayout(layout_offset_y)

        # 像素格式
        layout_fmt = QHBoxLayout()
        lbl_fmt = QLabel("像素格式:")
        self.combo_format = QComboBox()
        self.combo_format.addItems(["Mono8", "Mono10"])
        fmt_text = self.config.get("PixelFormat", "mono8").capitalize()
        idx = self.combo_format.findText(fmt_text)
        if idx >= 0:
            self.combo_format.setCurrentIndex(idx)
        self.combo_format.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.combo_format.currentTextChanged.connect(self.on_pixel_format_changed)
        layout_fmt.addWidget(lbl_fmt)
        layout_fmt.addWidget(self.combo_format)
        ctrl_layout.addLayout(layout_fmt)

        # 触发模式
        layout_trig_mode = QHBoxLayout()
        lbl_trig_mode = QLabel("触发模式:")
        self.combo_trigger_mode = QComboBox()
        self.combo_trigger_mode.addItems(["连续采集 (Free Run)", "外触发 (External)"])
        trig_mode_text = "外触发 (External)" if self.config.get("TriggerMode", "off") == "on" else "连续采集 (Free Run)"
        idx = self.combo_trigger_mode.findText(trig_mode_text)
        if idx >= 0:
            self.combo_trigger_mode.setCurrentIndex(idx)
        self.combo_trigger_mode.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.combo_trigger_mode.currentTextChanged.connect(self.on_trigger_mode_changed)
        layout_trig_mode.addWidget(lbl_trig_mode)
        layout_trig_mode.addWidget(self.combo_trigger_mode)
        ctrl_layout.addLayout(layout_trig_mode)

        # 触发源
        layout_trig_src = QHBoxLayout()
        lbl_trig_src = QLabel("触发源:")
        self.combo_trigger_source = QComboBox()
        self.combo_trigger_source.addItems(["Line0", "Line1", "Line2", "Line3", "Software"])
        trig_src_text = self.config.get("TriggerSource", "line0").capitalize()
        idx = self.combo_trigger_source.findText(trig_src_text)
        if idx >= 0:
            self.combo_trigger_source.setCurrentIndex(idx)
        self.combo_trigger_source.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.combo_trigger_source.currentTextChanged.connect(self.on_trigger_source_changed)
        layout_trig_src.addWidget(lbl_trig_src)
        layout_trig_src.addWidget(self.combo_trigger_source)
        ctrl_layout.addLayout(layout_trig_src)

        # 触发边沿
        layout_trig_act = QHBoxLayout()
        lbl_trig_act = QLabel("触发边沿:")
        self.combo_trigger_active = QComboBox()
        self.combo_trigger_active.addItems(["上升沿 (Rising)", "下降沿 (Falling)"])
        trig_act_text = "下降沿 (Falling)" if self.config.get("TriggerActivation", "rising") == "falling" else "上升沿 (Rising)"
        idx = self.combo_trigger_active.findText(trig_act_text)
        if idx >= 0:
            self.combo_trigger_active.setCurrentIndex(idx)
        self.combo_trigger_active.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.combo_trigger_active.currentTextChanged.connect(self.on_trigger_activation_changed)
        layout_trig_act.addWidget(lbl_trig_act)
        layout_trig_act.addWidget(self.combo_trigger_active)
        ctrl_layout.addLayout(layout_trig_act)

        # 触发延迟
        layout_trig_delay = QHBoxLayout()
        lbl_trig_delay = QLabel("触发延迟 (us):")
        self.spin_trigger_delay = QDoubleSpinBox()
        self.spin_trigger_delay.setRange(0.0, 1000000.0)
        self.spin_trigger_delay.setSingleStep(100.0)
        self.spin_trigger_delay.setDecimals(1)
        self.spin_trigger_delay.setValue(self.config.get("TriggerDelay", 0.0))
        self.spin_trigger_delay.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.spin_trigger_delay.valueChanged.connect(self.on_trigger_delay_changed)
        layout_trig_delay.addWidget(lbl_trig_delay)
        layout_trig_delay.addWidget(self.spin_trigger_delay)
        ctrl_layout.addLayout(layout_trig_delay)

        # 分割线
        line_io = QFrame()
        line_io.setFrameShape(QFrame.HLine)
        line_io.setStyleSheet("background-color: #444444;")
        ctrl_layout.addWidget(line_io)

        # 数字IO控制区域标题
        lbl_io_header = QLabel("数字IO控制")
        lbl_io_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ADB5;")
        ctrl_layout.addWidget(lbl_io_header)

        # 引脚选择
        layout_line_sel = QHBoxLayout()
        lbl_line_sel = QLabel("引脚选择:")
        self.combo_line_selector = QComboBox()
        self.combo_line_selector.addItems(["Line0", "Line1", "Line2", "Line3"])
        line_sel_text = self.config.get("LineSelector", "line0").capitalize()
        idx = self.combo_line_selector.findText(line_sel_text)
        if idx >= 0:
            self.combo_line_selector.setCurrentIndex(idx)
        self.combo_line_selector.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.combo_line_selector.currentTextChanged.connect(self.on_line_selector_changed)
        layout_line_sel.addWidget(lbl_line_sel)
        layout_line_sel.addWidget(self.combo_line_selector)
        ctrl_layout.addLayout(layout_line_sel)

        # 引脚方向
        layout_line_mode = QHBoxLayout()
        lbl_line_mode = QLabel("引脚方向:")
        self.combo_line_mode = QComboBox()
        self.combo_line_mode.addItems(["Input", "Output"])
        current_line = self.config.get("LineSelector", "line0").lower()
        modes_dict = self.config.get("LineModes", {"line0": "input", "line1": "input", "line2": "input", "line3": "input"})
        current_mode = modes_dict.get(current_line, "input").capitalize()
        idx_mode = self.combo_line_mode.findText(current_mode)
        if idx_mode >= 0:
            self.combo_line_mode.setCurrentIndex(idx_mode)
        self.combo_line_mode.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.combo_line_mode.currentTextChanged.connect(self.on_line_mode_changed)
        layout_line_mode.addWidget(lbl_line_mode)
        layout_line_mode.addWidget(self.combo_line_mode)
        ctrl_layout.addLayout(layout_line_mode)

        # 电平反转
        layout_line_inv = QHBoxLayout()
        lbl_line_inv = QLabel("电平反转:")
        self.combo_line_inverter = QComboBox()
        self.combo_line_inverter.addItems(["False", "True"])
        current_line = self.config.get("LineSelector", "line0").lower()
        inverters_dict = self.config.get("LineInverters", {"line0": False, "line1": False, "line2": False, "line3": False})
        is_inverted = inverters_dict.get(current_line, False)
        inv_idx = 1 if is_inverted else 0
        self.combo_line_inverter.setCurrentIndex(inv_idx)
        self.combo_line_inverter.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.combo_line_inverter.currentTextChanged.connect(self.on_line_inverter_changed)
        layout_line_inv.addWidget(lbl_line_inv)
        layout_line_inv.addWidget(self.combo_line_inverter)
        ctrl_layout.addLayout(layout_line_inv)

        # 软触发执行按钮
        self.btn_soft_trigger = QPushButton("手动发送软触发")
        self.btn_soft_trigger.setFixedHeight(30)
        self.btn_soft_trigger.setStyleSheet("""
            QPushButton { background-color: #17A2B8; color: white; font-size: 12px; font-weight: bold; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """)
        self.btn_soft_trigger.clicked.connect(self.send_software_trigger)
        ctrl_layout.addWidget(self.btn_soft_trigger)

        # 保存配置按钮
        self.btn_save_config = QPushButton("保存参数到配置")
        self.btn_save_config.setFixedHeight(30)
        self.btn_save_config.setStyleSheet("background-color: #17A2B8; color: white; font-size: 12px;")
        self.btn_save_config.clicked.connect(self.save_current_config)
        ctrl_layout.addWidget(self.btn_save_config)

        self.update_trigger_ui_states()

        # 分割线
        line3 = QFrame()
        line3.setFrameShape(QFrame.HLine)
        line3.setStyleSheet("background-color: #444444;")
        ctrl_layout.addWidget(line3)

        # 连续采图保存区域
        lbl_continuous_header = QLabel("连续采图保存")
        lbl_continuous_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF9800;")
        ctrl_layout.addWidget(lbl_continuous_header)

        self.btn_continuous_save = QPushButton("开始连续保存")
        self.btn_continuous_save.setFixedHeight(40)
        self.btn_continuous_save.setStyleSheet("background-color: #00ADB5; color: white; font-weight: bold;")
        self.btn_continuous_save.clicked.connect(self.toggle_continuous_save)
        ctrl_layout.addWidget(self.btn_continuous_save)

        # ================== DPC 重建与校准区域 ==================
        line_dpc = QFrame()
        line_dpc.setFrameShape(QFrame.HLine)
        line_dpc.setStyleSheet("background-color: #444444;")
        ctrl_layout.addWidget(line_dpc)

        lbl_dpc_header = QLabel("DPC 偏振重建")
        lbl_dpc_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #4DD0C8;")
        ctrl_layout.addWidget(lbl_dpc_header)

        # 参数：Crop Size
        layout_dpc_crop = QHBoxLayout()
        layout_dpc_crop.addWidget(QLabel("裁剪尺寸:"))
        self.spin_dpc_crop = QSpinBox()
        self.spin_dpc_crop.setRange(64, 2048)
        self.spin_dpc_crop.setSingleStep(64)
        self.spin_dpc_crop.setValue(256)
        self.spin_dpc_crop.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        layout_dpc_crop.addWidget(self.spin_dpc_crop)
        ctrl_layout.addLayout(layout_dpc_crop)

        # 参数：NA Radius
        layout_dpc_na = QHBoxLayout()
        layout_dpc_na.addWidget(QLabel("NA 半径:"))
        self.spin_dpc_na = QDoubleSpinBox()
        self.spin_dpc_na.setRange(1.0, 1000.0)
        self.spin_dpc_na.setSingleStep(10.0)
        self.spin_dpc_na.setValue(103.0)
        self.spin_dpc_na.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        layout_dpc_na.addWidget(self.spin_dpc_na)
        ctrl_layout.addLayout(layout_dpc_na)

        # 参数：TV Lambda
        layout_dpc_lambda = QHBoxLayout()
        layout_dpc_lambda.addWidget(QLabel("TV 正则 (λ):"))
        self.spin_dpc_lambda = QDoubleSpinBox()
        self.spin_dpc_lambda.setRange(0.0001, 10.0)
        self.spin_dpc_lambda.setDecimals(4)
        self.spin_dpc_lambda.setSingleStep(0.005)
        self.spin_dpc_lambda.setValue(0.03)
        self.spin_dpc_lambda.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        layout_dpc_lambda.addWidget(self.spin_dpc_lambda)
        ctrl_layout.addLayout(layout_dpc_lambda)

        # 参数：迭代步数
        layout_dpc_iter = QHBoxLayout()
        layout_dpc_iter.addWidget(QLabel("最大迭代次数:"))
        self.spin_dpc_iter = QSpinBox()
        self.spin_dpc_iter.setRange(1, 500)
        self.spin_dpc_iter.setValue(30)
        self.spin_dpc_iter.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        layout_dpc_iter.addWidget(self.spin_dpc_iter)
        ctrl_layout.addLayout(layout_dpc_iter)

        # 新增：跳过校准选项（直接使用已知角度）
        self.chk_dpc_skip_calib = QCheckBox("跳过校准 (直接使用已知角度)")
        self.chk_dpc_skip_calib.setStyleSheet("color: #FFFFFF; font-weight: bold; margin-top: 5px;")
        self.chk_dpc_skip_calib.toggled.connect(self.on_dpc_skip_calib_toggled)
        ctrl_layout.addWidget(self.chk_dpc_skip_calib)

        # 参数：已知旋转角
        self.layout_dpc_known_angle = QHBoxLayout()
        self.lbl_dpc_known_angle = QLabel("已知偏转角度 (°):")
        self.spin_dpc_known_angle = QDoubleSpinBox()
        self.spin_dpc_known_angle.setRange(-180.0, 180.0)
        self.spin_dpc_known_angle.setSingleStep(1.0)
        self.spin_dpc_known_angle.setValue(20.0)
        self.spin_dpc_known_angle.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        self.spin_dpc_known_angle.setEnabled(False) # 默认禁用
        self.layout_dpc_known_angle.addWidget(self.lbl_dpc_known_angle)
        self.layout_dpc_known_angle.addWidget(self.spin_dpc_known_angle)
        ctrl_layout.addLayout(self.layout_dpc_known_angle)

        # 扫描参数框组合，在跳过校准时隐藏/禁用
        self.dpc_scan_group = QFrame()
        self.dpc_scan_group.setStyleSheet("background: transparent; border: none;")
        scan_group_layout = QVBoxLayout(self.dpc_scan_group)
        scan_group_layout.setContentsMargins(0, 0, 0, 0)
        scan_group_layout.setSpacing(8)

        # 扫描起始角度
        layout_scan_min = QHBoxLayout()
        layout_scan_min.addWidget(QLabel("扫描起始角 (°):"))
        self.spin_dpc_x_min = QSpinBox()
        self.spin_dpc_x_min.setRange(-180, 180)
        self.spin_dpc_x_min.setValue(0)
        self.spin_dpc_x_min.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        layout_scan_min.addWidget(self.spin_dpc_x_min)
        scan_group_layout.addLayout(layout_scan_min)

        # 扫描终止角度
        layout_scan_max = QHBoxLayout()
        layout_scan_max.addWidget(QLabel("扫描终止角 (°):"))
        self.spin_dpc_x_max = QSpinBox()
        self.spin_dpc_x_max.setRange(-180, 180)
        self.spin_dpc_x_max.setValue(60)
        self.spin_dpc_x_max.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        layout_scan_max.addWidget(self.spin_dpc_x_max)
        scan_group_layout.addLayout(layout_scan_max)

        # 扫描步长
        layout_scan_step = QHBoxLayout()
        layout_scan_step.addWidget(QLabel("扫描步长 (°):"))
        self.spin_dpc_x_step = QSpinBox()
        self.spin_dpc_x_step.setRange(1, 45)
        self.spin_dpc_x_step.setValue(5)
        self.spin_dpc_x_step.setStyleSheet("background-color: #3D3D3D; color: white; border: 1px solid #555555; padding: 3px;")
        layout_scan_step.addWidget(self.spin_dpc_x_step)
        scan_group_layout.addLayout(layout_scan_step)

        ctrl_layout.addWidget(self.dpc_scan_group)

        # 保存 1_4.tif 按钮
        self.btn_dpc_save_tif = QPushButton("保存 1_4.tif (四通道)")
        self.btn_dpc_save_tif.setFixedHeight(35)
        self.btn_dpc_save_tif.setStyleSheet("""
            QPushButton { background-color: #673AB7; color: white; font-weight: bold; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """)
        self.btn_dpc_save_tif.setEnabled(False)
        self.btn_dpc_save_tif.clicked.connect(self.save_1_4_tiff_clicked)
        ctrl_layout.addWidget(self.btn_dpc_save_tif)

        # 一键 DPC 重建按钮
        self.btn_dpc_recon = QPushButton("运行 DPC 偏振重建")
        self.btn_dpc_recon.setFixedHeight(40)
        self.btn_dpc_recon.setStyleSheet("""
            QPushButton { background-color: #E91E63; color: white; font-weight: bold; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """)
        self.btn_dpc_recon.setEnabled(False)
        self.btn_dpc_recon.clicked.connect(self.run_live_dpc_reconstruction)
        ctrl_layout.addWidget(self.btn_dpc_recon)

        # 文件重建按钮
        self.btn_dpc_file_recon = QPushButton("选择本地文件重建...")
        self.btn_dpc_file_recon.setFixedHeight(35)
        self.btn_dpc_file_recon.setStyleSheet("background-color: #00ADB5; color: white;")
        self.btn_dpc_file_recon.clicked.connect(self.run_file_dpc_reconstruction)
        ctrl_layout.addWidget(self.btn_dpc_file_recon)

        # 分割线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #444444;")
        ctrl_layout.addWidget(line2)

        # 相机状态展示
        lbl_status_header = QLabel("相机状态参数")
        lbl_status_header.setStyleSheet("font-size: 14px; font-weight: bold; color: #AAAAAA;")
        ctrl_layout.addWidget(lbl_status_header)

        # 帧率
        layout_fps = QHBoxLayout()
        layout_fps.addWidget(QLabel("实时帧率:"))
        self.lbl_fps = QLabel("0.00 FPS")
        self.lbl_fps.setStyleSheet("color: #FF9800; font-weight: bold;")
        self.lbl_fps.setAlignment(Qt.AlignRight)
        layout_fps.addWidget(self.lbl_fps)
        ctrl_layout.addLayout(layout_fps)

        ctrl_layout.addStretch(1)

        ctrl_scroll = QScrollArea()
        ctrl_scroll.setObjectName("ControlPanelScroll")
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ctrl_scroll.setFrameShape(QFrame.NoFrame)
        ctrl_scroll.setMinimumWidth(276)
        ctrl_scroll.setMaximumWidth(292)
        ctrl_scroll.setStyleSheet("QScrollArea#ControlPanelScroll { background: transparent; border: none; }")
        ctrl_scroll.setWidget(ctrl_frame)
        main_layout.addWidget(ctrl_scroll)

    @Slot()
    def toggle_capture(self):
        """启动或停止采集切换"""
        if not self.capture_thread.isRunning():
            # 开启采集
            self.btn_toggle.setText("暂停采集")
            self.btn_toggle.setStyleSheet("background-color: #DC3545; color: white;")  # 红色
            self.btn_save.setEnabled(self.last_images[0] is not None)
            self.btn_dolp.setEnabled(False)  # 开启采集时禁用 DoLP 分析按钮
            self.btn_zoom_view.setEnabled(False)  # 开启采集时禁用细节浏览器按钮
            self.btn_dpc_save_tif.setEnabled(False)  # 开启采集时禁用 DPC 保存
            self.btn_dpc_recon.setEnabled(False)  # 开启采集时禁用 DPC 重建
            self.combo_format.setEnabled(False)  # 开启采集时禁用像素格式修改

            self.spin_width.setEnabled(False)
            self.spin_height.setEnabled(False)
            self.spin_offset_x.setEnabled(False)
            self.spin_offset_y.setEnabled(False)

            # 加载最新参数并启动后台拉流线程
            self.capture_thread.set_parameters(
                self.config["ExposureTime"],
                self.config["Gain"],
                self.config.get("PixelFormat", "mono8").lower(),
                self.config.get("TriggerMode", "off").lower(),
                self.config.get("TriggerSource", "line0").lower(),
                self.config.get("TriggerActivation", "rising").lower(),
                width=self.config.get("Width", 2448),
                height=self.config.get("Height", 2048),
                offset_x=self.config.get("OffsetX", 0),
                offset_y=self.config.get("OffsetY", 0),
                trigger_delay=self.config.get("TriggerDelay", 0.0)
            )
            self.capture_thread.set_line_parameters(
                self.config.get("LineSelector", "line0").lower(),
                self.config.get("LineInverters", {"line0": False, "line1": False, "line2": False, "line3": False}),
                self.config.get("LineModes", {"line0": "input", "line1": "input", "line2": "input", "line3": "input"})
            )
            self.capture_thread.start()
            self.update_trigger_ui_states()
        else:
            # 暂停采集
            self.reset_capture_ui_state()
    def on_disp_mode_changed(self, idx):
        is_3pol = (idx == 1)
        self.frame_rotation_angle.setVisible(is_3pol)
        is_synth = (idx == 2)
        self.frame_synth_angle.setVisible(is_synth)
        self.frame_synth_contrast.setVisible(is_synth)
        self.synth_frame.setVisible(is_synth)
        for frame in self.img_frames:
            frame.setVisible(not is_synth)
        self.config["DisplayMode"] = "synth-pol" if is_synth else ("3-pol" if is_3pol else "4-pol")
        self.update_display_titles_and_views()

    @Slot(int)
    def on_synth_angle_changed(self, slider_value):
        angle = slider_value / 10.0
        self.config["SynthesizedAngle"] = angle
        self.lbl_synth_angle.setText(f"{angle:g}°")
        self.lbl_synth_title.setText(f"合成偏振图像 · {angle:g}°")
        if self.last_images[0] is not None:
            self.update_synthesized_view(*self.last_images)

    @Slot(int)
    def on_synth_contrast_changed(self, value):
        black = self.slider_synth_black.value()
        white = self.slider_synth_white.value()
        if black >= white:
            if self.sender() is self.slider_synth_black:
                white = min(255, black + 1)
                self.slider_synth_white.setValue(white)
            else:
                black = max(0, white - 1)
                self.slider_synth_black.setValue(black)

        self.config["SynthContrastBlack"] = black
        self.config["SynthContrastWhite"] = white
        self.lbl_synth_black.setText(f"黑电平: {black}")
        self.lbl_synth_white.setText(f"白电平: {white}")
        if self.last_images[0] is not None and self.combo_disp_mode.currentIndex() == 2:
            self.update_synthesized_view(*self.last_images)

    @Slot()
    def reset_synth_contrast(self):
        self.slider_synth_black.setValue(0)
        self.slider_synth_white.setValue(255)
        self.on_synth_contrast_changed(0)

    def apply_synth_contrast(self, image):
        black = self.slider_synth_black.value()
        white = self.slider_synth_white.value()
        lut = np.clip(
            (np.arange(256, dtype=np.float32) - black) * (255.0 / (white - black)),
            0,
            255
        ).astype(np.uint8)
        return cv2.LUT(np.ascontiguousarray(image), lut)

    @Slot(float)
    def on_rotation_angle_changed(self, val):
        self.config["RotationAngle"] = val
        if hasattr(self, 'spin_dpc_known_angle'):
            self.spin_dpc_known_angle.blockSignals(True)
            self.spin_dpc_known_angle.setValue(val)
            self.spin_dpc_known_angle.blockSignals(False)
        self.update_display_titles_and_views()

    def update_display_titles_and_views(self):
        is_3pol = (hasattr(self, 'combo_disp_mode') and self.combo_disp_mode.currentIndex() == 1)
        is_synth = (hasattr(self, 'combo_disp_mode') and self.combo_disp_mode.currentIndex() == 2)
        if hasattr(self, 'synth_frame'):
            self.synth_frame.setVisible(is_synth)
            for frame in self.img_frames:
                frame.setVisible(not is_synth)
        if hasattr(self, 'frame_synth_angle'):
            self.frame_synth_angle.setVisible(is_synth)
            self.frame_synth_contrast.setVisible(is_synth)
        if is_3pol:
            x_deg = self.spin_rotation_angle.value()
            titles = [
                f"偏振通道 s1 (0° + {x_deg:.1f}°)",
                f"偏振通道 s2 (120° + {x_deg:.1f}°)",
                f"偏振通道 s3 (240° + {x_deg:.1f}°)",
                "偏振总强度 S0"
            ]
        elif is_synth:
            titles = ["偏振角度 90°", "偏振角度 45°", "偏振角度 135°", "偏振角度 0°"]
            self.lbl_synth_title.setText(f"合成偏振图像 · {self.slider_synth_angle.value() / 10:g}°")
        else:
            titles = ["偏振角度 90°", "偏振角度 45°", "偏振角度 135°", "偏振角度 0°"]

        for i in range(4):
            if hasattr(self, 'lbl_img_titles') and i < len(self.lbl_img_titles):
                self.lbl_img_titles[i].setText(titles[i])

        if hasattr(self, 'zoom_dialogs'):
            for dialog in list(self.zoom_dialogs):
                try:
                    dialog.update_combo_items()
                    dialog.load_image_to_view()
                except Exception:
                    pass

        if not getattr(self.capture_thread, 'is_capturing', False) and self.last_images[0] is not None:
            self.refresh_offline_display()

        if is_synth and self.last_images[0] is not None:
            self.update_synthesized_view(*self.last_images)

    def update_synthesized_view(self, img_90, img_45, img_135, img_0):
        self.last_synthesized_image = synthesize_polarization_image(
            img_90, img_45, img_135, img_0, self.slider_synth_angle.value() / 10.0
        )
        image = self.last_synthesized_image
        if image.dtype == np.uint16:
            image_display = (image >> 2).astype(np.uint8)
        else:
            image_display = image
        self.synth_view.update_image(self.apply_synth_contrast(image_display))

    def refresh_offline_display(self):
        if self.last_images[0] is None:
            return
        img_90, img_45, img_135, img_0 = self.last_images
        is_3pol = (hasattr(self, 'combo_disp_mode') and self.combo_disp_mode.currentIndex() == 1)
        if is_3pol:
            x_deg = self.spin_rotation_angle.value()
            s1, s2, s3 = unmix_3_channels_live(img_90, img_45, img_135, img_0, x_deg)
            img_s0 = calc_s0_image(img_90, img_45, img_135, img_0)
            imgs = [s1, s2, s3, img_s0]
        else:
            imgs = [img_90, img_45, img_135, img_0]

        self.last_rendered_imgs = imgs

        for i in range(4):
            img = imgs[i]
            if img.dtype == np.uint16:
                img_display = (img >> 2).astype(np.uint8)
            else:
                img_display = img

            self.img_views[i].update_image(img_display)

    @Slot()
    def on_frame_available(self):
        frame = self.capture_thread.take_latest_frame()
        if frame is not None:
            self.on_frame_processed(*frame)

    @Slot(np.ndarray, np.ndarray, np.ndarray, np.ndarray, float)
    def on_frame_processed(self, img_90, img_45, img_135, img_0, fps):
        """
        接收后台解复用完成的帧数据并转换为 Qt 纹理极速显示 (对应 C# WriteableBitmap 更新)
        """
        # 缓存图像，供暂停后保存使用
        self.last_images = [img_90, img_45, img_135, img_0]
        self.btn_save.setEnabled(True)

        # 更新 FPS 显示
        if fps > 0:
            self.lbl_fps.setText(f"{fps:.2f} FPS")

        # 偏振统计以 5 Hz 更新，抽样计算可避免影响实时图像显示。
        now = time.monotonic()
        if now - self.last_polarization_update >= 0.2:
            sensor_max = 1023.0 if img_0.dtype == np.uint16 else 255.0
            dolp, aolp = calculate_polarization_statistics(
                img_90, img_45, img_135, img_0,
                sample_step=4,
                sensor_max=sensor_max
            )
            self.lbl_dolp.setText(f"{dolp:.1f} %" if np.isfinite(dolp) else "-- %")
            self.lbl_aolp.setText(f"{aolp:.1f} °" if np.isfinite(aolp) else "-- °")
            self.last_polarization_update = now

        if self.combo_disp_mode.currentIndex() == 2:
            self.update_synthesized_view(img_90, img_45, img_135, img_0)
            return

        # 根据当前选择的显示模式 (4方向原生 vs 3方向解复用) 渲染 4 张画面
        is_3pol = (hasattr(self, 'combo_disp_mode') and self.combo_disp_mode.currentIndex() == 1)
        if is_3pol:
            x_deg = self.spin_rotation_angle.value()
            s1, s2, s3 = unmix_3_channels_live(img_90, img_45, img_135, img_0, x_deg)
            img_s0 = calc_s0_image(img_90, img_45, img_135, img_0)
            imgs = [s1, s2, s3, img_s0]
        else:
            imgs = [img_90, img_45, img_135, img_0]

        self.last_rendered_imgs = imgs

        for i in range(4):
            img = imgs[i]
            # 如果是 Mono10 图像（uint16），需要将其缩放到 8bit 进行显示
            if img.dtype == np.uint16:
                img_display = (img >> 2).astype(np.uint8)
            else:
                img_display = img

            self.img_views[i].update_image(img_display)

    def on_main_views_zoom(self, factor):
        if self.combo_disp_mode.currentIndex() == 2:
            views = [self.synth_view]
        else:
            views = self.img_views

        source = self.sender()
        current_zoom = source.zoom_factor if source in views else views[0].zoom_factor
        target_zoom = min(views[0].zoom_max, max(views[0].zoom_min, current_zoom * factor))
        applied_factor = target_zoom / current_zoom
        for view in views:
            view.scale(applied_factor, applied_factor)
            view.zoom_factor = target_zoom

        if len(views) > 1:
            self.sync_main_views_scroll(source)

    def on_main_views_pan(self):
        self.sync_main_views_scroll(self.sender())

    def sync_main_views_scroll(self, source_view=None):
        if getattr(self, '_is_syncing_main', False) or source_view is None:
            return
        self._is_syncing_main = True
        try:
            h_bar = source_view.horizontalScrollBar()
            v_bar = source_view.verticalScrollBar()
            h_max = h_bar.maximum()
            v_max = v_bar.maximum()
            h_ratio = h_bar.value() / h_max if h_max > 0 else 0.0
            v_ratio = v_bar.value() / v_max if v_max > 0 else 0.0

            for view in self.img_views:
                if view != source_view:
                    vh_max = view.horizontalScrollBar().maximum()
                    vv_max = view.verticalScrollBar().maximum()
                    view.horizontalScrollBar().setValue(int(h_ratio * vh_max))
                    view.verticalScrollBar().setValue(int(v_ratio * vv_max))
        finally:
            self._is_syncing_main = False

    @Slot(int)
    def toggle_zoom(self, index):
        """点击图像卡片时，弹出 4 视口高分辨率同步大图细节浏览器"""
        self.open_zoom_viewer(index)

    @Slot(str)
    def on_camera_error(self, message):
        """处理相机底层发出的错误异常信号"""
        self.reset_capture_ui_state()
        QMessageBox.critical(self, "相机错误", message)

    @Slot(float)
    def on_exposure_changed(self, val):
        self.config["ExposureTime"] = val
        self.capture_thread.set_parameters(self.config["ExposureTime"], self.config["Gain"])

    @Slot(float)
    def on_gain_changed(self, val):
        self.config["Gain"] = val
        self.capture_thread.set_parameters(self.config["ExposureTime"], self.config["Gain"])

    @Slot(int)
    def on_width_changed(self, val):
        self.config["Width"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            width=val
        )

    @Slot(int)
    def on_height_changed(self, val):
        self.config["Height"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            height=val
        )

    @Slot(int)
    def on_offset_x_changed(self, val):
        self.config["OffsetX"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            offset_x=val
        )

    @Slot(int)
    def on_offset_y_changed(self, val):
        self.config["OffsetY"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            offset_y=val
        )

    @Slot(str)
    def on_pixel_format_changed(self, text):
        self.config["PixelFormat"] = text.lower()
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            self.config["PixelFormat"]
        )

    @Slot()
    def save_images(self):
        """将当前显示的 4 幅偏振角度图像保存为 TIFF 文件"""
        if self.last_images[0] is None:
            QMessageBox.warning(self, "保存提示", "当前没有有效的图像可保存！")
            return

        if self.combo_disp_mode.currentIndex() == 2:
            angle = self.slider_synth_angle.value() / 10.0
            default_name = f"polarization_{time.strftime('%Y%m%d_%H%M%S')}_{angle:g}deg.tiff"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存当前合成偏振图像",
                os.path.join(self.config.get("LastSaveDir", os.getcwd()), default_name),
                "TIFF Image (*.tif *.tiff);;PNG Image (*.png)"
            )
            if not file_path:
                return

            image = synthesize_polarization_image(*self.last_images, angle)
            try:
                if not cv2.imwrite(file_path, image):
                    raise OSError("图像编码器未能写入文件")
                self.config["LastSaveDir"] = os.path.dirname(file_path)
                QMessageBox.information(
                    self, "保存成功",
                    f"{angle:g}° 合成偏振图像已保存至:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存图像时发生错误:\n{e}")
            return

        last_dir = self.config.get("LastSaveDir", os.getcwd())
        folder_path = QFileDialog.getExistingDirectory(self, "选择保存偏振图像的文件夹", last_dir)
        if not folder_path:
            return

        self.config["LastSaveDir"] = folder_path
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        is_3pol = (hasattr(self, 'combo_disp_mode') and self.combo_disp_mode.currentIndex() == 1)
        img_90, img_45, img_135, img_0 = self.last_images
        if is_3pol:
            x_deg = self.spin_rotation_angle.value()
            s1, s2, s3 = unmix_3_channels_live(img_90, img_45, img_135, img_0, x_deg)
            img_s0 = calc_s0_image(img_90, img_45, img_135, img_0)
            imgs = [s1, s2, s3, img_s0]
            names = [
                f"polarization_{timestamp_str}_s1_x{x_deg:.1f}deg.tiff",
                f"polarization_{timestamp_str}_s2_x{x_deg:.1f}deg.tiff",
                f"polarization_{timestamp_str}_s3_x{x_deg:.1f}deg.tiff",
                f"polarization_{timestamp_str}_S0.tiff"
            ]
        else:
            imgs = [img_90, img_45, img_135, img_0]
            names = [
                f"polarization_{timestamp_str}_90deg.tiff",
                f"polarization_{timestamp_str}_45deg.tiff",
                f"polarization_{timestamp_str}_135deg.tiff",
                f"polarization_{timestamp_str}_0deg.tiff"
            ]

        try:
            for i in range(4):
                full_path = os.path.join(folder_path, names[i])
                cv2.imwrite(full_path, imgs[i])

            QMessageBox.information(self, "保存成功", f"4 幅偏振图像已成功保存至:\n{folder_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存图像时发生错误:\n{e}")

    @Slot()
    def save_current_config(self):
        config_path = get_config_path()
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "保存成功", "参数已成功写入 config.json！")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存配置: {e}")

    def block_ui_signals(self, block):
        self.spin_exposure.blockSignals(block)
        self.spin_gain.blockSignals(block)
        self.spin_width.blockSignals(block)
        self.spin_height.blockSignals(block)
        self.spin_offset_x.blockSignals(block)
        self.spin_offset_y.blockSignals(block)
        self.combo_format.blockSignals(block)
        self.combo_trigger_mode.blockSignals(block)
        self.combo_trigger_source.blockSignals(block)
        self.combo_trigger_active.blockSignals(block)
        self.spin_trigger_delay.blockSignals(block)
        self.combo_line_selector.blockSignals(block)
        self.combo_line_mode.blockSignals(block)
        self.combo_line_inverter.blockSignals(block)
        if hasattr(self, 'combo_disp_mode'):
            self.combo_disp_mode.blockSignals(block)
            self.spin_rotation_angle.blockSignals(block)
            self.slider_synth_angle.blockSignals(block)
            self.slider_synth_black.blockSignals(block)
            self.slider_synth_white.blockSignals(block)

    @Slot(str)
    def on_camera_error(self, message):
        """处理相机底层发出的错误异常信号"""
        self.reset_capture_ui_state()
        QMessageBox.critical(self, "相机错误", message)

    @Slot(float)
    def on_exposure_changed(self, val):
        self.config["ExposureTime"] = val
        self.capture_thread.set_parameters(self.config["ExposureTime"], self.config["Gain"])

    @Slot(float)
    def on_gain_changed(self, val):
        self.config["Gain"] = val
        self.capture_thread.set_parameters(self.config["ExposureTime"], self.config["Gain"])

    @Slot(int)
    def on_width_changed(self, val):
        self.config["Width"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            width=val
        )

    @Slot(int)
    def on_height_changed(self, val):
        self.config["Height"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            height=val
        )

    @Slot(int)
    def on_offset_x_changed(self, val):
        self.config["OffsetX"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            offset_x=val
        )

    @Slot(int)
    def on_offset_y_changed(self, val):
        self.config["OffsetY"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            offset_y=val
        )

    @Slot(str)
    def on_pixel_format_changed(self, text):
        self.config["PixelFormat"] = text.lower()
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            self.config["PixelFormat"]
        )

    @Slot()
    def save_images(self):
        """将当前显示的 4 幅偏振角度图像保存为 TIFF 文件"""
        if self.last_images[0] is None:
            QMessageBox.warning(self, "保存提示", "当前没有有效的图像可保存！")
            return

        if self.combo_disp_mode.currentIndex() == 2:
            angle = self.slider_synth_angle.value() / 10.0
            default_name = f"polarization_{time.strftime('%Y%m%d_%H%M%S')}_{angle:g}deg.tiff"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存当前合成偏振图像",
                os.path.join(self.config.get("LastSaveDir", os.getcwd()), default_name),
                "TIFF Image (*.tif *.tiff);;PNG Image (*.png)"
            )
            if not file_path:
                return

            image = synthesize_polarization_image(*self.last_images, angle)
            try:
                if not cv2.imwrite(file_path, image):
                    raise OSError("图像编码器未能写入文件")
                self.config["LastSaveDir"] = os.path.dirname(file_path)
                QMessageBox.information(
                    self, "保存成功",
                    f"{angle:g}° 合成偏振图像已保存至:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存图像时发生错误:\n{e}")
            return

        last_dir = self.config.get("LastSaveDir", os.getcwd())
        folder_path = QFileDialog.getExistingDirectory(self, "选择保存偏振图像的文件夹", last_dir)
        if not folder_path:
            return

        self.config["LastSaveDir"] = folder_path
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        is_3pol = (hasattr(self, 'combo_disp_mode') and self.combo_disp_mode.currentIndex() == 1)
        img_90, img_45, img_135, img_0 = self.last_images
        if is_3pol:
            a1 = self.spin_custom_angle1.value()
            a2 = self.spin_custom_angle2.value()
            a3 = self.spin_custom_angle3.value()
            img_a1 = synthesize_polarization_image(img_90, img_45, img_135, img_0, a1)
            img_a2 = synthesize_polarization_image(img_90, img_45, img_135, img_0, a2)
            img_a3 = synthesize_polarization_image(img_90, img_45, img_135, img_0, a3)
            img_s0 = calc_s0_image(img_90, img_45, img_135, img_0)
            imgs = [img_a1, img_a2, img_a3, img_s0]
            names = [
                f"polarization_{timestamp_str}_{a1:.1f}deg.tiff",
                f"polarization_{timestamp_str}_{a2:.1f}deg.tiff",
                f"polarization_{timestamp_str}_{a3:.1f}deg.tiff",
                f"polarization_{timestamp_str}_S0.tiff"
            ]
        else:
            imgs = [img_90, img_45, img_135, img_0]
            names = [
                f"polarization_{timestamp_str}_90deg.tiff",
                f"polarization_{timestamp_str}_45deg.tiff",
                f"polarization_{timestamp_str}_135deg.tiff",
                f"polarization_{timestamp_str}_0deg.tiff"
            ]

        try:
            for i in range(4):
                full_path = os.path.join(folder_path, names[i])
                cv2.imwrite(full_path, imgs[i])

            QMessageBox.information(self, "保存成功", f"4 幅偏振图像已成功保存至:\n{folder_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存图像时发生错误:\n{e}")

    @Slot()
    def save_current_config(self):
        config_path = get_config_path()
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "保存成功", "参数已成功写入 config.json！")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存配置: {e}")

    def block_ui_signals(self, block):
        self.spin_exposure.blockSignals(block)
        self.spin_gain.blockSignals(block)
        self.spin_width.blockSignals(block)
        self.spin_height.blockSignals(block)
        self.spin_offset_x.blockSignals(block)
        self.spin_offset_y.blockSignals(block)
        self.combo_format.blockSignals(block)
        self.combo_trigger_mode.blockSignals(block)
        self.combo_trigger_source.blockSignals(block)
        self.combo_trigger_active.blockSignals(block)
        self.spin_trigger_delay.blockSignals(block)
        self.combo_line_selector.blockSignals(block)
        self.combo_line_mode.blockSignals(block)
        self.combo_line_inverter.blockSignals(block)
        if hasattr(self, 'combo_disp_mode'):
            self.combo_disp_mode.blockSignals(block)
            self.spin_rotation_angle.blockSignals(block)
            self.slider_synth_angle.blockSignals(block)

    @Slot(str)
    def on_trigger_mode_changed(self, text):
        val = "on" if "外触发" in text else "off"
        self.config["TriggerMode"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            self.config.get("PixelFormat", "mono8").lower(),
            trigger_mode=val
        )
        self.update_trigger_ui_states()

    @Slot(str)
    def on_trigger_source_changed(self, text):
        val = text.lower()
        self.config["TriggerSource"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            self.config.get("PixelFormat", "mono8").lower(),
            trigger_source=val
        )
        self.update_trigger_ui_states()

    @Slot(str)
    def on_trigger_activation_changed(self, text):
        val = "falling" if "下降沿" in text else "rising"
        self.config["TriggerActivation"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            self.config.get("PixelFormat", "mono8").lower(),
            trigger_activation=val
        )
        self.update_trigger_ui_states()

    @Slot(float)
    def on_trigger_delay_changed(self, val):
        self.config["TriggerDelay"] = val
        self.capture_thread.set_parameters(
            self.config["ExposureTime"],
            self.config["Gain"],
            trigger_delay=val
        )

    @Slot(str)
    def on_line_selector_changed(self, text):
        val = text.lower()
        self.config["LineSelector"] = val
        
        self.combo_line_mode.blockSignals(True)
        self.combo_line_inverter.blockSignals(True)
        
        modes_dict = self.config.setdefault("LineModes", {"line0": "input", "line1": "input", "line2": "input", "line3": "input"})
        inverters_dict = self.config.setdefault("LineInverters", {"line0": False, "line1": False, "line2": False, "line3": False})
        
        is_inverted = inverters_dict.setdefault(val, False)
        current_mode = modes_dict.setdefault(val, "input")
        
        if self.capture_thread.isRunning() and self.capture_thread.device:
            try:
                self.capture_thread._apply_line_selector_to_camera(val)
                if self.capture_thread.device.LineMode.is_implemented() and self.capture_thread.device.LineMode.is_readable():
                    _, real_mode_str = self.capture_thread.device.LineMode.get()
                    current_mode = real_mode_str.lower()
                    modes_dict[val] = current_mode
                
                if self.capture_thread.device.LineInverter.is_implemented() and self.capture_thread.device.LineInverter.is_readable():
                    real_val = self.capture_thread.device.LineInverter.get()
                    inverters_dict[val] = real_val
                    is_inverted = real_val
            except Exception as e:
                print(f"切换引脚并读取状态失败: {e}")
        
        idx_mode = self.combo_line_mode.findText(current_mode.capitalize())
        if idx_mode >= 0:
            self.combo_line_mode.setCurrentIndex(idx_mode)
            
        inv_idx = 1 if is_inverted else 0
        self.combo_line_inverter.setCurrentIndex(inv_idx)
        
        self.combo_line_mode.blockSignals(False)
        self.combo_line_inverter.blockSignals(False)

    @Slot(str)
    def on_line_mode_changed(self, text):
        val = text.lower()
        current_line = self.combo_line_selector.currentText().lower()
        
        modes_dict = self.config.setdefault("LineModes", {"line0": "input", "line1": "input", "line2": "input", "line3": "input"})
        modes_dict[current_line] = val
        inverters_dict = self.config.setdefault("LineInverters", {"line0": False, "line1": False, "line2": False, "line3": False})
        
        self.capture_thread.set_line_parameters(current_line, inverters_dict, modes_dict)

    @Slot(str)
    def on_line_inverter_changed(self, text):
        val = text == "True"
        current_line = self.combo_line_selector.currentText().lower()
        
        inverters_dict = self.config.setdefault("LineInverters", {"line0": False, "line1": False, "line2": False, "line3": False})
        inverters_dict[current_line] = val
        modes_dict = self.config.setdefault("LineModes", {"line0": "input", "line1": "input", "line2": "input", "line3": "input"})
        
        self.capture_thread.set_line_parameters(current_line, inverters_dict, modes_dict)

    @Slot()
    def send_software_trigger(self):
        try:
            self.capture_thread.send_software_trigger()
        except Exception as e:
            QMessageBox.warning(self, "触发失败", f"向相机发送软触发命令失败:\n{e}")

    def update_trigger_ui_states(self):
        is_running = self.capture_thread.isRunning()
        self.combo_trigger_mode.setEnabled(not is_running)
        
        mode = self.combo_trigger_mode.currentText()
        is_external = ("外触发" in mode)
        
        self.combo_trigger_source.setEnabled(is_external and not is_running)
        self.combo_trigger_active.setEnabled(is_external and not is_running)
        self.spin_trigger_delay.setEnabled(is_external and not is_running)
        
        source = self.combo_trigger_source.currentText()
        is_software = (source == "Software")
        
        self.btn_soft_trigger.setEnabled(is_external and is_software and is_running)

    @Slot()
    def toggle_continuous_save(self):
        if not self.capture_thread.is_saving:
            # 检查相机是否正在采集
            if not self.capture_thread.isRunning():
                QMessageBox.warning(self, "保存提示", "请先开始相机采集，再开启连续保存！")
                return

            # 打开文件夹选择框让用户指定路径
            last_dir = self.config.get("LastSaveDir", os.getcwd())
            folder_path = QFileDialog.getExistingDirectory(self, "选择连续保存图像的文件夹", last_dir)
            if not folder_path:
                return

            # 记住上一次选择的路径
            self.config["LastSaveDir"] = folder_path

            # 计算该文件夹内已存文件大小
            initial_size = 0
            try:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        fp = os.path.join(root, file)
                        if os.path.exists(fp):
                            initial_size += os.path.getsize(fp)
            except Exception as e:
                print(f"扫描文件夹大小失败: {e}")

            # 超过限额校验
            if initial_size >= 1024 * 1024 * 1024:
                QMessageBox.warning(self, "保存提示", "目标文件夹存储容量已超过 1G，请清理该文件夹或选择其他位置！")
                return

            # 启动多线程后台异步保存器
            self.capture_thread.start_saver(folder_path, initial_size)

            self.btn_continuous_save.setText("停止保存")
            self.btn_continuous_save.setStyleSheet("background-color: #DC3545; color: white; font-weight: bold;")
        else:
            self.capture_thread.stop_saver()
            self.btn_continuous_save.setText("开始连续保存")
            self.btn_continuous_save.setStyleSheet("background-color: #00ADB5; color: white; font-weight: bold;")

    @Slot(bool, str)
    def on_save_status_changed(self, is_saving, message):
        if not is_saving:
            self.btn_continuous_save.setText("开始连续保存")
            self.btn_continuous_save.setStyleSheet("background-color: #00ADB5; color: white; font-weight: bold;")
            if message:
                QMessageBox.warning(self, "自动停止保存", message)

    def reset_capture_ui_state(self):
        """重置采集 UI 状态至停止采集状态，避免启动失败时调用 toggle_capture 导致死循环"""
        self.capture_thread.is_capturing = False
        self.capture_thread.stop_saver()
        self.capture_thread.wait(2000)  # 等待线程安全退回 (配合 100ms timeout 确保能安全退出)

        self.btn_toggle.setText("开始采集")
        self.btn_toggle.setStyleSheet("background-color: #28A745; color: white;")  # 绿色
        self.lbl_fps.setText("0.00 FPS")
        self.lbl_dolp.setText("-- %")
        self.lbl_aolp.setText("-- °")
        self.last_polarization_update = 0.0
        self.combo_format.setEnabled(True)  # 停止采集时允许修改
        self.spin_width.setEnabled(True)
        self.spin_height.setEnabled(True)
        self.spin_offset_x.setEnabled(True)
        self.spin_offset_y.setEnabled(True)

        self.btn_continuous_save.setText("开始连续保存")
        self.btn_continuous_save.setStyleSheet("background-color: #00ADB5; color: white; font-weight: bold;")

        # 若已有捕获图像，暂停后允许保存、生成 DoLP 热力图以及进行交互式细节缩放浏览
        if self.last_images[0] is not None:
            self.btn_save.setEnabled(True)
            self.btn_dolp.setEnabled(True)
            self.btn_zoom_view.setEnabled(True)
            self.btn_dpc_save_tif.setEnabled(True)
            self.btn_dpc_recon.setEnabled(True)
        self.update_trigger_ui_states()

    @Slot()
    def save_1_4_tiff_clicked(self):
        """保存当前 live 图像为四通道 TIFF (1_4.tif)"""
        if self.last_images[0] is None:
            QMessageBox.warning(self, "保存提示", "当前没有有效的图像可保存！")
            return

        last_dir = self.config.get("LastSaveDir", os.getcwd())
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择保存四通道 TIFF 文件路径", 
            os.path.join(last_dir, "1_4.tif"), 
            "TIFF Image (*.tif *.tiff)"
        )
        if not file_path:
            return

        self.config["LastSaveDir"] = os.path.dirname(file_path)
        
        try:
            # 这里的 last_images 顺序为 90, 45, 135, 0，堆叠成 float32 四通道 TIFF
            stack = np.stack(self.last_images, axis=-1).astype(np.float32)
            save_float_tiff(file_path, stack)
            QMessageBox.information(self, "保存成功", f"四通道 TIFF 图像已保存至:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存四通道 TIFF 失败:\n{e}")

    @Slot(bool)
    def on_dpc_skip_calib_toggled(self, checked):
        """跳过校准选项勾选状态改变槽"""
        self.spin_dpc_known_angle.setEnabled(checked)
        self.dpc_scan_group.setEnabled(not checked)

    @Slot()
    def run_live_dpc_reconstruction(self):
        """对当前内存中的 live 图像运行 DPC 偏振重建（一键校准或直接已知角度重建）"""
        if self.last_images[0] is None:
            QMessageBox.warning(self, "重建提示", "当前没有有效的图像帧，无法运行 DPC 重建！")
            return

        last_dir = self.config.get("LastSaveDir", os.getcwd())
        save_dir = QFileDialog.getExistingDirectory(self, "选择 DPC 重建输出目录", last_dir)
        if not save_dir:
            return

        self.config["LastSaveDir"] = save_dir

        # 获取 DPC 参数
        length = self.spin_dpc_crop.value()
        na_radius = self.spin_dpc_na.value()
        tv_lambda = self.spin_dpc_lambda.value()
        tv_max_iter = self.spin_dpc_iter.value()

        # 根据是否跳过校准设定旋转角范围
        if self.chk_dpc_skip_calib.isChecked():
            known_angle = self.spin_dpc_known_angle.value()
            x_min = known_angle
            x_max = known_angle
            x_step = 1.0
        else:
            x_min = self.spin_dpc_x_min.value()
            x_max = self.spin_dpc_x_max.value()
            x_step = self.spin_dpc_x_step.value()

        # 创建并启动后台计算线程
        thread = DPCReconThread(
            mode='live',
            save_dir=save_dir,
            last_images=self.last_images,
            length=length,
            na_radius=na_radius,
            tv_lambda=tv_lambda,
            tv_max_iter=tv_max_iter,
            x_min=x_min,
            x_max=x_max,
            x_step=x_step
        )
        self.start_dpc_recon_pipeline(thread)

    @Slot()
    def run_file_dpc_reconstruction(self):
        """选择本地 RAW / TIFF 文件运行 DPC 偏振重建"""
        last_dir = self.config.get("LastSaveDir", os.getcwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择用于 DPC 重建的本地文件", 
            last_dir, 
            "Polarization Files (*.raw *.tif *.tiff)"
        )
        if not file_path:
            return

        self.config["LastSaveDir"] = os.path.dirname(file_path)

        # 如果是 raw，解析或者输入尺寸
        raw_w, raw_h = 2448, 2048
        if file_path.lower().endswith('.raw'):
            # 自动正则匹配解析尺寸，例如 Pic_..._W2448_H2040_...raw
            match_w = re.search(r'_W(\d+)_', file_path, re.IGNORECASE)
            match_h = re.search(r'_H(\d+)_', file_path, re.IGNORECASE)
            default_w = int(match_w.group(1)) if match_w else 2448
            default_h = int(match_h.group(1)) if match_h else 2040
            
            # 使用输入框确认尺寸
            size_str, ok = QInputDialog.getText(
                self, "输入 RAW 图像尺寸", 
                "请输入 RAW 图像尺寸 (格式: 宽,高):", 
                text=f"{default_w},{default_h}"
            )
            if not ok or not size_str:
                return
            try:
                raw_w, raw_h = map(int, size_str.split(","))
            except Exception:
                QMessageBox.critical(self, "输入错误", "尺寸格式错误，使用默认尺寸 (2448,2040)")
                raw_w, raw_h = default_w, default_h

        save_dir = QFileDialog.getExistingDirectory(self, "选择 DPC 重建输出目录", os.path.dirname(file_path))
        if not save_dir:
            return

        # 获取 DPC 参数
        length = self.spin_dpc_crop.value()
        na_radius = self.spin_dpc_na.value()
        tv_lambda = self.spin_dpc_lambda.value()
        tv_max_iter = self.spin_dpc_iter.value()

        # 根据是否跳过校准设定旋转角范围
        if self.chk_dpc_skip_calib.isChecked():
            known_angle = self.spin_dpc_known_angle.value()
            x_min = known_angle
            x_max = known_angle
            x_step = 1.0
        else:
            x_min = self.spin_dpc_x_min.value()
            x_max = self.spin_dpc_x_max.value()
            x_step = self.spin_dpc_x_step.value()

        # 创建并启动后台计算线程
        thread = DPCReconThread(
            mode='file',
            save_dir=save_dir,
            file_path=file_path,
            raw_width=raw_w,
            raw_height=raw_h,
            length=length,
            na_radius=na_radius,
            tv_lambda=tv_lambda,
            tv_max_iter=tv_max_iter,
            x_min=x_min,
            x_max=x_max,
            x_step=x_step
        )
        self.start_dpc_recon_pipeline(thread)

    def start_dpc_recon_pipeline(self, thread):
        """配置并启动进度框和 DPC 后台线程"""
        self._dpc_finished = False  # 防止 canceled 信号干扰正常完成流程

        self.dpc_progress_dialog = QProgressDialog("正在准备重建...", "取消", 0, 100, self)
        self.dpc_progress_dialog.setWindowTitle("DPC 偏振重建进度")
        self.dpc_progress_dialog.setWindowModality(Qt.WindowModal)
        self.dpc_progress_dialog.setAutoClose(False)   # 禁止自动关闭，防止触发 canceled
        self.dpc_progress_dialog.setAutoReset(False)    # 禁止自动重置
        self.dpc_progress_dialog.setMinimumDuration(0)
        self.dpc_progress_dialog.show()

        # 保存线程引用
        self.current_dpc_thread = thread
        
        # 关联信号
        thread.progress.connect(self.on_dpc_progress_update)
        thread.finished.connect(self.on_dpc_recon_finished)
        
        # 用户点击取消处理
        self.dpc_progress_dialog.canceled.connect(self.cancel_dpc_reconstruction)
        
        thread.start()

    @Slot(int, str)
    def on_dpc_progress_update(self, percent, msg):
        if hasattr(self, 'dpc_progress_dialog') and self.dpc_progress_dialog:
            self.dpc_progress_dialog.setValue(percent)
            self.dpc_progress_dialog.setLabelText(msg)

    def cancel_dpc_reconstruction(self):
        # 如果已经正常完成，忽略 canceled 信号
        if getattr(self, '_dpc_finished', False):
            return
        if hasattr(self, 'current_dpc_thread') and self.current_dpc_thread:
            self.current_dpc_thread.terminate()
            self.current_dpc_thread.wait()
            self.current_dpc_thread = None
        QMessageBox.warning(self, "取消重建", "用户已中止 DPC 偏振重建流程！")

    @Slot(bool, str, dict)
    def on_dpc_recon_finished(self, success, message, result_dict):
        self._dpc_finished = True  # 标记已正常完成，屏蔽后续 canceled 信号

        # 安全等待后台线程完全结束后再释放线程引用，防止 QThread 被 GC 时底层仍在运行而导致程序自动退出/崩溃
        if hasattr(self, 'current_dpc_thread') and self.current_dpc_thread is not None:
            try:
                self.current_dpc_thread.wait(2000)
            except Exception:
                pass
            self.current_dpc_thread = None

        # 安全关闭进度窗口：先断开 canceled 信号再关闭
        if hasattr(self, 'dpc_progress_dialog') and self.dpc_progress_dialog:
            try:
                self.dpc_progress_dialog.canceled.disconnect(self.cancel_dpc_reconstruction)
            except (RuntimeError, TypeError):
                pass
            self.dpc_progress_dialog.close()
            self.dpc_progress_dialog = None

        if not success:
            QMessageBox.critical(self, "重建失败", f"DPC 算法发生错误:\n{message}")
            return

        QMessageBox.information(self, "重建成功", message)

        # 自动将估算的最佳角度填入已知偏转角度与 3 通道实时显示旋转角输入框
        x_est = result_dict.get("x_est")
        if x_est is not None:
            self.spin_dpc_known_angle.setValue(x_est)
            if hasattr(self, 'spin_rotation_angle'):
                self.spin_rotation_angle.setValue(x_est)

        # 弹出 DPC 重建结果多维度可视化对话框
        try:
            self.dpc_result_dialog = DPCResultDialog(result_dict, self)
            self.dpc_result_dialog.exec()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "结果显示错误", f"展示重建结果对话框时发生异常:\n{e}")

    @Slot()
    def analyze_dolp(self):
        """分析并弹出线偏振度 (DoLP) 伪彩可视化窗口"""
        if self.last_images[0] is None:
            QMessageBox.warning(self, "分析提示", "当前没有捕获到的图像帧，无法生成偏振度图像！")
            return

        dialog = DoLPAnalysisDialog(
            self.last_images[0],
            self.last_images[1],
            self.last_images[2],
            self.last_images[3],
            parent=self
        )
        dialog.exec()

    @Slot()
    @Slot(int)
    def open_zoom_viewer(self, index=0):
        """弹出交互式 4 视口同步高分辨率偏振大图细节浏览器 (多开 modeless 模式)"""
        if self.last_images[0] is None:
            QMessageBox.warning(self, "浏览提示", "当前没有捕获到的图像帧，无法浏览细节！")
            return
        # 创建 modeless 窗口，并指定 parent=self。
        # 同时必须将引用追加到 self.zoom_dialogs 中，否则方法结束时 dialog 对象就会被 GC 释放导致窗口闪退。
        dialog = ImageZoomViewerDialog(self.last_images, self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        
        # 窗口关闭时，从 MainWindow 的缓存列表中移除
        dialog.destroyed.connect(lambda: self.zoom_dialogs.remove(dialog) if dialog in self.zoom_dialogs else None)
        self.zoom_dialogs.append(dialog)
        dialog.show()

    def closeEvent(self, event):
        """窗口关闭时卸载资源，防止进程悬挂"""
        # 关闭所有尚存的大图查看器窗口
        for dialog in list(self.zoom_dialogs):
            try:
                dialog.close()
            except:
                pass
        self.zoom_dialogs.clear()
        
        self.config_timer.stop()
        self.capture_thread.is_capturing = False
        self.capture_thread.wait(1000)
        event.accept()
        QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 防止关闭对话框（如进度条或结果窗口）时误触发 QApplication 自动退出应用
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
