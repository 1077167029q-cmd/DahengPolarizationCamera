# Daheng Polarization Camera

大恒偏振相机采集与分析软件，提供实时图像采集、四偏振通道显示、偏振参数分析和 DPC 重建。

## 环境要求

- Windows
- Python 3.10 或更高版本
- 大恒 Galaxy SDK 与匹配的相机驱动
- 已连接并正确配置的大恒偏振相机

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

安装并配置大恒 Galaxy SDK 后，在项目目录启动：

```powershell
python main.py
```

## 功能模块

- `main.py`：相机界面、实时采集与偏振图像分析
- `gxipy/`：大恒相机 Python 接口封装
- `dpc/`、`dpc_pipeline.py`：偏振解复用、DPC 重建与图像处理
- `docs/`：偏振通道和分析结果示意图

实时采集时可在“显示模式”中选择“任意角度合成”，用角度滑块以 0.1° 步进选择 0°–180° 检偏方向。保存按钮可在采集过程中将当前合成原始强度图保存为 TIFF 或 PNG；Mono10 相机数据会以 16 位 TIFF 保存。

任意角度合成模式支持鼠标滚轮缩放合成图，以及通过黑电平/白电平滑块手动调整显示对比度。Mono10 时电平范围为 0–1023，Mono8 时为 0–255。显示调整不会修改原始数据或保存图像。

在任意角度合成模式下，点击“打开实时 DoLP 图”可在独立窗口查看当前相机帧的线偏振度伪彩图。窗口每约 0.2 秒更新一次，支持滚轮缩放和鼠标拖动。

相机连接和功能可用性取决于本机 Galaxy SDK、驱动版本及相机型号。
