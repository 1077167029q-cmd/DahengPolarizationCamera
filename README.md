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

相机连接和功能可用性取决于本机 Galaxy SDK、驱动版本及相机型号。
