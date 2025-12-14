# -pytorch-om-npu--Pytorch-to-Ascend
解决华为昇腾模型转换痛点，将一切复杂命令转换为图形化傻瓜操作，具备完整校验功能能和导出功能，以及无脑int8量化功能，支持图形化模型量化校准以及各项指标验证。支持混合精度导出fp32/fp16，支持int8量化主干后导出fp32模型供fp32推理脚本上运行（无需重写推理脚本以支持量化模型且精度稳定）。实验设备是香橙派AIpro 20T。
## PTH/PT -> ONNX -> OM 转换工具（`pth2om_gui.py`）使用说明

本目录提供 **一键转换工具** `pth2om_gui.py`，用于将 PyTorch 权重文件转换为 Ascend 310B1 可用的 OM，并可选做数值一致性校验。

> 设计目标：工具脚本本身是“纯个体”，不依赖仓库其它 `model.py`。  
> 当 checkpoint 内包含完整模型结构时无需模板；当仅包含 `state_dict` 时通过 `templates/` 插件目录扩展模板。

---

### 1) 文件结构

```
asc_sample/tools/
├── pth2om_gui.py                  # 转换主工具（GUI，可在 Linux/Windows 使用）
└── templates/
    └── sobel_resnet50_4ch.py      # 示例模板（可照此扩展 YOLO 等）
```

---

### 2) 运行环境要求

#### 2.1 Windows / 非 Ascend 主机（用于导出 ONNX）
- **必需**：Python + `torch` + `torchvision` + `onnx` + `onnxruntime` + `tkinter`
- **可选**：CUDA（如果你的 onnxruntime 支持 CUDA provider，校验会更快）
- **注意**：Windows 端通常不具备 `atc`，因此工具会自动跳过 ATC/OM/ACL 校验阶段。

#### 2.2 Linux + Ascend（用于 ATC 转 OM 与可选 ACL 校验）
- **必需**：`atc` 可用（不同镜像的 `atc` 可能不支持 `--version`，建议用 `atc --help | head` 或读取 `ascend_toolkit_install.info` 确认版本）
- **导出 ONNX 仍需要**：`torch` + `torchvision` + `onnx`
- **ONNX 校验需要**：`onnxruntime`（如果你板上 ORT 不完整，可关闭 Validate ONNX）
- **OM 校验需要**：`acl` Python 包可导入（`python -c "import acl"`）

---

### 3) 工具使用流程（推荐）

#### 3.1 Step A：选择 checkpoint
工具支持两种 checkpoint：

1. **包含完整结构的 checkpoint（推荐）**  
例如通过 `torch.save(model)` 保存的 `.pt/.pth`。  
工具会显示 “Checkpoint: full module (no template required).”，并自动禁用模板选择。

2. **仅 state_dict 的 checkpoint**  
例如 `{model_state_dict: ...}` 或直接保存 `state_dict`。  
工具会显示 “Checkpoint: state_dict only (template required).”，你需要在下拉框选择一个模板来重建结构。

#### 3.2 Step B：导出 ONNX
在 GUI 中设置：
- **Batch**：建议与 `atc --input_shape` 保持一致（通常固定 `1`）
- **In channels**：输入通道数（会在选择 checkpoint 后自动探测并回填；也可手动修改）
- **Input size**：与训练/预处理一致（例如 `384`）
- **OPSET**：默认 `17`，若遇到导出/兼容问题，脚本会自动尝试降级（17→16→15→14→13）
- **Dynamic axes**：若你准备固定输入，建议关闭（更容易 `atc`）
- **启用混合精度（内部FP16/FP32）**：勾选后会自动给 `atc` 添加 `--precision_mode=allow_mix_precision`（常用于加速；推荐保持 I/O 为 FP32）
- **编译前先做ATC预检查**：勾选后会在真正编译前调用 `atc --mode=3` 预检查，提前发现不支持算子/shape 等问题

#### 3.3 Step C：可选校验（建议按环境选择）
- **Validate ONNX**：比较 PyTorch vs ONNX 输出
  - 如果你板子上的 onnxruntime 报算子不支持（例如 MaxPool/element type），请取消勾选。
- **Validate OM (Linux+ACL)**：比较 PyTorch vs OM 输出
  - 若 ACL 绑定存在兼容问题，可先关闭校验，确保 OM 能生成用于部署。

#### 3.4 Step D：ATC 转 OM（仅 Linux）
工具内部会调用类似命令：
```bash
atc --model=xxx.onnx --framework=5 --output=xxx \
    --input_format=NCHW --input_shape=input:<batch>,<in_ch>,<img>,<img> \
    --output_type=FP32 --soc_version=Ascend310B1 \
    --op_select_implmode=high_precision
```

可选加速/排障参数（可填在 “ATC额外参数”）：
- `--precision_mode=allow_mix_precision`：内部混合精度（常见提速手段）
- `--log=info`：输出更多编译日志，便于定位失败原因

---

### 3.5 INT8 后量化（PTQ）页（单独 Tab）

工具新增了一个独立页签：**“INT8后量化(PTQ)”**，用于把导出的 FP32 ONNX 先做后量化，再转 OM。

说明：
- **量化步骤依赖外部量化工具**（例如昇腾环境中的 `amct_onnx`）。GUI 会执行你在“量化命令模板”里填的命令。
- 为了方便新手，PTQ 页提供两种模式：
  - **简易模式（默认）**：选择量化工具（目前内置 `amct_onnx`）+ 填路径即可，界面会自动拼出命令并展示“命令预览”。
  - **高级模式**：允许自定义“量化命令模板”（适配不同量化工具或参数习惯）。
- PTQ 页新增 **量化方式**（无命令行也能用）：
  - **运行量化工具生成量化ONNX（推荐）**：在设备上存在量化工具时使用（例如已安装 `amct_onnx`）。
  - **我已有量化ONNX（跳过量化）**：当边缘定制镜像缺少 AMCT/你不想写命令行时使用——你只需选择“量化ONNX输入”，工具会直接执行 `atc` 编译生成 OM。
- 若你的镜像/环境中 **没有 `amct_onnx`**：
  - 建议在设备上安装 AMCT(ONNX) 组件后再用“简易模式”；或
  - 直接切换到**高级模式**，调用你已有的量化工具命令。
  - 如果你完全不想/不会使用命令行：可以在其它环境先生成“量化 ONNX”，然后在 PTQ 页选择“我已有量化ONNX（跳过量化）”，只让设备执行 `atc` 编译生成 OM。
- 什么时候用 PTH？什么时候用 ONNX？
  - **用 PTH**：只有 `.pt/.pth`，或希望工具先导出 FP32 ONNX（并可选做 PyTorch vs ONNX 校验）。
  - **用 ONNX**：已经有一份稳定的 FP32 ONNX（常见做法：在 PC/训练机导出后拷贝到板子），板子侧只做 PTQ+ATC。
- ONNX 要量化前还是量化后？
  - **输入**：必须是“量化前 FP32 ONNX”。
  - **输出**：量化工具会生成“量化后 ONNX”，再交给 `atc` 生成 OM。
- 量化页提供 3 个关键路径：
  - **FP32 ONNX 输入**：PTQ 的输入 ONNX；来源可选“从PTH导出ONNX”或“使用已有ONNX”。
  - **校准数据目录（可选）**：由你的量化工具使用（内容格式取决于量化工具本身）。可通过“使用校准集”勾选控制是否启用。
  - **量化 ONNX 输出**：量化工具生成的 ONNX 路径。
  - **INT8 OM 输出**：后续 `atc` 产物路径。
- “量化命令模板”支持占位符：
  - `{onnx_in_q}`：FP32 ONNX（带 shell 引号）
  - `{onnx_out_q}`：量化 ONNX 输出（带 shell 引号）
  - `{calib_dir_q}`：校准数据目录（带 shell 引号）

> 注意：INT8 PTQ 的数值误差通常大于 FP32/FP16，建议最终用任务指标（分类 Top1 / 检测 mAP）验收。

### 4) 中文界面开关

默认 UI 为中文。若你希望切换为英文，可通过环境变量设置：

- Linux:
```bash
export PTH2OM_LANG=en
python pth2om_gui.py
```

- Windows PowerShell:
```powershell
setx PTH2OM_LANG en
python pth2om_gui.py
```

---

### 5) 扩展模型模板（核心）

当你的 checkpoint 只有 `state_dict`，必须提供模板来重建网络结构。模板通过放置 `templates/*.py` 插件实现。

#### 5.1 模板文件规范
在 `asc_sample/tools/templates/` 新建一个 `*.py`，需要提供：

- `TEMPLATE_NAME`：显示在下拉框里的名字
- `build(num_classes: int, pretrained: bool=False) -> torch.nn.Module`：返回 PyTorch 模型

示例（见 `templates/sobel_resnet50_4ch.py`）：

```python
TEMPLATE_NAME = "SobelResNet50(4ch)"

def build(num_classes: int, pretrained: bool = False) -> nn.Module:
    ...
    return model
```

#### 5.2 YOLO 模板怎么做（建议）
YOLO 版本分支多（v5/v8/自定义 head），仅凭 state_dict 很难“猜结构”。最佳实践：
- **优先使用保存完整模型结构的 `.pt`**（例如 Ultralytics 的 `.pt`），这样无需模板。
- 如果必须用 state_dict：
  - 你需要在模板里完整定义 YOLO 网络结构（Backbone/Neck/Head/NMS 是否入图）
  - 在 `build()` 内根据 `num_classes` 组装 head，并返回模型对象

---

### 6) 常见问题（FAQ）

#### 6.1 “Compare close=false” 但 max diff 很小
属于浮点误差。脚本内已将比较阈值放宽到 `atol=5e-3, rtol=5e-3`，一般无需担心。

#### 6.2 ATC 成功但 ACL 校验后进程崩溃（segfault）
多发生在 ACL 释放阶段（binding 差异导致 double free）。建议：
- 优先关闭 `Validate OM`，以保证 OM 产物可用于部署；
- 或升级/统一 ACL Python 与 CANN 版本。

#### 6.3 onnxruntime 提示某算子未实现
说明该环境的 ORT build 不完整（板载环境常见）。关闭 `Validate ONNX` 即可继续转 OM。

---

### 7) 最小命令行建议（无桌面环境）

如果目标 Linux 没有桌面/X11，Tkinter GUI 可能无法启动。建议在有桌面机器上完成导出 ONNX 与 ATC 转 OM，或后续将工具补充 CLI 模式（可以继续在本项目上扩展）。


