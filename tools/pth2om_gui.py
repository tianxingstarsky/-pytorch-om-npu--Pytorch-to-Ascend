#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter GUI tool: PTH/PT -> ONNX -> (Linux/Ascend) atc -> OM.

Notes:
- Standalone script: does not import any external model.py from a repo.
- If the checkpoint stores a full torch module (torch.save(model) / torch.jit.save), it will load structure+weights directly.
- If the checkpoint stores only a state_dict, choose a built-in template to rebuild the network (default: SobelResNet50 4-channel).
- OM numeric validation is only available on Linux + ACL. Windows will skip.
"""

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

try:
    import onnx  # noqa: F401
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import acl  # type: ignore
except ImportError:
    acl = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def get_lang() -> str:
    # Default to Chinese UI. You can override by setting env: PTH2OM_LANG=en
    lang = os.getenv("PTH2OM_LANG", "").strip().lower()
    return lang or "zh"


LANG = get_lang()


def t(en: str, zh: str) -> str:
    return zh if LANG.startswith("zh") else en


class SobelResNet50(nn.Module):
    """Built-in template: 4-channel input ResNet50 with configurable classifier head."""

    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            4,
            original_conv.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight[:, :3, :, :] = original_conv.weight
                backbone.conv1.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def detect_num_classes(state_dict: dict, default: int = 1000) -> int:
    for key in ("model.fc.weight", "fc.weight", "classifier.weight", "head.weight"):
        if key in state_dict:
            return state_dict[key].shape[0]
    return default


def strip_classifier(state_dict: dict) -> None:
    head_keys = ("model.fc.weight", "model.fc.bias", "fc.weight", "fc.bias", "head.weight", "head.bias")
    for key in head_keys:
        state_dict.pop(key, None)


def _load_checkpoint_any(model_path: Path, device: torch.device):
    """Load from PTH/PT: prefer a full torch module object, otherwise load a state_dict."""
    obj = torch.load(model_path, map_location=device)
    # 1) saved nn.Module
    if isinstance(obj, nn.Module):
        return "module", obj
    # 2) dict contains a torch module
    if isinstance(obj, dict):
        for key in ("model", "net", "module"):
            if key in obj and isinstance(obj[key], nn.Module):
                return "module", obj[key]
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return "state_dict", obj[key]
    # 3) raw state_dict
    if isinstance(obj, dict):
        return "state_dict", obj
    raise RuntimeError("Unsupported checkpoint format. Please provide a torch-saved module or a state_dict.")


TemplateBuilder = Callable[[int, bool], nn.Module]


def _builtin_templates() -> Dict[str, TemplateBuilder]:
    return {"SobelResNet50(4ch)": lambda n, p: SobelResNet50(num_classes=n, pretrained=p)}


def _load_external_templates(template_dir: Path) -> Dict[str, TemplateBuilder]:
    templates: Dict[str, TemplateBuilder] = {}
    if not template_dir.exists() or not template_dir.is_dir():
        return templates

    import importlib.util

    for py in sorted(template_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"pth2om_tpl_{py.stem}", py)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        name = getattr(mod, "TEMPLATE_NAME", py.stem)
        build_fn = getattr(mod, "build", None)
        if callable(build_fn):
            templates[str(name)] = build_fn
    return templates


def list_templates() -> Dict[str, TemplateBuilder]:
    templates = _builtin_templates()
    templates.update(_load_external_templates(TEMPLATES_DIR))
    return templates


def load_pth_model(
    model_path: Path,
    device: torch.device,
    manual_classes: Optional[int],
    template: str,
    pretrained_template: bool,
    log_fn,
) -> Tuple[nn.Module, int]:
    kind, payload = _load_checkpoint_any(model_path, device)

    if kind == "module":
        model = payload
        model.to(device)
        model.eval()
        log_fn(t("Loaded full torch module from checkpoint.", "已加载完整Torch模型（含结构）"))
        return model, manual_classes or -1

    state_dict = payload
    inferred_classes = detect_num_classes(state_dict)
    target_classes = manual_classes or inferred_classes
    strict = True
    if manual_classes and manual_classes != inferred_classes:
        log_fn(
            t(
                f"Head classes in checkpoint={inferred_classes}, user specified={manual_classes}. Rebuilding classifier head.",
                f"权重分类头={inferred_classes}，手动指定={manual_classes}，重建分类头",
            )
        )
        strip_classifier(state_dict)
        strict = False

    templates = list_templates()
    if template not in templates:
        raise RuntimeError(f"Template not found: {template}. Put template .py files in: {TEMPLATES_DIR}")
    model = templates[template](target_classes, pretrained_template)
    result = model.load_state_dict(state_dict, strict=strict)
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    if missing:
        log_fn(t(f"Missing keys: {missing}", f"缺失参数: {missing}"))
    if unexpected:
        log_fn(t(f"Unexpected keys: {unexpected}", f"多余参数: {unexpected}"))

    model.to(device)
    model.eval()
    return model, target_classes


def export_onnx(model: nn.Module, dummy: torch.Tensor, onnx_path: Path, dynamic: bool, opset: int, log_fn) -> None:
    kwargs = {
        "input_names": ["input"],
        "output_names": ["output"],
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if dynamic:
        kwargs["dynamic_axes"] = {"input": {0: "batch"}, "output": {0: "batch"}}
    log_fn(f"Export ONNX -> {onnx_path}")
    # Prefer legacy exporter to avoid onnxscript version_converter failures on some setups
    try:
        torch.onnx.export(model, dummy, onnx_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, onnx_path, **kwargs)


def run_torch(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        output = model(tensor)
    return output.detach().cpu().numpy()


def run_onnx_runtime(onnx_path: Path, tensor: torch.Tensor, use_cuda: bool, log_fn) -> np.ndarray:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed; cannot validate ONNX.")
    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if use_cuda and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    log_fn(f"onnxruntime providers: {providers} (available={available})")
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor.detach().cpu().numpy()})
    return outputs[0]


def compare_arrays(ref: np.ndarray, target: np.ndarray) -> dict:
    diff = np.abs(ref - target)
    return {
        "close": bool(np.allclose(ref, target, atol=5e-3, rtol=5e-3)),
        "max": float(diff.max()),
        "mean": float(diff.mean()),
    }


def build_atc_command(
    onnx_path: Path,
    om_path: Path,
    batch: int,
    in_channels: int,
    img_size: int,
    soc_version: str,
    output_type: str = "FP32",
    enable_mix_precision: bool = False,
    extra_atc_args: Optional[str] = None,
) -> Tuple[list, Path]:
    output_prefix = om_path.with_suffix("")
    output_type = (output_type or "FP32").strip().upper()
    if output_type not in ("FP32", "FP16"):
        output_type = "FP32"
    cmd = [
        "atc",
        f"--model={onnx_path}",
        "--framework=5",
        f"--output={output_prefix}",
        "--input_format=NCHW",
        f"--input_shape=input:{batch},{in_channels},{img_size},{img_size}",
        f"--output_type={output_type}",
        f"--soc_version={soc_version}",
        "--op_select_implmode=high_precision",
    ]
    # 混合精度（内部尽量用 FP16/FP32 混合）
    if enable_mix_precision:
        cmd.append("--precision_mode=allow_mix_precision")
    cmd.extend(_split_cmdline(extra_atc_args or ""))
    return cmd, output_prefix.with_suffix(".om")


def _infer_in_channels_from_module(model: nn.Module) -> Optional[int]:
    try:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "weight") and getattr(m.weight, "ndim", 0) == 4:
                return int(m.weight.shape[1])
    except Exception:
        return None
    return None


def _infer_in_channels_from_state_dict(state_dict: dict) -> Optional[int]:
    try:
        for key in ("conv1.weight", "model.conv1.weight", "backbone.conv1.weight"):
            w = state_dict.get(key, None)
            if hasattr(w, "shape") and len(w.shape) == 4:
                return int(w.shape[1])
        # fallback: first 4D weight tensor
        for _, w in state_dict.items():
            if hasattr(w, "shape") and len(w.shape) == 4:
                return int(w.shape[1])
    except Exception:
        return None
    return None


def _infer_num_classes_from_module(model: nn.Module) -> Optional[int]:
    # Heuristic: common classifier heads
    for attr in ("fc", "classifier", "head"):
        layer = getattr(model, attr, None)
        if isinstance(layer, nn.Linear):
            return int(layer.out_features)
    # Try nested common patterns (e.g., model.model.fc)
    for attr in ("model", "backbone", "net"):
        inner = getattr(model, attr, None)
        if inner is None:
            continue
        for inner_attr in ("fc", "classifier", "head"):
            layer = getattr(inner, inner_attr, None)
            if isinstance(layer, nn.Linear):
                return int(layer.out_features)
    return None


def _split_cmdline(cmdline: str) -> list:
    cmdline = (cmdline or "").strip()
    if not cmdline:
        return []
    try:
        return shlex.split(cmdline)
    except Exception:
        # fallback: very naive split
        return cmdline.split()


def _shell_quote(s: str) -> str:
    s = str(s)
    if sys.platform.startswith("win"):
        # PowerShell/cmd compatible enough for our usage (GUI users can still override).
        if '"' in s:
            s = s.replace('"', '\\"')
        return f'"{s}"'
    return shlex.quote(s)


def _detect_quant_tools() -> list:
    """
    Detect available PTQ quantization CLI tools on current host.
    NOTE: We keep this conservative; users can still use Advanced mode for any custom tool.
    """
    candidates = ["amct_onnx"]
    available = [c for c in candidates if shutil.which(c)]
    return available


def parse_atc_log(log_text: str) -> list:
    failed_ops = []
    for line in log_text.splitlines():
        if "opType" in line and ("opName" in line or "op_name" in line):
            failed_ops.append(line.strip())
    return failed_ops


def is_linux() -> bool:
    return sys.platform.startswith("linux")


class AclOmRunner:
    """Minimal OM runner using ACL python API (single input only)."""

    ACL_MEM_MALLOC_NORMAL_ONLY = 0
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2

    @staticmethod
    def _ret_code(ret) -> int:
        """
        Normalize return code across different ACL python bindings.
        Some APIs may return:
        - int ret
        - (ptr, ret) where ptr is a big integer address
        - (ret, ptr)
        - (something, 0) where 0 means success
        """
        if isinstance(ret, tuple) and ret:
            ints = [int(x) for x in ret if isinstance(x, int)]
            if not ints:
                return 0
            # If any element is 0, treat as success.
            if 0 in ints:
                return 0
            # Heuristic: ret code is usually the smallest-magnitude integer.
            ints_sorted = sorted(ints, key=lambda v: abs(v))
            return ints_sorted[0]
        return int(ret)

    @staticmethod
    def _unpack_ptr_ret(result):
        # normalize (ptr, ret) vs (ret, ptr)
        if isinstance(result, tuple) and len(result) == 2:
            a, b = result
            if isinstance(a, int) and not isinstance(b, int):
                return b, a
            return a, b
        return result, 0

    def __init__(self, model_path: Path, device_id: int = 0, log_fn: Optional[Callable[[str], None]] = None):
        if acl is None:
            raise RuntimeError("acl python package is missing; cannot validate OM output.")
        if not is_linux():
            raise RuntimeError("OM validation is only supported on Linux + Ascend.")
        self.model_path = model_path
        self.device_id = device_id
        self.log_fn = log_fn
        self._released = False
        self._acl_inited = False
        self._init_resources()

    def _check_ret(self, ret, msg: str):
        code = self._ret_code(ret)
        if code != 0:
            raise RuntimeError(f"{msg} failed, ret={ret}")

    def _init_resources(self):
        self._check_ret(acl.init(), "acl.init")
        self._acl_inited = True
        self._check_ret(acl.rt.set_device(self.device_id), "rt.set_device")
        ctx, ret = self._unpack_ptr_ret(acl.rt.create_context(self.device_id))
        self.context = ctx
        self._check_ret(ret, "rt.create_context")
        stream, ret = self._unpack_ptr_ret(acl.rt.create_stream())
        self.stream = stream
        self._check_ret(ret, "rt.create_stream")
        mid, ret = self._unpack_ptr_ret(acl.mdl.load_from_file(str(self.model_path)))
        self.model_id = mid
        self._check_ret(ret, "mdl.load_from_file")
        self.model_desc = acl.mdl.create_desc()
        self._check_ret(acl.mdl.get_desc(self.model_desc, self.model_id), "get_desc")

    def _create_dataset(self, num, size_getter):
        dataset = acl.mdl.create_dataset()
        buffers = []
        for idx in range(num):
            size = size_getter(idx)
            buf, ret = self._unpack_ptr_ret(acl.rt.malloc(size, self.ACL_MEM_MALLOC_NORMAL_ONLY))
            self._check_ret(ret, "rt.malloc")
            data_buffer = acl.create_data_buffer(buf, size)
            if data_buffer is None:
                raise RuntimeError("Failed to create data buffer.")
            add_ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
            self._check_ret(add_ret, "mdl.add_dataset_buffer")
            buffers.append((buf, data_buffer, size))
        return dataset, buffers

    def infer(self, input_array: np.ndarray, output_shape: Tuple[int, ...]) -> np.ndarray:
        input_array = np.ascontiguousarray(input_array.astype(np.float32))
        input_num = acl.mdl.get_num_inputs(self.model_desc)
        output_num = acl.mdl.get_num_outputs(self.model_desc)

        input_dataset, input_buffers = self._create_dataset(
            input_num, lambda idx: acl.mdl.get_input_size_by_index(self.model_desc, idx)
        )
        output_dataset, output_buffers = self._create_dataset(
            output_num, lambda idx: acl.mdl.get_output_size_by_index(self.model_desc, idx)
        )

        if self.log_fn:
            self.log_fn(f"ACL input_size={input_buffers[0][2]} bytes, np_nbytes={input_array.nbytes}")

        # Prefer bytes_to_ptr to avoid deprecated numpy_to_ptr (binding differences across versions).
        # Some acl python builds require a bytes-like object (bytes/bytearray) as input.
        if hasattr(acl.util, "bytes_to_ptr"):
            in_bytes = input_array.tobytes()
            host_ptr = acl.util.bytes_to_ptr(in_bytes)
        else:
            host_ptr = acl.util.numpy_to_ptr(input_array)
        ret = acl.rt.memcpy(
            input_buffers[0][0],
            input_array.nbytes,
            host_ptr,
            input_array.nbytes,
            self.ACL_MEMCPY_HOST_TO_DEVICE,
        )
        self._check_ret(ret, "rt.memcpy H2D")

        self._check_ret(
            acl.mdl.execute_async(self.model_id, input_dataset, output_dataset, self.stream),
            "mdl.execute_async",
        )
        self._check_ret(acl.rt.synchronize_stream(self.stream), "rt.synchronize_stream")

        out_bytes_len = int(output_buffers[0][2])
        if hasattr(acl.util, "bytes_to_ptr"):
            # Use a writable bytearray as host buffer to receive D2H memcpy safely.
            out_buf = bytearray(out_bytes_len)
            out_ptr = acl.util.bytes_to_ptr(out_buf)
        else:
            host_out = np.zeros((out_bytes_len // 4,), dtype=np.float32)
            out_ptr = acl.util.numpy_to_ptr(host_out)
        ret = acl.rt.memcpy(
            out_ptr,
            out_bytes_len,
            output_buffers[0][0],
            out_bytes_len,
            self.ACL_MEMCPY_DEVICE_TO_HOST,
        )
        self._check_ret(ret, "rt.memcpy D2H")

        self._destroy_dataset(input_dataset)
        self._destroy_dataset(output_dataset)
        for buf, data_buf, _ in input_buffers + output_buffers:
            acl.destroy_data_buffer(data_buf)
            acl.rt.free(buf)

        if hasattr(acl.util, "bytes_to_ptr"):
            host_out = np.frombuffer(out_buf, dtype=np.float32)
        return host_out.reshape(output_shape)

    @staticmethod
    def _destroy_dataset(dataset):
        if dataset is None:
            return
        acl.mdl.destroy_dataset(dataset)

    def release(self):
        # Make release idempotent to avoid double-free / double-finalize segfaults.
        if self._released:
            return
        self._released = True
        try:
            if hasattr(self, "model_id"):
                acl.mdl.unload(self.model_id)
            if hasattr(self, "model_desc"):
                acl.mdl.destroy_desc(self.model_desc)
            if hasattr(self, "stream"):
                acl.rt.destroy_stream(self.stream)
            if hasattr(self, "context"):
                acl.rt.destroy_context(self.context)
            # reset_device/finalize may crash if called multiple times across bindings; guard with flag.
            try:
                acl.rt.reset_device(self.device_id)
            except Exception:
                pass
            if self._acl_inited:
                try:
                    acl.finalize()
                except Exception:
                    pass
        except Exception:
            # Never raise from release; avoid crashing interpreter shutdown
            pass

    def __del__(self):
        # Do not crash during interpreter shutdown
        self.release()


@dataclass
class GuiConfig:
    img_size: int = 384
    batch_size: int = 1
    opset: int = 17
    soc_version: str = "Ascend310B1"


class ConversionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(t("PTH-ONNX-OM Converter", "PTH-ONNX-OM 转换工具"))
        self.geometry("960x720")
        self.resizable(True, True)

        self.model_path = tk.StringVar()
        self.onnx_path = tk.StringVar()
        self.om_path = tk.StringVar()
        self.om_output_type_var = tk.StringVar(value="FP32")
        self.enable_mix_precision_var = tk.BooleanVar(value=True)
        self.atc_extra_args_var = tk.StringVar(value="")
        self.atc_precheck_var = tk.BooleanVar(value=True)
        self._atc_precheck_status_var = tk.StringVar(value=t("ATC pre-check: not run", "ATC预检查：未运行"))

        # INT8 PTQ page
        self.ptq_source_var = tk.StringVar(value="from_pth")  # from_pth | from_onnx
        self.ptq_onnx_in_path = tk.StringVar()
        # PTQ quantization step control: run tool vs use existing quantized onnx
        self.ptq_quant_flow_var = tk.StringVar(value="auto")  # auto | run_tool | use_existing
        self.ptq_existing_quant_onnx_var = tk.StringVar()
        self.ptq_use_calib_var = tk.BooleanVar(value=True)
        self.ptq_mode_var = tk.StringVar(value="easy")  # easy | advanced
        self.ptq_quant_tool_var = tk.StringVar(value="")
        self.ptq_tool_status_var = tk.StringVar(value="")
        self.ptq_quant_extra_args_var = tk.StringVar(value="")
        self.ptq_calib_dir_var = tk.StringVar()
        self.ptq_quant_onnx_path = tk.StringVar()
        self.ptq_om_path = tk.StringVar()
        self.ptq_om_output_type_var = tk.StringVar(value="FP32")
        self.ptq_atc_extra_args_var = tk.StringVar(value="")
        self.sample_image = tk.StringVar()
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        self.manual_classes = tk.StringVar()
        self.batch_var = tk.StringVar(value="1")
        self.in_channels_var = tk.StringVar(value="4")
        self.img_size_var = tk.StringVar(value="384")
        self.soc_var = tk.StringVar(value="Ascend310B1")
        self.opset_var = tk.StringVar(value="17")
        self.dynamic_axes = tk.BooleanVar(value=False)
        self.validate_om = tk.BooleanVar(value=True)
        self.validate_onnx = tk.BooleanVar(value=True)
        self.device_id_var = tk.StringVar(value="0")
        self.template_var = tk.StringVar(value="SobelResNet50(4ch)")
        self.template_pretrained_var = tk.BooleanVar(value=False)
        self._is_linux = is_linux()
        self._ckpt_info_var = tk.StringVar(value=t("Checkpoint: not selected.", "权重：未选择"))

        # Defaults for PTQ page outputs
        self.ptq_quant_onnx_path.set(str(PROJECT_ROOT / "quantized_int8.onnx"))
        self.ptq_om_path.set(str(PROJECT_ROOT / "converted_int8.om"))
        self.ptq_onnx_in_path.set(str(PROJECT_ROOT / "ptq_input_fp32.onnx"))
        self.ptq_existing_quant_onnx_var.set("")

        self.log_text = tk.Text(self, height=20)
        self._build_widgets()

    def _build_widgets(self):
        # 顶部工具栏：放全局按钮（如“帮助”）
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", padx=10, pady=6)
        ttk.Button(top_bar, text=t("Help", "帮助"), command=self._show_help).pack(side="right")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="x", padx=10, pady=5)

        tab_convert = ttk.Frame(notebook)
        tab_ptq = ttk.Frame(notebook)
        notebook.add(tab_convert, text=t("Basic convert", "基础转换"))
        notebook.add(tab_ptq, text=t("INT8 PTQ", "INT8后量化(PTQ)"))

        # ===== Tab 1: basic convert =====
        frm_paths = ttk.LabelFrame(tab_convert, text=t("Paths", "路径"))
        frm_paths.pack(fill="x", padx=10, pady=5)

        self._add_file_selector(frm_paths, t("PTH model", "PTH模型"), self.model_path, 0, self._select_pth)
        self._add_file_selector(
            frm_paths, t("ONNX output", "ONNX输出"), self.onnx_path, 1, self._select_onnx, save=True, def_ext=".onnx"
        )
        self._add_file_selector(frm_paths, t("OM output", "OM输出"), self.om_path, 2, self._select_om, save=True, def_ext=".om")
        ttk.Label(frm_paths, textvariable=self._ckpt_info_var).grid(row=3, column=1, sticky="w", padx=5, pady=2)

        frm_opts = ttk.LabelFrame(tab_convert, text=t("Options", "选项"))
        frm_opts.pack(fill="x", padx=10, pady=5)

        ttk.Label(frm_opts, text=t("Torch device", "Torch设备")).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(frm_opts, textvariable=self.device_var, values=["cpu", "cuda"], width=10, state="readonly").grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(frm_opts, text=t("Batch", "Batch")).grid(row=0, column=2, padx=5)
        ttk.Entry(frm_opts, textvariable=self.batch_var, width=6).grid(row=0, column=3, sticky="w")

        ttk.Label(frm_opts, text=t("Input size", "输入尺寸")).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frm_opts, textvariable=self.img_size_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(frm_opts, text=t("In channels", "输入通道")).grid(row=1, column=2, padx=5, sticky="w")
        ttk.Entry(frm_opts, textvariable=self.in_channels_var, width=6).grid(row=1, column=3, sticky="w")
        ttk.Label(frm_opts, text="OPSET").grid(row=1, column=4)
        ttk.Entry(frm_opts, textvariable=self.opset_var, width=6).grid(row=1, column=5, sticky="w")

        ttk.Label(frm_opts, text=t("SoC", "SoC型号")).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frm_opts, textvariable=self.soc_var, width=16).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(frm_opts, text=t("Dynamic axes", "动态维度"), variable=self.dynamic_axes).grid(row=2, column=2, sticky="w", padx=5)

        ttk.Label(frm_opts, text=t("Template (for state_dict)", "模板(仅state_dict需要)")).grid(row=2, column=3, padx=5, pady=5, sticky="w")
        self.template_combo = ttk.Combobox(
            frm_opts,
            textvariable=self.template_var,
            values=list(list_templates().keys()),
            width=22,
            state="readonly",
        )
        self.template_combo.grid(row=2, column=4, sticky="w")
        ttk.Checkbutton(frm_opts, text=t("Template pretrained", "模板预训练"), variable=self.template_pretrained_var).grid(
            row=2, column=5, sticky="w", padx=5
        )

        ttk.Label(frm_opts, text=t("Manual classes (optional)", "手动类别数(可选)")).grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frm_opts, textvariable=self.manual_classes, width=10).grid(row=3, column=1, sticky="w")
        ttk.Checkbutton(frm_opts, text=t("Validate ONNX", "校验ONNX"), variable=self.validate_onnx).grid(row=3, column=2, sticky="w", padx=5)
        ttk.Checkbutton(frm_opts, text=t("Validate OM (Linux+ACL)", "校验OM(ACL)"), variable=self.validate_om).grid(row=3, column=3, sticky="w", padx=5)
        ttk.Label(frm_opts, text=t("ACL device id", "ACL设备ID")).grid(row=3, column=4, padx=5, sticky="w")
        ttk.Entry(frm_opts, textvariable=self.device_id_var, width=6).grid(row=3, column=5, sticky="w")

        ttk.Label(frm_opts, text=t("OM output type", "OM输出精度")).grid(row=4, column=0, padx=5, pady=5, sticky="w")
        ttk.Combobox(frm_opts, textvariable=self.om_output_type_var, values=["FP32", "FP16"], width=8, state="readonly").grid(
            row=4, column=1, sticky="w"
        )
        ttk.Checkbutton(
            frm_opts,
            text=t("Enable FP16/FP32 mixed precision (internal)", "启用混合精度（内部FP16/FP32）"),
            variable=self.enable_mix_precision_var,
        ).grid(row=4, column=2, sticky="w", padx=5)

        ttk.Label(frm_opts, text=t("ATC extra args", "ATC额外参数")).grid(row=5, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frm_opts, textvariable=self.atc_extra_args_var, width=72).grid(row=5, column=1, columnspan=5, sticky="w")

        ttk.Checkbutton(
            frm_opts,
            text=t("Run ATC pre-check before build", "编译前先做ATC预检查"),
            variable=self.atc_precheck_var,
        ).grid(row=6, column=0, sticky="w", padx=5, pady=2)
        ttk.Button(frm_opts, text=t("ATC pre-check now", "立即预检查"), command=self._run_atc_precheck_async).grid(
            row=6, column=1, sticky="w", padx=5, pady=2
        )
        ttk.Label(frm_opts, textvariable=self._atc_precheck_status_var).grid(row=6, column=2, columnspan=4, sticky="w", padx=5, pady=2)

        ttk.Button(tab_convert, text=t("Run", "开始"), command=self._run_async).pack(pady=10)

        # ===== Tab 2: INT8 PTQ =====
        frm_ptq = ttk.LabelFrame(tab_ptq, text=t("INT8 post-training quantization", "INT8 后训练量化（PTQ）"))
        frm_ptq.pack(fill="x", padx=10, pady=5)

        # Source selection: from PTH export ONNX vs use existing ONNX
        src_row = 0
        ttk.Label(frm_ptq, text=t("Source", "来源")).grid(row=src_row, column=0, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(
            frm_ptq,
            text=t("Export ONNX from PTH", "从PTH导出ONNX"),
            variable=self.ptq_source_var,
            value="from_pth",
            command=self._refresh_ptq_ui_state,
        ).grid(row=src_row, column=1, sticky="w", padx=5)
        ttk.Radiobutton(
            frm_ptq,
            text=t("Use existing ONNX", "使用已有ONNX"),
            variable=self.ptq_source_var,
            value="from_onnx",
            command=self._refresh_ptq_ui_state,
        ).grid(row=src_row, column=2, sticky="w", padx=5)

        # PTH model input (used when source=from_pth)
        ttk.Label(frm_ptq, text=t("PTH model", "PTH模型")).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self._ptq_pth_entry = ttk.Entry(frm_ptq, textvariable=self.model_path, width=70)
        self._ptq_pth_entry.grid(row=1, column=1, padx=5, pady=5, columnspan=2, sticky="w")
        self._ptq_pth_btn = ttk.Button(frm_ptq, text=t("Browse", "浏览"), command=self._select_pth)
        self._ptq_pth_btn.grid(row=1, column=3, padx=5, pady=5)

        # ONNX input path (always needed for PTQ stage)
        ttk.Label(frm_ptq, text=t("FP32 ONNX input", "FP32 ONNX输入")).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frm_ptq, textvariable=self.ptq_onnx_in_path, width=70).grid(row=2, column=1, padx=5, pady=5, columnspan=2, sticky="w")
        ttk.Button(frm_ptq, text=t("Browse", "浏览"), command=self._select_ptq_onnx_in).grid(row=2, column=3, padx=5, pady=5)

        # Quantization flow selection (no-CLI friendly)
        ttk.Label(frm_ptq, text=t("Quantization", "量化方式")).grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(
            frm_ptq,
            text=t("Run quant tool to generate quantized ONNX", "运行量化工具生成量化ONNX（推荐）"),
            variable=self.ptq_quant_flow_var,
            value="run_tool",
            command=self._refresh_ptq_ui_state,
        ).grid(row=3, column=1, sticky="w", padx=5)
        ttk.Radiobutton(
            frm_ptq,
            text=t("I already have quantized ONNX", "我已有量化ONNX（跳过量化）"),
            variable=self.ptq_quant_flow_var,
            value="use_existing",
            command=self._refresh_ptq_ui_state,
        ).grid(row=3, column=2, sticky="w", padx=5)

        ttk.Label(frm_ptq, text=t("Quantized ONNX input", "量化ONNX输入")).grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self._ptq_existing_quant_entry = ttk.Entry(frm_ptq, textvariable=self.ptq_existing_quant_onnx_var, width=70)
        self._ptq_existing_quant_entry.grid(row=4, column=1, padx=5, pady=5, columnspan=2, sticky="w")
        self._ptq_existing_quant_btn = ttk.Button(frm_ptq, text=t("Browse", "浏览"), command=self._select_ptq_existing_quant_onnx)
        self._ptq_existing_quant_btn.grid(row=4, column=3, padx=5, pady=5)

        # Calibration toggle + dir
        # 说明：PTQ“校准”只需要前向统计，不需要反向传播算子。
        ttk.Checkbutton(
            frm_ptq, text=t("Use calibration set", "使用校准集"), variable=self.ptq_use_calib_var, command=self._refresh_ptq_ui_state
        ).grid(row=5, column=0, sticky="w", padx=5, pady=5)
        # move calib dir to row=2 for new layout
        self._ptq_calib_label = ttk.Label(frm_ptq, text=t("Calibration data dir", "校准数据目录"))
        self._ptq_calib_label.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self._ptq_calib_entry = ttk.Entry(frm_ptq, textvariable=self.ptq_calib_dir_var, width=62)
        self._ptq_calib_entry.grid(row=5, column=2, padx=5, pady=5, sticky="w")
        self._ptq_calib_btn = ttk.Button(frm_ptq, text=t("Browse", "浏览"), command=self._select_ptq_calib_dir)
        self._ptq_calib_btn.grid(row=5, column=3, padx=5, pady=5)

        ttk.Label(frm_ptq, text=t("Quantized ONNX output", "量化ONNX输出")).grid(row=6, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frm_ptq, textvariable=self.ptq_quant_onnx_path, width=70).grid(row=6, column=1, padx=5, pady=5, columnspan=2, sticky="w")
        ttk.Button(frm_ptq, text=t("Browse", "浏览"), command=self._select_ptq_quant_onnx).grid(row=6, column=3, padx=5, pady=5)

        ttk.Label(frm_ptq, text=t("INT8 OM output", "INT8 OM输出")).grid(row=7, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frm_ptq, textvariable=self.ptq_om_path, width=70).grid(row=7, column=1, padx=5, pady=5, columnspan=2, sticky="w")
        ttk.Button(frm_ptq, text=t("Browse", "浏览"), command=self._select_ptq_om).grid(row=7, column=3, padx=5, pady=5)

        ttk.Label(frm_ptq, text=t("OM output type", "OM输出精度")).grid(row=8, column=0, sticky="w", padx=5, pady=5)
        ttk.Combobox(frm_ptq, textvariable=self.ptq_om_output_type_var, values=["FP32", "FP16"], width=8, state="readonly").grid(
            row=8, column=1, sticky="w"
        )
        ttk.Label(frm_ptq, text=t("ATC extra args", "ATC额外参数")).grid(row=8, column=2, sticky="w", padx=5, pady=5)
        ttk.Entry(frm_ptq, textvariable=self.ptq_atc_extra_args_var, width=30).grid(row=8, column=3, sticky="w", padx=5, pady=5)

        # Quantization mode: easy vs advanced
        # 简易模式：自动拼量化命令（默认 amct_onnx）；高级模式：允许用户手写命令模板。
        ttk.Label(frm_ptq, text=t("PTQ mode", "量化模式")).grid(row=9, column=0, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(frm_ptq, text=t("Easy", "简易"), variable=self.ptq_mode_var, value="easy", command=self._refresh_ptq_ui_state).grid(
            row=9, column=1, sticky="w", padx=5
        )
        ttk.Radiobutton(
            frm_ptq, text=t("Advanced (custom command)", "高级（自定义命令）"), variable=self.ptq_mode_var, value="advanced", command=self._refresh_ptq_ui_state
        ).grid(row=9, column=2, columnspan=2, sticky="w", padx=5)

        # Easy mode: pick tool + extra args, show preview
        self._ptq_easy_frame = ttk.Frame(frm_ptq)
        self._ptq_easy_frame.grid(row=10, column=0, columnspan=4, sticky="w", padx=5, pady=5)
        ttk.Label(self._ptq_easy_frame, text=t("Quant tool", "量化工具")).grid(row=0, column=0, sticky="w")
        self._ptq_tool_combo = ttk.Combobox(
            self._ptq_easy_frame, textvariable=self.ptq_quant_tool_var, values=[], width=18, state="readonly"
        )
        self._ptq_tool_combo.grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(self._ptq_easy_frame, textvariable=self.ptq_tool_status_var).grid(row=0, column=2, columnspan=2, sticky="w", padx=5)
        ttk.Label(self._ptq_easy_frame, text=t("Tool extra args", "工具额外参数")).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(self._ptq_easy_frame, textvariable=self.ptq_quant_extra_args_var, width=72).grid(row=1, column=1, columnspan=3, sticky="w")
        ttk.Label(self._ptq_easy_frame, text=t("Command preview:", "命令预览：")).grid(row=2, column=0, sticky="w", pady=2)
        self._ptq_cmd_preview = tk.Text(self._ptq_easy_frame, height=2, width=94)
        self._ptq_cmd_preview.grid(row=3, column=0, columnspan=4, sticky="w")
        self._ptq_cmd_preview.configure(state="disabled")

        # Detect quantization tools on host and update UI
        tools = _detect_quant_tools()
        if tools:
            self._ptq_tool_combo.configure(values=tools)
            if not self.ptq_quant_tool_var.get().strip() or self.ptq_quant_tool_var.get().strip() not in tools:
                self.ptq_quant_tool_var.set(tools[0])
            self.ptq_tool_status_var.set(t("Detected", "已检测到"))
        else:
            none_label = t("(no tool)", "（未检测到量化工具）")
            self._ptq_tool_combo.configure(values=[none_label])
            self.ptq_quant_tool_var.set(none_label)
            self.ptq_tool_status_var.set(t("amct_onnx missing; use Advanced or install AMCT.", "缺少 amct_onnx：请用“高级模式”或安装 AMCT。"))

        # Advanced mode: custom command template
        self._ptq_adv_frame = ttk.Frame(frm_ptq)
        self._ptq_adv_frame.grid(row=11, column=0, columnspan=4, sticky="w", padx=5, pady=5)
        ttk.Label(self._ptq_adv_frame, text=t("Quant command template", "量化命令模板")).grid(row=0, column=0, sticky="nw", padx=0, pady=0)
        self.ptq_cmd_text = tk.Text(self._ptq_adv_frame, height=5, width=86)
        self.ptq_cmd_text.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        default_tpl = "amct_onnx --model {onnx_in_q} --output {onnx_out_q} --calibration_data {calib_dir_q}"
        self.ptq_cmd_text.insert("1.0", default_tpl)
        ttk.Label(
            self._ptq_adv_frame,
            text=t("Placeholders: {onnx_in_q} {onnx_out_q} {calib_dir_q}", "占位符：{onnx_in_q} {onnx_out_q} {calib_dir_q}"),
        ).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Button(tab_ptq, text=t("Run INT8 PTQ", "开始(INT8后量化)"), command=self._run_ptq_async).pack(pady=10)

        # ===== Shared log =====
        frm_log = ttk.LabelFrame(self, text=t("Log", "日志"))
        frm_log.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text.pack(in_=frm_log, fill="both", expand=True)

        # Initialize PTQ UI state & preview
        self._refresh_ptq_ui_state()
        for var in (
            self.ptq_onnx_in_path,
            self.ptq_quant_onnx_path,
            self.ptq_calib_dir_var,
            self.ptq_use_calib_var,
            self.ptq_quant_tool_var,
            self.ptq_quant_extra_args_var,
        ):
            try:
                var.trace_add("write", lambda *_: self._refresh_ptq_ui_state())
            except Exception:
                pass

    def _add_file_selector(self, parent, label, var, row, command, save=False, def_ext=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(parent, textvariable=var, width=70).grid(row=row, column=1, padx=5, pady=5)
        ttk.Button(parent, text=t("Browse", "浏览"), command=command).grid(row=row, column=2, padx=5, pady=5)
        if save:
            var.set(str(PROJECT_ROOT / f"converted_{row}{def_ext}"))

    def _select_pth(self):
        path = filedialog.askopenfilename(
            title=t("Select PTH/PT", "选择PTH/PT"),
            filetypes=[(t("PyTorch", "PyTorch"), "*.pth *.pt"), (t("All", "全部"), "*.*")],
        )
        if path:
            self.model_path.set(path)
            self._probe_checkpoint_async(Path(path))

    def _probe_checkpoint_async(self, model_path: Path) -> None:
        def _worker():
            try:
                kind, payload = _load_checkpoint_any(model_path, torch.device("cpu"))
                inferred_classes = None
                inferred_in_ch = None
                if kind == "module":
                    inferred_classes = _infer_num_classes_from_module(payload)
                    inferred_in_ch = _infer_in_channels_from_module(payload)
                else:
                    inferred_classes = detect_num_classes(payload)
                    inferred_in_ch = _infer_in_channels_from_state_dict(payload)

                info_parts = []
                if kind == "module":
                    info_parts.append(t("Checkpoint: full module", "权重：完整模型"))
                else:
                    info_parts.append(t("Checkpoint: state_dict only", "权重：仅state_dict"))
                if inferred_in_ch is not None:
                    info_parts.append(t(f"in_ch={inferred_in_ch}", f"输入通道={inferred_in_ch}"))
                if inferred_classes is not None and int(inferred_classes) > 0:
                    info_parts.append(t(f"classes={inferred_classes}", f"类别={inferred_classes}"))

                def _update_ui():
                    self._ckpt_info_var.set(" | ".join(info_parts))
                    if kind == "module":
                        self.template_combo.configure(state="disabled")
                    else:
                        self.template_combo.configure(state="readonly")
                    if inferred_in_ch is not None:
                        self.in_channels_var.set(str(int(inferred_in_ch)))

                self.after(0, _update_ui)
            except Exception as exc:  # noqa: broad-except
                def _update_ui_err():
                    self._ckpt_info_var.set(t(f"Checkpoint probe failed: {exc}", f"权重检查失败: {exc}"))
                    self.template_combo.configure(state="readonly")

                self.after(0, _update_ui_err)

        threading.Thread(target=_worker, daemon=True).start()

    def _select_onnx(self):
        path = filedialog.asksaveasfilename(
            title=t("Save ONNX", "保存ONNX"),
            defaultextension=".onnx",
            filetypes=[("ONNX", "*.onnx"), (t("All", "全部"), "*.*")],
        )
        if path:
            self.onnx_path.set(path)

    def _select_om(self):
        path = filedialog.asksaveasfilename(
            title=t("Save OM", "保存OM"),
            defaultextension=".om",
            filetypes=[("OM", "*.om"), (t("All", "全部"), "*.*")],
        )
        if path:
            self.om_path.set(path)

    def _select_ptq_calib_dir(self):
        path = filedialog.askdirectory(title=t("Select calibration data directory", "选择校准数据目录"))
        if path:
            self.ptq_calib_dir_var.set(path)

    def _select_ptq_onnx_in(self):
        path = filedialog.askopenfilename(
            title=t("Select FP32 ONNX", "选择FP32 ONNX"),
            filetypes=[("ONNX", "*.onnx"), (t("All", "全部"), "*.*")],
        )
        if path:
            self.ptq_onnx_in_path.set(path)

    def _select_ptq_quant_onnx(self):
        path = filedialog.asksaveasfilename(
            title=t("Save quantized ONNX", "保存量化ONNX"),
            defaultextension=".onnx",
            filetypes=[("ONNX", "*.onnx"), (t("All", "全部"), "*.*")],
        )
        if path:
            self.ptq_quant_onnx_path.set(path)

    def _select_ptq_existing_quant_onnx(self):
        path = filedialog.askopenfilename(
            title=t("Select quantized ONNX", "选择量化ONNX"),
            filetypes=[("ONNX", "*.onnx"), (t("All", "全部"), "*.*")],
        )
        if path:
            self.ptq_existing_quant_onnx_var.set(path)

    def _select_ptq_om(self):
        path = filedialog.asksaveasfilename(
            title=t("Save INT8 OM", "保存INT8 OM"),
            defaultextension=".om",
            filetypes=[("OM", "*.om"), (t("All", "全部"), "*.*")],
        )
        if path:
            self.ptq_om_path.set(path)

    def _log(self, msg: str):
        # Tk widgets must be updated in the main thread
        def _append():
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")

        try:
            self.after(0, _append)
        except RuntimeError:
            pass
        print(msg)

    def _show_help(self):
        """弹出帮助窗口（使用说明）。"""
        win = tk.Toplevel(self)
        win.title(t("Help", "使用说明"))
        win.geometry("920x720")
        win.resizable(True, True)

        frm = ttk.Frame(win)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        txt = tk.Text(frm, wrap="word")
        txt.pack(side="left", fill="both", expand=True)
        ybar = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        ybar.pack(side="right", fill="y")
        txt.configure(yscrollcommand=ybar.set)

        # 尽量用“任务导向”的说明，降低新手使用门槛。
        help_text = """
【工具用途】
把 PyTorch 模型转换为 Ascend 310B1 可运行的 OM：
- 基础转换：PTH/PT → ONNX → ATC → OM
- INT8 后量化（PTQ）：FP32 ONNX →（量化工具）→ 量化 ONNX → ATC → INT8 OM

【什么时候用 PTH？什么时候用 ONNX？】
- 用 PTH：你手头只有 .pt/.pth，或还没把导出 ONNX 这件事“固定住”（希望工具一条龙导出并可选校验）。
- 用 ONNX：你已经有稳定的 FP32 ONNX（常见做法：在 PC/训练机导出后拷贝到板子），板子只负责 PTQ+ATC 编译。

【ONNX 需要量化前还是量化后？】
- PTQ 的输入必须是“量化前的 FP32 ONNX”。
- 量化工具（例如 amct_onnx）会生成“量化后的 ONNX”。
- atc 以“量化后的 ONNX”为输入，生成最终 INT8 OM。

============================================================
一、基础转换（基础转换页）
1) 选择 PTH模型（.pt/.pth）
   - 如果 checkpoint 保存了完整结构（torch.save(model)），无需模板
   - 如果只有 state_dict，需要在“模板”下拉框选对应结构模板

2) 设置关键输入参数
   - Batch：通常部署固定 1
   - 输入通道：会自动探测并回填（3/4 等），也可手动改
   - 输入尺寸：与部署一致（如 384/640）
   - OPSET：默认 17，脚本会自动尝试降级（17→16→15→14→13）

3) 校验（可选）
   - 校验ONNX：用 onnxruntime 对齐 PyTorch 输出（更可靠）
   - 校验OM：在 Linux+ACL 上对齐 OM 输出（部署端验收）

4) 生成 OM
   - 需要 Linux 且已安装 atc（CANN 工具链）
   - “OM输出精度”建议先用 FP32（I/O FP32 更稳）

============================================================
二、INT8 后量化（PTQ页）——新手推荐“简易模式”
【核心概念】
PTQ 校准不需要反向传播算子，只做前向统计（scale/zero-point）。
校准集的关键是“输入分布要像真实业务”，不需要标签。

1) 来源（两种入口）
   - 从PTH导出ONNX：填 PTH模型 + 设置 Batch/通道/尺寸，工具会先导出 FP32 ONNX
   - 使用已有ONNX：直接选择 FP32 ONNX输入（跳过 PTH 导出）

【边缘镜像缺少 amct_onnx 时怎么办？】
若系统没有安装 amct_onnx（边缘定制镜像常见），你仍然可以用 GUI 完成“无命令行”的后半段：
1) 在其它环境生成“量化后的 ONNX”（例如在开发机/服务器上用 AMCT 量化好）；
2) 回到本工具 PTQ 页，选择“我已有量化ONNX（跳过量化）”，选中量化ONNX输入；
3) 点击开始，工具会直接执行 ATC → 生成 OM。

2) 是否使用校准集
   - 勾选：选择“校准数据目录”，量化工具会用它做统计（推荐）
   - 不勾选：命令里不会带 --calibration_data（仅在你明确知道工具支持无校准时使用）

3) 简易模式（默认）
   - 量化工具：优先使用 amct_onnx（若系统未安装，简易模式会提示；请切换高级模式或安装 AMCT）
   - 工具额外参数：只在你需要特殊开关时填写
   - 命令预览：工具会把最终要执行的命令展示出来（便于复制/排障）

4) 高级模式（可选）
   - 允许自定义“量化命令模板”以适配不同工具
   - 支持占位符：{onnx_in_q} {onnx_out_q} {calib_dir_q}

5) ATC额外参数（可选）
   - 追加到 atc 命令末尾，用于控制编译策略/日志
   - 示例：--log=info
   - 示例：--precision_mode=allow_mix_precision（有助于内部混合精度提速）

============================================================
三、常见坑
1) 输入 bytes 不匹配：说明 OM 的输入 shape/dtype 与你喂入的不一致（Batch/通道/尺寸/输出精度）
2) INT8 输出误差更大：PTQ 常见现象，最终用任务指标（分类 Top1 / 检测 mAP）验收
3) atc 转换失败：看日志中不支持算子/shape 报错，必要时先固定输入、关闭动态维度

"""
        txt.insert("1.0", help_text.strip() + "\n")
        txt.configure(state="disabled")

    def _run_async(self):
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _run_ptq_async(self):
        thread = threading.Thread(target=self._run_ptq_pipeline, daemon=True)
        thread.start()

    def _run_atc_precheck_async(self):
        thread = threading.Thread(target=self._run_atc_precheck, daemon=True)
        thread.start()

    def _run_pipeline(self):
        try:
            self._convert_and_validate()
            messagebox.showinfo(t("Done", "完成"), t("Pipeline finished. See log for details.", "转换流程已完成，详情见日志。"))
        except Exception as exc:  # noqa: broad-except
            self._log(f"ERROR: {exc}")
            messagebox.showerror(t("Error", "错误"), str(exc))

    def _run_ptq_pipeline(self):
        try:
            self._convert_ptq_int8()
            messagebox.showinfo(t("Done", "完成"), t("INT8 PTQ pipeline finished. See log for details.", "INT8后量化流程已完成，详情见日志。"))
        except Exception as exc:  # noqa: broad-except
            self._log(f"ERROR: {exc}")
            messagebox.showerror(t("Error", "错误"), str(exc))

    def _run_atc_precheck(self):
        try:
            ok, msg = self._atc_precheck()
            self._atc_precheck_status_var.set(msg)
            if ok:
                self._log(t("[ATC pre-check] PASS", "[ATC预检查] 通过"))
            else:
                self._log(t(f"[ATC pre-check] FAIL: {msg}", f"[ATC预检查] 失败：{msg}"))
        except Exception as exc:  # noqa: broad-except
            self._atc_precheck_status_var.set(t(f"ATC pre-check error: {exc}", f"ATC预检查出错：{exc}"))
            self._log(t(f"[ATC pre-check] ERROR: {exc}", f"[ATC预检查] 出错：{exc}"))

    def _convert_and_validate(self):
        model_path = Path(self.model_path.get())
        onnx_path = Path(self.onnx_path.get())
        om_path = Path(self.om_path.get())
        if not model_path.exists():
            raise FileNotFoundError(t("PTH/PT file does not exist.", "PTH/PT 文件不存在。"))
        device = torch.device(self.device_var.get() if torch.cuda.is_available() else "cpu")
        manual_classes = int(self.manual_classes.get()) if self.manual_classes.get() else None
        try:
            batch = int(self.batch_var.get())
        except Exception:
            batch = 1
        try:
            in_ch = int(self.in_channels_var.get())
        except Exception:
            in_ch = 4
        try:
            img_size = int(self.img_size_var.get())
        except Exception:
            img_size = 384
        try:
            opset = int(self.opset_var.get())
        except Exception:
            opset = 17
        soc = self.soc_var.get().strip()

        self._log(t("=== 1. Load checkpoint ===", "=== 1. 加载权重 ==="))
        model, num_classes = load_pth_model(
            model_path,
            device,
            manual_classes,
            template=self.template_var.get(),
            pretrained_template=self.template_pretrained_var.get(),
            log_fn=self._log,
        )
        self._log(f"Model classes: {num_classes}")

        dummy = torch.randn(batch, in_ch, img_size, img_size, device=device)
        input_np = dummy.detach().cpu().numpy()
        torch_output = run_torch(model, dummy)

        self._log(t("=== 2. Export ONNX ===", "=== 2. 导出 ONNX ==="))
        # Auto downgrade opset: try 17->16->15->14->13 to improve compatibility
        candidate_opsets = [opset] + [v for v in (17, 16, 15, 14, 13) if v != opset]
        last_err = None
        for try_opset in candidate_opsets:
            try:
                export_onnx(model, dummy, onnx_path, self.dynamic_axes.get(), try_opset, self._log)
                if self.validate_onnx.get():
                    self._log(t("=== 3. Validate PyTorch vs ONNX ===", "=== 3. 校验 PyTorch vs ONNX ==="))
                    onnx_output = run_onnx_runtime(onnx_path, dummy, device.type == "cuda", self._log)
                    cmp_result = compare_arrays(torch_output, onnx_output)
                    self._log(f"Compare: {json.dumps(cmp_result, ensure_ascii=False)} (opset={try_opset})")
                    if cmp_result["close"]:
                        self._log(f"ONNX validation passed with opset={try_opset}")
                        break
                    last_err = RuntimeError("ONNX output mismatch")
                else:
                    self._log(f"Skip ONNX validation (opset={try_opset})")
                    break
            except Exception as exc:  # noqa: broad-except
                last_err = exc
                self._log(f"ONNX export/validate failed with opset={try_opset}: {exc}")
        else:
            raise RuntimeError(f"ONNX export/validation failed. Last error: {last_err}")

        # 可选：ATC 预检查（仅 Linux + atc 可用时）
        if self.atc_precheck_var.get():
            ok, msg = self._atc_precheck(onnx_override=onnx_path, batch=batch, in_ch=in_ch, img_size=img_size, soc=soc)
            self._atc_precheck_status_var.set(msg)
            if not ok:
                raise RuntimeError(msg)

        self._log(t("=== 4. ATC -> OM ===", "=== 4. ATC 转 OM ==="))
        if not self._is_linux:
            self._log(t("Non-Linux host: skip ATC/OM. Copy ONNX to Ascend Linux device and run ATC there.",
                        "非 Linux 主机：跳过 ATC/OM。请把 ONNX 拷贝到昇腾 Linux 设备上再执行 ATC。"))
            return

        atc_cmd, real_om = build_atc_command(
            onnx_path,
            om_path,
            batch,
            in_ch,
            img_size,
            soc,
            output_type=self.om_output_type_var.get(),
            enable_mix_precision=self.enable_mix_precision_var.get(),
            extra_atc_args=self.atc_extra_args_var.get(),
        )
        self._log(" ".join(atc_cmd))
        proc = subprocess.run(atc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._log(proc.stdout)
        if proc.returncode != 0:
            self._log(proc.stderr)
            failed_ops = parse_atc_log(proc.stderr)
            if failed_ops:
                self._log(t("ATC failed operators:", "ATC 失败算子："))
                for line in failed_ops:
                    self._log(f"  - {line}")
            raise RuntimeError(t("ATC conversion failed. Please check logs above.", "ATC 转换失败，请查看上方日志。"))
        self._log(f"OM file generated: {real_om}")

        if self.validate_om.get():
            self._log(t("=== 5. OM validation (ACL) ===", "=== 5. OM 校验（ACL） ==="))
            runner = AclOmRunner(real_om, device_id=int(self.device_id_var.get()), log_fn=self._log)
            try:
                om_output = runner.infer(input_np, torch_output.shape)
                cmp_result = compare_arrays(torch_output, om_output)
                self._log(f"OM compare: {json.dumps(cmp_result, ensure_ascii=False)}")
                if not cmp_result["close"]:
                    self._log("WARN: OM output differs from PyTorch; check op mapping/precision.")
            finally:
                runner.release()

    def _get_ptq_cmd_template(self) -> str:
        try:
            return self.ptq_cmd_text.get("1.0", "end").strip()
        except Exception:
            return ""

    def _build_easy_ptq_cmd(self, onnx_in: Path, onnx_out: Path, calib_dir: Optional[Path]) -> str:
        tool = (self.ptq_quant_tool_var.get() or "amct_onnx").strip()
        if tool.startswith("(") or tool.startswith("（") or tool.endswith("tool)"):
            raise RuntimeError(t("No quantization tool detected. Use Advanced mode or install AMCT.", "未检测到量化工具。请使用“高级模式”或安装 AMCT。"))
        if shutil.which(tool) is None and not Path(tool).exists():
            raise RuntimeError(t(f"Quantization tool not found: {tool}", f"未找到量化工具：{tool}"))
        extra = (self.ptq_quant_extra_args_var.get() or "").strip()
        parts = [
            tool,
            "--model",
            _shell_quote(str(onnx_in)),
            "--output",
            _shell_quote(str(onnx_out)),
        ]
        if self.ptq_use_calib_var.get() and calib_dir is not None:
            parts.extend(["--calibration_data", _shell_quote(str(calib_dir))])
        if extra:
            parts.extend(_split_cmdline(extra))
        # Build a shell command string
        return " ".join(str(p) for p in parts).strip()

    def _refresh_ptq_ui_state(self):
        # advanced/easy frames visibility
        mode = (self.ptq_mode_var.get() or "easy").strip().lower()
        show_adv = mode == "advanced"
        try:
            if show_adv:
                self._ptq_easy_frame.grid_remove()
                self._ptq_adv_frame.grid()
            else:
                self._ptq_adv_frame.grid_remove()
                self._ptq_easy_frame.grid()
        except Exception:
            pass

        # calibration widgets enable/disable
        use_calib = bool(self.ptq_use_calib_var.get())
        state = "normal" if use_calib else "disabled"
        try:
            self._ptq_calib_entry.configure(state=state)
            self._ptq_calib_btn.configure(state=state)
        except Exception:
            pass

        # source widgets enable/disable
        source = (self.ptq_source_var.get() or "from_pth").strip().lower()
        pth_state = "normal" if source == "from_pth" else "disabled"
        try:
            self._ptq_pth_entry.configure(state=pth_state)
            self._ptq_pth_btn.configure(state=pth_state)
        except Exception:
            pass

        # quantization flow widgets enable/disable
        flow = (self.ptq_quant_flow_var.get() or "run_tool").strip().lower()
        use_existing = flow == "use_existing"
        try:
            self._ptq_existing_quant_entry.configure(state=("normal" if use_existing else "disabled"))
            self._ptq_existing_quant_btn.configure(state=("normal" if use_existing else "disabled"))
        except Exception:
            pass
        # If using existing quantized ONNX, disable calibration + tool options to avoid confusion
        if use_existing:
            try:
                self._ptq_calib_entry.configure(state="disabled")
                self._ptq_calib_btn.configure(state="disabled")
            except Exception:
                pass

        # preview command in easy mode (only meaningful when we run quant tool)
        try:
            onnx_in = Path(self.ptq_onnx_in_path.get().strip()) if self.ptq_onnx_in_path.get().strip() else None
            onnx_out = Path(self.ptq_quant_onnx_path.get().strip()) if self.ptq_quant_onnx_path.get().strip() else None
            calib_dir = Path(self.ptq_calib_dir_var.get().strip()) if self.ptq_calib_dir_var.get().strip() else None
            preview = ""
            flow = (self.ptq_quant_flow_var.get() or "run_tool").strip().lower()
            if flow != "use_existing" and mode != "advanced" and onnx_in and onnx_out:
                preview = self._build_easy_ptq_cmd(onnx_in, onnx_out, calib_dir if use_calib else None)
            self._ptq_cmd_preview.configure(state="normal")
            self._ptq_cmd_preview.delete("1.0", "end")
            self._ptq_cmd_preview.insert("1.0", preview)
            self._ptq_cmd_preview.configure(state="disabled")
        except Exception:
            pass

    def _render_ptq_cmd(self, template: str, onnx_in: Path, onnx_out: Path, calib_dir: Path) -> str:
        mapping = {
            "onnx_in": str(onnx_in),
            "onnx_out": str(onnx_out),
            "calib_dir": str(calib_dir),
            "onnx_in_q": _shell_quote(str(onnx_in)),
            "onnx_out_q": _shell_quote(str(onnx_out)),
            "calib_dir_q": _shell_quote(str(calib_dir)),
        }
        # very small templating: {key}
        cmd = template
        for k, v in mapping.items():
            cmd = cmd.replace("{" + k + "}", v)
        return cmd.strip()

    def _atc_precheck(
        self,
        onnx_override: Optional[Path] = None,
        batch: Optional[int] = None,
        in_ch: Optional[int] = None,
        img_size: Optional[int] = None,
        soc: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        使用 ATC 的 mode=3 做预检查：尽量在真正编译前判断“能否按当前设置编译”。
        说明：预检查依赖 Linux + atc；Windows 会直接跳过。
        """
        if not self._is_linux:
            return True, t("ATC pre-check: skipped (non-Linux)", "ATC预检查：已跳过（非Linux）")
        if shutil.which("atc") is None:
            return False, t("ATC pre-check: atc not found in PATH", "ATC预检查：未找到 atc（PATH中无）")

        try:
            b = batch if batch is not None else int(self.batch_var.get())
        except Exception:
            b = 1
        try:
            c = in_ch if in_ch is not None else int(self.in_channels_var.get())
        except Exception:
            c = 4
        try:
            s = img_size if img_size is not None else int(self.img_size_var.get())
        except Exception:
            s = 384
        soc_v = (soc if soc is not None else self.soc_var.get()).strip()

        onnx_path = onnx_override if onnx_override is not None else Path(self.onnx_path.get())
        if not onnx_path.exists():
            return False, t("ATC pre-check: ONNX not found", "ATC预检查：找不到 ONNX 文件")

        # atc mode=3 只预检查：需要 model/framework/input_shape/soc_version 等参数。
        # 为了最大程度贴近真实编译参数，我们复用 build_atc_command 的关键入参。
        temp_output = (PROJECT_ROOT / "_atc_precheck_tmp.om")
        atc_cmd, _ = build_atc_command(
            onnx_path,
            temp_output,
            b,
            c,
            s,
            soc_v,
            output_type=self.om_output_type_var.get(),
            enable_mix_precision=self.enable_mix_precision_var.get(),
            extra_atc_args=self.atc_extra_args_var.get(),
        )
        # Insert mode=3 as early arg
        atc_cmd.insert(1, "--mode=3")

        # Use a deterministic report path
        report = PROJECT_ROOT / "check_result.json"
        atc_cmd.append(f"--check_report={report}")

        self._log(t("[ATC pre-check] running: ", "[ATC预检查] 执行：") + " ".join(atc_cmd))
        proc = subprocess.run(atc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.stdout:
            self._log(proc.stdout)
        if proc.returncode == 0:
            return True, t("ATC pre-check: PASS", "ATC预检查：通过")
        # Best-effort error message
        err = (proc.stderr or "").strip()
        if err:
            self._log(err)
            return False, t("ATC pre-check: FAIL (see log)", "ATC预检查：失败（见日志）")
        return False, t("ATC pre-check: FAIL", "ATC预检查：失败")

    def _convert_ptq_int8(self):
        source = (self.ptq_source_var.get() or "from_pth").strip().lower()
        flow = (self.ptq_quant_flow_var.get() or "run_tool").strip().lower()
        model_path = Path(self.model_path.get()) if self.model_path.get().strip() else None
        onnx_path = Path(self.ptq_onnx_in_path.get()) if self.ptq_onnx_in_path.get().strip() else None
        quant_onnx_path = Path(self.ptq_quant_onnx_path.get()) if self.ptq_quant_onnx_path.get().strip() else None
        om_path = Path(self.ptq_om_path.get())
        calib_dir = Path(self.ptq_calib_dir_var.get()) if self.ptq_calib_dir_var.get().strip() else None
        use_calib = bool(self.ptq_use_calib_var.get())

        # Common shape params
        try:
            batch = int(self.batch_var.get())
        except Exception:
            batch = 1
        try:
            in_ch = int(self.in_channels_var.get())
        except Exception:
            in_ch = 4
        try:
            img_size = int(self.img_size_var.get())
        except Exception:
            img_size = 384
        try:
            opset = int(self.opset_var.get())
        except Exception:
            opset = 17
        soc = self.soc_var.get().strip()

        torch_output = None
        device = torch.device("cpu")
        dummy = torch.randn(batch, in_ch, img_size, img_size, device=device)

        if source == "from_pth":
            if model_path is None or not model_path.exists():
                raise FileNotFoundError(t("PTH/PT file does not exist.", "PTH/PT 文件不存在。"))
            if onnx_path is None:
                raise FileNotFoundError(t("FP32 ONNX path is empty.", "FP32 ONNX 路径为空。"))

            device = torch.device(self.device_var.get() if torch.cuda.is_available() else "cpu")
            manual_classes = int(self.manual_classes.get()) if self.manual_classes.get() else None
            self._log(t("=== [PTQ] 1. Load checkpoint ===", "=== [PTQ] 1. 加载权重 ==="))
            model, num_classes = load_pth_model(
                model_path,
                device,
                manual_classes,
                template=self.template_var.get(),
                pretrained_template=self.template_pretrained_var.get(),
                log_fn=self._log,
            )
            self._log(f"Model classes: {num_classes}")

            dummy = torch.randn(batch, in_ch, img_size, img_size, device=device)
            torch_output = run_torch(model, dummy)

            self._log(t("=== [PTQ] 2. Export FP32 ONNX ===", "=== [PTQ] 2. 导出 FP32 ONNX ==="))
            candidate_opsets = [opset] + [v for v in (17, 16, 15, 14, 13) if v != opset]
            last_err = None
            for try_opset in candidate_opsets:
                try:
                    export_onnx(model, dummy, onnx_path, self.dynamic_axes.get(), try_opset, self._log)
                    if self.validate_onnx.get():
                        self._log(
                            t(
                                "=== [PTQ] 3. Validate PyTorch vs ONNX (FP32) ===",
                                "=== [PTQ] 3. 校验 PyTorch vs ONNX（FP32） ===",
                            )
                        )
                        onnx_output = run_onnx_runtime(onnx_path, dummy, device.type == "cuda", self._log)
                        cmp_result = compare_arrays(torch_output, onnx_output)
                        self._log(f"Compare: {json.dumps(cmp_result, ensure_ascii=False)} (opset={try_opset})")
                        if cmp_result["close"]:
                            self._log(f"ONNX validation passed with opset={try_opset}")
                            break
                        last_err = RuntimeError("ONNX output mismatch")
                    else:
                        self._log(f"Skip ONNX validation (opset={try_opset})")
                        break
                except Exception as exc:  # noqa: broad-except
                    last_err = exc
                    self._log(f"ONNX export/validate failed with opset={try_opset}: {exc}")
            else:
                raise RuntimeError(f"ONNX export/validation failed. Last error: {last_err}")
        else:
            # from_onnx
            if onnx_path is None or not onnx_path.exists():
                raise FileNotFoundError(t("FP32 ONNX file does not exist.", "FP32 ONNX 文件不存在。"))
            self._log(t("=== [PTQ] Using existing ONNX ===", "=== [PTQ] 使用已有ONNX ==="))

        if use_calib and flow != "use_existing":
            if calib_dir is None or not calib_dir.exists():
                raise FileNotFoundError(t("Calibration data directory does not exist.", "校准数据目录不存在。"))

        # Decide how to obtain quantized ONNX:
        # - run_tool: run quantization CLI (easy/advanced) to produce quant_onnx_path
        # - use_existing: user provides an already-quantized ONNX and we skip quantization
        if flow == "use_existing":
            qin = Path(self.ptq_existing_quant_onnx_var.get()) if self.ptq_existing_quant_onnx_var.get().strip() else None
            if qin is None or not qin.exists():
                raise FileNotFoundError(t("Quantized ONNX input does not exist.", "量化ONNX输入文件不存在。"))
            quant_onnx_path = qin
            self._log(t("=== [PTQ] 4. Skip quantization (use existing quantized ONNX) ===", "=== [PTQ] 4. 跳过量化（使用已有量化ONNX） ==="))
        else:
            if quant_onnx_path is None:
                raise FileNotFoundError(t("Quantized ONNX output path is empty.", "量化ONNX输出路径为空。"))

            self._log(t("=== [PTQ] 4. Run INT8 PTQ quantizer (external) ===", "=== [PTQ] 4. 执行 INT8 PTQ 量化工具（外部） ==="))
            mode = (self.ptq_mode_var.get() or "easy").strip().lower()
            if mode == "advanced":
                tpl = self._get_ptq_cmd_template()
                if not tpl:
                    raise RuntimeError(t("PTQ command template is empty.", "量化命令模板为空。"))

                # Optional: quick check for first token existence
                first_tokens = _split_cmdline(tpl)
                if first_tokens:
                    exe = first_tokens[0]
                    if shutil.which(exe) is None and not (Path(exe).exists()):
                        self._log(t(f"WARN: command not found in PATH: {exe}", f"警告：命令不在 PATH 中：{exe}"))

                if not use_calib:
                    self._log(
                        t(
                            "WARN: calibration is disabled; please remove calibration args in custom command if needed.",
                            "警告：已关闭校准集；如量化工具不支持无校准，请在自定义命令中移除校准参数。",
                        )
                    )
                cmd = self._render_ptq_cmd(tpl, onnx_path, quant_onnx_path, calib_dir or Path(""))
            else:
                cmd = self._build_easy_ptq_cmd(onnx_path, quant_onnx_path, calib_dir if use_calib else None)

            self._log(cmd)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
            if proc.stdout:
                self._log(proc.stdout)
            if proc.returncode != 0:
                if proc.stderr:
                    self._log(proc.stderr)
                raise RuntimeError(t("PTQ quantization command failed.", "PTQ量化命令执行失败。"))
            if not quant_onnx_path.exists():
                raise FileNotFoundError(t("Quantized ONNX not found.", "量化ONNX未生成。"))

        self._log(t("=== [PTQ] 5. ATC -> OM ===", "=== [PTQ] 5. ATC 转 OM ==="))
        if not self._is_linux:
            self._log(t("Non-Linux host: skip ATC/OM. Copy ONNX to Ascend Linux device and run PTQ+ATC there.",
                        "非 Linux 主机：跳过 ATC/OM。请把 ONNX 拷贝到昇腾 Linux 设备上执行 PTQ+ATC。"))
            return

        atc_cmd, real_om = build_atc_command(
            quant_onnx_path,
            om_path,
            batch,
            in_ch,
            img_size,
            soc,
            output_type=self.ptq_om_output_type_var.get(),
            extra_atc_args=self.ptq_atc_extra_args_var.get(),
        )
        self._log(" ".join(atc_cmd))
        proc = subprocess.run(atc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._log(proc.stdout)
        if proc.returncode != 0:
            self._log(proc.stderr)
            failed_ops = parse_atc_log(proc.stderr)
            if failed_ops:
                self._log(t("ATC failed operators:", "ATC 失败算子："))
                for line in failed_ops:
                    self._log(f"  - {line}")
            raise RuntimeError(t("ATC conversion failed. Please check logs above.", "ATC 转换失败，请查看上方日志。"))
        self._log(f"INT8 OM file generated: {real_om}")

        if self.validate_om.get():
            self._log(
                t(
                    "=== [PTQ] 6. OM validation (ACL, tolerance may be larger for INT8) ===",
                    "=== [PTQ] 6. OM 校验（ACL，INT8 误差通常更大） ===",
                )
            )
            input_np = dummy.detach().cpu().numpy()
            runner = AclOmRunner(real_om, device_id=int(self.device_id_var.get()), log_fn=self._log)
            try:
                ref = None
                output_shape = None
                if torch_output is not None:
                    ref = torch_output
                    output_shape = torch_output.shape
                else:
                    # ONNX-only mode: use onnxruntime output as shape reference if possible
                    if ort is not None and onnx_path is not None:
                        try:
                            ref = run_onnx_runtime(onnx_path, dummy, False, self._log)
                            output_shape = ref.shape
                            self._log(t("Use ONNXRuntime output as reference (first output).", "使用 ONNXRuntime 输出作为参考（仅第一个输出）。"))
                        except Exception as exc:  # noqa: broad-except
                            self._log(t(f"WARN: ONNXRuntime reference failed: {exc}", f"警告：ONNXRuntime 参考输出失败：{exc}"))
                    if output_shape is None:
                        self._log(
                            t(
                                "Skip OM compare: no reference shape available (disable Validate OM or provide PTH).",
                                "跳过OM对比：无法获取参考输出形状（请关闭校验OM或提供PTH）。",
                            )
                        )
                        return

                om_output = runner.infer(input_np, output_shape)
                if ref is not None:
                    cmp_result = compare_arrays(ref, om_output)
                    self._log(f"OM compare: {json.dumps(cmp_result, ensure_ascii=False)}")
                    if not cmp_result["close"]:
                        self._log(
                            t(
                                "WARN: OM output differs from reference (PTQ/INT8 often has larger error). Validate with task metrics.",
                                "警告：OM 输出与参考不一致（PTQ/INT8 误差通常更大）。建议用任务指标验收。",
                            )
                        )
            finally:
                runner.release()


def main():
    app = ConversionApp()
    app.mainloop()


if __name__ == "__main__":
    main()

