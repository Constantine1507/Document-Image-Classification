"""CUDA diagnostics and native-only AMP selection for document backbones."""
import platform

import torch
import torchvision


def native_bf16_supported():
    """Reject emulated BF16; use the device capability on older CUDA APIs."""
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return bool(torch.version.cuda and torch.cuda.get_device_capability()[0] >= 8)


def selected_amp_dtype(device, enabled=True):
    """Return the shared autocast/scaler policy, or None for full precision."""
    if torch.device(device).type != "cuda" or not enabled:
        return None
    return torch.bfloat16 if native_bf16_supported() else torch.float16


def runtime_diagnostics():
    """Describe the active runtime without claiming a full training test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = selected_amp_dtype(device)
    return {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "capability": list(torch.cuda.get_device_capability()) if device.type == "cuda" else None,
        "native_bf16": native_bf16_supported(), "amp_dtype": str(dtype),
        "gradient_scaler": dtype == torch.float16,
        "memory_format": "contiguous NCHW", "cudnn_benchmark": False,
    }


def probe_cuda():
    """Test CUDA stem/depthwise forward and backward before expensive training."""
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA unavailable; GPU kernels were not tested"}
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    dtype = selected_amp_dtype(device)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 96, 4, stride=4),
        torch.nn.Conv2d(96, 96, 7, padding=3, groups=96),
        torch.nn.GELU(), torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(), torch.nn.Linear(96, 16),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    try:
        with torch.autocast("cuda", dtype=dtype):
            loss = torch.nn.functional.cross_entropy(
                model(torch.randn(2, 3, 64, 64, device=device)),
                torch.tensor([0, 1], device=device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite CUDA probe loss")
        return {"status": "passed", "amp_dtype": str(dtype), "scope": "small convolution probe"}
    except RuntimeError as exc:
        raise RuntimeError(
            "CUDA convolution probe failed. Restart the Kaggle session and use its matched "
            "torch/torchvision installation. See docs/TROUBLESHOOTING.md."
        ) from exc
    finally:
        del model, optimizer, scaler
        torch.cuda.empty_cache()
