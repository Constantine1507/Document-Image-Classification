"""Verify native-only BF16 selection without requiring a CUDA device in CI."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from docvision_v2.runtime import native_bf16_supported, probe_cuda, selected_amp_dtype
from docvision_v2.utils import write_json


def main():
    """Check CPU, T4-like native BF16 rejection, and supported GPU selection."""
    assert selected_amp_dtype(torch.device("cpu")) is None
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=False) as check:
            assert selected_amp_dtype("cuda") == torch.float16
            check.assert_called_once_with(including_emulation=False)
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            assert selected_amp_dtype("cuda") == torch.bfloat16
        with patch("torch.cuda.is_bf16_supported", side_effect=TypeError):
            with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
                assert not native_bf16_supported()
    assert selected_amp_dtype("cuda", enabled=False) is None
    report = {"status": "passed", "policy_checks": "CPU and mocked CUDA capability",
              "actual_kernel_probe": probe_cuda()}
    if len(sys.argv) > 1:
        write_json(sys.argv[1], report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
