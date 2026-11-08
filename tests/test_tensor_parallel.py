"""
Cross-Platform 2-Way Tensor Parallelism Test Suite.
Verifies multi-GPU communication and tensor parallel execution.
Gated with @pytest.mark.skipif(_core.cuda_device_count() < 2) for single-GPU environments.
"""

import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_engine import Engine, _core

def test_cuda_device_count_query():
    count = _core.cuda_device_count()
    assert count >= 1, f"Expected at least 1 CUDA device, found {count}"

def test_single_device_communicator_execution():
    model_path = "models/qwen25_1.5b_w4a16.qwenbin"
    if not os.path.exists(model_path):
        pytest.skip(f"Model {model_path} not found")

    # Single-device initialization via device_ids list
    eng = _core.NativeEngine(model_path, [0])
    assert eng.device_id() == 0

    prompt = [151644, 872, 198, 10838, 151645]
    tok = eng.prefill(prompt, 0.0)
    assert tok > 0

    next_tok = eng.decode_step()
    assert next_tok > 0

@pytest.mark.skipif(_core.cuda_device_count() < 2, reason="Requires >= 2 CUDA devices for 2-way Tensor Parallelism")
def test_dual_gpu_tensor_parallel_execution():
    model_path = "models/qwen25_1.5b_w4a16.qwenbin"
    if not os.path.exists(model_path):
        pytest.skip(f"Model {model_path} not found")

    eng_tp = _core.NativeEngine(model_path, [0, 1])
    assert eng_tp.device_id() == 0

    prompt = [151644, 872, 198, 10838, 151645]
    tok = eng_tp.prefill(prompt, 0.0)
    assert tok > 0

    for _ in range(8):
        step_tok = eng_tp.decode_step()
        assert step_tok > 0

def test_invalid_device_id_validation():
    model_path = "models/qwen25_1.5b_w4a16.qwenbin"
    if not os.path.exists(model_path):
        pytest.skip(f"Model {model_path} not found")

    with pytest.raises(ValueError):
        Engine(model_path=model_path, device_ids=[9999])

