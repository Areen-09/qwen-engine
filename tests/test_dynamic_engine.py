import os
import sys
from pathlib import Path
import tempfile
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_engine import Engine, _core
from tests.test_gguf_converter import create_synthetic_gguf
from tools.convert_gguf_to_qwenbin import convert_gguf_to_qwenbin

def test_qwen25_dynamic_engine_parity():
    model_path = "models/qwen25_1.5b_w4a16.qwenbin"
    if not os.path.exists(model_path):
        pytest.skip(f"Model {model_path} not found")

    native_eng = _core.NativeEngine(model_path, 0)
    assert native_eng.num_gqa_layers() == 28
    assert native_eng.num_deltanet_layers() == 0

    stats = native_eng.get_memory_stats()
    assert stats.kv_cache_bytes > 0

    # Prefill and decode steps
    prompt = [151644, 872, 198, 10838, 151645, 198, 151644, 77091, 198]
    tok1 = native_eng.prefill(prompt, 0.0)
    assert tok1 > 0

    tok2 = native_eng.decode_step()
    tok3 = native_eng.decode_step()
    assert tok2 > 0
    assert tok3 > 0

    cur_pos = native_eng.current_seq_pos()
    assert cur_pos == len(prompt) + 2

    # Test rollback
    native_eng.rollback_to(len(prompt))
    assert native_eng.current_seq_pos() == len(prompt)

def test_qwen35_hybrid_deltanet_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_path = os.path.join(tmpdir, "hybrid_model.gguf")
        qwenbin_path = os.path.join(tmpdir, "hybrid_model.qwenbin")

        # 4 layers: 3 DeltaNet, 1 GQA (3:1 hybrid stacking ratio)
        create_synthetic_gguf(gguf_path, num_layers=4, hidden_size=256, intermediate_size=512)
        convert_gguf_to_qwenbin(gguf_path, qwenbin_path, mode="fp16")

        native_eng = _core.NativeEngine(qwenbin_path, 0)
        assert native_eng.num_gqa_layers() == 1
        assert native_eng.num_deltanet_layers() == 3

        stats = native_eng.get_memory_stats()
        # For 1 GQA layer: 1 layer * 2 kv_heads * 512 max_seq * 128 head_dim * 2 (half) * 2 (K and V) = 524,288 bytes = 0.5 MB
        # An equivalent 4-layer pure GQA model would allocate 4 * 0.5 MB = 2 MB (2,097,152 bytes).
        # Thus KV cache is exactly 25% of a 4-layer pure attention model!
        assert stats.kv_cache_bytes == 524288

        # Execute prefill
        prompt = [5, 10, 15, 20]
        tok1 = native_eng.prefill(prompt, 0.0)

        # Autoregressive decode steps
        tok2 = native_eng.decode_step()
        tok3 = native_eng.decode_step()

        assert native_eng.current_seq_pos() == len(prompt) + 2

        # Test rollback on hybrid DeltaNet model
        native_eng.rollback_to(len(prompt))
        assert native_eng.current_seq_pos() == len(prompt)

        # Generating again after rollback is deterministic
        tok_replay_2 = native_eng.decode_step()
        assert tok_replay_2 == tok2

        # Explicitly release engine before tempdir cleanup on Windows
        del native_eng
