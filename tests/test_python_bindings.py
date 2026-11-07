"""
Functional & integration test suite for qwen_engine.
"""
import os
import sys
import pytest
from pathlib import Path

# Ensure package is imported
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

import qwen_engine
from qwen_engine import (
    Engine,
    ConcurrencyError,
    ContextWindowExceededError,
    ModelNotFoundError,
    download,
    get_default_cache_dir,
)
from qwen_engine._core import NativeTokenizer, NativeEngine

MODEL_PATH = "models/qwen25_1.5b_w4a16.qwenbin"
TOKENIZER_PATH = "models/tokenizer.bin"

HAS_MODEL = os.path.isfile(MODEL_PATH) and os.path.isfile(TOKENIZER_PATH)


def test_tokenizer_encode_decode():
    """Verify tokenizer encodes and decodes ASCII and Unicode text."""
    if not os.path.isfile(TOKENIZER_PATH):
        pytest.skip("tokenizer.bin not found")

    tok = NativeTokenizer(TOKENIZER_PATH)
    text = "Hello, world! 🚀 Quantum computing 你好世界"
    tokens = tok.encode(text)
    assert len(tokens) > 0

    decoded = tok.decode(tokens)
    assert decoded == text


def test_tokenizer_chatml_formatting():
    """Verify ChatML deterministic formatting."""
    if not os.path.isfile(TOKENIZER_PATH):
        pytest.skip("tokenizer.bin not found")

    tok = NativeTokenizer(TOKENIZER_PATH)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi!"},
    ]
    tokens = tok.format_chatml(messages)
    assert len(tokens) > 0
    decoded = tok.decode(tokens)
    assert "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" in decoded
    assert "<|im_start|>user\nHi!<|im_end|>\n" in decoded


def test_tokenizer_multibyte_utf8_streaming():
    """Verify UTF-8 code points split across BPE boundaries reconstruct cleanly."""
    if not os.path.isfile(TOKENIZER_PATH):
        pytest.skip("tokenizer.bin not found")

    tok = NativeTokenizer(TOKENIZER_PATH)
    complex_text = "你好世界！🚀✨ Anthropic & OpenAI Quantum Computing."
    tokens = tok.encode(complex_text)

    # Decode token-by-token with carry buffer
    reconstructed = ""
    carry = ""
    for t in tokens:
        piece, carry = tok.decode_streaming(t, carry)
        reconstructed += piece

    assert "\ufffd" not in reconstructed
    assert reconstructed == complex_text


@pytest.mark.skipif(not HAS_MODEL, reason="Model weights not found")
def test_engine_init_and_stats():
    """Verify Engine initializes and exposes accurate memory stats."""
    eng = Engine(MODEL_PATH, TOKENIZER_PATH)
    assert eng.device_id == 0
    stats = eng.get_memory_stats()
    assert stats.weights_bytes > 700 * 1024 * 1024
    assert stats.kv_cache_bytes == 112 * 1024 * 1024
    assert stats.total_allocated_bytes > 800 * 1024 * 1024


@pytest.mark.skipif(not HAS_MODEL, reason="Model weights not found")
def test_deterministic_greedy_generation():
    """Verify greedy generation produces deterministic output."""
    eng = Engine(MODEL_PATH, TOKENIZER_PATH)
    eng.reset_session()
    out1 = eng.generate("The capital of France is", max_tokens=10, temperature=0.0)
    eng.reset_session()
    out2 = eng.generate("The capital of France is", max_tokens=10, temperature=0.0)
    assert len(out1) > 0
    assert out1 == out2
    assert "Paris" in out1


@pytest.mark.skipif(not HAS_MODEL, reason="Model weights not found")
def test_stream_rollback_on_interruption():
    """Verify stream interruption triggers rollback to prompt length."""
    eng = Engine(MODEL_PATH, TOKENIZER_PATH)
    eng.reset_session()

    prompt = "Write an essay about the history of mathematics."
    prompt_tokens = eng.encode(prompt)
    prompt_len = len(prompt_tokens)

    # Generate 5 tokens and break early
    count = 0
    for piece in eng.stream(prompt, max_tokens=50, rollback_on_interrupt=True):
        count += 1
        if count == 5:
            break

    # Prompt boundary should be preserved via O(1) rollback
    assert eng.current_seq_pos == prompt_len
    assert not eng.is_generating


@pytest.mark.skipif(not HAS_MODEL, reason="Model weights not found")
def test_intentional_early_stop_commit():
    """Verify rollback_on_interrupt=False preserves generated tokens in KV cache."""
    eng = Engine(MODEL_PATH, TOKENIZER_PATH)
    eng.reset_session()

    prompt = "1, 2, 3, 4, 5,"
    prompt_len = len(eng.encode(prompt))

    count = 0
    for piece in eng.stream(prompt, max_tokens=50, rollback_on_interrupt=False):
        count += 1
        if count == 5:
            break

    # Sequence position should have advanced by (count - 1) decode steps and remained committed
    assert eng.current_seq_pos == prompt_len + (count - 1)
    assert not eng.is_generating


@pytest.mark.skipif(not HAS_MODEL, reason="Model weights not found")
def test_concurrency_lease_blocking():
    """Verify concurrent generation on an active instance raises ConcurrencyError."""
    eng = Engine(MODEL_PATH, TOKENIZER_PATH)
    eng.reset_session()

    stream = eng.stream("Tell me a story.", max_tokens=20)
    # Start the stream by pulling first item
    first_item = next(stream)
    assert len(first_item) > 0

    # Engine is now actively locked under session lease
    assert eng.is_generating
    with pytest.raises(ConcurrencyError):
        eng.generate("Another query concurrently")

    # Clean up stream
    stream.close()
    assert not eng.is_generating


def test_downloader_local_resolution():
    """Verify downloader resolves local model paths without network access."""
    if not HAS_MODEL:
        pytest.skip("Local model not present")

    m_path, t_path = download("models/qwen25_1.5b_w4a16.qwenbin")
    assert os.path.isfile(m_path)
    assert os.path.isfile(t_path)
