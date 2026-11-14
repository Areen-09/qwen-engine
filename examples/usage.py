"""
Standalone Qwen 2.5-1.5B-Instruct (W4A16) usage example using qwen-engine.
Runnable anywhere (Google Colab, Kaggle, Windows, Linux).
"""
import subprocess
import sys
import time
from pathlib import Path

RELEASE_TAG = "v0.2.1"
WHEEL_VER = "0.2.0"


def install_qwen_engine_from_release() -> None:
    """Install the prebuilt qwen-engine wheel from GitHub Releases if missing."""
    try:
        import qwen_engine  # noqa: F401
        return
    except ImportError:
        pass

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    base = f"https://github.com/Areen-09/qwen-engine/releases/download/{RELEASE_TAG}"

    if sys.platform == "win32" and (3, 9) <= sys.version_info[:2] <= (3, 13):
        url = f"{base}/qwen_engine-{WHEEL_VER}-{py_tag}-{py_tag}-win_amd64.whl"
    elif sys.platform.startswith("linux") and (3, 9) <= sys.version_info[:2] <= (3, 13):
        url = f"{base}/qwen_engine-{WHEEL_VER}-{py_tag}-{py_tag}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    else:
        url = f"{base}/qwen_engine-{WHEEL_VER}.tar.gz"

    print(f"Installing qwen-engine from {url} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", url])


def load_engine():
    from qwen_engine import Engine

    local_model = Path("models/qwen25_1.5b_w4a16.qwenbin")
    local_tok = Path("models/tokenizer.bin")
    if local_model.is_file() and local_tok.is_file():
        return Engine(str(local_model), str(local_tok), device_id=0)

    # Auto-download from Hugging Face cache when running outside a local models/ dir
    return Engine.from_pretrained("qwen2.5-1.5b-w4a16", device_id=0)


def main() -> None:
    install_qwen_engine_from_release()
    import qwen_engine

    print(f"qwen_engine version: {qwen_engine.__version__}")
    engine = load_engine()

    stats = engine.get_memory_stats()
    print(
        f"VRAM -> Weights: {stats.weights_bytes / 1024**2:.2f} MB | "
        f"KV Cache: {stats.kv_cache_bytes / 1024**2:.2f} MB | "
        f"Workspace: {stats.workspace_bytes / 1024**2:.2f} MB | "
        f"Total: {stats.total_allocated_bytes / 1024**2:.2f} MB"
    )

    # 1. Streaming ChatML generation
    engine.reset_session()
    messages = [
        {"role": "system", "content": "You are a helpful, concise AI assistant."},
        {"role": "user", "content": "Explain why CUDA Graphs speed up LLM decoding in 3 bullet points."},
    ]

    print("\n--- Streaming Chat ---")
    t0 = time.perf_counter()
    chunks = 0
    for piece in engine.stream(messages, max_tokens=192, temperature=0.0):
        print(piece, end="", flush=True)
        chunks += 1
    elapsed = time.perf_counter() - t0
    print(f"\n[{chunks} tokens in {elapsed:.2f}s -> {chunks / elapsed:.1f} tok/s]")

    # 2. One-shot generation with rollback checkpointing
    engine.reset_session()
    checkpoint = engine.current_seq_pos
    reply = engine.generate("List 3 benefits of W4A16 quantization:", max_tokens=96, temperature=0.0)
    print("\n--- One-Shot Generation ---")
    print(reply)

    engine.rollback_to(checkpoint)
    print(f"Rolled back KV cache position to: {engine.current_seq_pos}")


if __name__ == "__main__":
    main()
