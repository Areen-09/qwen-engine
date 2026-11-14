"""
Head-to-head benchmark: qwen-engine (W4A16 G128) vs. llama.cpp (Q4_K_M).
Runnable anywhere (Google Colab, Kaggle, Windows, Linux).
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RELEASE_TAG = "v0.2.1"
WHEEL_VER = "0.2.0"
GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def ensure_packages() -> None:
    """Install qwen-engine from GitHub Release v0.2.1 and benchmark dependencies."""
    try:
        import qwen_engine  # noqa: F401
    except ImportError:
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

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        print("Installing llama-cpp-python (CUDA) ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python"],
            env=env,
        )


def load_qwen_engine():
    from qwen_engine import Engine

    if Path("models/qwen25_1.5b_w4a16.qwenbin").is_file():
        return Engine("models/qwen25_1.5b_w4a16.qwenbin", "models/tokenizer.bin", device_id=0)
    return Engine.from_pretrained("qwen2.5-1.5b-w4a16", device_id=0)


def resolve_gguf_path() -> str:
    local = Path("models") / GGUF_FILE
    if local.is_file():
        return str(local)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)


def bench_qwen_engine(engine, prompt_tokens: int = 128, gen_tokens: int = 128, runs: int = 5):
    # Build exact-length synthetic prompt
    seed_ids = engine.encode("The quick brown fox jumps over the lazy dog. ")
    prompt_ids = (seed_ids * ((prompt_tokens // len(seed_ids)) + 1))[:prompt_tokens]
    prompt_text = engine.decode(prompt_ids)

    # Warmup
    engine.reset_session()
    for _ in engine.stream(prompt_text, max_tokens=16, temperature=0.0):
        pass

    prefill_tps, decode_tps = [], []
    for _ in range(runs):
        engine.reset_session()
        t0 = time.perf_counter()
        it = engine.stream(prompt_text, max_tokens=gen_tokens, temperature=0.0)
        next(it)
        t_first = time.perf_counter()
        count = 1
        for _ in it:
            count += 1
        t_end = time.perf_counter()

        prefill_tps.append(len(engine.encode(prompt_text)) / (t_first - t0))
        if count > 1:
            decode_tps.append((count - 1) / (t_end - t_first))

    return sum(prefill_tps) / len(prefill_tps), sum(decode_tps) / len(decode_tps)


def bench_llamacpp(gguf_path: str, prompt_tokens: int = 128, gen_tokens: int = 128, runs: int = 5):
    from llama_cpp import Llama

    llm = Llama(model_path=gguf_path, n_gpu_layers=-1, n_ctx=2048, flash_attn=True, verbose=False)
    seed = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = llm.tokenize(seed.encode("utf-8"))[:prompt_tokens]
    prompt = llm.detokenize(tokens).decode("utf-8", errors="ignore")

    # Warmup
    llm(prompt, max_tokens=16, temperature=0.0)

    prefill_tps, decode_tps = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        stream = llm(prompt, max_tokens=gen_tokens, temperature=0.0, stream=True)
        next(stream)
        t_first = time.perf_counter()
        count = 1
        for _ in stream:
            count += 1
        t_end = time.perf_counter()

        prefill_tps.append(len(tokens) / (t_first - t0))
        if count > 1:
            decode_tps.append((count - 1) / (t_end - t_first))

    return sum(prefill_tps) / len(prefill_tps), sum(decode_tps) / len(decode_tps)


def main() -> None:
    ensure_packages()

    print("Loading qwen-engine (W4A16 G128)...")
    qe = load_qwen_engine()
    qe_pre, qe_dec = bench_qwen_engine(qe, prompt_tokens=128, gen_tokens=128, runs=5)
    del qe

    gguf_path = resolve_gguf_path()
    print(f"Loading llama.cpp ({Path(gguf_path).name})...")
    lc_pre, lc_dec = bench_llamacpp(gguf_path, prompt_tokens=128, gen_tokens=128, runs=5)

    print("\n============================================================")
    print("  Qwen 2.5-1.5B Benchmark (p=128, n=128, CUDA)")
    print("============================================================")
    print(f"{'Engine':<28} {'Prefill (tok/s)':>15} {'Decode (tok/s)':>15}")
    print("-" * 60)
    print(f"{'qwen-engine (W4A16 G128)':<28} {qe_pre:>15.2f} {qe_dec:>15.2f}")
    print(f"{'llama.cpp (Q4_K_M)':<28} {lc_pre:>15.2f} {lc_dec:>15.2f}")
    print("-" * 60)
    print(f"Speedup -> Prefill: {qe_pre / lc_pre:.2f}x | Decode: {qe_dec / lc_dec:.2f}x")


if __name__ == "__main__":
    main()
