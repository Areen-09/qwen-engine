# Walkthrough: Using the `qwen-engine` Python Package

This guide provides a comprehensive walkthrough for using **`qwen-engine`**, the bare-metal, zero-dependency C++20/CUDA Python package for `Qwen2.5-1.5B`.

---

## 1. Key Architectural Features

* **Zero Framework Bloat:** No PyTorch, no libtorch, no Hugging Face transformers, no cuBLAS or CUTLASS at runtime. The precompiled package is standalone and only requires an NVIDIA display driver (`sm_75` to `sm_90`).
* **Ultra-Low VRAM Footprint:** Uses a single static GPU arena of **879.34 MB** (W4A16 quantized weights, 4096-token KV cache, and workspace buffers).
* **Blazing Inference Speed:** Runs at **>148 tokens/second** decoding and **<38 ms TTFT** on laptop GPUs (e.g., RTX 3050).
* **Idiomatic Python Streaming & $O(1)$ Rollback:** Real-time token streaming with automatic multi-byte UTF-8 boundary assembly and deterministic sequence rollback if the stream is interrupted early.
* **Concurrency Safety & Session Leasing:** Prevents memory corruption on shared GPU workspaces via deterministic generator session leasing.
* **Multi-GPU Ready:** Per-instance device routing (`device_id`) with RAII `cudaSetDevice` guards.

---

## 2. Installation

### From Pre-compiled Wheel (Recommended)

```bash
pip install qwen-engine
```
Or using `uv`:
```bash
uv add qwen-engine
```

### From Source (Local Development)

Ensure you have CMake 3.26+, a C++20 compiler (MSVC 2022 on Windows or GCC 11+ on Linux), and the NVIDIA CUDA Toolkit installed:

```bash
# Clone the repository
git clone https://github.com/Areen-09/qwen-engine.git
cd qwen-engine

# Build and install locally
uv pip install -e . --no-build-isolation
```

---

## 3. Quickstart: Zero-Setup Text Generation

If you don't have local weight files, `Engine.from_pretrained` automatically downloads pre-quantized W4A16 weights and tokenizer vocabulary using the standard Python library:

```python
import qwen_engine

# Automatically downloads and caches weights (~759 MB)
engine = qwen_engine.Engine.from_pretrained("qwen2.5-1.5b-w4a16")

# Generate completion
response = engine.generate("Explain quantum computing in one simple sentence.", max_tokens=64)
print(response)
```

If you already have local model files:

```python
from qwen_engine import Engine

engine = Engine(
    model_path="models/qwen25_1.5b_w4a16.qwenbin",
    tokenizer_path="models/tokenizer.bin",  # Optional if located adjacent to model
    device_id=0                            # Target GPU index
)

output = engine.generate("What is the speed of light?", max_tokens=32, temperature=0.0)
print(output)
```

---

## 4. Real-Time Token Streaming

`engine.stream()` returns a generator that yields text fragments as each token is generated. Multi-byte UTF-8 characters (e.g. emojis and Chinese characters) are reconstructed smoothly without unicode replacement characters (`\ufffd`):

```python
from qwen_engine import Engine

engine = Engine("models/qwen25_1.5b_w4a16.qwenbin")

print("Streaming: ", end="", flush=True)
for piece in engine.stream("Write a short poem about the ocean:", max_tokens=100):
    print(piece, end="", flush=True)
print("\n[Done]")
```

---

## 5. Early Interruption & Sequence Rollback Semantics

### Scenario A: Early Client Disconnect with $O(1)$ Rollback (Default)

In web servers (FastAPI / SSE / WebSockets), clients often disconnect before generation finishes. By default, `rollback_on_interrupt=True`:

```python
# Starts a 100-token stream, but aborts after 10 tokens
prompt = "Draft an extensive 5-page research paper on black holes."
prompt_len = len(engine.encode(prompt))

for i, piece in enumerate(engine.stream(prompt, max_tokens=100, rollback_on_interrupt=True)):
    print(piece, end="", flush=True)
    if i == 10:
        print("\n[Client Disconnected!]")
        break  # Generator teardown triggers stream.close(rollback=True)

# Sequence position was instantly rolled back to the prompt boundary:
assert engine.current_seq_pos == prompt_len
print(f"Sequence rolled back cleanly to prompt position: {engine.current_seq_pos}")
```

### Scenario B: Intentional Early Stop with KV-Cache Commit

When executing multi-turn dialog or stopping on specific delimiter sequences (e.g., `\n` or `<|im_end|>`), you may want to keep the generated tokens in the KV-cache:

```python
response = []
# Preserve generated tokens in KV-cache when exiting early:
for piece in engine.stream("User: Hello\nAssistant:", rollback_on_interrupt=False):
    response.append(piece)
    if "\n" in piece:
        break  # Generation stops, but tokens remain committed in KV-cache!

print(f"Committed tokens in KV-cache: {engine.current_seq_pos}")
```

---

## 6. Multi-Turn ChatML Conversation

The tokenizer supports deterministic formatting for Qwen's ChatML template (`<|im_start|>role\ncontent<|im_end|>\n`):

```python
from qwen_engine import Engine

engine = Engine("models/qwen25_1.5b_w4a16.qwenbin")

messages = [
    {"role": "system", "content": "You are a concise, helpful coding assistant."},
    {"role": "user", "content": "How do I reverse a list in Python?"}
]

# Pass message list directly to generate or stream:
reply = engine.generate(messages, max_tokens=64)
print(reply)
```

---

## 7. Asynchronous Streaming (`asyncio` / FastAPI)

For non-blocking asynchronous web servers, use `stream_async()`:

```python
import asyncio
from qwen_engine import Engine

async def main():
    engine = Engine("models/qwen25_1.5b_w4a16.qwenbin")
    
    async for piece in engine.stream_async("Count from 1 to 10:", max_tokens=30):
        print(piece, end="", flush=True)

asyncio.run(main())
```

---

## 8. Multi-GPU Data-Parallel Routing

To scale throughput across multiple GPUs on a single server, instantiate independent `Engine` instances targeting different `device_id` indices:

```python
import threading
from qwen_engine import Engine

# GPU 0 Engine
engine_gpu0 = Engine("models/qwen25_1.5b_w4a16.qwenbin", device_id=0)

# GPU 1 Engine (if available)
# engine_gpu1 = Engine("models/qwen25_1.5b_w4a16.qwenbin", device_id=1)

def run_worker(engine, prompt):
    print(f"[Device {engine.device_id}] Running: {prompt}")
    output = engine.generate(prompt, max_tokens=32)
    print(f"[Device {engine.device_id}] Result: {output}")

# The Python GIL is released during GPU compute, enabling full concurrency across threads:
t1 = threading.Thread(target=run_worker, args=(engine_gpu0, "What is machine learning?"))
t1.start()
t1.join()
```

---

## 9. Inspecting Discrete VRAM & Hardware Stats

You can query the static GPU arena allocations and device memory at any time:

```python
from qwen_engine import Engine

engine = Engine("models/qwen25_1.5b_w4a16.qwenbin")
stats = engine.stats

print(f"Total Arena Allocated: {stats.total_allocated_bytes / (1024**2):.2f} MB")
print(f"Weights Memory:        {stats.weights_bytes / (1024**2):.2f} MB")
print(f"Static KV Cache:       {stats.kv_cache_bytes / (1024**2):.2f} MB")
print(f"Workspace Buffers:     {stats.workspace_bytes / (1024**2):.2f} MB")
print(f"Device Free Memory:    {stats.device_free_bytes / (1024**2):.2f} MB")
```

---

## 10. Concurrency & Safety Rules

> [!IMPORTANT]
> Each `Engine` instance manages a single pre-allocated GPU workspace arena. **Do not attempt to run multiple concurrent queries on the same `Engine` instance simultaneously.**

If a second thread attempts to invoke `generate()` or `stream()` while an engine session is active, a `ConcurrencyError` is immediately raised:

```python
import qwen_engine

engine = qwen_engine.Engine("models/qwen25_1.5b_w4a16.qwenbin")

# In Thread A:
stream = engine.stream("Draft a long paper...")
next(stream) # Acquires session lease

# In Thread B (or same thread):
try:
    engine.generate("Another question")
except qwen_engine.ConcurrencyError as e:
    print("Caught concurrency violation:", e)

stream.close() # Releases session lease
```

---

## 11. Exception Reference

All exceptions inherit from `qwen_engine.QwenError` (alias `QwenEngineError`):

| Exception | Base | When Raised |
| :--- | :--- | :--- |
| `QwenError` | `RuntimeError` | Base class for all engine exceptions. |
| `ConcurrencyError` | `QwenError` | Raised if multiple threads attempt inference on the same `Engine` instance simultaneously. |
| `ContextWindowExceededError` | `QwenError` | Raised if prompt + generation length exceeds `MAX_SEQ_LEN` (4096 tokens). |
| `CudaOutOfMemoryError` | `QwenError` | Raised if GPU allocation fails during engine startup. |
| `ModelNotFoundError` | `FileNotFoundError`, `QwenError` | Raised if `.qwenbin` or `tokenizer.bin` cannot be located locally or remotely. |

---

## 12. Verification & Running Tests

To verify the installation and run the full test suite on your local GPU:

```bash
pytest tests/test_python_bindings.py -v
```
All 9 unit and integration tests (including deterministic parity, multi-byte UTF-8 streaming, rollback semantics, concurrency leasing, and downloader checks) will execute and report status.

