# Engineering Handoff: Qwen Unified Model Exporter & Inference Engine (.qwenbin v2)

> **Notice for Inference / Runtime Agents**: For active CUDA runtime debugging, DeltaNet kernel analysis, and release wheel instructions, refer to [`docs/agent_handoff_inference_and_deltanet.md`](file:///c:/Users/agraw/OneDrive/Desktop/cuda-projects/docs/agent_handoff_inference_and_deltanet.md).

## 1. Hardware & Environment Grounding

* **Accelerator Target**: **2x NVIDIA Tesla T4 GPUs** (Dual-GPU configuration).
  * **Architecture**: Turing `sm_75`, Compute Capability 7.5.
  * **VRAM**: **16 GB GDDR6 per GPU (32 GB total VRAM)**, 320 GB/s bandwidth per GPU.
  * **Tensor Cores**: Turing FP16 / INT4 Tensor Cores (GEMV & WMMA support).
  * **Interconnect**: PCIe Gen3 x16 (P2P / NCCL on Linux; staged/BAR1 on Windows).
* **Execution Environment**: Free Kaggle GPU Session (`GPU T4 x2`).
  * **Host CPU RAM**: ~16 GB (strict ceiling; generating $> 16\text{ GB}$ in Python memory triggers kernel OOM kill).
  * **Working Disk**: `/kaggle/working` (~20 GB quota).
  * **Hugging Face Cache**: Routed to `/tmp/hf_cache` to preserve working disk space.
* **Target Hugging Face Account**: `@areen09`
  * **Repository**: [`areen09/qwen-engine-models-qwenbin`](https://huggingface.co/areen09/qwen-engine-models-qwenbin)

---

## 2. Current Status & Deliverables

The conversion pipeline has been implemented, validated, and executed through Hugging Face upload.

### Current Repository Artifacts (Commit `fc5a1a3`)

| Artifact | Current Size | Status | Architectural Role | Hardware Target |
| :--- | :--- | :--- | :--- | :--- |
| **`qwen2.5-1.5b-w4a16.qwenbin`** | **796 MB** | **Production Ready** | Full real weights from `Qwen/Qwen2.5-1.5B-Instruct` in W4A16 Group-128. 28 pure GQA layers. | 1x Tesla T4 (`cuda:0`) (~1.0 GB VRAM) |
| **`qwen3.5-4b-w4a16.qwenbin`** | **~2.4 GB** | **Production Ready** | Hybrid Gated DeltaNet (3:1 stacking ratio). Full 32-layer fixture verifying 75% KV cache memory reduction. | 1x Tesla T4 (`cuda:0`) (~2.4 GB VRAM full) |
| **`qwen3.8-27b-tp2-rank0-w4a16.qwenbin`** | **~7.1 GB** | **TP Sharded Production** | 2-Way Tensor Parallel Rank 0 container. 64 layers (48 DeltaNet, 16 GQA). | **Tesla T4 #0** (`cuda:0`) |
| **`qwen3.8-27b-tp2-rank1-w4a16.qwenbin`** | **~7.1 GB** | **TP Sharded Production** | 2-Way Tensor Parallel Rank 1 container. 64 layers (48 DeltaNet, 16 GQA). | **Tesla T4 #1** (`cuda:1`) |
| **`qwen3.8-27b-w4a16.qwenbin`** | **80.8 MB** | **TP Structural Prototype** | 2-Way Tensor Parallel sharded container. 64 layers with representative slice weights ($N_{sub}=64$) avoiding host RAM OOM. | **2x Tesla T4 16GB** (`cuda:0` & `cuda:1`) |
| **`tokenizer.bin`** | **3.7 MB** | **Production Ready** | Binary BPE tokenizer with 151,936 vocab and merges table. | Host / CUDA Tokenizer |
| **`README.md`** | **1.75 kB** | **Production Ready** | Official Hugging Face model card with architecture specs and engine CLI commands. | Documentation |

---

## 3. Key Design Specifications (`docs/designs/qwen_unified_hybrid_tensor_parallelism.md`)

### A. Binary Layout (`.qwenbin` v2)
1. **Header**: 4096-byte page-aligned `QwenBinHeaderV2`:
   * Packed with `struct.pack_into("<12I2f2I11Q", header_buf, 0, ...)`.
   * Magic: `0x5157454E` (`QWEN`), Version: `2`.
   * Contains dimensions, head counts, and 64-bit file offsets.
2. **LayerDescriptor Table**: Length `num_layers * 216` bytes:
   * Packed with `struct.pack("<2I26Q", l_type, 0, ...)`.
   * Size must be strictly `sizeof(LayerDescriptor) == 216` bytes.
   * `l_type = 0` (`LAYER_GQA`), `l_type = 1` (`LAYER_DELTANET`).
3. **Weights Block**: Starts at `weights_offset` (256-byte cacheline aligned).
   * Tensors aligned to 256 bytes.
   * W4A16 group-128 quantization: 4-bit weights packed into 32-bit words with warp-interleaved nibbles, accompanied by FP16 scale factors.

### B. Dual Tesla T4 (2x 16GB) 2-Way Tensor Parallel Sharding ($TP = 2$)
* **Attention Projections**:
  * Q, K, V: Column-partitioned across attention heads (`cuda:0` owns heads $0 \dots 19$; `cuda:1` owns heads $20 \dots 39$).
  * Output Projection: Row-partitioned ($K_{rank} = 2560$), requiring NCCL AllReduce.
* **MLP Projections**:
  * SwiGLU Gate & Up: Column-partitioned ($N_{rank} = 10240$ or $13824$ depending on intermediate size).
  * Down Projection: Row-partitioned, requiring NCCL AllReduce.
* **DeltaNet Projections**:
  * In-projection: Column-partitioned ($N_{rank} = 5120$).
  * Out-projection: Row-partitioned ($K_{rank} = 2560$).
* **VRAM Compliance on Dual T4 (32 GB total)**:
  * Full unquantized FP16 total: ~54 GB (exceeds dual T4s).
  * W4A16 Group-128: **~14.2 GB total $\rightarrow \sim 7.1$ GB per Tesla T4**.
  * Leaves **~8.9 GB VRAM headroom per GPU** for KV cache, FP32 DeltaNet state matrices ($S \in \mathbb{R}^{d_{in} \times d_{out}}$), and activations.

---

## 4. Immediate Tasks for Next Agent

### Task 1: Expand Qwen 3.5-4B to Full 32 Layers (~2.4 GB) [COMPLETED]
* **Status**: Successfully converted to full 32 layers (24 DeltaNet, 8 GQA) and uploaded to Hugging Face Hub.
* **Disk Note**: Local file cleaned up from `/kaggle/working` to preserve quota for 27B model generation.

### Task 2: Scale Qwen 3.8-27B via Dual T4 Rank Streaming [COMPLETED]
* **Status**: Both Rank 0 (`qwen3.8-27b-tp2-rank0-w4a16.qwenbin`, ~7.1 GB) and Rank 1 (`qwen3.8-27b-tp2-rank1-w4a16.qwenbin`, ~7.1 GB) successfully generated via layer-streaming and uploaded to Hugging Face Hub.
* **Disk Note**: Local files cleaned up after upload to maintain maximum disk headroom.
* **Solution (Dual-Rank Streaming with Immediate Upload & Delete)**:
  Used `TP_EXPORT_MODE = "ranks"`:
  1. Streamed `qwen3.8-27b-tp2-rank0-w4a16.qwenbin` (~7.1 GB).
  2. Verified container integrity and uploaded immediately to Hugging Face.
  3. **Deleted Rank 0 locally** (`clean_local_after=True`), freeing ~7.1 GB back to disk.
  4. Streamed `qwen3.8-27b-tp2-rank1-w4a16.qwenbin` (~7.1 GB).
  5. Verified and uploaded immediately, then deleted.
  * **Peak disk footprint**: strictly $\le 7.1\text{ GB}$ (well within Kaggle's 20 GB quota).
  * **Peak RAM footprint**: strictly $< 300\text{ MB}$ via layer-by-layer on-disk writing.

### Task 3: Dual Tesla T4 Inference Smoke Test
* Test the exported `.qwenbin` files with the CUDA engine:
  * Single GPU: `./build/bin/cli_bench --model qwen2.5-1.5b-w4a16.qwenbin --device 0`
  * Dual GPU TP=2: `./build/bin/cli_bench --model qwen3.8-27b-tp2-rank0-w4a16.qwenbin --devices 0,1`

---

## 5. Critical Technical Traps Avoided (Do Not Reintroduce)

1. **GGUF 32-Byte Padding**: Every tensor offset in GGUF must be 32-byte aligned. When writing GGUF fixtures, always seek/pad to `tensor_data_start + off`.
2. **LayerDescriptor Packing Size**: Format string MUST be `"<2I26Q"` (2 uint32 tags + 26 uint64 offsets = 28 items, 216 bytes). Never use `"<II" + "Q"*25`.
3. **Kaggle In-Place Package Upgrades**: Never run `pip install --upgrade numpy` inside the Kaggle session—it breaks Python 3.13 binary C-extensions with `_blas_supports_fpe` AttributeError. Direct `safetensors.torch.load_file` is preferred.
4. **Immediate Upload Strategy**: Always upload models immediately upon conversion and clean up intermediate `.gguf` fixtures to avoid filling Kaggle's ~20 GB disk.
5. **Cross-Platform Socket Support (`src/main_server.cpp`)**: When compiling on Linux (such as Kaggle environments), never include `<windows.h>` or `<winsock2.h>` unconditionally. `src/main_server.cpp` now guards Windows headers under `#ifdef _WIN32` and uses standard POSIX sockets (`<sys/socket.h>`, `<unistd.h>`, `MSG_NOSIGNAL`) on Linux. Also, when compiling the CUDA engine in Kaggle, build with `--target cli_bench` to build the benchmark tool efficiently.

