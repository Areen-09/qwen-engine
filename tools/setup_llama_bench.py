import os
import sys
import json
import urllib.request
import zipfile

def download_file(url, dest_path):
    print(f"Downloading {url} -> {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100
                print(f"\r  {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
    print("\nDownload complete.")

def main():
    dest_dir = os.path.abspath("tools/llama_cpp")
    os.makedirs(dest_dir, exist_ok=True)
    models_dir = os.path.abspath("models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Fetch release info for tag b12175
    api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b12175"
    print(f"Fetching llama.cpp release info from {api_url}...")
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        release_data = json.loads(resp.read().decode("utf-8"))

    tag_name = release_data.get("tag_name", "b12175")
    print(f"Using llama.cpp release: {tag_name}")

    bin_url = None
    cudart_url = None
    for asset in release_data.get("assets", []):
        name = asset["name"]
        if name == "llama-b12175-bin-win-cuda-13.5-x64.zip":
            bin_url = asset["browser_download_url"]
        elif name == "cudart-llama-bin-win-cuda-13.5-x64.zip":
            cudart_url = asset["browser_download_url"]

    if not bin_url:
        print("Could not find win-cuda-13.5 x64 asset.")
        sys.exit(1)

    bin_zip = os.path.join(dest_dir, "llama-bin-win-cuda-13.5-x64.zip")
    if not os.path.exists(bin_zip):
        download_file(bin_url, bin_zip)
    print("Extracting binary zip...")
    with zipfile.ZipFile(bin_zip, "r") as z:
        z.extractall(dest_dir)

    if cudart_url:
        cudart_zip = os.path.join(dest_dir, "cudart-llama-bin-win-cuda-13.5-x64.zip")
        if not os.path.exists(cudart_zip):
            download_file(cudart_url, cudart_zip)
        print("Extracting cudart zip...")
        with zipfile.ZipFile(cudart_zip, "r") as z:
            z.extractall(dest_dir)

    # 2. Download Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
    gguf_path = os.path.join(models_dir, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    if not os.path.exists(gguf_path) or os.path.getsize(gguf_path) < 900 * 1024 * 1024:
        gguf_url = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        download_file(gguf_url, gguf_path)
    else:
        print(f"GGUF model already present: {gguf_path} ({os.path.getsize(gguf_path)} bytes)")

    print("\nSetup complete! You can now run llama-bench.")

if __name__ == "__main__":
    main()
