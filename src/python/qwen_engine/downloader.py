"""
Zero-dependency weight downloader using standard library urllib and hashlib.
"""
import os
import sys
import hashlib
import uuid
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import urllib.error
from .exceptions import ModelNotFoundError, QwenEngineError

DEFAULT_HF_REPO = "cuda-projects/qwen2.5-1.5b-w4a16"

KNOWN_MODELS = {
    "qwen2.5-1.5b-w4a16": {
        "repo": DEFAULT_HF_REPO,
        "weight_file": "qwen25_1.5b_w4a16.qwenbin",
        "tokenizer_file": "tokenizer.bin",
    },
    "qwen25-1.5b-w4a16": {
        "repo": DEFAULT_HF_REPO,
        "weight_file": "qwen25_1.5b_w4a16.qwenbin",
        "tokenizer_file": "tokenizer.bin",
    },
    "qwen2.5-1.5b-fp16": {
        "repo": "cuda-projects/qwen2.5-1.5b-fp16",
        "weight_file": "qwen25_1.5b_fp16.qwenbin",
        "tokenizer_file": "tokenizer.bin",
    },
}


def get_default_cache_dir() -> Path:
    """Resolve the default cache directory according to OS conventions."""
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            cache_dir = Path(local_app_data) / "qwen_engine" / "Cache"
        else:
            cache_dir = Path.home() / ".cache" / "qwen_engine"
    else:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            cache_dir = Path(xdg_cache_home) / "qwen_engine"
        else:
            cache_dir = Path.home() / ".cache" / "qwen_engine"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_file(
    url: str,
    dest_path: Path,
    expected_sha256: Optional[str] = None,
    chunk_size: int = 1024 * 1024
) -> None:
    """Download a file with atomic replacement and live progress display."""
    if dest_path.exists():
        if expected_sha256:
            hasher = hashlib.sha256()
            with open(dest_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            if hasher.hexdigest().lower() == expected_sha256.lower():
                return
        else:
            # File exists and size > 0, assume cached
            if dest_path.stat().st_size > 0:
                return

    dest_dir = dest_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    tmp_filename = f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp_path = dest_dir / tmp_filename

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "qwen-engine/0.1.0"}
    )

    try:
        with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_file:
            content_length = resp.headers.get("Content-Length")
            total_bytes = int(content_length) if content_length and content_length.isdigit() else None
            downloaded = 0
            hasher = hashlib.sha256() if expected_sha256 else None

            filename = dest_path.name
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if hasher:
                    hasher.update(chunk)

                if total_bytes and sys.stdout.isatty():
                    percent = (downloaded / total_bytes) * 100.0
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_bytes / (1024 * 1024)
                    bar_len = 30
                    filled = int(bar_len * downloaded // total_bytes)
                    bar = "=" * filled + "-" * (bar_len - filled)
                    sys.stdout.write(f"\rDownloading {filename}: [{bar}] {percent:5.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                    sys.stdout.flush()

            if sys.stdout.isatty() and total_bytes:
                sys.stdout.write("\n")
                sys.stdout.flush()

        if expected_sha256 and hasher:
            actual_sha = hasher.hexdigest().lower()
            if actual_sha != expected_sha256.lower():
                if tmp_path.exists():
                    tmp_path.unlink()
                raise QwenEngineError(
                    f"SHA256 mismatch for {dest_path.name}: expected {expected_sha256}, got {actual_sha}"
                )

        # Atomic replacement into final destination
        os.replace(tmp_path, dest_path)

    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def download(
    model_id_or_path: str,
    cache_dir: Optional[str | Path] = None,
    revision: str = "main",
) -> Tuple[str, str]:
    """
    Resolve or download model weights (.qwenbin) and tokenizer (.bin).

    Returns:
        Tuple of (model_path, tokenizer_path) as absolute string paths.
    """
    path_obj = Path(model_id_or_path)

    # 1. Direct local file or directory checks
    if path_obj.is_file():
        # Check if tokenizer is adjacent
        cand_tok = path_obj.parent / "tokenizer.bin"
        if cand_tok.is_file():
            return str(path_obj.resolve()), str(cand_tok.resolve())
        # Check models/ directory
        cwd_cand_tok = Path("models") / "tokenizer.bin"
        if cwd_cand_tok.is_file():
            return str(path_obj.resolve()), str(cwd_cand_tok.resolve())
        return str(path_obj.resolve()), ""

    if path_obj.is_dir():
        # Directory containing .qwenbin and tokenizer.bin
        qwenbins = list(path_obj.glob("*.qwenbin"))
        toks = list(path_obj.glob("tokenizer*.bin"))
        if qwenbins:
            model_file = qwenbins[0]
            tok_file = toks[0] if toks else (path_obj / "tokenizer.bin")
            return str(model_file.resolve()), str(tok_file.resolve())

    # Check relative to working directory models/
    local_model = Path("models") / model_id_or_path
    if local_model.is_file():
        local_tok = Path("models") / "tokenizer.bin"
        return str(local_model.resolve()), str(local_tok.resolve())

    # 2. Check known models or remote Hugging Face repository
    target_cache = Path(cache_dir) if cache_dir else get_default_cache_dir()
    target_cache.mkdir(parents=True, exist_ok=True)

    model_key = model_id_or_path.lower().strip()
    if model_key in KNOWN_MODELS:
        meta = KNOWN_MODELS[model_key]
        repo = meta["repo"]
        weight_name = meta["weight_file"]
        tok_name = meta["tokenizer_file"]
    elif "/" in model_id_or_path:
        repo = model_id_or_path
        weight_name = "qwen25_1.5b_w4a16.qwenbin"
        tok_name = "tokenizer.bin"
    else:
        # Check if local model folder matches
        for cand_dir in [Path("."), Path("models")]:
            cands = list(cand_dir.glob(f"*{model_id_or_path}*.qwenbin"))
            if cands:
                cand_tok = cand_dir / "tokenizer.bin"
                return str(cands[0].resolve()), str(cand_tok.resolve())
        raise ModelNotFoundError(
            f"Could not locate model '{model_id_or_path}'. Ensure it is a valid local path or known model identifier."
        )

    weight_url = f"https://huggingface.co/{repo}/resolve/{revision}/{weight_name}"
    tok_url = f"https://huggingface.co/{repo}/resolve/{revision}/{tok_name}"

    dest_weight = target_cache / weight_name
    dest_tok = target_cache / tok_name

    try:
        _download_file(weight_url, dest_weight)
        _download_file(tok_url, dest_tok)
    except urllib.error.URLError as e:
        # If download fails, check if we have local fallbacks in ./models/
        local_weight = Path("models") / weight_name
        local_tok = Path("models") / tok_name
        if local_weight.is_file() and local_tok.is_file():
            return str(local_weight.resolve()), str(local_tok.resolve())
        raise QwenEngineError(f"Network error while downloading weights from {repo}: {e}") from e

    return str(dest_weight.resolve()), str(dest_tok.resolve())

