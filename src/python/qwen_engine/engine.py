"""
High-level Pythonic Engine interface for qwen_engine.
"""
from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Generator, AsyncGenerator, List, Dict, Union, Optional, Any
from dataclasses import dataclass

from . import _core
from .exceptions import (
    QwenEngineError,
    ConcurrencyError,
    ModelNotFoundError,
    ContextWindowExceededError,
)
from .downloader import download


@dataclass
class SamplingParams:
    """Sampling and decoding configuration parameters."""
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float = 1.0
    top_k: int = 50


class Engine:
    """
    High-level inference engine interface for Qwen2.5.
    
    Encapsulates custom CUDA kernels, memory mapping, BPE tokenization,
    streaming generation, and KV cache rollback management.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        device_id: int = 0,
        device_ids: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Initialize the inference engine on the designated GPU(s).

        Args:
            model_path: Path to the quantized .qwenbin weight binary.
            tokenizer_path: Path to the binary tokenizer.bin. If None,
                searches adjacent to model_path or in standard paths.
            device_id: Target CUDA device ID (default: 0).
            device_ids: Optional list of target CUDA device IDs for multi-GPU execution.
        """
        model_p = Path(model_path).resolve()
        if not model_p.is_file():
            raise ModelNotFoundError(f"Model file not found: {model_p}")

        if tokenizer_path is None:
            # Check adjacent
            cand = model_p.parent / "tokenizer.bin"
            if cand.is_file():
                tokenizer_p = cand
            else:
                cwd_cand = Path("models") / "tokenizer.bin"
                if cwd_cand.is_file():
                    tokenizer_p = cwd_cand.resolve()
                else:
                    raise ModelNotFoundError(
                        f"tokenizer_path was not specified and could not be found adjacent to {model_p}"
                    )
        else:
            tokenizer_p = Path(tokenizer_path).resolve()
            if not tokenizer_p.is_file():
                raise ModelNotFoundError(f"Tokenizer file not found: {tokenizer_p}")

        self._model_path = str(model_p)
        self._tokenizer_path = str(tokenizer_p)

        # Resolve device IDs
        if device_ids is not None:
            if isinstance(device_ids, int):
                self._device_ids = [device_ids]
            else:
                self._device_ids = list(device_ids)
        else:
            self._device_ids = [device_id]

        available_devices = _core.cuda_device_count()
        for d in self._device_ids:
            if d < 0 or (available_devices > 0 and d >= available_devices):
                raise ValueError(f"Invalid CUDA device ID {d}; available devices: {available_devices}")

        self._device_id = self._device_ids[0]

        self._tokenizer = _core.NativeTokenizer(self._tokenizer_path)
        if len(self._device_ids) > 1:
            self._engine = _core.NativeEngine(self._model_path, self._device_ids)
        else:
            self._engine = _core.NativeEngine(self._model_path, self._device_id)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        device_id: int = 0,
        device_ids: Optional[Union[int, List[int]]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Engine:
        """
        Instantiate an engine from local file paths, an existing directory, or
        by automatically downloading weights from Hugging Face.

        Args:
            model_id_or_path: Model alias (e.g. 'qwen2.5-1.5b-w4a16'),
                Hugging Face repo ID, or local file/directory path.
            tokenizer_path: Path to the binary tokenizer.bin.
            device_id: Target GPU device ID (default: 0).
            device_ids: Optional list of target CUDA device IDs for multi-GPU execution.
            cache_dir: Custom directory to store downloaded models.
            model_path: Alias for model_id_or_path.
        """
        target_model = model_path or model_id_or_path
        if target_model is None:
            raise ValueError("model_id_or_path or model_path must be provided.")

        target_p = Path(target_model)
        if target_p.is_file():
            return cls(model_path=target_p, tokenizer_path=tokenizer_path, device_id=device_id, device_ids=device_ids)

        resolved_model, resolved_tok = download(str(target_model), cache_dir=cache_dir)
        actual_tok = tokenizer_path or resolved_tok
        return cls(model_path=resolved_model, tokenizer_path=actual_tok, device_id=device_id, device_ids=device_ids)

    @property
    def device_id(self) -> int:
        """Get the bound CUDA device ID."""
        return self._engine.device_id()

    @property
    def current_seq_pos(self) -> int:
        """Current sequence position in the KV cache."""
        return self._engine.current_seq_pos()

    @property
    def is_generating(self) -> bool:
        """Whether the engine currently holds an active inference lease."""
        return self._engine.is_generating()

    def get_memory_stats(self) -> _core.ArenaStats:
        """Retrieve discrete VRAM allocation and device memory statistics."""
        return self._engine.get_memory_stats()

    @property
    def stats(self) -> _core.ArenaStats:
        """Convenience property for memory and arena statistics."""
        return self.get_memory_stats()

    def encode(self, text: str) -> List[int]:
        """Encode input string to token IDs."""
        return self._tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to a string."""
        return self._tokenizer.decode(tokens)

    def format_chatml(self, messages: List[Dict[str, str]]) -> List[int]:
        """Format and tokenize a multi-turn ChatML conversation."""
        return self._tokenizer.format_chatml(messages)

    def reset_session(self) -> None:
        """Reset sequence positions, KV cache bounds, and session state."""
        self._engine.reset_session()

    def rollback_to(self, target_seq_pos: int) -> None:
        """Roll back KV-cache pointers and sequence position to target_seq_pos."""
        self._engine.rollback_to(target_seq_pos)

    def _prepare_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> List[int]:
        if isinstance(prompt, list):
            return self._tokenizer.format_chatml(prompt)
        elif isinstance(prompt, str):
            return self._tokenizer.encode(prompt)
        else:
            raise TypeError(f"Prompt must be a string or list of dicts, got {type(prompt).__name__}")

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        sampling_params: Optional[SamplingParams] = None,
    ) -> str:
        """
        Run complete forward prefill and autoregressive decode, returning the text.

        Args:
            prompt: Text prompt or list of ChatML messages (e.g. [{'role': 'user', 'content': '...'}]).
            max_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (0.0 = greedy argmax).
            sampling_params: Optional SamplingParams object overriding max_tokens/temperature.

        Returns:
            Decoded completion string.
        """
        if sampling_params is not None:
            max_tokens = sampling_params.max_tokens
            temperature = sampling_params.temperature
        if not self._engine.try_acquire_session():
            raise ConcurrencyError(
                "Engine is already executing another inference session. "
                "Concurrency on a single GPU arena is not permitted; use separate Engine instances."
            )

        try:
            tokens = self._prepare_tokens(prompt)
            if not tokens:
                return ""

            first_tok = self._engine.prefill(tokens, temperature)
            if self._tokenizer.is_eos(first_tok):
                return ""

            gen_tokens = [first_tok]
            for _ in range(max_tokens - 1):
                next_tok = self._engine.decode_step()
                if self._tokenizer.is_eos(next_tok):
                    break
                gen_tokens.append(next_tok)

            return self._tokenizer.decode(gen_tokens)

        finally:
            self._engine.release_session()

    def stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        rollback_on_interrupt: bool = True,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Generator[str, None, None]:
        """
        Generate tokens as a streaming generator with multi-byte UTF-8 buffering.

        If generation is interrupted (e.g. client disconnect or generator break),
        an O(1) rollback preserves prompt cache bounds when rollback_on_interrupt=True.
        If rollback_on_interrupt=False, the generated tokens remain committed in the KV cache.

        Args:
            prompt: Input text string or ChatML message list.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            rollback_on_interrupt: Whether to roll back KV cache if stream is closed early.
            sampling_params: Optional SamplingParams object overriding max_tokens/temperature.

        Yields:
            Decoded UTF-8 string fragments as tokens are generated.
        """
        if sampling_params is not None:
            max_tokens = sampling_params.max_tokens
            temperature = sampling_params.temperature
        if not self._engine.try_acquire_session():
            raise ConcurrencyError(
                "Engine is already executing another inference session. "
                "Concurrency on a single GPU arena is not permitted; use separate Engine instances."
            )

        native_stream: Optional[_core.PyEngineStream] = None
        carry_buffer = ""
        prompt_len = 0
        normal_completion = False

        try:
            tokens = self._prepare_tokens(prompt)
            if not tokens:
                return

            prompt_len = len(tokens)
            first_tok = self._engine.prefill(tokens, temperature)

            if self._tokenizer.is_eos(first_tok):
                flushed, carry_buffer = self._tokenizer.decode_streaming(first_tok, carry_buffer)
                if flushed:
                    yield flushed
                normal_completion = True
                return

            first_piece, carry_buffer = self._tokenizer.decode_streaming(first_tok, carry_buffer)
            if first_piece:
                yield first_piece

            remaining_tokens = max_tokens - 1
            if remaining_tokens > 0:
                # Construct native stream wrapper with carry buffer
                native_stream = _core.PyEngineStream(
                    self._engine,
                    self._tokenizer,
                    prompt_len,
                    remaining_tokens,
                    rollback_on_interrupt,
                    carry_buffer,
                )
                for piece in native_stream:
                    yield piece

            normal_completion = True

        finally:
            if native_stream is not None:
                # If finished normally, we commit without rolling back
                should_rollback = rollback_on_interrupt if not normal_completion else False
                native_stream.close(rollback=should_rollback)
            else:
                if not normal_completion and rollback_on_interrupt and prompt_len > 0:
                    self._engine.rollback_to(prompt_len)
                self._engine.release_session()

    async def stream_async(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        rollback_on_interrupt: bool = True,
        sampling_params: Optional[SamplingParams] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronous generator interface wrapping the synchronous token stream in worker threads.
        """
        if sampling_params is not None:
            max_tokens = sampling_params.max_tokens
            temperature = sampling_params.temperature
        gen = self.stream(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            rollback_on_interrupt=rollback_on_interrupt,
        )

        def _get_next():
            try:
                return next(gen), False
            except StopIteration:
                return "", True

        try:
            while True:
                piece, done = await asyncio.to_thread(_get_next)
                if done:
                    break
                yield piece
        finally:
            await asyncio.to_thread(gen.close)
