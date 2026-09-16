"""Thin LLM adapter — provider-agnostic interface for chat completions.

Concrete implementation for OpenAI-compatible APIs (OpenAI, Groq, Together,
local Ollama, etc). Swap providers by changing the base URL and API key.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI

from app.config import settings


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class LLMToolCall:
    """A tool call parsed from the LLM response."""
    id: str
    name: str
    arguments: dict[str, Any]
    extra_content: dict[str, Any] | None = None


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    content: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str = ""
    raw_message: dict[str, Any] | None = None
    provider: str = ""
    latency_ms: float = 0.0
    request_latency_ms: float = 0.0
    retry_count: int = 0
    retry_delay_ms: float = 0.0
    retry_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol (for future alternative implementations)
# ---------------------------------------------------------------------------

class LLMAdapter(Protocol):
    """Abstract interface for LLM chat completions."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible implementation
# ---------------------------------------------------------------------------

class OpenAIAdapter:
    """Adapter for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.openai_api_key
        self._base_url = base_url or settings.llm_base_url
        self._model = model or settings.llm_model
        self._client: AsyncOpenAI | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        base_url = self._base_url.lower()
        if "generativelanguage.googleapis.com" in base_url:
            return "gemini"
        if "api.openai.com" in base_url:
            return "openai"
        return self._base_url.split("/")[2] if "://" in self._base_url else "custom"

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-initialize the OpenAI client (avoids eager key validation on import)."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "OpenAI API key not configured. Set OPENAI_API_KEY in .env "
                    "or pass api_key to OpenAIAdapter."
                )
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=60.0,  # 60s timeout — Gemini compat endpoint can be slow
                max_retries=0,  # Handle retries in our own loop with adaptive backoff
            )
        return self._client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a chat completion request and parse the response."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        max_retries = 5
        backoff = 3.0
        response = None
        request_started = time.perf_counter()
        request_latency_ms = 0.0
        retry_delay_ms = 0.0
        retry_count = 0
        retry_errors: list[str] = []

        for attempt in range(max_retries):
            try:
                attempt_started = time.perf_counter()
                response = await self._get_client().chat.completions.create(**kwargs)
                request_latency_ms += (time.perf_counter() - attempt_started) * 1000
                break
            except Exception as e:
                request_latency_ms += (time.perf_counter() - attempt_started) * 1000
                err_str = str(e)
                retry_errors.append(err_str)
                err_lower = err_str.lower()
                is_rate_limit = (
                    (hasattr(e, "status_code") and e.status_code == 429)
                    or "429" in err_lower
                    or "resource_exhausted" in err_lower
                    or "quota" in err_lower
                )
                if is_rate_limit and attempt < max_retries - 1:
                    retry_count += 1
                    wait_time = backoff
                    import re
                    match = re.search(r'retry in (\d+(?:\.\d+)?)s', err_str, re.IGNORECASE)
                    if match:
                        wait_time = float(match.group(1)) + 1.5
                    else:
                        match_delay = re.search(r'retryDelay[\'":\s]+(\d+)s', err_str)
                        if match_delay:
                            wait_time = float(match_delay.group(1)) + 1.5
                        else:
                            wait_time = max(backoff, 6.0)
                    retry_delay_ms += wait_time * 1000
                    await asyncio.sleep(wait_time)
                    backoff *= 1.5
                else:
                    raise

        choice = response.choices[0]
        message = choice.message

        # Parse tool calls if present
        parsed_tool_calls: list[LLMToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                extra = getattr(tc, "extra_content", None)
                parsed_tool_calls.append(
                    LLMToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                        extra_content=extra,
                    )
                )

        raw_msg = None
        try:
            raw_msg = message.model_dump(exclude_unset=True)
        except Exception:
            pass

        return LLMResponse(
            content=message.content,
            tool_calls=parsed_tool_calls,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            finish_reason=choice.finish_reason or "",
            raw_message=raw_msg,
            provider=self.provider,
            latency_ms=(time.perf_counter() - request_started) * 1000,
            request_latency_ms=request_latency_ms,
            retry_count=retry_count,
            retry_delay_ms=retry_delay_ms,
            retry_errors=retry_errors,
        )


# ---------------------------------------------------------------------------
# Local Hugging Face Open-Source Model Adapter (with optional LoRA)
# ---------------------------------------------------------------------------

class LocalHuggingFaceAdapter:
    """Adapter for local Hugging Face CausalLM models (e.g. Qwen2.5) with optional LoRA adapter."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path: str | None = None,
        device: str | None = None,
        torch_dtype: Any | None = None,
    ) -> None:
        self._base_model_name = base_model_name
        self._adapter_path = adapter_path
        self._device_str = device
        self._dtype = torch_dtype
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        adapter_suffix = f"+lora({Path(adapter_path).name})" if adapter_path else ""
        self._display_name = f"{base_model_name}{adapter_suffix}"

    @property
    def model(self) -> str:
        return self._display_name

    @property
    def provider(self) -> str:
        return "local_huggingface"

    def _ensure_loaded(self) -> None:
        """Lazy-initialize model and tokenizer on first chat invocation."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        target_device = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = self._dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._base_model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self._base_model_name,
            dtype=target_dtype,
            device_map="auto" if target_device == "cuda" else None,
            trust_remote_code=True,
        )

        if self._adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(base_model, self._adapter_path)
        else:
            self._model = base_model

        self._model.eval()
        self._device = next(self._model.parameters()).device

    def _sync_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Synchronous chat completion with native tool call extraction."""
        import torch

        self._ensure_loaded()
        request_started = time.perf_counter()

        # Format prompt using Qwen chat template with native tools if provided
        if tools:
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)
        prompt_tokens = inputs.input_ids.shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": 512,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][prompt_tokens:]
        completion_tokens = len(new_tokens)
        decoded_text = self._tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Parse tool calls from <tool_call> tags
        # Format: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
        parsed_tool_calls: list[LLMToolCall] = []
        call_blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', decoded_text, flags=re.DOTALL)
        for block in call_blocks:
            try:
                call_data = json.loads(block)
                name = call_data.get("name", "")
                arguments = call_data.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {}
                parsed_tool_calls.append(
                    LLMToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=name,
                        arguments=arguments,
                    )
                )
            except Exception:
                pass

        # Strip tool call tags and EOS special tokens to extract clean content
        cleaned_content = re.sub(r'<tool_call>.*?</tool_call>', '', decoded_text, flags=re.DOTALL)
        cleaned_content = cleaned_content.replace("<|im_end|>", "").strip()
        final_content = cleaned_content if cleaned_content else None

        latency_ms = (time.perf_counter() - request_started) * 1000

        return LLMResponse(
            content=final_content,
            tool_calls=parsed_tool_calls,
            model=self._display_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="tool_calls" if parsed_tool_calls else "stop",
            provider=self.provider,
            latency_ms=latency_ms,
            request_latency_ms=latency_ms,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Asynchronously invoke local model generation in a background worker thread."""
        return await asyncio.to_thread(self._sync_chat, messages, tools, temperature)

