"""Quick test — verify Gemini API connectivity via OpenAI-compatible endpoint."""

import asyncio
from app.agent.llm_adapter import OpenAIAdapter


async def main():
    adapter = OpenAIAdapter()
    print(f"Model: {adapter._model}")
    print(f"Base URL: {adapter._base_url}")
    print("Sending test message...")

    try:
        response = await adapter.chat(
            messages=[
                {"role": "user", "content": "Say hello in exactly 5 words."}
            ],
            temperature=0.0,
        )
        print(f"Response: {response.content}")
        print(f"Model used: {response.model}")
        print(f"Tokens: prompt={response.prompt_tokens}, completion={response.completion_tokens}")
        print("[OK] Gemini API working!")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")


asyncio.run(main())
