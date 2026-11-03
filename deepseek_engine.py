"""DeepSeek vision OCR runner (cloud API; needs DEEPSEEK_API_KEY in .env).

DeepSeek's only image-capable API model is ``deepseek-flash``; override with
DEEPSEEK_MODEL. The API is OpenAI-compatible, so this uses the ``openai`` SDK
pointed at https://api.deepseek.com (override with DEEPSEEK_BASE_URL).
"""
from __future__ import annotations

import os

from engines.vision_llm import MAX_OUTPUT_TOKENS, OCR_PROMPT, PageResult, cli, load_env, run_engine

ENGINE = "DeepSeek"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def _model() -> str:
    load_env()
    return os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)


def extract(file_bytes: bytes, filename: str) -> dict:
    from openai import OpenAI

    model = _model()
    client = None

    def call_page(image_b64: str, mime: str) -> PageResult:
        nonlocal client
        client = client or OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            timeout=300,
        )
        resp = client.chat.completions.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            }],
        )
        choice = resp.choices[0]
        return PageResult(
            text=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            truncated=choice.finish_reason == "length",
        )

    return run_engine(ENGINE, model, "DEEPSEEK_API_KEY", file_bytes, filename, call_page)


if __name__ == "__main__":
    cli(extract)
