"""OpenAI vision OCR runner (cloud API; needs OPENAI_API_KEY in .env).

Model defaults to the flagship ``gpt-6-astra``; override with OPENAI_MODEL.
Each page is sent as a rasterised image via Chat Completions with detail="high".
"""
from __future__ import annotations

import os

from engines.vision_llm import MAX_OUTPUT_TOKENS, OCR_PROMPT, PageResult, cli, load_env, run_engine

ENGINE = "OpenAI"
DEFAULT_MODEL = "gpt-6-astra"


def _model() -> str:
    load_env()
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def extract(file_bytes: bytes, filename: str) -> dict:
    from openai import OpenAI

    model = _model()
    client = None

    def call_page(image_b64: str, mime: str) -> PageResult:
        nonlocal client
        client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300)
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{image_b64}", "detail": "high"}},
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

    return run_engine(ENGINE, model, "OPENAI_API_KEY", file_bytes, filename, call_page)


if __name__ == "__main__":
    cli(extract)
