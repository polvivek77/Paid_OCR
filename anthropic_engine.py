"""Anthropic Claude vision OCR runner (cloud API; needs ANTHROPIC_API_KEY in .env).

Model defaults to the flagship ``claude-opus-5-5``; override with ANTHROPIC_MODEL.
Each page is sent as a rasterised base64 image block via the Messages API.
"""
from __future__ import annotations

import os

from engines.vision_llm import MAX_OUTPUT_TOKENS, OCR_PROMPT, PageResult, cli, load_env, run_engine

ENGINE = "Anthropic"
DEFAULT_MODEL = "claude-opus-5-5"


def _model() -> str:
    load_env()
    return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


def extract(file_bytes: bytes, filename: str) -> dict:
    import anthropic

    model = _model()
    client = None

    def call_page(image_b64: str, mime: str) -> PageResult:
        nonlocal client
        client = client or anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300)
        # Streaming avoids the SDK's long-request guard on large max_tokens values.
        with client.messages.stream(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": mime, "data": image_b64}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }],
        ) as stream:
            msg = stream.get_final_message()
        text = "".join(block.text for block in msg.content if block.type == "text")
        return PageResult(
            text=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            truncated=msg.stop_reason == "max_tokens",
        )

    return run_engine(ENGINE, model, "ANTHROPIC_API_KEY", file_bytes, filename, call_page)


if __name__ == "__main__":
    cli(extract)
