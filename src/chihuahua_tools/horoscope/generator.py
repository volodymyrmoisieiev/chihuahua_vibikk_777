from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT_HOROSCOPE

load_dotenv()


def generate_chihuahua_horoscope(
    name: str,
    details: Optional[str] = None,
) -> str:
    """
    Generate a short funny horoscope for a chihuahua (UA, 3–4 sentences).
    Uses SYSTEM_PROMPT_HOROSCOPE and OPENAI_API_KEY from environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (check your .env in repo root).")

    user_prompt = f"Згенеруй гороскоп для чіхуахуа на ім'я {name}."
    if details:
        user_prompt += f"\nДодаткові деталі: {details}"

    
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_HOROSCOPE},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=250,
        )
        return (resp.choices[0].message.content or "").strip()
    except ImportError:
        import openai  # type: ignore

        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_HOROSCOPE},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=250,
        )
        return resp["choices"][0]["message"]["content"].strip()
