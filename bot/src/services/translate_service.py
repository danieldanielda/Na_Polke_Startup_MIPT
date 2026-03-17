import asyncio
import logging
from openai import OpenAI

from config import BotSettings

settings = BotSettings()
logger = logging.getLogger(__name__)

async def translate_text(text: str, target_lang: str) -> str | None:
    """
    Перевод текста через AITunnel (Gemini).
    target_lang: 'ru' или 'en'
    """
    try:
        client = OpenAI(
            api_key=settings.model_api_key,
            base_url=settings.model_api
        )

        if target_lang == "ru":
            prompt = f"Переведи следующий текст на русский язык:\n\n{text}"
        else:
            prompt = f"Translate the following text to English:\n\n{text}"

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=settings.translate_model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        return None