import html
import logging
import cv2
import numpy as np
import httpx
import json

from pyzbar import pyzbar

from config import BotSettings

logger = logging.getLogger(__name__)
settings = BotSettings()


def decode_barcode_with_cv2(image_path: str) -> str | None:
    """
    Decode barcode from image using multiple enhancement techniques.
    Returns the barcode string or None if decoding fails.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        attempts = [
            gray,
            cv2.equalizeHist(gray),
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        ]

        for i, img in enumerate(attempts):
            barcodes = pyzbar.decode(img)
            if barcodes:
                logger.info(f"Barcode decoded on attempt {i+1}: {barcodes[0].data.decode('utf-8')}")
                return barcodes[0].data.decode("utf-8")

        # Try scaling up for better detection
        for scale in [1.2, 1.5, 2.0]:
            resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            barcodes = pyzbar.decode(resized)
            if barcodes:
                logger.info(f"Barcode decoded with scale {scale}: {barcodes[0].data.decode('utf-8')}")
                return barcodes[0].data.decode("utf-8")

        logger.warning(f"No barcode found in image: {image_path}")
        return None
    
    except Exception as e:
        logger.error(f"Error decoding barcode from {image_path}: {e}")
        return None


async def get_product_analysis(product_info: str, analysis_type: str = "description", is_barcode: bool = True) -> str | None:
    """
    Получает результат анализа продукта: сначала parser (search_barcode) если это штрихкод, затем analyze_product.
    Возвращает отформатированный текст для пользователя или None при ошибке.
    """
    analyze_url = f"{settings.agents_api_base}/api/v1/crew/analyze_product"
    try:
        timeout = httpx.Timeout(
            connect=10.0,   # таймаут подключения
            read=300.0,     # таймаут чтения
            write=60.0,     # таймаут записи (можно меньше)
            pool=60.0       # таймаут ожидания соединения из пула
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            if is_barcode:
                if not product_info or not product_info.strip().isdigit():
                    return None
                product_info = product_info.strip()
                search_url = f"{settings.agents_api_base}/api/v1/crew/search_barcode"
                search_resp = await client.post(search_url, json={"barcode": product_info})
                if search_resp.status_code != 200:
                    logger.error(f"search_barcode failed: {search_resp.status_code} {search_resp.text}")
                    return None
                data = search_resp.json()
                product_info = data.get("product_info") or ""
                if not product_info.strip():
                    return None
            
            # 2. Запрос в analyze_product
            analyze_resp = await client.post(
                analyze_url,
                json={
                    "product_info": product_info,
                    "collection_id": settings.rag_collection_id,
                    "analysis_type": analysis_type,
                },
            )
            if analyze_resp.status_code != 200:
                logger.error(f"analyze_product failed: {analyze_resp.status_code} {analyze_resp.text}")
                return None
            analysis_data = analyze_resp.json()
            
            if "summary" in analysis_data:
                return analysis_data["summary"][:4096]

            product_description = (
                analysis_data.get("product_description")
                or analysis_data.get("summary")
                or "Описание продукта недоступно."
            )

            safe_ingredients = analysis_data.get("safe_ingredients", [])
            neutral_ingredients = analysis_data.get("neutral_ingredients", [])
            caution_ingredients = analysis_data.get("caution_ingredients", [])
            avoid_ingredients = analysis_data.get("avoid_ingredients", [])

            response_text = f"**Описание продукта:**\n{product_description}\n\n"

            if safe_ingredients:
                response_text += "🟢 Безопасные ингредиенты: 🟢\n"
                for ing in safe_ingredients:
                    response_text += f"- {ing['name']}\n"
            if neutral_ingredients:
                response_text += "🟡 Нейтральные ингредиенты: 🟡\n"
                for ing in neutral_ingredients:
                    response_text += f"- {ing['name']}\n"
            if caution_ingredients:
                response_text += "🟠 Ингредиенты, требующие осторожности: 🟠\n"
                for ing in caution_ingredients:
                    response_text += f"- {ing['name']}\n"
            if avoid_ingredients:
                response_text += "🔴 Ингредиенты, которых следует избегать: 🔴\n"
                for ing in avoid_ingredients:
                    response_text += f"- {ing['name']}\n"
            
            return response_text

    except httpx.RequestError as e:
        logger.error(f"HTTP request error in get_product_analysis: {e}", exc_info=True)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in get_product_analysis: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_product_analysis: {e}", exc_info=True)
    return None


async def get_product_analysis_by_barcode(barcode: str, analysis_type: str = "description") -> str | None:
    return await get_product_analysis(barcode, analysis_type, is_barcode=True)


async def get_product_analysis_by_name(product_name: str, analysis_type: str = "description") -> str | None:
    return await get_product_analysis(product_name, analysis_type, is_barcode=False)


async def format_analysis_for_telegram(analysis_text: str) -> str:
    lines = analysis_text.splitlines()
    html_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            html_lines.append("")  
        elif line.startswith("## "):
            html_lines.append(f"<b>{html.escape(line[3:])}</b>")
        elif line.startswith("# "):
            html_lines.append(f"<b>{html.escape(line[2:])}</b>")
        elif line.startswith("*"):
            # Убираем звёздочки, оставляем только текст
            content = line.strip("* ").strip()
            if content:  # игнорируем пустые
                html_lines.append(f"<i>{html.escape(content)}</i>")
        elif line.startswith("---"):
            html_lines.append("——————————")
        elif line.startswith("- "):
            html_lines.append(f"• {html.escape(line[2:])}")
        else:
            # для остальных строк экранируем HTML, но оставляем эмодзи
            html_lines.append(html.escape(line).replace("&amp;", "&"))

    return "\n".join(html_lines)