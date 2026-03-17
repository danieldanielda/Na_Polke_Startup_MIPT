from crewai.tools import BaseTool
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
import logging

from src.settings.config import CrewSettings

logger = logging.getLogger(__name__)
settings = CrewSettings()

class SonarSearchTool(BaseTool):
    name: str = "Sonar Barcode Product Lookup"
    description: str = (
        "Use this tool to find the product name, brand, and category by providing a barcode. "
        "Input must be a valid numeric barcode string (e.g., '8809576261752')."
    )

    def _run(self, barcode: str) -> str:
        if not barcode.isdigit():
            logger.warning(f"Invalid barcode format: {barcode}")
            return "Product not found."
        
        api_key = settings.model_api_key
        if not api_key:
            logger.error("SONAR_API_KEY not set in environment variables")
            return "Error: Internal configuration error - API key not available."

        # Создаём клиент внутри _run
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=settings.model_api_base
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return "Error: Failed to initialize search service."

        prompt = (
            f"Find the exact commercial product name for barcode {barcode}. "
            "Return ONLY the product name in plain text, without any formatting, explanations, or extra words. "
            "If the product is not found, return exactly: Product not found."
        )

        try:
            response = client.chat.completions.create(
                model=settings.model_search_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0,
            )
            result = response.choices[0].message.content.strip()
            if "Product not found" in result:
                logger.info(f"Product not found for barcode: {barcode}")
                return "Product not found."
            logger.info(f"Successfully found product for barcode {barcode}: {result}")
            return result
        
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded for Sonar API: {e}")
            return "Error: Search service temporarily unavailable (rate limit exceeded)."
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            return "Error: Unable to connect to product search service."
        except APIError as e:
            logger.error(f"API error for barcode {barcode}: {e}")
            if e.status_code == 404:
                return "Product not found."
            return "Error: Search service returned an error."
        except Exception as e:
            logger.error(f"Unexpected error searching for barcode {barcode}: {e}")
            return "Error: An unexpected error occurred while searching for the product."