from crewai.tools import BaseTool
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
import logging

from src.settings.config import CrewSettings

logger = logging.getLogger(__name__)
settings = CrewSettings()

class SonarGeneralSearchTool(BaseTool):
    name: str = "Sonar General Search"
    description: str = (
        "Use this tool to perform a general internet search using the Sonar API. "
        "Input should be a string representing the search query."
    )

    def _run(self, query: str) -> str:
        api_key = settings.model_api_key
        if not api_key:
            logger.error("SONAR_API_KEY not set in environment variables")
            return "Error: Internal configuration error - API key not available."

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=settings.model_api_base
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return "Error: Failed to initialize search service."

        prompt = (
            f"Perform a general internet search for: {query}. "
            "Return a concise summary of the search results, including relevant links if available. "
            "If no relevant results are found, return exactly: No relevant results found."
        )

        try:
            response = client.chat.completions.create(
                model=settings.model_search_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0,
            )
            result = response.choices[0].message.content.strip()
            if "No relevant results found" in result:
                logger.info(f"No relevant results found for query: {query}")
                return "No relevant results found."
            logger.info(f"Successfully found results for query {query}: {result}")
            return result
        
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded for Sonar API: {e}")
            return "Error: Search service temporarily unavailable (rate limit exceeded)."
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            return "Error: Unable to connect to search service."
        except APIError as e:
            logger.error(f"API error for query {query}: {e}")
            return "Error: Search service returned an error."
        except Exception as e:
            logger.error(f"Unexpected error searching for query {query}: {e}")
            return "Error: An unexpected error occurred while searching."
