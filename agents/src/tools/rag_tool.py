from crewai.tools import BaseTool
import httpx
import os
import json
from typing import Optional

from src.settings.config import CrewSettings

settings = CrewSettings()


class RAGQueryTool(BaseTool):
    """
    Tool for querying the global RAG (Retrieval-Augmented Generation) system.
    All queries are executed against a single shared collection containing all documents.
    """
    name: str = "RAG Query Tool"
    description: str = (
        "Use this tool to query the RAG (Retrieval-Augmented Generation) system with product information. "
        "Input should be a JSON string with 'query' (required) and 'system_prompt' (optional, default: 'system_common_prompt'). "
        "The tool queries the global RAG collection which contains all uploaded documents. "
        "Example: {\"query\": \"What are the ingredients of this product?\", \"system_prompt\": \"system_common_prompt\"}"
    )

    def _run(self, query_input: str) -> str:
        """
        Query the global RAG system
        
        Args:
            query_input: JSON string with query and optional system_prompt
            
        Returns:
            Response from RAG system
        """
        try:
            # Try to parse as JSON first
            try:
                params = json.loads(query_input)
                query = params.get("query", "")
                system_prompt = params.get("system_prompt", "system_common_prompt")
            except json.JSONDecodeError:
                # If not JSON, treat as plain query string
                query = query_input
                system_prompt = "system_common_prompt"
            
            if not query or not query.strip():
                return "Error: Query cannot be empty."
            
            # Get RAG API URL from environment or use default
            # Note: The /rag/rag/ path is correct because the router prefix is /api/v1/rag
            # and the endpoint is now defined at /rag/ask (no collection_id parameter)
            
            rag_url = f"http://{settings.rag_host}:{settings.rag_port}/api/v1/rag/ask"
            print(rag_url)
            with httpx.Client(timeout=90.0) as client:
                response = client.post(
                    rag_url,
                    json={"query": query},
                    params={"system_prompt": system_prompt}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "No response from RAG system.")
                elif response.status_code == 404:
                    return "Error: RAG index not found. Please upload documents first."
                else:
                    error_msg = response.json().get("error", "Unknown error")
                    return f"Error querying RAG system: {error_msg}"
        
        except httpx.TimeoutException:
            return "Error: Request to RAG system timed out."
        except httpx.RequestError as e:
            return f"Error connecting to RAG system: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
