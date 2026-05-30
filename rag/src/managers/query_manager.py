import logging

from typing import Dict, Any, List
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

class QueryManager:
    """
    Executing queries and formatting responses.
    Attributes:
        engine: RetrieverQueryEngine
    """
    engine: RetrieverQueryEngine

    def __init__(self, query_engine: RetrieverQueryEngine) -> None:

        self.engine = query_engine

    async def run_query(self, query_str: str) -> Dict[str, Any]:
        try:

            result = await self.engine.aquery(query_str)

            input_tokens = 0
            output_tokens = 0
            
            # Serialize source nodes (unchanged)
            source_nodes = await self._serialize_nodes(result.source_nodes)
            metadata = {}
            if hasattr(result, 'metadata') and result.metadata is not None:
                metadata = dict(result.metadata)
                score = metadata.get('score', 0.0)
            else:
                score = 0.0

            dict_response = {
                "response": str(result.response),
                "tokens": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },

                "source_nodes": source_nodes,
                "metadata": {
                    "score": float(score)}
            }
            return dict_response
        except Exception as e:

            logger.error(f"Query execution failed: {str(e)}", exc_info=True)
            return {
                "response": f"An error occurred while processing the request: {str(e)}",
                "source_nodes": [],
                "metadata": {"error": str(e)}
            }

    @staticmethod
    async def _serialize_nodes(nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        serialized = []
        for node in nodes:
            try:
                original_metadata = getattr(node.node, "_original_metadata", {})

                serialized.append({
                    "id": node.node.node_id if hasattr(node.node, 'node_id') else '',
                    "text": node.node.text,
                    "score": float(node.score) if hasattr(node, 'score') else 0.0,
                    "metadata": original_metadata  # Use original metadata
                })
            except Exception as e:
                logger.warning(f"Failed to serialize node: {str(e)}", exc_info=True)
                continue
        return serialized