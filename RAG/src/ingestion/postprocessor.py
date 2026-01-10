from llama_index.core.schema import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor

class MetadataStripperPostprocessor(BaseNodePostprocessor):
    """Clears metadata for LLM but preserves it for return in response"""

    def _postprocess_nodes(self, nodes, query_bundle: QueryBundle = None):
        from llama_index.core.schema import NodeWithScore, TextNode
        clean_nodes = []
        for node in nodes:
            # Save original metatadata
            original_metadata = node.node.metadata.copy()

            # Empty new node for llm
            clean_node = TextNode(
                text=node.node.text,  #] only text for llm
                id_=node.node.node_id,
                metadata={},  # empty metadata for LLM
                text_template="{content}",
                metadata_template=""
            )

            setattr(clean_node, "_original_metadata", original_metadata)
            clean_nodes.append(NodeWithScore(node=clean_node, score=node.score))
        return clean_nodes