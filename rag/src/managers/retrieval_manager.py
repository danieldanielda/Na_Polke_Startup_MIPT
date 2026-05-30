import logging
from typing import List

from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.postprocessor.tei_rerank import TextEmbeddingInference as TEIR
from llama_index.core import Settings

from src.settings.config import RagSettings
from src.services.chroma_manager import ChromaManager
from src.ingestion.prompt_constructor import PromptConstructor
from src.ingestion.postprocessor import MetadataStripperPostprocessor

logger = logging.getLogger(__name__)

settings = RagSettings()

class RetrievalManager:
    """
    Manages and configures hybrid retrieval logic for a document index.
    Filters based on 'collection_id' have been removed to support global collection queries.
    """

    chroma_manager: ChromaManager
    index: VectorStoreIndex
    nodes: List[BaseNode]
    reranker: TEIR

    def __init__(self, index: VectorStoreIndex, nodes: List[TextNode], system_prompt: str) -> None:
        if index is None and nodes is None:
            raise ValueError("Needs either index or nodes")
    
        self.chroma_manager = ChromaManager()
        self.index = index
        self.nodes = nodes
        self.reranker = TEIR(
                top_n=5,
                auth_token=f"Bearer {settings.reranker_api_key}",
                model_name=settings.ranker_model,
                base_url=settings.ranker_api,
                timeout=60,
            )
        self.system_prompt = system_prompt
    
    
    def _check_similarity_top_k(self, similarity_top_k: int) -> int:
        if not self.nodes:
            return similarity_top_k
        actual_top_k = min(similarity_top_k, len(self.nodes))
        if actual_top_k < similarity_top_k:
            logger.warning(f"Reduced top k from {similarity_top_k} to {actual_top_k} (corpus size: {len(self.nodes)})")
        return actual_top_k


    def setup_vector_retriever(self, similarity_top_k: int = 10) -> VectorIndexRetriever:
        """
        Sets up a vector retriever using the vector index.
        NO filters are applied. Retrieves from the entire loaded index/collection.
        """
        if not self.nodes:
            # Если узлов нет в памяти, но индекс есть, ретривер все равно сработает через индекс
            logger.warning("No nodes loaded in memory, but attempting to create retriever from index.")
    
        actual_top_k = self._check_similarity_top_k(similarity_top_k=similarity_top_k)

        return VectorIndexRetriever(
            index=self.index,
            similarity_top_k=actual_top_k,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            embed_model=Settings.embed_model
        )


    def setup_bm25_retriever(self, similarity_top_k: int = 10) -> BM25Retriever:
        """
        Sets up a BM25 retriever based on ALL available nodes.
        No filtering by collection_id is performed.
        """
        if not self.nodes:
            raise ValueError("No nodes available for BM25 retrieval.")
            
        actual_top_k = self._check_similarity_top_k(similarity_top_k=similarity_top_k)
        
        return BM25Retriever.from_defaults(
            nodes=self.nodes, 
            similarity_top_k=actual_top_k, 
            language='ru'
        )


    def setup_fusion_retriever(self, similarity_top_k=10) -> QueryFusionRetriever:
        """
        Sets up a fusion retriever that combines vector and BM25 retrievers.
        """
        individual_top_k = min(similarity_top_k * 2, 20)

        vector_retriever = self.setup_vector_retriever(similarity_top_k=individual_top_k)
        bm25_retriever = self.setup_bm25_retriever(similarity_top_k=individual_top_k)

        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=False # Set to True only for debugging
        )  
        logger.info("Fusion retriever setup completed (Global Collection Mode)")
        return fusion_retriever


    async def build_query_engine(self) -> RetrieverQueryEngine:
        """
        Async method: builds the query engine.
        """
        prompt_template = await PromptConstructor().change_llama_index_prompt(path=settings.rag_prompt_path)

        response_synthesizer = get_response_synthesizer(
            llm=Settings.llm,
            response_mode="tree_summarize",
            text_qa_template=prompt_template,
            streaming=False,
            use_async=True
        )
    
        fusion_retriever = self.setup_fusion_retriever(similarity_top_k=10)

        metadata_stripper = MetadataStripperPostprocessor()

        engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            node_postprocessors=[metadata_stripper, self.reranker],
            response_synthesizer=response_synthesizer
        )

        engine.update_prompts({
            "response_synthesizer:summary_template": prompt_template
        })
        logger.info("Hybrid query engine setup completed successfully")
        return engine