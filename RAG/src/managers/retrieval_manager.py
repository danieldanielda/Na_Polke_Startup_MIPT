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
#from llama_index.postprocessor.tei_rerank import TextEmbeddingInference as TEIR
from llama_index.core import Settings

from src.settings.config import RagSettings
# from src.evaluation.retrievers_eval import AdvancedAnswerEvaluator
# from src.evaluation.course_eval import CourseRetrievalEvaluator
from src.services.chroma_manager import ChromaManager
from src.ingestion.prompt_constructor import PromptConstructor
from src.ingestion.postprocessor import MetadataStripperPostprocessor

logger = logging.getLogger(__name__)

settings = RagSettings()

class RetrievalManager:
    """
    Manages and configures hybrid retrieval logic for a document index,
    including vector-based, BM25, and router-based retrievers.

    Also sets up reranking and builds a retriever-powered query engine
    suitable for hybrid RAG pipelines.

    Attributes:
        index (VectorStoreIndex): The vector index built from processed document nodes.
        nodes (List[TextNode]): A list of document nodes used for BM25 retrieval.
        reranker (SentenceTransformerRerank): A reranker that uses a sentence transformer model to rerank retrieved results.
    """

    chroma_manager: ChromaManager
    index: VectorStoreIndex
    nodes: List[BaseNode]
    reranker: SentenceTransformerRerank

    def __init__(self, index: VectorStoreIndex, nodes: List[TextNode], system_prompt: str) -> None:
        if index is None and nodes is None:
            raise ValueError("Needs either index or nodes")
    
        self.chroma_manager = ChromaManager()
        self.index = index
        self.nodes = nodes
        #self.reranker = TEIR(
        #        top_n=5,
        #       auth_token=f"Bearer {settings.reranker_api_key}",
        #      model_name=settings.ranker_model, # MUST BE THE SAME AS IN OFFICIAL DOCS
        #      base_url=settings.ranker_api,
        #        timeout=60,
        #    )
        self.reranker = SentenceTransformerRerank(top_n=5, model=settings.ranker_model)
        self.system_prompt = system_prompt
    
    def _check_similarity_top_k(self, similarity_top_k: int) -> int:
        actual_top_k = min(similarity_top_k, len(self.nodes))
        if actual_top_k < similarity_top_k:
            logger.warning(f"Reduced top k from {similarity_top_k} to {actual_top_k} (corpus size: {len(self.nodes)})")
        return actual_top_k

    def setup_vector_retriever(self, similarity_top_k: int = 10) -> VectorIndexRetriever:
        """
        Sets up a vector retriever using the vector index.

        Args:
            similarity_top_k (int): Number of top similar results to retrieve.

        Returns:
            VectorIndexRetriever: Configured vector retriever instance.
        """
        if not self.nodes:
            raise ValueError("No nodes available for retrieval.")
    
        collection_id = self.nodes[0].metadata.get("collection_id")
        if not collection_id:
            raise ValueError("No collection_id found in node metadata.")
    
        filters = self.postgres_manager.get_collection_filter(user_id=collection_id)
        logger.debug(filters)
        actual_top_k = self._check_similarity_top_k(similarity_top_k=similarity_top_k)
        return VectorIndexRetriever(index=self.index,
                                    filters=filters,
                                    similarity_top_k=actual_top_k,
                                    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                                    embed_model=Settings.embed_model)

    def setup_bm25_retriever(self, similarity_top_k: int = 10) -> BM25Retriever:
        """
        Sets up a BM25 retriever based on the indexed nodes.

        Args:
            similarity_top_k (int): Number of top similar results to retrieve.

        Returns:
            BM25Retriever: Configured BM25 retriever instance.
        """
        collection_id = self.nodes[0].metadata.get("collection_id")
        filtered_nodes = [
            node for node in self.nodes
            if node.metadata.get("collection_id") == collection_id
        ]
        actual_top_k = self._check_similarity_top_k(similarity_top_k=similarity_top_k)
        return BM25Retriever.from_defaults(nodes=filtered_nodes, similarity_top_k=actual_top_k, language='ru')

    def setup_fusion_retriever(self, similarity_top_k=10) -> QueryFusionRetriever:
        """
        Sets up a fusion retriever that combines vector and BM25 retrievers
        using reciprocal rank fusion for improved ranking.
        """
        individual_top_k = min(similarity_top_k * 2, 20)

        vector_retriever = self.setup_vector_retriever(similarity_top_k=individual_top_k)
        bm25_retriever = self.setup_bm25_retriever(similarity_top_k=individual_top_k)

        # Use QueryFusionRetriever с Reciprocal Rank Fusion
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=1, # use original query without extension
            mode="reciprocal_rerank", # Algorithm RRF for fusion
            use_async=True,
            verbose=True
        )  
        logger.info("Fusion retriever setup completed with RRF ranking")
        return fusion_retriever

    async def build_query_engine(self) -> RetrieverQueryEngine:
        """
        Async method: builds the query engine.
        """
        prompt_template = await PromptConstructor().change_llama_index_prompt(path=settings.rag_prompt_path, prompt=self.system_prompt)

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
        logger.debug(prompt_template)
        logger.info("Hybrid query engine setup completed")
        return engine

    # ТОЛЬКО ДЛЯ ТЕСТИРОАВНИЯ ДАТАСЕТОВ

    # async def evaluate_retrieval_advanced(self, dataset: List[Dict], k_values: List[int] = [1, 3, 5, 10]) -> Dict:
    #    """Rating search systems using your embed_model"""
    #    retriever = self.setup_fusion_retriever(similarity_top_k=max(k_values))
    #    logger.debug("Start retrieval evaluation!")
    #    evaluator = AdvancedAnswerEvaluator(
    #        dataset=dataset,
    #        similarity_threshold=0.3
    #    )
    #    results = await evaluator.evaluate_retriever(retriever, k_values)
    #    logger.debug("Getting results from evaluation")
    #    return results

    """async def aretrieve(self, query: str) -> List[TextNode]:
        #Implements the async retrieval interface expected by CourseRetrievalEvaluator.
        #Uses the fusion retriever to fetch relevant nodes.
        top_k_for_retrieval = 10  # достаточно для оценки до @10
        fusion_retriever = self.setup_fusion_retriever(similarity_top_k=top_k_for_retrieval)
        nodes = await fusion_retriever.aretrieve(query)
        return nodes

    async def evaluate_retrieval_advanced_course(self, dataset: List[Dict], k_values: List[int] = [1, 3, 5, 10]):
        evaluator = CourseRetrievalEvaluator(
            dataset=dataset,
            similarity_threshold=0.3,
            no_answer_threshold=0.1
        )
        return await evaluator.evaluate_retriever(self, k_values=k_values)"""