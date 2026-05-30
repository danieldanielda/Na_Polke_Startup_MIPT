import re
import numpy as np
from typing import List, Dict, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core.schema import TextNode

from src.services.embedding_inference import CustomTextEmbeddingsInference
from src.settings.config import RagSettings

settings = RagSettings()

class AdvancedAnswerEvaluator:

    def __init__(self, dataset: List[Dict], similarity_threshold: float = 0.3, no_answer_threshold: float = 0.1):
        self.dataset = dataset
        self.similarity_threshold = similarity_threshold
        self.no_answer_threshold = no_answer_threshold  # Threshold for "high confidence" in cases without response
        self.embed_model = CustomTextEmbeddingsInference(

            model_name=settings.emb_model,
            base_url=settings.emb_api,
            auth_token=settings.emb_api_key,
            timeout=60, 
            embed_batch_size=10,
        )

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text"""
        cleaned = re.sub(r'\[\d+\.\d+\]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    async def _calculate_semantic_relevance(self, retrieved_nodes: List[TextNode], ground_truth_answer: List[str]) -> Tuple[List[bool], List[float]]:
        """Calculates semantic relevance"""
        if not ground_truth_answer or not retrieved_nodes:
            return [False] * len(retrieved_nodes), [0.0] * len(retrieved_nodes)

        # We combine ground truth answers into one text
        gt_text = " ".join([self._preprocess_text(text) for text in ground_truth_answer])

        is_relevant = []
        relevance_scores = []

        # Getting an embedding for ground truth (one time)
        gt_embedding = await self._get_embedding(gt_text)
        for node in retrieved_nodes:
            node_text = self._preprocess_text(node.text)
            # Calculate semantic similarity
            similarity = await self._calculate_semantic_similarity(node_text, gt_embedding)

            is_rel = similarity >= self.similarity_threshold
            is_relevant.append(is_rel)
            relevance_scores.append(similarity)
            
        return is_relevant, relevance_scores

    async def _get_embedding(self, text: str) -> List[float]:
        """Get test embedding"""
        try:
            embeddings = await self.embed_model.aget_text_embedding(text)
            return embeddings

        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * 768
        

    async def _calculate_semantic_similarity(self, text: str, gt_embedding: List[float]) -> float:
        """Calculates the cosine similarity between text and ground truth embedding"""
        try:
            text_embedding = await self._get_embedding(text)
            similarity = cosine_similarity([text_embedding], [gt_embedding])[0][0]
            return float(similarity)

        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0


    async def evaluate_no_answer_confidence(self, retrieved_nodes: List[TextNode]) -> Dict:
        """Estimates the system's confidence for cases without a correct answer"""

        if not retrieved_nodes:
            return {
                'max_confidence': 0.0,
                'mean_confidence': 0.0,
                'has_high_confidence': False,
                'retrieved_count': 0
            }

        confidence_scores = []
        for node in retrieved_nodes:
            # Extract score from nodes
            score = getattr(node, 'score', 0.0) or getattr(node, 'similarity_score', 0.0) or 0.0
            confidence_scores.append(float(score))

        # If there are no scoring nodes, we use default values
        if not confidence_scores or all(score == 0.0 for score in confidence_scores):
            confidence_scores = [0.1] * len(retrieved_nodes)  # small base speed


        max_confidence = max(confidence_scores) if confidence_scores else 0.0
        mean_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        return {

            'max_confidence': max_confidence,
            'mean_confidence': mean_confidence,
            'has_high_confidence': max_confidence > self.no_answer_threshold,
            'retrieved_count': len(retrieved_nodes)
        }
        

    async def calculate_recall_at_k(self, retrieved_nodes: List[TextNode], ground_truth_answer: List[str], k: int) -> float:
        """Recall@k: Is there at least one relevant chunk in the top-K?"""
        if not ground_truth_answer or not retrieved_nodes:
            return 0.0

        k = min(k, len(retrieved_nodes))
        is_relevant, _ = await self._calculate_semantic_relevance(retrieved_nodes[:k], ground_truth_answer)

        return 1.0 if any(is_relevant) else 0.0


    async def calculate_mrr_at_k(self, retrieved_nodes: List[TextNode], ground_truth_answer: List[str], k: int) -> float:
        """MRR@k: inverse rank of the first relevant chunk"""
        if not ground_truth_answer or not retrieved_nodes:
            return 0.0


        k = min(k, len(retrieved_nodes))
        is_relevant, _ = await self._calculate_semantic_relevance(retrieved_nodes[:k], ground_truth_answer)
        
        for i, relevant in enumerate(is_relevant):
            if relevant:
                return 1.0 / (i + 1)

        return 0.0
    

    async def calculate_ndcg_at_k(self, retrieved_nodes: List[TextNode], ground_truth_answer: List[str], k: int) -> float:
        """NDCG@k: evaluates ranking quality based on relevance"""
        if not ground_truth_answer or not retrieved_nodes:
            return 0.0

        k = min(k, len(retrieved_nodes))
        _, relevance_scores = await self._calculate_semantic_relevance(retrieved_nodes[:k], ground_truth_answer)


        if not relevance_scores:
            return 0.0

        # Calculate DCG
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 1)


        # Calculate ideal DCG
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = ideal_scores[0]
        for i in range(1, len(ideal_scores)):
            idcg += ideal_scores[i] / np.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    async def evaluate_retriever(self, retriever, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Full evaluation of the retriever on the entire dataset"""
        results = {
            'per_query': [],
            'with_answers': {k: {'recall': [], 'mrr': [], 'ndcg': []} for k in k_values},
            'without_answers': {'confidence_stats': []},
            'statistics': {
                'total_questions': len(self.dataset),
                'questions_with_answers': 0,
                'questions_without_answers': 0,
                'total_chunks_processed': 0
            }
        }

        for item in self.dataset:
            question = item['question']
            ground_truth_answer = item.get('relevant_text', [])

            # Get results
            try:
                retrieved_nodes = await retriever.aretrieve(question)
                results['statistics']['total_chunks_processed'] += len(retrieved_nodes)

            except Exception as e:
                print(f"Error retrieving for question '{question}': {e}")
                retrieved_nodes = []

            query_result = {
                'question': question,
                'has_ground_truth': bool(ground_truth_answer),
                'retrieved_count': len(retrieved_nodes),
                'metrics': {},
                'confidence_metrics': {}  # Adding confidence metrics for all cases
            }

            if ground_truth_answer:
                results['statistics']['questions_with_answers'] += 1
                query_result['ground_truth_answer'] = ground_truth_answer

                for k in k_values:
                    recall = await self.calculate_recall_at_k(retrieved_nodes, ground_truth_answer, k)
                    mrr = await self.calculate_mrr_at_k(retrieved_nodes, ground_truth_answer, k)
                    ndcg = await self.calculate_ndcg_at_k(retrieved_nodes, ground_truth_answer, k)

                    query_result['metrics'][f'recall@{k}'] = recall
                    query_result['metrics'][f'mrr@{k}'] = mrr
                    query_result['metrics'][f'ndcg@{k}'] = ndcg
                    results['with_answers'][k]['recall'].append(recall)
                    results['with_answers'][k]['mrr'].append(mrr)
                    results['with_answers'][k]['ndcg'].append(ndcg)

                # also calculate confidence metrics for cases with responses
                confidence_metrics = await self.evaluate_no_answer_confidence(retrieved_nodes)
                query_result['confidence_metrics'] = confidence_metrics

            else:
                # Case without relevance answer
                results['statistics']['questions_without_answers'] += 1
                query_result['ground_truth_answer'] = []

                # Evaluate the system's confidence for cases without a response
                confidence_metrics = await self.evaluate_no_answer_confidence(retrieved_nodes)
                query_result['confidence_metrics'] = confidence_metrics
                results['without_answers']['confidence_stats'].append(confidence_metrics)

                # For no-answer cases, the default metrics are always 0
                for k in k_values:
                    query_result['metrics'][f'recall@{k}'] = 0.0
                    query_result['metrics'][f'mrr@{k}'] = 0.0
                    query_result['metrics'][f'ndcg@{k}'] = 0.0

            results['per_query'].append(query_result)

        # We calculate the average values ​​for cases with answers
        for k in k_values:
            if results['with_answers'][k]['recall']:
                results['with_answers'][k]['mean_recall'] = np.mean(results['with_answers'][k]['recall'])
                results['with_answers'][k]['mean_mrr'] = np.mean(results['with_answers'][k]['mrr'])
                results['with_answers'][k]['mean_ndcg'] = np.mean(results['with_answers'][k]['ndcg'])
            else:
                results['with_answers'][k]['mean_recall'] = 0.0
                results['with_answers'][k]['mean_mrr'] = 0.0
                results['with_answers'][k]['mean_ndcg'] = 0.0

        # Analyzing unanswered cases
        if results['without_answers']['confidence_stats']:
            confidences = [stats['max_confidence'] for stats in results['without_answers']['confidence_stats']]
            results['without_answers']['mean_max_confidence'] = np.mean(confidences)
            results['without_answers']['median_max_confidence'] = np.median(confidences)
            results['without_answers']['high_confidence_ratio'] = np.mean(
                [1 if stats['has_high_confidence'] else 0 for stats in results['without_answers']['confidence_stats']]
            )

        else:
            results['without_answers']['mean_max_confidence'] = 0.0
            results['without_answers']['median_max_confidence'] = 0.0
            results['without_answers']['high_confidence_ratio'] = 0.0
        return results