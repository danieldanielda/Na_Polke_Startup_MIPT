"""
BERTScore и Answer Relevancy (через эмбеддинги) требуют наличия Reference Text (эталонного ответа).
В нашем случае эталона в виде готового текста нет, есть только критерии. Поэтому мы будем использовать подход LLM-as-Judge для генерации "Идеального ответа" или сравнения с фактами из базы.
Groundedness (Обоснованность) — это проверка того, не выдумал ли бот ничего лишнего.
Это делается через сравнение ответа с retrieved контекстом (chunks из RAG).


LLM-as-Judge будет использоваться для расчета Factual Accuracy и Hallucination Rate (самые важные метрики для ТЗ).
Embeddings API будет использоваться для Answer Relevancy и Groundedness.
BERTScore останется как дополнительная семантическая метрика.
"""

import json
import os
import requests
import numpy as np
import time
import logging
from typing import List, Dict, Tuple


try:
    from bert_score import score as bert_score_calc
except ImportError:
    print("Warning: bert-score not installed. Install with: pip install bert-score")
    bert_score_calc = None


from config import EvalSettings
settings = EvalSettings()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://195.209.219.147:8000" 
DATASET_H1_PATH = "evals/eval_dataset_nl_queries.jsonl"
GOLDAPPLE_DB_PATH = "data/parser/goldapple_dataset.json" 

# Embedding API Config
EMB_API_URL = "http://172.17.0.1:8080/embed"
EMB_API_KEY = "secret"
EMB_MODEL = "BAAI/bge-m3"

# ==============================================================================
# 1. LLM-as-Judge Logic (Factual Accuracy & Hallucinations)
# ==============================================================================

def decompose_answer_into_claims(answer: str) -> List[str]:
    """Разбивает ответ на атомарные утверждения (предложения)."""
    if not answer:
        return []
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10 and not s.strip().startswith(('http', 'www'))]
    return sentences

def call_llm_judge(claim: str, reference_text: str) -> str:
    """Отправляет утверждение Судье (LLM) для вердикта."""
    prompt = f"""
    Ты — строгий эксперт-аудитор фактов.
    Проверь утверждение на соответствие Reference.
    
    Утверждение (Claim): "{claim}"
    Reference (Факты): "{reference_text}"
    
    Верни ТОЛЬКО одно слово:
    SUPPORTED (если подтверждается)
    CONTRADICTED (если противоречит)
    UNVERIFIED (если нет информации)
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    
    payload = {
        "model": settings.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    
    try:
        response = requests.post(settings.model_api, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        verdict = data['choices'][0]['message']['content'].strip().upper()
        
        if "SUPPORTED" in verdict: return "SUPPORTED"
        if "CONTRADICTED" in verdict: return "CONTRADICTED"
        return "UNVERIFIED"
    except Exception as e:
        logger.error(f"Judge API Error: {e}")
        return "ERROR"

def calculate_factual_accuracy_metrics(claims: List[str], reference_text: str) -> Dict[str, float]:
    """Считает Accuracy и Hallucination Rate через Judge."""
    if not claims:
        return {"factual_accuracy": 0.0, "hallucination_rate": 0.0, "total_claims": 0}
        
    results = []
    for claim in claims:
        verdict = call_llm_judge(claim, reference_text)
        results.append(verdict)
        # Можно раскомментировать для отладки:
        # print(f"    Claim: '{claim[:40]}...' -> {verdict}")
        
    total = len(results)
    supported = results.count("SUPPORTED")
    contradicted = results.count("CONTRADICTED")
    
    checkable = supported + contradicted
    accuracy = supported / checkable if checkable > 0 else 0.0
    hallucination_rate = contradicted / checkable if checkable > 0 else 0.0
    
    return {
        "factual_accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "total_claims": total
    }

# ==============================================================================
# 2. Embedding-based Metrics (Relevancy & Groundedness)
# ==============================================================================

def get_embeddings_api(texts: List[str]) -> np.ndarray:
    """Получает эмбеддинги через внешний API."""
    if not texts:
        return np.array([])
    if isinstance(texts, str):
        texts = [texts]
    
    clean_texts = [t if t else " " for t in texts]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {EMB_API_KEY}"}
    payload = {"inputs": clean_texts, "model": EMB_MODEL}
    
    try:
        response = requests.post(EMB_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return np.array(data)
        return np.zeros((len(clean_texts), 1024))
    except Exception as e:
        logger.error(f"Embedding API Error: {e}")
        return np.zeros((len(clean_texts), 1024))

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.size == 0 or vec2.size == 0: return 0.0
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return float(dot_product / (norm1 * norm2))

def calculate_answer_relevancy_api(question: str, answer: str) -> float:
    """Cosine Similarity between Question and Answer embeddings."""
    q_emb = get_embeddings_api([question])
    a_emb = get_embeddings_api([answer])
    if q_emb.size == 0 or a_emb.size == 0: return 0.0
    return calculate_cosine_similarity(q_emb[0], a_emb[0])

def calculate_groundedness_api(answer: str, context_chunks: List[str]) -> float:
    """Проверяет, насколько ответ обоснован контекстом."""
    if not context_chunks or not answer:
        return 0.0
        
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 15]
    if not sentences: return 0.0
        
    all_texts = sentences + context_chunks
    all_embs = get_embeddings_api(all_texts)
    
    if all_embs.size == 0: return 0.0
        
    n_sentences = len(sentences)
    sent_embs = all_embs[:n_sentences]
    chunk_embs = all_embs[n_sentences:]
    
    scores = []
    for s_emb in sent_embs:
        sims = np.array([calculate_cosine_similarity(s_emb, c_emb) for c_emb in chunk_embs])
        max_sim = np.max(sims) if sims.size > 0 else 0
        scores.append(max_sim)
        
    return float(np.mean(scores))

# ==============================================================================
# 3. BERTScore (Semantic Similarity)
# ==============================================================================

def calculate_bert_score_safe(candidates: List[str], references: List[str]) -> Dict[str, float]:
    if not bert_score_calc or not candidates or not references:
        return {"bert_f1": 0.0}
    try:
        P, R, F1 = bert_score_calc(candidates, references, lang="ru", verbose=False, device='cpu')
        return {"bert_f1": F1.mean().item()}
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return {"bert_f1": 0.0}

# ==============================================================================
# 4. Data Loading & API Calls
# ==============================================================================

def load_goldapple_db():
    if not os.path.exists(GOLDAPPLE_DB_PATH):
        return {}
    with open(GOLDAPPLE_DB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    db_index = {}
    for item in data:
        key = item.get('article') or item.get('sku')
        if key:
            db_index[str(key)] = item
    return db_index

def call_rag_system(query: str) -> Tuple[str, List[str]]:
    url = f"{API_BASE_URL}/recommend_products"
    payload = {"query": query, "collection_id": "global_collection"}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("recommendations", "")
        # Если API возвращает контекст, раскомментируй:
        # context = data.get("context_chunks", [])
        context = [] 
        return answer, context
    except Exception as e:
        logger.error(f"RAG API Error: {e}")
        return "", []

def call_baseline_llm(query: str) -> str:
    # ЗАГЛУШКА: В реальности здесь запрос к LLM без RAG
    return "Для ухода за кожей рекомендуется использовать увлажняющие средства."

# ==============================================================================
# 5. Main Experiment Runner
# ==============================================================================

def run_experiment_b_full(limit_queries: int = 5):
    print("\n=== EXPERIMENT B: FULL QUALITY EVALUATION (Judge + Embeddings + BERT) ===")
    
    if not os.path.exists(DATASET_H1_PATH):
        print("ERROR: Dataset H1 not found.")
        return
    
    queries = [json.loads(l) for l in open(DATASET_H1_PATH)]
    db_index = load_goldapple_db()
    test_queries = queries[:limit_queries]
    
    results = []
    
    for i, q in enumerate(test_queries):
        query_text = q['query']
        gt_ids = q['ground_truth_product_ids']
        
        print(f"\n[{i+1}/{len(test_queries)}] Processing: '{query_text}'")
        
        # 1. Получаем ответы
        rag_answer, rag_context = call_rag_system(query_text)
        baseline_answer = call_baseline_llm(query_text)
        
        # 2. Готовим Reference (описание продукта из GT)
        ref_text = ""
        if gt_ids and db_index:
            product = db_index.get(str(gt_ids[0]))
            if product:
                ref_text = product.get('description', '') + " " + product.get('ingredients', '')
        
        if not ref_text:
            print("  Skipping: No reference text found.")
            continue
            
        # 3. Метрики LLM-as-Judge (Только для RAG, так как Baseline не имеет доступа к фактам из DB)
        claims = decompose_answer_into_claims(rag_answer)
        judge_metrics = calculate_factual_accuracy_metrics(claims, ref_text)
        
        # 4. Метрики Embeddings (Relevancy & Groundedness)
        rel_rag = calculate_answer_relevancy_api(query_text, rag_answer)
        rel_base = calculate_answer_relevancy_api(query_text, baseline_answer)
        ground_rag = calculate_groundedness_api(rag_answer, rag_context)
        
        # 5. BERTScore (Семантическая близость к Reference)
        bert_rag = calculate_bert_score_safe([rag_answer], [ref_text])
        bert_base = calculate_bert_score_safe([baseline_answer], [ref_text])
        
        print(f"  Judge Accuracy: {judge_metrics['factual_accuracy']:.2f}")
        print(f"  Relevancy (RAG/Base): {rel_rag:.2f} / {rel_base:.2f}")
        
        results.append({
            "query": query_text,
            "judge_acc_rag": judge_metrics['factual_accuracy'],
            "hallucination_rate_rag": judge_metrics['hallucination_rate'],
            "relevancy_rag": rel_rag,
            "relevancy_base": rel_base,
            "groundedness_rag": ground_rag,
            "bert_f1_rag": bert_rag['bert_f1'],
            "bert_f1_base": bert_base['bert_f1']
        })
        
        time.sleep(2) # Пауза, чтобы не перегружать API

    # --- ИТОГОВАЯ ТАБЛИЦА ---
    if results:
        print("\n--- FINAL SUMMARY TABLE ---")
        print(f"{'Metric':<25} | {'RAG Avg':<10} | {'Baseline Avg':<10}")
        print("-" * 55)
        
        avg_judge_acc = np.mean([r['judge_acc_rag'] for r in results])
        avg_halluc = np.mean([r['hallucination_rate_rag'] for r in results])
        avg_rel_rag = np.mean([r['relevancy_rag'] for r in results])
        avg_rel_base = np.mean([r['relevancy_base'] for r in results])
        avg_ground = np.mean([r['groundedness_rag'] for r in results])
        avg_bert_rag = np.mean([r['bert_f1_rag'] for r in results])
        avg_bert_base = np.mean([r['bert_f1_base'] for r in results])
        
        print(f"{'Factual Accuracy (Judge)':<25} | {avg_judge_acc:<10.4f} | {'N/A':<10}")
        print(f"{'Hallucination Rate (Judge)':<25} | {avg_halluc:<10.4f} | {'N/A':<10}")
        print(f"{'Answer Relevancy (Emb)':<25} | {avg_rel_rag:<10.4f} | {avg_rel_base:<10.4f}")
        print(f"{'Groundedness (Emb)':<25} | {avg_ground:<10.4f} | {'N/A':<10}")
        print(f"{'BERTScore F1':<25} | {avg_bert_rag:<10.4f} | {avg_bert_base:<10.4f}")

if __name__ == "__main__":
    run_experiment_b_full()
