"""
Адаптирует датасет H1 под реальную базу goldapple_dataset.json.
Заменяет синтетические ground_truth_product_ids на реальные article ID.
"""

import json
from pathlib import Path
import re
import os
import random
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "data" / "parser" / "goldapple_dataset.json"
H1_INPUT_PATH = PROJECT_ROOT / "evals" / "eval_dataset_nl_queries.jsonl"
H1_OUTPUT_PATH = PROJECT_ROOT / "evals" / "eval_dataset_nl_queries.jsonl"

STOP_WORDS = {
    "для", "кожи", "лица", "тела", "глаз", "с", "от", "и", "в", "на", "по", 
    "без", "под", "над", "крем", "средство", "гель", "сыворотка", "тоник",
    "умывалка", "очищающее", "защитный", "ночной", "дневной", "мягкий", "нежный"
}

random.seed(42)


def load_db(path: str) -> List[Dict]:
    print(f"📦 Loading DB from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} products")
    return data


def load_h1(path: str) -> List[Dict]:
    print(f"📝 Loading H1 queries from {path}...")
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    print(f"✅ Loaded {len(queries)} queries")
    return queries


def extract_keywords(text: str) -> List[str]:
    words = re.findall(r'[а-яёa-z]{4,}', text.lower(), re.I)
    return [w for w in words if w not in STOP_WORDS]


def score_product(query: str, product: Dict) -> float:
    """Считает релевантность товара запросу. Защита от None в полях."""
    query_keywords = set(extract_keywords(query))
    if not query_keywords:
        return 0.0
    
    # Безопасное извлечение: (value or '') превращает None в пустую строку
    title = (product.get('title') or '').lower()
    desc = (product.get('description') or '').lower()
    ingredients = (product.get('ingredients') or '').lower()
    
    searchable_text = f"{title} {desc} {ingredients}"
    
    score = 0.0
    for kw in query_keywords:
        if kw in title:
            score += 3.0  # Название в приоритете
        elif kw in searchable_text:
            score += 1.0
            
    # Бонус за категорию
    category = (product.get('category') or '').lower()
    if category and any(kw in category for kw in query_keywords):
        score += 2.0
        
    # Бонус за характеристики
    chars = product.get('characteristics') or {}
    if isinstance(chars, dict):
        chars_text = ' '.join(str(v).lower() for v in chars.values() if v)
        for kw in query_keywords:
            if kw in chars_text:
                score += 1.5
                
    return score


def find_ground_truth(query: str, db: List[Dict], top_k: int = 5) -> List[str]:
    """Находит топ-K релевантных товаров и возвращает их article ID"""
    scored = []
    
    for prod in db:
        article = prod.get('article')
        if not article:  # Пропускаем товары без артикула
            continue
            
        score = score_product(query, prod)
        if score > 0:
            scored.append((score, str(article)))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [article for score, article in scored[:top_k]]


def adapt_h1_dataset(db: List[Dict], queries: List[Dict]) -> List[Dict]:
    adapted = []
    
    for i, q in enumerate(queries):
        query_text = q.get('query', '')
        gt_ids = find_ground_truth(query_text, db, top_k=5)
        
        if not gt_ids:
            # Fallback: случайные товары, если эвристика не нашла совпадений
            random_articles = [p.get('article') for p in random.sample(db, 5) if p.get('article')]
            gt_ids = [a for a in random_articles if a][:5]
            print(f"  ⚠️  [{i+1}/{len(queries)}] No keyword matches for '{query_text[:40]}...', using random GT")
        else:
            print(f"  ✅ [{i+1}/{len(queries)}] '{query_text[:40]}...' -> GT: {gt_ids[:2]}")
        
        new_q = q.copy()
        new_q['ground_truth_product_ids'] = gt_ids
        new_q['source'] = 'adapted_real_db'
        adapted.append(new_q)
    
    return adapted


def main():
    print("🚀 Starting H1 dataset adaptation for real GoldApple DB\n")
    db = load_db(DB_PATH)
    h1_queries = load_h1(H1_INPUT_PATH)
    
    print(f"\n🔍 Adapting {len(h1_queries)} queries...")
    adapted_h1 = adapt_h1_dataset(db, h1_queries)
    
    os.makedirs(os.path.dirname(H1_OUTPUT_PATH), exist_ok=True)
    with open(H1_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in adapted_h1:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n💾 Adapted dataset saved to: {H1_OUTPUT_PATH}")
    print("✅ Done! Update DATASET_H1_PATH in your eval script to this new file.")


if __name__ == "__main__":
    main()