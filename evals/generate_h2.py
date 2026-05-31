"""
эталон формируется на основе знаний LLM о регуляторных базах (EWG, CIR, EU).

"""

import json
import os
import re
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
from config import EvalSettings

random.seed(2026)

GOLDAPPLE_DB_PATH = Path("data/parser/goldapple_dataset.json")
OUTPUT_DIR = Path("evals")
OUTPUT_DIR.mkdir(exist_ok=True)

settings = EvalSettings()
llm_client = OpenAI(
    api_key=settings.model_api_key,
    base_url=settings.model_api,
    timeout=60,
    max_retries=3
)

INGREDIENT_CACHE_PATH = OUTPUT_DIR / "inci_judge_cache.json"

def normalize_ingredient_name(name: str) -> str:
    """Приводит название ингредиента к каноническому виду"""
    # Убираем скобки, приводим к нижнему регистру, убираем лишние пробелы
    clean = re.split(r'\s*\(', name.strip())[0].strip().lower()
    # Убираем лишние запятые и точки в конце
    clean = clean.rstrip(',. ')
    return clean


def load_ingredient_cache() -> Dict[str, str]:
    """Загружает кэш разметки ингредиентов"""
    if INGREDIENT_CACHE_PATH.exists():
        with open(INGREDIENT_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_ingredient_cache(cache: Dict[str, str]):
    """Сохраняет кэш разметки"""
    with open(INGREDIENT_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def classify_ingredient_with_llm(ingredient_name: str, cache: Dict[str, str]) -> Optional[str]:
    """
    Классифицирует ингредиент через LLM-as-Judge на основе регуляторных знаний.
    Возвращает: safe / neutral / caution / avoid / None (если ошибка)
    """
    normalized = normalize_ingredient_name(ingredient_name)
    
    # Проверяем кэш
    if normalized in cache:
        return cache[normalized]
    
    # Промпт для независимой классификации
    prompt = f"""Ты — эксперт-косметолог и регуляторный аналитик.
Оцени уровень безопасности косметического ингредиента на основе баз:
- EU Cosmetics Regulation 1223/2009 (Annex II-VI)
- EWG Skin Deep Database
- Cosmetic Ingredient Review (CIR)

Ингредиент: "{ingredient_name}"

Верни ТОЛЬКО одно слово из четырёх:
- safe: безопасен при обычном использовании
- neutral: нейтрален, нет данных о вреде или пользе
- caution: требует осторожности (потенциальный аллерген, раздражитель)
- avoid: рекомендуется избегать (токсичен, запрещён, высокий риск)

Ответ (только одно слово):"""
    
    try:
        response = llm_client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": "Отвечай строго по инструкции. Только одно слово: safe, neutral, caution или avoid."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        verdict = response.choices[0].message.content.strip().lower()
        
        # Нормализация ответа
        if verdict in ["safe", "neutral", "caution", "avoid"]:
            cache[normalized] = verdict
            return verdict
        else:
            # Если вернул что-то другое, пробуем найти ключевое слово
            for cat in ["safe", "neutral", "caution", "avoid"]:
                if cat in verdict:
                    cache[normalized] = cat
                    return cat
            return None
            
    except Exception as e:
        print(f"⚠️  Error classifying '{ingredient_name}': {e}")
        return None


def parse_ingredients_from_text(ingredients_text: str) -> List[str]:
    """Парсит строку INCI-состава в список ингредиентов"""
    if not ingredients_text:
        return []
    
    # Разбиваем по запятой, чистим каждый ингредиент
    ingredients = []
    for ing in ingredients_text.split(','):
        cleaned = ing.strip()
        if cleaned and len(cleaned) > 2:  # Фильтруем слишком короткие
            ingredients.append(cleaned)
    
    return ingredients


def load_goldapple_products() -> List[Dict]:
    """Загружает реальные продукты из базы"""
    if not GOLDAPPLE_DB_PATH.exists():
        print(f"❌ Database not found: {GOLDAPPLE_DB_PATH}")
        return []
    
    with open(GOLDAPPLE_DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def rebuild_h2_dataset(num_products: int = 100, min_ingredients: int = 5, max_ingredients: int = 50):
    """
    Пересобирает датасет H2 с независимой LLM-разметкой.
    Собирает ровно num_products подходящих товаров.
    """
    print(f"🚀 Rebuilding H2 dataset: target {num_products} products...")
    print(f"   Criteria: {min_ingredients}-{max_ingredients} ingredients per product.")
    
    # Загружаем продукты и кэш
    products = load_goldapple_products()
    cache = load_ingredient_cache()
    
    if not products:
        print("❌ No products found")
        return
    
    # Фильтруем продукты с валидным составом и article
    valid_products = [
        p for p in products 
        if p.get('ingredients') and p.get('article')
    ]
    
    print(f"📦 Found {len(valid_products)} total products with ingredients in DB.")
    
    # Перемешиваем, чтобы выборка была случайной
    random.shuffle(valid_products)
    
    dataset = []
    total_classifications = 0
    products_scanned = 0
    
    # Идем по списку, пока не наберем нужное количество продуктов в датасет
    for product in valid_products:
        # Если набрали 100 — выходим
        if len(dataset) >= num_products:
            break
            
        products_scanned += 1
        article = product['article']
        title = product.get('title', 'Unknown')
        ingredients_text = product.get('ingredients', '')
        
        # Парсим состав
        inci_list = parse_ingredients_from_text(ingredients_text)
        
        # Фильтруем по количеству ингредиентов
        if not (min_ingredients <= len(inci_list) <= max_ingredients):
            continue # Пропускаем, если не подходит по длине
        
        # Классифицируем каждый ингредиент через LLM
        ground_truth = {}
        # Выводим прогресс: [1/100], [2/100]...
        current_idx = len(dataset) + 1
        print(f"[{current_idx}/{num_products}] Product: {title[:50]}... ({len(inci_list)} ingrs)")
        
        for ing in inci_list:
            normalized = normalize_ingredient_name(ing)
            if normalized in ground_truth:
                continue  # Дубликаты в составе
            
            rating = classify_ingredient_with_llm(ing, cache)
            if rating:
                ground_truth[normalized] = rating
                total_classifications += 1
            else:
                # Если не удалось классифицировать — пропускаем ингредиент
                # print(f"   ⚠️  Skipped unclassified: {ing}")
                pass
        
        # Сохраняем кэш периодически (каждые 10 продуктов)
        if current_idx % 10 == 0:
            save_ingredient_cache(cache)
            time.sleep(1)  # Rate limiting для API
        
        if ground_truth:  # Добавляем только если есть хотя бы одна классификация
            dataset.append({
                "product_id": article,  # Используем реальный article
                "product_name": title,
                "inci_list": inci_list,  # Оригинальный список для отправки агенту
                "ground_truth_categories": ground_truth,  # Независимая разметка
                "annotator_1": "LLM-as-Judge (EWG/CIR/EU)",
                "source": "goldapple_real_products",
                "num_ingredients": len(ground_truth)
            })
        
        # Небольшая пауза, чтобы не спамить API слишком часто
        time.sleep(0.2)
    
    # Если прошли всю базу, но не набрали 100
    if len(dataset) < num_products:
        print(f"⚠️  Warning: Only found {len(dataset)} suitable products in the database.")
    
    # Сохраняем финальный кэш
    save_ingredient_cache(cache)
    
    # Сохраняем датасет
    output_path = OUTPUT_DIR / "eval_dataset_inci.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Статистика
    print(f"\n✅ Saved {len(dataset)} products to {output_path}")
    print(f"📊 Total ingredient classifications: {total_classifications}")
    
    # Распределение классов
    if dataset:
        class_counts = {"safe": 0, "neutral": 0, "caution": 0, "avoid": 0}
        for record in dataset:
            for cat in record["ground_truth_categories"].values():
                class_counts[cat] = class_counts.get(cat, 0) + 1
        
        print(f"📈 Class distribution:")
        for cat, count in class_counts.items():
            pct = count / total_classifications * 100 if total_classifications > 0 else 0
            print(f"   {cat}: {count} ({pct:.1f}%)")
    
    # Пример записи
    if dataset:
        sample = dataset[0]
        print(f"\n📋 Sample entry:")
        print(f"   Product: {sample['product_name']}")
        print(f"   Article: {sample['product_id']}")
        print(f"   Ingredients: {sample['inci_list'][:5]}...")
        print(f"   Ground Truth (first 5): {dict(list(sample['ground_truth_categories'].items())[:5])}")


if __name__ == "__main__":
    rebuild_h2_dataset(num_products=100, min_ingredients=5, max_ingredients=50)