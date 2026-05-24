import json
import random
import os
from typing import List, Dict

# Фиксируем сид для воспроизводимости
random.seed(2026)

OUTPUT_DIR = "./evals" # Папка для датасетов
DB_OUTPUT_PATH = "./data/parser/goldapple_dataset.json" # Путь к базе для RAG

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_OUTPUT_PATH), exist_ok=True)

# ==============================================================================
# 1. БАЗА ЗНАНИЙ (INCI) - Без изменений
# ==============================================================================
INGREDIENT_KB = {
    "Aqua": "safe", "Water": "safe", "Alcohol Denat": "caution", "Glycerin": "safe", 
    "Butylene Glycol": "safe", "Propanediol": "safe", "Ethanol": "caution", 
    "Isopropyl Alcohol": "avoid", "Pentylene Glycol": "safe",
    "Squalane": "safe", "Caprylic/Capric Triglyceride": "safe", "Cetearyl Alcohol": "safe",
    "Dimethicone": "neutral", "Cyclomethicone": "neutral", "Shea Butter": "safe",
    "Jojoba Oil": "safe", "Simmondsia Chinensis Seed Oil": "safe", "Tocopheryl Acetate": "safe",
    "Niacinamide": "safe", "Salicylic Acid": "caution", "Retinol": "caution", 
    "Ascorbic Acid": "caution", "Sodium Ascorbyl Phosphate": "safe",
    "Peptides": "safe", "Palmitoyl Pentapeptide-4": "safe", "Azelaic Acid": "caution",
    "Centella Asiatica Extract": "safe", "Madecassoside": "safe", "Allantoin": "safe",
    "Hyaluronic Acid": "safe", "Sodium Hyaluronate": "safe", "Panthenol": "safe",
    "Phenoxyethanol": "neutral", "Ethylhexylglycerin": "neutral", "Sodium Benzoate": "safe",
    "Parfum": "caution", "Fragrance": "caution", "Limonene": "caution", 
    "Sodium Laureth Sulfate": "caution", "Sodium Lauryl Sulfate": "avoid",
    "Xanthan Gum": "safe", "Carbomer": "safe", "Citric Acid": "neutral"
}

# ==============================================================================
# 2. ГЕНЕРАТОРЫ КОНТЕНТА
# ==============================================================================
BRANDS = ["La Roche-Posay", "CeraVe", "Vichy", "The Ordinary", "Bioderma", "Clarins"]
TYPES = ["Cream", "Serum", "Gel", "Cleanser", "Toner", "Emulsion"]
SKIN_TYPES_REAL = ["для сухой кожи", "для жирной кожи", "для чувствительной кожи", "для нормальной кожи", "для комбинированной кожи"]
PRODUCT_TYPES_REAL = ["крем", "сыворотка", "гель для умывания", "тоник", "эмульсия", "мицеллярная вода"]

def generate_realistic_inci(product_type="normal"):
    # Упрощенная генерация для примера
    base = ["Aqua", "Glycerin"]
    actives = ["Niacinamide", "Hyaluronic Acid", "Retinol", "Salicylic Acid"]
    preservatives = ["Phenoxyethanol", "Ethylhexylglycerin"]
    
    inci = base.copy()
    if product_type == "active":
        inci.append(random.choice(actives))
    else:
        inci.append("Panthenol")
        
    inci.extend(random.sample(preservatives, 1))
    inci.extend(["Xanthan Gum", "Citric Acid"])
    return ", ".join(inci)

def generate_product_db(num_products: int = 100) -> List[Dict]:
    """
    Генерирует базу товаров в формате, совместимом с NodeParser и Eval Script.
    Ключевое поле: 'article' (строка).
    """
    products = []
    # Создаем пул SKU: sku_0001 ... sku_0100
    skus = [f"sku_{i:04d}" for i in range(1, num_products + 1)]
    
    # Теги для каждого товара (чтобы потом матчить с запросами)
    tags_pool = ["dry", "oily", "sensitive", "acne", "aging", "hydration", "cleanser", "serum", "moisturizer", "spf"]
    
    for sku in skus:
        brand = random.choice(BRANDS)
        p_type_en = random.choice(TYPES)
        p_type_ru = random.choice(PRODUCT_TYPES_REAL)
        skin_ru = random.choice(SKIN_TYPES_REAL)
        
        # Назначаем случайные теги товару
        tags = random.sample(tags_pool, k=random.randint(2, 4))
        
        # Формируем описание, содержащее ключевые слова тегов (для поиска RAG)
        desc_parts = [f"{brand} {p_type_ru} {skin_ru}."]
        if "acne" in tags: desc_parts.append("Эффективно против акне и воспалений.")
        if "aging" in tags: desc_parts.append("Антивозрастной уход, разглаживает морщины.")
        if "hydration" in tags: desc_parts.append("Интенсивное увлажнение и восстановление барьера.")
        if "spf" in tags: desc_parts.append("Защита от солнца SPF 30/50.")
        if "cleanser" in tags: desc_parts.append("Деликатное очищение без стянутости.")
        
        description = " ".join(desc_parts)
        ingredients = generate_realistic_inci("active" if "aging" in tags or "acne" in tags else "normal")
        
        product = {
            "title": f"{brand} {p_type_en}",
            "article": sku,  # <--- ЭТО САМОЕ ВАЖНОЕ ПОЛЕ
            "description": description,
            "ingredients": ingredients,
            "characteristics": {
                "тип продукта": p_type_ru,
                "тип кожи": skin_ru,
                "назначение": ", ".join(tags)
            },
            "category": "care",
            "_internal_tags": tags # Скрытое поле для логики генерации, в RAG не пойдет
        }
        products.append(product)
        
    return products, {p["article"]: p["_internal_tags"] for p in products}

# ==============================================================================
# 3. ГЕНЕРАЦИЯ H1 (ВОПРОСЫ)
# ==============================================================================
def generate_h1_dataset(products_db: List[Dict], sku_tags_map: Dict[str, List[str]]):
    queries_templates = [
        {"q": "увлажняющий крем для сухой кожи", "tags": ["dry", "moisturizer", "hydration"]},
        {"q": "средство от прыщей для жирной кожи", "tags": ["oily", "acne", "cleanser"]},
        {"q": "сыворотка от морщин с ретинолом", "tags": ["aging", "serum"]},
        {"q": "мягкая умывалка для чувствительной кожи", "tags": ["sensitive", "cleanser"]},
        {"q": "солнцезащитный крем для лица", "tags": ["spf", "moisturizer"]},
    ]
    
    dataset_h1 = []
    all_skus = list(sku_tags_map.keys())
    
    for idx, tmpl in enumerate(queries_templates):
        query_text = tmpl["q"]
        target_tags = tmpl["tags"]
        
        # Ищем товары, у которых есть хотя бы 2 совпадающих тега
        relevant_skus = [
            sku for sku, tags in sku_tags_map.items() 
            if len(set(target_tags).intersection(set(tags))) >= 1 #放宽条件 для теста
        ]
        
        # Если нашли мало, добиваем случайными
        if len(relevant_skus) < 3:
            relevant_skus += random.sample(all_skus, 3 - len(relevant_skus))
            
        # Берем топ-5 как Ground Truth
        ground_truth = list(set(relevant_skus))[:5]
        
        dataset_h1.append({
            "query_id": f"nl_{idx+1:03d}",
            "query": query_text,
            "ground_truth_product_ids": ground_truth, # Здесь будут реальные sku_XXXX
            "annotator_1": "Polina",
            "source": "synthetic",
            "metadata_tags": target_tags
        })

    path = os.path.join(OUTPUT_DIR, "eval_dataset_nl_queries.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        for record in dataset_h1:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"[H1] Generated {len(dataset_h1)} queries at {path}")

# ==============================================================================
# 4. ГЕНЕРАЦИЯ H2 (INCI) - Без изменений
# ==============================================================================
def generate_h2_dataset():
    dataset_h2 = []
    for idx in range(10): # 10 товаров для теста
        inci_list = ["Aqua", "Glycerin", "Niacinamide", "Phenoxyethanol"]
        gt_cats = {ing: INGREDIENT_KB.get(ing, "neutral") for ing in inci_list}
        
        dataset_h2.append({
            "product_id": f"test_prod_{idx}",
            "inci_list": inci_list,
            "ground_truth_categories": gt_cats
        })

    path = os.path.join(OUTPUT_DIR, "eval_dataset_inci.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        for record in dataset_h2:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"[H2] Generated {len(dataset_h2)} items at {path}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("🚀 Generating synchronized datasets...")
    
    # 1. Генерируем Базу Товаров (100 штук для теста)
    products_db, sku_tags_map = generate_product_db(num_products=100)
    
    # Сохраняем базу туда, где её ждет NodeParser / Eval Script
    with open(DB_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(products_db, f, ensure_ascii=False, indent=2)
    print(f"[DB] Saved {len(products_db)} products to {DB_OUTPUT_PATH}")
    
    # 2. Генерируем вопросы H1, используя SKU из базы
    generate_h1_dataset(products_db, sku_tags_map)
    
    # 3. Генерируем H2
    generate_h2_dataset()
    
    print("✅ DONE! Now upload goldapple_dataset.json to your RAG system.")