import json
import random
import os
import itertools

OUTPUT_DIR = "./" # Лучше хранить в папке eval
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- КОНСТАНТЫ ДЛЯ ГЕНЕРАЦИИ H2 ---

# Расширенная база знаний ингредиентов (INCI -> Category)
INGREDIENT_KB = {
    # Базовые растворители/основы
    "Aqua": "safe", "Water": "safe", "Alcohol Denat": "caution", "Glycerin": "safe", "Butylene Glycol": "safe",
    "Propanediol": "safe", "Ethanol": "caution", "Isopropyl Alcohol": "avoid",
    
    # Эмоленты и увлажнители
    "Squalane": "safe", "Caprylic/Capric Triglyceride": "safe", "Cetearyl Alcohol": "safe",
    "Dimethicone": "neutral", "Cyclomethicone": "neutral", "Shea Butter": "safe",
    "Jojoba Oil": "safe", "Hyaluronic Acid": "safe", "Sodium Hyaluronate": "safe",
    
    # Активные вещества (Anti-age/Acne)
    "Niacinamide": "safe", "Salicylic Acid": "caution", "Retinol": "caution", 
    "Retinyl Palmitate": "caution", "Ascorbic Acid": "caution", "Vitamin C": "safe",
    "Peptides": "safe", "Palmitoyl Pentapeptide-4": "safe", "Azelaic Acid": "caution",
    "Benzoyl Peroxide": "caution", "Adapalene": "caution",
    
    # Консерванты (часто вызывают вопросы)
    "Phenoxyethanol": "neutral", "Ethylhexylglycerin": "neutral", "Sodium Benzoate": "safe",
    "Potassium Sorbate": "safe", "Methylparaben": "caution", "Propylparaben": "caution",
    "Triclosan": "avoid", "Dmdm Hydantoin": "avoid", "Methylisothiazolinone": "avoid",
    "Chlorphenesin": "neutral", "Benzyl Alcohol": "neutral",
    
    # Отдушки и аллергены
    "Parfum": "caution", "Fragrance": "caution", "Limonene": "caution", 
    "Linalool": "caution", "Citronellol": "caution", "Geraniol": "caution",
    "Eugenol": "caution", "Coumarin": "caution",
    
    # ПАВы и загустители
    "Sodium Laureth Sulfate": "caution", "Sodium Lauryl Sulfate": "avoid",
    "Cocamidopropyl Betaine": "safe", "Xanthan Gum": "safe", "Carbomer": "safe"
}

# Компоненты для сборки продуктов (Combinatorial Generation)
BASES = ["Aqua", "Glycerin", "Butylene Glycol", "Propanediol"]
EMOLLIENTS = ["Squalane", "Caprylic/Capric Triglyceride", "Dimethicone", "Shea Butter"]
ACTIVES_SAFE = ["Niacinamide", "Hyaluronic Acid", "Peptides", "Panthenol"]
ACTIVES_CAUTION = ["Retinol", "Salicylic Acid", "Ascorbic Acid", "Azelaic Acid"]
PRESERVATIVES_SAFE = ["Phenoxyethanol", "Ethylhexylglycerin", "Sodium Benzoate"]
PRESERVATIVES_AVOID = ["Triclosan", "Dmdm Hydantoin", "Methylisothiazolinone"] # Для проблемных продуктов
FRAGRANCES = ["Parfum", "Limonene", "Linalool"]

def generate_unique_inci_list(product_type="normal"):
    """Генерирует уникальный список ингредиентов."""
    inci = []
    
    # 1. Основа (всегда есть)
    inci.append(random.choice(BASES))
    inci.append(random.choice(["Glycerin", "Sodium Hyaluronate"]))
    
    # 2. Эмоленты (1-2 шт)
    inci.extend(random.sample(EMOLLIENTS, random.randint(1, 2)))
    
    # 3. Активы
    if product_type == "problematic":
        inci.append(random.choice(ACTIVES_CAUTION))
        inci.append(random.choice(PRESERVATIVES_AVOID)) # Опасный консервант
    elif product_type == "active":
        inci.extend(random.sample(ACTIVES_CAUTION, 2))
        inci.append(random.choice(PRESERVATIVES_SAFE))
    else: # normal / sensitive
        inci.extend(random.sample(ACTIVES_SAFE, random.randint(1, 2)))
        inci.append(random.choice(PRESERVATIVES_SAFE))
        
    # 4. Отдушки (иногда)
    if random.random() > 0.6:
        inci.append(random.choice(FRAGRANCES))
        
    # 5. Технические добавки (загустители и т.д.)
    inci.append("Xanthan Gum")
    inci.append("Citric Acid")
    
    # Перемешиваем, чтобы порядок был как в реальном INCI (по убыванию массы примерно)
    random.shuffle(inci)
    return list(dict.fromkeys(inci)) # Убираем дубликаты, сохраняя порядок

# --- ГЕНЕРАЦИЯ H1 ---

def generate_h1_dataset():
    """
    Генерирует 50+ запросов с более сложной логикой Ground Truth.
    """
    # Расширенный пул запросов
    base_queries = [
        # Сложные/Смешанные запросы
        {"query": "увлажняющий крем для сухой кожи без отдушек", "tags": ["dry", "moisturizer", "fragrance-free"]},
        {"query": "сыворотка от пигментации с витамином с", "tags": ["brightening", "vitamin-c", "serum"]},
        {"query": "очищающее средство для чувствительной кожи розацеа", "tags": ["sensitive", "rosacea", "cleanser"]},
        {"query": "солнцезащитный крем для жирной кожи не комедогенный", "tags": ["oily", "spf", "non-comedogenic"]},
        {"query": "ночной крем с ретинолом для возрастной кожи", "tags": ["anti-age", "retinol", "night"]},
        {"query": "тоник с кислотами для проблемной кожи", "tags": ["acne", "exfoliant", "toner"]},
        {"query": "масло для снятия макияжа с водостойкой туши", "tags": ["makeup-remover", "oil", "eyes"]},
        {"query": "гель умывалка без сульфатов sls sles", "tags": ["cleanser", "sulfate-free", "gentle"]},
        {"query": "крем барьерный восстанавливающий церамиды", "tags": ["repair", "ceramides", "sensitive"]},
        {"query": "пилинг энзимный мягкий для лица", "tags": ["exfoliant", "enzyme", "gentle"]},
        
        # Сленг и опечатки (симуляция реальных логов)
        {"query": "крим от прыщей", "tags": ["acne", "moisturizer"]},
        {"query": "умывалка для жирной кожи", "tags": ["oily", "cleanser"]},
        {"query": "гиалуронка сыворотка", "tags": ["hydration", "serum"]},
        {"query": "санскрин для лица спф 50", "tags": ["spf", "face"]},
        {"query": "мицеллярка для глаз", "tags": ["cleanser", "eyes"]},
    ]
    
    # Добираем синтетикой до 50
    categories = ["cleanser", "toner", "serum", "moisturizer", "mask", "spf"]
    skin_types = ["dry", "oily", "sensitive", "normal", "combination"]
    concerns = ["acne", "aging", "pigmentation", "hydration", "redness"]
    
    synthetic_count = 0
    while len(base_queries) < 50:
        cat = random.choice(categories)
        skin = random.choice(skin_types)
        conc = random.choice(concerns)
        
        templates = [
            f"{cat} for {skin} skin with {conc}",
            f"best {cat} for {conc} on {skin} skin",
            f"{skin} skin {cat} against {conc}"
        ]
        # Простой перевод на русский для примера (в реальности лучше иметь русские шаблоны)
        ru_cat = {"cleanser": "средство для умывания", "toner": "тоник", "serum": "сыворотка", "moisturizer": "крем", "mask": "маска", "spf": "санскрин"}[cat]
        ru_skin = {"dry": "сухой", "oily": "жирной", "sensitive": "чувствительной", "normal": "нормальной", "combination": "комбинированной"}[skin]
        ru_conc = {"acne": "акне", "aging": "старения", "pigmentation": "пигментации", "hydration": "увлажнения", "redness": "покраснений"}[conc]
        
        query = f"{ru_cat} для {ru_skin} кожи от {ru_conc}"
        
        base_queries.append({
            "query": query,
            "tags": [skin, cat, conc]
        })
        synthetic_count += 1

    # Генерация пула SKU с тегами для реалистичного GT
    # В реальности это делается по базе, здесь симулируем
    all_skus = [f"sku_{i:04d}" for i in range(1, 2501)]
    # Присваиваем каждому SKU случайные теги (симуляция базы данных)
    sku_tags_map = {}
    for sku in all_skus:
        # У каждого товара есть 2-4 случайных тега
        possible_tags = ["dry", "oily", "sensitive", "normal", "cleanser", "serum", "moisturizer", "acne", "aging", "fragrance-free", "spf"]
        sku_tags_map[sku] = random.sample(possible_tags, random.randint(2, 4))

    dataset_h1 = []
    for idx, item in enumerate(base_queries):
        query_tags = item["tags"]
        
        # Находим "релевантные" SKU (у которых есть пересечение тегов)
        relevant_skus = []
        for sku, tags in sku_tags_map.items():
            # Если хотя бы 2 тега совпадают - считаем релевантным
            if len(set(query_tags).intersection(set(tags))) >= 2:
                relevant_skus.append(sku)
        
        # Если нашли мало, доберем случайных (для симуляции)
        if len(relevant_skus) < 3:
            relevant_skus += random.sample(all_skus, 3)
            
        ground_truth = list(set(relevant_skus))[:5] # Берем топ-5 релевантных

        record = {
            "query_id": f"nl_{idx+1:03d}",
            "query": item["query"],
            "ground_truth_product_ids": ground_truth,
            "annotator_1": "Polina",
            "annotator_2": "Expert_AI_Sim",
            "agreement": "full",
            "source": "synthetic" if idx >= 10 else "logs",
            "metadata_tags": query_tags
        }
        dataset_h1.append(record)

    path = os.path.join(OUTPUT_DIR, "eval_dataset_nl_queries.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        for record in dataset_h1:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"[H1] Dataset generated: {path} ({len(dataset_h1)} records)")

# --- ГЕНЕРАЦИЯ H2 ---

def generate_h2_dataset():
    """
    Генерирует 70 уникальных продуктов с помощью комбинаторики ингредиентов.
    """
    dataset_h2 = []
    
    # Типы продуктов для генерации
    product_types = ["normal"] * 40 + ["active"] * 20 + ["problematic"] * 10
    
    # Бренды для разнообразия names
    brands = ["La Roche-Posay", "CeraVe", "Vichy", "The Ordinary", "Bioderma", "Clinique", "Custom Brand"]
    
    for idx, p_type in enumerate(product_types):
        # Генерируем уникальный состав
        inci_list = generate_unique_inci_list(product_type=p_type)
        
        # Формируем Ground Truth на основе INGREDIENT_KB
        ground_truth_cats = {}
        for ing in inci_list:
            # Если ингредиент есть в базе - берем категорию, иначе neutral
            ground_truth_cats[ing] = INGREDIENT_KB.get(ing, "neutral")
            
        brand = random.choice(brands)
        prod_name = f"{brand} Product Type {p_type.upper()} #{idx+1}"
        
        record = {
            "product_id": f"sku_test_{idx+1:04d}",
            "product_name": prod_name,
            "inci_list": inci_list,
            "ground_truth_categories": ground_truth_cats,
            "annotator_1": "Polina",
            "annotator_2": "Expert_Cosmetologist_Sim",
            "kappa": 0.85,
            "sources": ["EU 1223/2009", "EWG Skin Deep", "CIR"],
            "product_type_gen": p_type # метка для внутреннего анализа
        }
        dataset_h2.append(record)

    path = os.path.join(OUTPUT_DIR, "eval_dataset_inci.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        for record in dataset_h2:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"[H2] Dataset generated: {path} ({len(dataset_h2)} records)")

if __name__ == "__main__":
    generate_h1_dataset()
    generate_h2_dataset()
    print("Datasets ready for evaluation.")