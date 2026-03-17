#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INCI Beauty Full Scraper - OPTIMIZED VERSION
Параллельный парсинг с ретраями и очередью повторных попыток
"""

import json
import time
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm  # pip install tqdm

# ==================== НАСТРОЙКИ ====================
HTML_FILE = "table.html"
OUTPUT_FILE = "incibeauty_ingredients_full.json"
BASE_URL = "https://incibeauty.com"

# Параллелизм
MAX_WORKERS = 5  # Количество потоков (не ставьте >10, чтобы не блокировали)
DELAY_BETWEEN_REQUESTS = 0.5  # Задержка внутри потока (сек)
TIMEOUT = 30

# Ретраи
MAX_RETRIES = 3  # Максимум попыток на один URL
RETRY_BACKOFF = 2  # Множитель задержки между ретраями (экспоненциальный)
INITIAL_RETRY_DELAY = 1  # Начальная задержка перед первым ретраем (сек)

# Маппинг рейтингов
RATING_MAP = {
    'inci_vert.png': 'safe',
    'inci_jaune.png': 'neutral',
    'inci_orange.png': 'caution',
    'inci_orange_4.png': 'caution',
    'inci_rouge.png': 'avoid',
}

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Глобальная блокировка для потокобезопасного логгирования прогресса
progress_lock = threading.Lock()


def create_session() -> requests.Session:
    """Создаёт сессию с retry-логикой на уровне HTTP"""
    session = requests.Session()
    retry = Retry(
        total=2,  # 2 автоматических ретрая на уровне requests
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504, 403],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,ru;q=0.8',
        'Connection': 'keep-alive',
    })
    return session


def extract_rating(img_src: str) -> str:
    """Определяет рейтинг по имени изображения"""
    if not img_src:
        return 'unknown'
    for img_name, rating in RATING_MAP.items():
        if img_name in img_src:
            return rating
    return 'unknown'


def parse_table_from_file(filepath: str) -> List[Dict[str, str]]:
    """Парсит HTML-файл и извлекает список ингредиентов с ссылками"""
    ingredients_list = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    table = soup.find('table', class_='table-inci')
    
    if not table:
        logger.error("❌ Таблица не найдена в файле")
        return ingredients_list
    
    rows = table.find_all('tr')
    logger.info(f"📋 Найдено строк в таблице: {len(rows)}")
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 2:
            continue
        
        rating_img = cells[0].find('img')
        rating = 'unknown'
        if rating_img and rating_img.get('src'):
            rating = extract_rating(rating_img['src'])
        
        link = cells[1].find('a')
        if not link or not link.get('href'):
            continue
        
        inci_name = link.get_text(strip=True)
        href = link['href'].strip()
        
        common_name = ""
        if len(cells) >= 3:
            common_name = cells[2].get_text(strip=True)
        
        ingredients_list.append({
            'inci_name': inci_name,
            'common_name': common_name,
            'url': href if href.startswith('http') else urljoin(BASE_URL, href),
            'rating_preview': rating
        })
    
    logger.info(f"✅ Извлечено {len(ingredients_list)} ингредиентов из таблицы")
    return ingredients_list


def scrape_ingredient_with_retries(
    item: Dict[str, str], 
    session: requests.Session,
    max_retries: int = MAX_RETRIES
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Парсит ингредиент с ретраями.
    Возвращает: (данные_или_None, успех)
    """
    url = item['url']
    inci_name = item['inci_name']
    
    for attempt in range(1, max_retries + 1):
        try:
            # Задержка между попытками (экспоненциальная)
            if attempt > 1:
                delay = INITIAL_RETRY_DELAY * (RETRY_BACKOFF ** (attempt - 2))
                logger.debug(f"🔄 Ретрай #{attempt} для {inci_name} через {delay:.1f}с")
                time.sleep(delay)
            
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # === Парсинг данных ===
            h1 = soup.find('h1')
            inci_name_parsed = h1.get_text(strip=True) if h1 else item['inci_name']
            
            cas_number = ""
            cas_match = re.search(r'CAS number:\s*([^\n<]+)', soup.get_text())
            if cas_match:
                cas_number = cas_match.group(1).strip()
            
            einecs_number = ""
            for li in soup.find_all('li'):
                text = li.get_text(strip=True)
                if 'EINECS' in text or 'ELINCS' in text:
                    parts = text.split(':', 1)
                    if len(parts) > 1:
                        einecs_number = parts[1].strip()
                    break
            
            rating = 'unknown'
            img = soup.find('img', class_='inci-fleur')
            if img and img.get('src'):
                rating = extract_rating(img['src'])
            
            functions = []
            func_section = soup.find('strong', string=lambda t: t and 'function' in t.lower())
            if func_section:
                func_container = func_section.find_parent('div')
                if func_container:
                    func_list = func_container.find('ul', class_='fonctions-inci')
                    if func_list:
                        for li in func_list.find_all('li'):
                            func_text = li.get_text(strip=True)
                            func_name = func_text.split(':')[0].strip() if ':' in func_text else func_text
                            if func_name and func_name not in functions:
                                functions.append(func_name)
            
            origin = ""
            classification = ""
            for li in soup.find_all('li'):
                strong = li.find('strong')
                if not strong:
                    continue
                label = strong.get_text(strip=True).lower()
                value_elem = strong.find_next_sibling()
                value = value_elem.get_text(strip=True) if value_elem else ""
                if 'origin' in label:
                    origin = value
                elif 'classification' in label:
                    classification = value
            
            name = inci_name_parsed.title() if inci_name_parsed else ""
            description_parts = [f"{name} — косметический ингредиент."]
            if origin:
                description_parts.append(f"Происхождение: {origin}.")
            
            note_parts = []
            if rating == 'safe':
                note_parts.append("Считается безопасным для использования в косметике.")
            elif rating == 'neutral':
                note_parts.append("Low penalty in all categories.")
            elif rating == 'caution':
                note_parts.append("Требует осторожности при использовании.")
            elif rating == 'avoid':
                note_parts.append("Рекомендуется избегать в косметических продуктах.")
            if classification.lower() == 'regulated':
                note_parts.append("Ингредиент регулируется в ЕС и Великобритании.")
            
            note = " ".join(note_parts) if note_parts else "Данных недостаточно для оценки."
            
            return {
                "name": name,
                "inci_name": inci_name_parsed.upper() if inci_name_parsed else "",
                "cas_number": cas_number,
                "einecs_number": einecs_number,
                "description": " ".join(description_parts),
                "rating": rating,
                "functions": functions if functions else ["Не указано"],
                "prevalence": "Данные недоступны",
                "common_product_types": [],
                "note": note
            }, True
            
        except requests.exceptions.RequestException as e:
            status_code = getattr(e.response, 'status_code', 'N/A') if hasattr(e, 'response') else 'N/A'
            logger.warning(f"⚠️ [{attempt}/{max_retries}] {inci_name}: HTTP {status_code} - {e}")
            if attempt == max_retries:
                logger.error(f"❌ Не удалось после {max_retries} попыток: {url}")
                return None, False
        except Exception as e:
            logger.warning(f"⚠️ [{attempt}/{max_retries}] {inci_name}: Ошибка парсинга - {e}")
            if attempt == max_retries:
                logger.error(f"❌ Критическая ошибка после {max_retries} попыток: {url}")
                return None, False
    
    return None, False


def worker_task(item: Dict, session: requests.Session, delay: float) -> Tuple[Dict, Optional[Dict], bool]:
    """Задача для воркера: парсинг + задержка"""
    result = scrape_ingredient_with_retries(item, session)
    time.sleep(delay)  # Вежливость к серверу
    return item, result[0], result[1]


def scrape_all_parallel(
    ingredients_list: List[Dict], 
    max_workers: int = MAX_WORKERS
) -> Tuple[List[Dict], List[Dict]]:
    """
    Параллельный сбор данных с очередью ретраев.
    Возвращает: (успешные, неудачные)
    """
    successful = []
    failed = []
    
    # Создаём сессию для каждого потока (thread-safe)
    def get_session():
        return create_session()
    
    print(f"\n🚀 Запуск параллельного сбора ({max_workers} потоков, {len(ingredients_list)} ссылок)...")
    
    # Первый проход: параллельная обработка
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Создаём сессию для каждого воркера
        sessions = {i: get_session() for i in range(max_workers)}
        
        # Отправляем задачи
        future_to_item = {
            executor.submit(
                worker_task, 
                item, 
                sessions[i % max_workers], 
                DELAY_BETWEEN_REQUESTS
            ): item 
            for i, item in enumerate(ingredients_list)
        }
        
        # Прогресс-бар
        with tqdm(total=len(ingredients_list), desc="📦 Обработка") as pbar:
            for future in as_completed(future_to_item):
                item, data, success = future.result()
                
                if success and data:
                    successful.append(data)
                    with progress_lock:
                        pbar.set_postfix_str(f"✓ {data['rating']}")
                else:
                    failed.append(item)
                    with progress_lock:
                        pbar.set_postfix_str("✗ failed")
                
                pbar.update(1)
    
    # Второй проход: ретраи для неудачных
    if failed:
        print(f"\n🔄 Второй проход: ретраи для {len(failed)} неудачных запросов...")
        session = create_session()
        
        for i, item in enumerate(failed, 1):
            print(f"[{i}/{len(failed)}] Ретрай: {item['inci_name']}...", end=" ")
            data, success = scrape_ingredient_with_retries(item, session, max_retries=2)
            
            if success and data:
                successful.append(data)
                print("✓")
            else:
                print("✗")
                # Сохраняем в список окончательно неудачных для отчёта
                failed.remove(item)  # убираем из failed, добавим в отдельный список
                failed.append({**item, 'error': 'max_retries_exceeded'})
            
            time.sleep(DELAY_BETWEEN_REQUESTS * 2)  # Чуть больше задержка для ретраев
    
    return successful, failed


def save_results(successful: List[Dict], failed: List[Dict], output_file: str):
    """Сохраняет результаты и отчёт о неудачах"""
    # Основной JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(successful, f, ensure_ascii=False, indent=2)
    
    # Отчёт о неудачах (если есть)
    if failed:
        failed_file = output_file.replace('.json', '_failed.json')
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        logger.warning(f"⚠️ {len(failed)} записей не удалось собрать: {failed_file}")
    
    logger.info(f"✅ Сохранено {len(successful)} записей в {output_file}")


def main():
    """Точка входа"""
    print("🧪 INCI Beauty Scraper - OPTIMIZED")
    print(f"📁 Вход: {HTML_FILE} | Выход: {OUTPUT_FILE}")
    print(f"⚙️  Потоки: {MAX_WORKERS} | Ретраи: {MAX_RETRIES}")
    print("-" * 60)
    
    if not Path(HTML_FILE).exists():
        logger.error(f"❌ Файл {HTML_FILE} не найден!")
        return
    
    ingredients_list = parse_table_from_file(HTML_FILE)
    if not ingredients_list:
        print("❌ Не удалось извлечь ингредиенты")
        return
    
    # Запуск параллельного сбора
    successful, failed = scrape_all_parallel(ingredients_list, max_workers=MAX_WORKERS)
    
    # Сохранение
    save_results(successful, failed, OUTPUT_FILE)
    
    # Итоговая статистика
    total = len(ingredients_list)
    success_rate = len(successful) / total * 100 if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 ИТОГИ:")
    print(f"   Всего ссылок: {total}")
    print(f"   ✅ Успешно: {len(successful)} ({success_rate:.1f}%)")
    print(f"   ❌ Неудачно: {len(failed)}")
    
    if successful:
        ratings = {}
        for ing in successful:
            r = ing['rating']
            ratings[r] = ratings.get(r, 0) + 1
        print(f"\n📈 Распределение рейтингов:")
        for rating, count in sorted(ratings.items()):
            print(f"   {rating}: {count} ({count/len(successful)*100:.1f}%)")
        
        print(f"\n📄 Пример записи:")
        print(json.dumps(successful[0], ensure_ascii=False, indent=2)[:500] + "...")
    
    if failed:
        print(f"\n⚠️  Неудачные URL (проверьте {OUTPUT_FILE.replace('.json', '_failed.json')}):")
        for item in failed[:5]:  # Показать первые 5
            print(f"   • {item.get('inci_name', 'N/A')}: {item.get('url', 'N/A')}")
        if len(failed) > 5:
            print(f"   ... и ещё {len(failed) - 5}")


if __name__ == "__main__":
    main()