# RAG Consultant Agent

## Описание

RAG Consultant Agent - это агент, который использует результат от parser агента (анализ штрихкода) для запроса информации в RAG системе.

## Компоненты

### 1. RAG Tool (`src/tools/rag_tool.py`)
Инструмент для работы с RAG API. Принимает запрос в формате JSON или простую строку запроса.

### 2. RAG Consultant Crew (`rag_crew.py`)
CrewAI crew, содержащий агента `rag_consultant`, который использует RAG Tool для запросов к базе знаний.

### 3. API Endpoint (`src/api/v1/router.py`)
REST API endpoint `/rag_query` для использования RAG агента.

## Использование

### Через API

```python
import requests

# Product info from barcode parser agent
product_info = """
Product Name: L'Oreal Paris Revitalift Anti-Wrinkle Cream
Brand: L'Oreal Paris
Category: Skincare / Anti-Aging
"""

response = requests.post(
    "http://localhost:8000/api/v1/crew/rag_query",
    json={
        "product_info": product_info,
        "collection_id": "default",  # optional
        "system_prompt": "system_common_prompt"  # optional
    }
)

result = response.json()
print(result["rag_response"])
```

### Напрямую через Crew

```python
from rag_crew import RAGConsultantCrew

inputs = {
    "product_info": "Product Name: ...",
    "collection_id": "default",
    "system_prompt": "system_common_prompt"
}

result = RAGConsultantCrew().crew().kickoff(inputs=inputs)
print(result)
```

## Настройка

### Переменные окружения

Для настройки подключения к RAG API используйте переменные окружения:

- `RAG_HOST` - хост RAG сервиса (по умолчанию: `localhost`)
- `RAG_PORT` - порт RAG сервиса (по умолчанию: `8001`)

### Параметры запроса

- `product_info` (обязательный) - информация о продукте от parser агента
- `collection_id` (опционально) - ID коллекции в RAG системе (по умолчанию: `"default"`)
- `system_prompt` (опционально) - системный промпт для RAG (по умолчанию: `"system_common_prompt"`)

## Пример полного workflow

1. Получить информацию о продукте через barcode parser:
   ```python
   barcode_response = requests.post(
       "http://localhost:8000/api/v1/crew/search_barcode",
       json={"barcode": "1234567890123"}
   )
   product_info = barcode_response.json()["product_info"]
   ```

2. Использовать результат для запроса в RAG:
   ```python
   rag_response = requests.post(
       "http://localhost:8000/api/v1/crew/rag_query",
       json={
           "product_info": product_info,
           "collection_id": "default"
       }
   )
   rag_result = rag_response.json()["rag_response"]
   ```

## Структура файлов

```
agents/
├── rag_crew.py                    # RAG Consultant Crew
├── src/
│   ├── tools/
│   │   └── rag_tool.py           # RAG Query Tool
│   ├── config/
│   │   ├── agents.yaml           # Конфигурация агента rag_consultant
│   │   └── tasks.yaml            # Конфигурация задачи rag_query_task
│   └── api/
│       └── v1/
│           ├── router.py         # API endpoints
│           └── schemas.py        # Pydantic схемы
└── example_rag_usage.py          # Пример использования
```
