# node_parser.py
import json
import uuid
from typing import List
from pathlib import Path
from datetime import datetime, timezone
from llama_index.core.schema import TextNode, Document
import logging

from src.services.chroma_manager import GLOBAL_COLLECTION_NAME

logger = logging.getLogger(__name__)

class NodeParser:
    """
    Парсер для JSON-файлов с косметикой.
    """
    async def aload_documents(self, file_paths: List[str]) -> List[Document]:
        documents = []
        current_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        for file_path in file_paths:
            if not file_path.endswith(".json"):
                logger.warning(f"Skipping non-JSON file: {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                items = data if isinstance(data, list) else [data]
                file_name = Path(file_path).name
                
                for obj in items:
                    text_parts = []

                    if title := obj.get("title"):
                        text_parts.append(f"Название: {title}")


                    if desc := obj.get("description"):
                        text_parts.append(f"Описание: {desc}")

                    if ing := obj.get("ingredients"):
                        text_parts.append(f"Состав: {ing}")

                    if chars := obj.get("characteristics"):
                        if product_type := chars.get("тип продукта"):
                            text_parts.append(f"Тип продукта: {product_type}")
                        if skin_type := chars.get("тип кожи"):
                            text_parts.append(f"Тип кожи: {skin_type}")

                    full_text = "\n".join(text_parts) or json.dumps(obj, ensure_ascii=False)

                    doc = Document(
                        text=full_text,
                        id_=str(uuid.uuid4()),
                        metadata={
                            "source_file": file_path,
                            "file_name": file_name,
                            "creation_date": current_date_utc,
                            "object_type": "cosmetic_product",
                            "product_name": obj.get("title", ""),
                            "product_type": obj.get("characteristics", {}).get("тип продукта", ""),
                            "skin_type": obj.get("characteristics", {}).get("тип кожи", ""),
                            "article": obj.get("article", "")
                        }
                    )
                    documents.append(doc)
                    logger.debug(f"Document created with metadata: {doc.metadata}")

            except Exception as e:
                logger.error(f"Failed to load JSON {file_path}: {e}")
                continue

        return documents

    async def acreate_json_nodes(self, documents: List[Document]) -> List[TextNode]:
        """
        Создаёт ноды для JSON с косметикой (без чанкинга).
        ВАЖНО: ChromaDB принимает только простые типы в метаданных (str, int, float, None).
        """
        collection_id = GLOBAL_COLLECTION_NAME
        nodes = []
        
        for doc in documents:
            # Фильтруем метаданные, оставляя только простые типы
            safe_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, type(None))):
                    safe_metadata[key] = value
                else:
                    # Конвертируем сложные типы в строки
                    safe_metadata[key] = str(value)
            
            node = TextNode(
                text=doc.text,
                id_=str(uuid.uuid4()),
                metadata={
                    "parent_id": doc.id_,
                    "collection_id": collection_id,
                    "object_type": "cosmetic_product",
                    **safe_metadata
                },
                # Исключаем сложные метаданные из эмбеддинга и LLM
                excluded_embed_metadata_keys=["parent_id", "collection_id", "source_file"],
                excluded_llm_metadata_keys=["collection_id", "parent_id", "source_file"]
            )
            nodes.append(node)
            logger.debug(f"Node created with safe_metadata: {safe_metadata}")
        
        return nodes