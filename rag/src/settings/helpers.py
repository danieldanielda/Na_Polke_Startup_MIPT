from datetime import datetime
import json
from typing import Dict, List
import yaml
from src.settings.config import RagSettings

settings = RagSettings()

async def load_prompts_from_yaml(yaml_path: str):
    """
    Loads prompts from a YAML file.
    
    Args:
        file_path (str): Path to the YAML file.
        
    Returns:
        dict: A dictionary containing the prompts
    """
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        prompts = yaml.safe_load(file)
        
    return prompts

async def load_golden_dataset(golden_path: str=settings.golden_answer_path) -> List[Dict]:
    with open(golden_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        return dataset