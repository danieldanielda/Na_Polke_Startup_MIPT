import json
from crewai.tools import BaseTool

from src.settings.config import CrewSettings

settings = CrewSettings()

class InciQueryTool(BaseTool):
    name: str = "Inci Query Tool"
    description: str = "A tool to query INCI files for ingredient information, aggregating results from multiple sources."

    def _run(self, ingredient_name: str) -> str:
        results = []

        # Search in incibeauty_ingredients_full_most.json first
        try:
            with open(settings.most_used_inci_path, 'r', encoding='utf-8') as f:
                most_used_data = json.load(f)
            for ingredient in most_used_data:
                if ingredient['inci_name'].lower() == ingredient_name.lower():
                    results.append(ingredient)
        except FileNotFoundError:
            pass # Handle case where file might not exist

        # Search in inci.json
        try:
            with open(settings.inci_path, 'r', encoding='utf-8') as f:
                inci_data = json.load(f)
            for ingredient in inci_data:
                if ingredient['inci_name'].lower() == ingredient_name.lower():
                    results.append(ingredient)
        except FileNotFoundError:
            pass # Handle case where file might not exist

        if results:
            return json.dumps(results, ensure_ascii=False, indent=2)
        return "Ingredient not found in any INCI file."
