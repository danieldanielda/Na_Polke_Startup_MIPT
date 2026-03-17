from llama_index.core import PromptTemplate


from src.settings.helpers import load_prompts_from_yaml
from src.settings.config import RagSettings

settings = RagSettings()

class PromptConstructor:  

    async def change_llama_index_prompt(self, path: str) -> PromptTemplate:
        """
        Usage example:
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": new_summary_tmpl}
        )
        """
        yaml_prompts = await load_prompts_from_yaml(yaml_path=path)
        new_prompt = yaml_prompts['context_prompts'][settings.rag_system_prompt]['template']
        new_template = PromptTemplate(new_prompt)
        return new_template

    async def change_summary_prompt(self, path: str) -> str:
        """Returns summary prompt for summary case"""
        yaml_prompts = await load_prompts_from_yaml(yaml_path=path)
        new_summary_prompt = yaml_prompts['summary_prompts']['summary']['template']
        return new_summary_prompt