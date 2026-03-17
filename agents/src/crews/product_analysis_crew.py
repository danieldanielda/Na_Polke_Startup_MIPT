# src/my_project/product_analysis_crew.py
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from src.tools.rag_tool import RAGQueryTool
from src.tools.sonar_general_search_tool import SonarGeneralSearchTool
from src.tools.inci_tool import InciQueryTool
from src.settings.config import CrewSettings
from src.api.v1.schemas import ProductAnalysisResponse

settings = CrewSettings()

llm = LLM(
    model=settings.model_name,
    temperature=0.0,
    api_base=settings.model_api_base,
    api_key=settings.model_api_key
)
rag_tool = RAGQueryTool()
inci_tool = InciQueryTool()
sonar_general_search_tool = SonarGeneralSearchTool()

@CrewBase
class ProductAnalysisCrew():
    """Crew for querying RAG system and analyzing product ingredients"""

    @agent
    def rag_consultant(self) -> Agent:
        return Agent(
            config=self.agents_config['rag_consultant'],
            verbose=True,
            llm=llm,
            tools=[rag_tool, sonar_general_search_tool],
        )

    @agent
    def ingredient_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['ingredient_analyzer'],
            verbose=True,
            llm=llm,
            tools=[inci_tool],
        )

    @agent
    def ingredient_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['ingredient_summarizer'],
            verbose=True,
            llm=llm,
            tools=[inci_tool], # The summarizer will also need the inci_tool to fetch ingredient descriptions
        )

    @task
    def rag_query_task(self) -> Task:
        return Task(
            config=self.tasks_config['rag_query_task'],
        )

    @task
    def ingredient_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['ingredient_analysis_task'],
            output_json=ProductAnalysisResponse
        )

    @task
    def ingredient_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config['ingredient_summary_task'],
        )

    @crew
    def crew(self, analysis_type: str = "description") -> Crew:
        tasks = [
            self.rag_query_task(),
            self.ingredient_analysis_task(),
        ]
        if analysis_type == "summary":
            tasks.append(self.ingredient_summary_task())

        return Crew(
            agents=self.agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )
