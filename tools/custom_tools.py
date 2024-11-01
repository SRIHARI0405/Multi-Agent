from langchain_core.tools import BaseTool
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
import os
from typing import Any, Optional, Union

class TavilySearchTool(BaseTool):
    name: str = "tavily_search"  
    description: str = "Search the internet for current information"
    client: Optional[Any] = None

    def __init__(self):
        super().__init__()
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def _run(self, query: str) -> str:
        search_result = self.client.search(query)
        return str(search_result)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("TavilySearchTool does not support async")

class KaggleDatasetSearchTool(BaseTool):
    name: str = "kaggle_dataset_search" 
    description: str = "Search for datasets on Kaggle"

    def _run(self, query: str) -> str:
        url = f"https://www.kaggle.com/datasets?search={query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        datasets = soup.find_all('div', class_='dataset-card')
        results = []
        for dataset in datasets[:5]:
            title = dataset.find('div', class_='dataset-card-title').text.strip()
            link = "https://www.kaggle.com" + dataset.find('a')['href']
            results.append(f"- [{title}]({link})")
        return "\n".join(results)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("KaggleDatasetSearchTool does not support async")

class HuggingFaceSearchTool(BaseTool):
    name: str = "huggingface_search"
    description: str = "Search for models and datasets on HuggingFace"

    def _run(self, query: str) -> str:
        url = f"https://huggingface.co/api/models?search={query}"
        response = requests.get(url)
        return self._format_results(response.json())

    def _format_results(self, results: Union[list, dict]) -> str:
        formatted_results = []
        items = results if isinstance(results, list) else results.get('items', [])
        for item in items[:5]:
            name = item.get('modelId', 'Unknown')
            url = f"https://huggingface.co/{name}"
            formatted_results.append(f"- [{name}]({url})")
        return "\n".join(formatted_results)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("HuggingFaceSearchTool does not support async")

class GitHubSearchTool(BaseTool):
    name: str = "github_search"  
    description: str = "Search for repositories and code examples on GitHub"

    def _run(self, query: str) -> str:
        headers = {}
        if github_token := os.getenv("GITHUB_TOKEN"):
            headers["Authorization"] = f"token {github_token}"
        
        url = f"https://api.github.com/search/repositories?q={query}"
        response = requests.get(url, headers=headers)
        return self._format_results(response.json())

    def _format_results(self, results: dict) -> str:
        formatted_results = []
        for item in results.get('items', [])[:5]:
            name = item.get('full_name', 'Unknown')
            url = item.get('html_url', '')
            description = item.get('description', 'No description')
            formatted_results.append(f"- [{name}]({url}): {description}")
        return "\n".join(formatted_results)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("GitHubSearchTool does not support async")