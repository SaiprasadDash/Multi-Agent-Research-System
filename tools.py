from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily_client  = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information about a topic.
    Returns the title, URL, and snippet of relevant search results.
    """

    response = tavily_client.search(
        query=query,
        # search_depth="advanced",
        topic="general",
        max_results=5,
        include_answer=False
    )

    results = response.get("results", [])

    if not results:
        return "No relevant search results found."

    output = []

    for i, result in enumerate(results, start=1):
        output.append(
            f"""
            Result {i}
            Title: {result.get("title", "N/A")}
            URL: {result.get("url", "N/A")}
            Snippet: {result.get("content", "N/A")}
            Relevance Score: {result.get("score", "N/A")}
            """
        )

    return "\n".join(output)

# print(web_search.invoke("Give latest news of auguest"))

