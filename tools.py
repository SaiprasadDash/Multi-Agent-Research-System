from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
# from rich import print
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


@tool
def scrap_url(url: str) -> str:
    """
    Fetch the content of a given URL and return cleaned, readable text
    extracted from the page. Useful for getting full page content when
    a search snippet isn't enough detail.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"

    soup = BeautifulSoup(response.content, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "iframe", "svg"]):
        tag.decompose()

    main_content = soup.find("article") or soup.find("main") or soup.body or soup
    text = main_content.get_text(separator="\n")

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned_text = "\n".join(lines)

    max_chars = 2500
    if len(cleaned_text) > max_chars:
        cleaned_text = cleaned_text[:max_chars] + "\n...[truncated]"

    return cleaned_text

result = scrap_url.invoke("https://en.wikipedia.org/wiki/Real_Madrid_CF")
# print(result)