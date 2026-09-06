from agent import build_search_agent, build_reader_agent, writer_chain, critic_chain


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # Step 1 - search agent
    print("\n" + "=" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_result"] = search_result['messages'][-1].content
    print("\nsearch result:\n", state['search_result'])

    # Step 2 - reader agent scraping
    print("\n" + "=" * 50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""
Based on the following search results about '{topic}', identify the most
relevant and reliable URL and use the scrap_url tool to scrape it for
deeper information.

Search Results:
{state['search_result'][:800]}

After scraping the URL:
- Extract the most important facts and details related to the topic.
- Ignore irrelevant website content.
- Preserve important dates, numbers, names, and key findings.
- Mention the source URL.
- If the first URL cannot be scraped, try another relevant URL from the search results.

Return a clear and concise research summary that can be used by the writer agent.
"""
            )
        ]
    })
    state['scraped_content'] = reader_result['messages'][-1].content
    print("\nscraped content:\n", state['scraped_content'])

    # Step 3 - writer chain
    print("\n" + "=" * 50)
    print("step 3 - Writer is drafting the report ...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS : \n{state['search_result']}\n\n"
        f"DETAILED SCRAPED CONTENT : \n{state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })
    print("\nFinal report:\n", state["report"])

    # Step 4 - critic
    print("\n" + "=" * 50)
    print("step 4 - critic is reviewing the report")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })
    print("\ncritic feedback:\n", state['feedback'])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)