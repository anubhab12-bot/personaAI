from duckduckgo_search import DDGS

def fetch_duckduckgo_links(query):
    search_results = list(DDGS().text(query, max_results=3))
    links = [{"title": result["title"], "url": result["href"]} for result in search_results]
    return links