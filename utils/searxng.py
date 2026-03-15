import requests
import os
import ollama

SEARXNG_URL =  os.getenv('SEARXNG_URL', "http://localhost:8080/search")

def searxng_search(query, num_results=5):
    params = {
        "q": query,
        "format": "json"
    }

    res = requests.get(SEARXNG_URL, params=params)
    data = res.json()

    results = []

    for r in data.get("results", [])[:num_results]:
        results.append({
            "title": r.get("title"),
            "content": r.get("content"),
            "url": r.get("url")
        })

    return results

def should_search(question, model="mistral"):

    prompt = f"""
    Decide if the user question requires internet search.

    Search if the question involves:
    - latest news
    - current events
    - sports results
    - prices
    - weather
    - anything after 2024

    Question: {question}

    Respond ONLY with YES or NO.
    """

    result = ollama.generate(model=model, prompt=prompt)

    decision = result["response"].lower()

    return "yes" in decision