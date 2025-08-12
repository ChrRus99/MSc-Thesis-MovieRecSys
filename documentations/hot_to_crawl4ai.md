# Movie Search Crawler Functions Documentation

This document explains, step by step, how the two core functions—`search_movie_info` and `search_movie_reviews`—work in the Movie Search Crawler module. Each section breaks down the process and includes examples where applicable.

---

## 1. `fetch_html(url: str) -> str`

**Purpose:**
Fetches the raw HTML content of a given URL using an HTTP-only asynchronous crawler.

**Process:**
1.  Creates an instance of `AsyncWebCrawler` configured with `AsyncHTTPCrawlerStrategy`, which uses `aiohttp` under the hood (no Playwright subprocess).
2.  Calls `crawler.arun(url=url)` to perform the HTTP GET request.
3.  Returns the `html` attribute of the result, or an empty string if fetching failed.

```python
async def fetch_html(url: str) -> str:
    async with AsyncWebCrawler(
        crawler_strategy=AsyncHTTPCrawlerStrategy(browser_config=http_config)
    ) as crawler:
        result = await crawler.arun(url=url)
        return result.html if (result and hasattr(result, 'html')) else ""
```

**Example:**

```python
html = await fetch_html('https://example.com')
print(html[:200])  # Prints first 200 characters of the page
```

## 2. `extract_plot(html: str) -> str`

**Purpose:**
Parses HTML to extract the movie’s plot or synopsis text.

**Process:**

1.  Uses BeautifulSoup to parse the HTML.
2.  Searches for headings (`<h2>`, `<h3>`, `<h4>`) containing the words “plot” or “synopsis” (case-insensitive).
3.  If found, collects all `<p>` paragraphs immediately following that heading until the next heading.
4.  If no explicit section is detected, falls back to returning the first 300 words of the page’s text content.

```python
import re
from bs4 import BeautifulSoup

def extract_plot(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for header in soup.find_all(['h2', 'h3', 'h4']):
        if re.search(r'plot|synopsis', header.get_text(), re.I):
            paragraphs = []
            for sibling in header.find_next_siblings():
                if sibling.name in ['h2', 'h3', 'h4']:
                    break
                if sibling.name == 'p':
                    paragraphs.append(sibling.get_text(strip=True))
            if paragraphs:
                return "\n\n".join(paragraphs)
    # Fallback: first 300 words
    full_text = soup.get_text(separator=' ', strip=True)
    words = full_text.split()
    return ' '.join(words[:300]) + ('...' if len(words) > 300 else '')
```

**Example:**

```python
plot = extract_plot(html)
print(plot)
```

## 3. `extract_reviews(html: str, max_reviews: int = 5) -> list[str]`

**Purpose:**
Parses HTML to extract up to `max_reviews` user review snippets.

**Process:**

1.  Parses HTML with BeautifulSoup.
2.  Finds elements (`<div>`, `<section>`, `<article>`, `<p>`) where the `class` or `id` attribute contains “review”.
3.  Extracts text from each such element (minimum length 50 characters) until reaching `max_reviews`.
4.  If none are found, falls back to the first `max_reviews` `<p>` tags of at least 50 characters.

```python
from bs4 import BeautifulSoup

def extract_reviews(html: str, max_reviews: int = 5) -> list[str]:
    soup = BeautifulSoup(html, 'html.parser')
    reviews = []
    # Find elements likely containing reviews based on class/id
    for tag in soup.find_all(
        lambda tag: (tag.name in ['div', 'section', 'article', 'p']) and (
            (tag.get('class') and any('review' in c.lower() for c in tag.get('class'))) or
            (tag.get('id') and 'review' in tag.get('id').lower())
        )
    ):
        text = tag.get_text(separator=' ', strip=True)
        if len(text) > 50: # Basic filter for meaningful content
            reviews.append(text)
        if len(reviews) >= max_reviews:
            break

    # Fallback: if no specific review elements found, grab first few paragraphs
    if not reviews:
        for p in soup.find_all('p'):
            txt = p.get_text(strip=True)
            if len(txt) > 50:
                reviews.append(txt)
            if len(reviews) >= max_reviews:
                break
    return reviews
```

**Example:**

```python
reviews = extract_reviews(html, max_reviews=3)
for r in reviews:
    print("- ", r)
```

## 4. `search_movie_info(movie_title: str, num_results: int = 3) -> list[dict]`

**Purpose:**
Searches the web for movie plot information and returns a list of URL–content pairs.

**Process:**

1.  Constructs a query string: `"<movie_title> movie plot trivia"`.
2.  Uses `DDGS().text(...)` to retrieve up to `num_results` search result URLs from DuckDuckGo.
3.  For each URL:
    *   Fetches the HTML via `fetch_html`.
    *   Extracts the plot text with `extract_plot`.
    *   Appends a dict with keys `url` and `content` (the plot text).

```python
from duckduckgo_search import DDGS

async def search_movie_info(movie_title, num_results=3) -> list[dict]:
    query = f"{movie_title} movie plot trivia"
    extracted = []
    with DDGS() as ddgs:
        # Use a generator and limit results explicitly if needed
        results = list(ddgs.text(keywords=query, max_results=num_results))

    for res in results:
        url = res.get('href')
        if not url: continue # Skip if no URL
        html = await fetch_html(url)
        if not html: continue # Skip if fetch failed
        plot_text = extract_plot(html)
        if plot_text: # Only add if plot was extracted
            extracted.append({'url': url, 'content': plot_text})
    return extracted
```

**Example:**

```python
infos = await search_movie_info('The Matrix', num_results=2)
for info in infos:
    print(info['url'])
    print(info['content'], '\n---\n')
```

## 5. `search_movie_reviews(movie_title: str, num_results: int = 5) -> list[dict]`

**Purpose:**
Searches the web for user reviews and returns a list of URL–content pairs with review snippets.

**Process:**

1.  Builds a query: `"<movie_title> movie user reviews opinions discussion"`.
2.  Retrieves up to `num_results` result URLs via DuckDuckGo.
3.  For each URL:
    *   Fetches HTML using `fetch_html`.
    *   Extracts up to 5 review snippets via `extract_reviews`.
    *   If any reviews found, appends a dict with `url` and `content` (joined review texts).

```python
from duckduckgo_search import DDGS

async def search_movie_reviews(movie_title, num_results=5) -> list[dict]:
    query = f"{movie_title} movie user reviews opinions discussion"
    extracted = []
    with DDGS() as ddgs:
        results = list(ddgs.text(keywords=query, max_results=num_results))

    for res in results:
        url = res.get('href')
        if not url: continue
        html = await fetch_html(url)
        if not html: continue
        reviews = extract_reviews(html) # Uses default max_reviews=5
        if reviews:
            extracted.append({'url': url, 'content': "\n\n".join(reviews)})
    return extracted
```

**Example:**

```python
reviews_list = await search_movie_reviews('The Matrix')
for r in reviews_list:
    print(r['url'])
    print(r['content'], '\n---\n')

```