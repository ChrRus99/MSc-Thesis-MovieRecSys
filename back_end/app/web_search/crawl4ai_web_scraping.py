import asyncio
import platform
import re
import sys
import time # Import time for sleep
from bs4 import BeautifulSoup, Tag # Import Tag for type hinting
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy
from crawl4ai import HTTPCrawlerConfig
from duckduckgo_search import DDGS
from pathlib import Path
from typing import Optional, List, Dict, Any # Added List, Dict, Any


# Ensure Windows uses ProactorEventLoop for subprocess support, necessary for certain async operations.
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Configure HTTP-only crawler (no Playwright) for faster, simpler HTML fetching.
http_config = HTTPCrawlerConfig(
    method="GET",
    headers={"User-Agent": "MyCrawler/1.0"}, # Set a user agent to identify the crawler.
    follow_redirects=True,
    verify_ssl=True
)


async def _fetch_html(url: str) -> str:
    """
    Fetch raw HTML content for a given URL using AsyncWebCrawler. Uses the pre-configured HTTP-only 
    strategy.
    """
    # Initialize the crawler with the HTTP-only strategy
    async with AsyncWebCrawler(
        crawler_strategy=AsyncHTTPCrawlerStrategy(browser_config=http_config)
    ) as crawler:
        # Perform the crawl operation to fetch HTML content
        result = await crawler.arun(url=url)
        return result.html if (result and hasattr(result, 'html')) else ""


def _extract_plot(html: str) -> str:
    """
    Extracts the movie plot or synopsis from HTML by looking for headings containing 'Plot' or
    'Synopsis'. Falls back to finding main content areas or the first ~1000 words if specific
    sections aren't found.
    """
    soup = BeautifulSoup(html, 'html.parser')
    content_found = False

    # Priority 1: Search for common section headers for plot/synopsis
    for header in soup.find_all(['h2', 'h3', 'h4']):
        if re.search(r'plot|synopsis', header.get_text(), re.I):
            paragraphs = []
            for sibling in header.find_next_siblings():
                if sibling.name in ['h2', 'h3', 'h4']:
                    break
                if sibling.name == 'p':
                    paragraphs.append(sibling.get_text(strip=True))
            if paragraphs:
                print("Extracted content using Plot/Synopsis header.")
                return "\n\n".join(paragraphs)

    # Priority 2: Look for common main content containers
    main_content_selectors = ['main', '#content', '#main', '#article', '.post-content', '.entry-content']
    for selector in main_content_selectors:
        main_area = soup.select_one(selector)
        if main_area:
            print(f"Found main content area using selector: '{selector}'. Extracting text.")
            # Extract text, trying to preserve some structure
            main_text = main_area.get_text(separator='\n', strip=True)
            # Basic cleanup: remove excessive blank lines
            cleaned_text = re.sub(r'\n{3,}', '\n\n', main_text).strip()
            if len(cleaned_text.split()) > 50: # Check if it has substantial content
                 return cleaned_text
            else:
                 print(f"Main area '{selector}' found but content seems too short. Trying next selector.")


    # Fallback: Extract the first ~1000 words from the entire page text if no specific section found
    print("No specific plot/synopsis section or main content area found. Falling back to first ~1000 words.")
    full_text = soup.get_text(separator=' ', strip=True)
    words = full_text.split()
    limit = 1000 # Increased word limit
    return ' '.join(words[:limit]) + ('...' if len(words) > limit else '')


def _extract_imdb_plot(html: str) -> str:
    """
    Extracts the movie plot specifically from an IMDb plot summary page.
    Prioritizes the section titled 'Description'/'Descrizione' and extracts text from list items within the next sibling.
    Includes fallbacks for other known IMDb structures.
    """
    soup = BeautifulSoup(html, 'html.parser')
    plot_text = ""

    # Strategy 1: User-specified detailed extraction for 'Description'/'Descrizione' section
    sections = soup.find_all('section', class_=lambda x: x and 'ipc-page-section' in x.split())

    for section in sections:
        # Find the title element within the section
        title_div = section.find('div', class_=lambda x: x and 'ipc-title' in x.split())
        if not title_div:
            continue

        title_elem = title_div.find(class_='ipc-title__text')
        if title_elem:
            title_text = title_elem.get_text(strip=True)
            # Check if the title matches 'Description' or 'Descrizione' (case-insensitive)
            if title_text.lower() in ['description', 'descrizione', 'synopsis']: # Added Synopsis here too
                print(f"Found section with title: '{title_text}'")
                # Find the container sibling holding the plot content
                # The structure might be title_div -> next_sibling or title_div's parent -> next_sibling
                plot_container = title_div.find_next_sibling()
                # Sometimes the relevant sibling is of the parent section, check that too
                if not plot_container:
                     parent_section_sibling = section.find_next_sibling()
                     if parent_section_sibling and parent_section_sibling.find('ul'):
                         plot_container = parent_section_sibling

                # If plot_container is still None, try finding sibling of title's parent div
                if not plot_container and title_div.parent:
                    plot_container = title_div.parent.find_next_sibling()


                if plot_container and isinstance(plot_container, Tag):
                    # Find the UL element within this container
                    ul_element = plot_container.find('ul')
                    if ul_element:
                        # Extract text from all direct child LI elements within the UL
                        list_items = ul_element.find_all('li', recursive=False) # Direct children LI
                        if list_items:
                            plot_parts = [li.get_text(separator=' ', strip=True) for li in list_items]
                            plot_text = "\n\n".join(plot_parts)
                            print("Successfully extracted plot using Description/Synopsis section strategy (UL/LI).")
                            break # Found the plot, exit the loop
                        else:
                             print("Found UL, but no LI elements inside.")
                    else:
                        # Fallback if structure is slightly different (e.g., text directly in container div)
                        content_div = plot_container.find('div', class_='ipc-html-content-inner-div')
                        if content_div:
                             plot_text = content_div.get_text(separator='\n', strip=True)
                             print("Extracted plot from container sibling's inner div.")
                             break
                        else:
                            # Check for direct text as last resort within container
                            direct_text = plot_container.get_text(separator='\n', strip=True)
                            if len(direct_text) > 100: # Check if it looks like a plot
                                plot_text = direct_text
                                print("Extracted plot from container sibling directly (no UL/LI/div found).")
                                break
                            else:
                                print("Found plot container sibling, but no UL or significant direct text/div.")
                else:
                    print(f"Found '{title_text}' title, but couldn't find a valid next sibling container.")

    # If Strategy 1 failed, try other fallbacks
    if not plot_text:
        print("Could not extract plot using the Description/Synopsis section strategy. Trying other fallbacks.")
        # Strategy 2: Look for the section with data-testid="sub-section-synopsis"
        synopsis_section = soup.find('section', attrs={'data-testid': 'sub-section-synopsis'})
        if synopsis_section:
            content_div = synopsis_section.find('div', class_='ipc-html-content-inner-div')
            if content_div:
                print("Found synopsis using data-testid strategy.")
                plot_text = content_div.get_text(separator='\n', strip=True)

    # Strategy 3: Older h4 Synopsis header (less likely now but kept as final fallback)
    # This was part of the previous logic, keeping it just in case.
    if not plot_text:
        synopsis_header = soup.find('h4', string='Synopsis')
        if synopsis_header:
            print("Found synopsis using H4 'Synopsis' header strategy.")
            paragraphs = []
            current = synopsis_header.find_next_sibling()
            while current:
                if current.name == 'p':
                    paragraphs.append(current.get_text(strip=True))
                elif current.name in ['h4', 'hr']: # Stop at the next section
                    break
                current = current.find_next_sibling()
            if paragraphs:
                plot_text = "\n\n".join(paragraphs)


    if not plot_text:
        print("Could not find synopsis section using any known strategy.")

    return plot_text


def _extract_triviaforyou_trivia(html: str) -> str:
    """
    Extracts trivia content specifically from a triviaforyou.com page.
    Targets the main content area identified by 'entry-content'.
    """
    soup = BeautifulSoup(html, 'html.parser')
    trivia_text = ""

    # Find the main content div
    entry_content_div = soup.find('div', class_='entry-content')

    if entry_content_div:
        print("Found 'entry-content' div on triviaforyou.com.")
        # Extract all text within this div, preserving paragraphs roughly
        trivia_text = entry_content_div.get_text(separator='\n', strip=True)
        # Basic cleanup: remove excessive blank lines
        trivia_text = re.sub(r'\n{3,}', '\n\n', trivia_text).strip()
    else:
        print("Could not find 'entry-content' div on triviaforyou.com page.")

    return trivia_text


def _extract_imdb_reviews(html: str) -> list[str]:
    """
    Extracts user reviews specifically from an IMDb reviews page.
    Targets potential review containers and extracts text content, cleaning up initial lines.
    """
    soup = BeautifulSoup(html, 'html.parser')
    reviews = []
    
    # ... existing code to find main_content and review_containers ...
    main_content = soup.find('div', id='main')
    if not main_content:
        main_content = soup # Fallback to searching the whole document if #main isn't found

    review_containers = main_content.find_all('div', class_='review-container')
    if not review_containers:
        review_containers = main_content.find_all('div', class_='lister-item')
    if not review_containers:
        review_containers = main_content.find_all('div', attrs={'data-testid': lambda x: x and 'review' in x})

    print(f"Found {len(review_containers)} potential review containers using combined strategies on IMDb page.")

    for container in review_containers:
        review_text = ""
        processed = False # Flag to check if review was processed by primary methods

        # Extract text from within the container using previous logic
        review_text_div = container.find('div', class_=lambda x: x and 'text' in x.split() and 'show-more__control' in x.split())
        
        if not review_text_div:
             content_div = container.find('div', class_='content')
             if content_div:
                 review_text_div_nested = content_div.find('div', class_=lambda x: x and 'text' in x.split())
                 if review_text_div_nested:
                      review_text_div = review_text_div_nested
                 else:
                      direct_text = content_div.get_text(strip=True)
                      if len(direct_text) > 50:
                           # Clean up initial newlines/spaces more aggressively
                           cleaned_text = re.sub(r'^\s+', '', direct_text) # Remove leading whitespace
                           cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text) # Consolidate multiple newlines
                           # Attempt to merge rating/title line if present at the very beginning
                           lines = cleaned_text.split('\n')
                           if len(lines) > 1 and (re.match(r'\d+\s*/\s*10', lines[0].strip()) or len(lines[0].strip()) < 30): # Check if first line is likely rating/short title
                               cleaned_text = lines[0].strip() + " " + "\n".join(lines[1:]).strip()

                           if len(cleaned_text) > 30:
                               reviews.append(cleaned_text)
                               processed = True
                               continue # Skip further processing for this container

        if review_text_div and not processed:
            raw_text = review_text_div.get_text(separator='\n', strip=True)
            # Clean up initial newlines/spaces more aggressively
            cleaned_text = re.sub(r'^\s+', '', raw_text) # Remove leading whitespace
            cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text) # Consolidate multiple newlines
             # Attempt to merge rating/title line if present at the very beginning
            lines = cleaned_text.split('\n')
            if len(lines) > 1 and (re.match(r'\d+\s*/\s*10', lines[0].strip()) or len(lines[0].strip()) < 30): # Check if first line is likely rating/short title
                cleaned_text = lines[0].strip() + " " + "\n".join(lines[1:]).strip()

            if len(cleaned_text) > 30:
                reviews.append(cleaned_text)
                processed = True

        if not processed:
            # Fallback: Extract text directly from container, filtering metadata
            container_text = container.get_text(separator='\n', strip=True)
            lines = container_text.split('\n')
            start_index = 0
            # ... (existing logic to find start_index based on metadata) ...
            for i, line in enumerate(lines):
                 # More robust check for metadata lines (ratings, date, permalink, title)
                if 'Permalink' in line or \
                   re.match(r'\d+\s*/\s*10', line) or \
                   re.match(r'\d+\s+out\s+of\s+\d+', line, re.I) or \
                   len(line) < 15 and re.search(r'\d{1,2}\s+\w+\s+\d{4}', line): # Date check
                    start_index = max(start_index, i + 1)

            title_elem = container.find(['h1', 'h2', 'h3', 'a'], class_='title')
            if title_elem:
                 title_text = title_elem.get_text(strip=True)
                 for i in range(start_index, len(lines)):
                     if title_text in lines[i]:
                         start_index = max(start_index, i + 1)
                         break

            # Join remaining lines and clean
            review_text = "\n".join(lines[start_index:]).strip()
            cleaned_text = re.sub(r'^\s+', '', review_text) # Remove leading whitespace
            cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text) # Consolidate multiple newlines

            if len(cleaned_text) > 50:
                 print(f"Using fallback text extraction for container: {cleaned_text[:100]}...")
                 reviews.append(cleaned_text)


    print(f"Extracted {len(reviews)} reviews from IMDb page.")
    return reviews


def _extract_reviews(html: str, max_reviews: int = 5) -> list[str]:
    """
    Extracts user review snippets from HTML.
    Priority 1: Find elements with 'review' in class/id.
    Priority 2: Find common main content areas.
    Priority 3: Fallback to first ~1000 words.
    """
    soup = BeautifulSoup(html, 'html.parser')
    reviews = []

    # Priority 1: Look for common containers likely containing specific reviews
    review_selectors = [
        'div[class*="review"]', 'section[class*="review"]', 'article[class*="review"]', 'p[class*="review"]',
        'div[id*="review"]', 'section[id*="review"]', 'article[id*="review"]', 'p[id*="review"]',
        '.comment-body', '.comment-content', '.user-review', '.review-text' # Add more specific selectors if known
    ]
    found_specific_reviews = False
    for selector in review_selectors:
        try:
            tags = soup.select(selector)
            for tag in tags:
                text = tag.get_text(separator=' ', strip=True)
                # Filter out very short text snippets
                if len(text.split()) > 30: # Use word count for better filtering
                    reviews.append(text)
                    found_specific_reviews = True
                if len(reviews) >= max_reviews:
                    break
            if len(reviews) >= max_reviews:
                break
        except Exception as e:
            print(f"Error processing selector '{selector}': {e}") # Handle potential invalid CSS selectors
    
    if found_specific_reviews:
        print(f"Extracted {len(reviews)} specific review snippets.")
        return reviews[:max_reviews]

    # Priority 2: Look for common main content containers if no specific reviews found
    print("No specific review elements found. Looking for main content areas.")
    main_content_selectors = ['main', '#content', '#main', '#article', '.post-content', '.entry-content', '.article-body']
    for selector in main_content_selectors:
        main_area = soup.select_one(selector)
        if main_area:
            print(f"Found main content area using selector: '{selector}'. Extracting text.")
            main_text = main_area.get_text(separator='\n', strip=True)
            cleaned_text = re.sub(r'\n{3,}', '\n\n', main_text).strip()
            if len(cleaned_text.split()) > 50:
                 # Treat the whole main content as one review snippet in this fallback
                 return [cleaned_text]
            else:
                 print(f"Main area '{selector}' found but content seems too short. Trying next selector.")

    # Priority 3: Fallback to first ~1000 words if nothing else worked
    print("No specific reviews or main content area found. Falling back to first ~1000 words.")
    full_text = soup.get_text(separator=' ', strip=True)
    words = full_text.split()
    limit = 1000
    fallback_text = ' '.join(words[:limit]) + ('...' if len(words) > limit else '')
    if len(words) > 50: # Only return if there's substantial text
        return [fallback_text]
    else:
        return [] # Return empty list if even the fallback is too short


async def _ddgs_text_search_with_retry(query: str, max_results: int, retries: int = 4, initial_delay: float = 5.0) -> List[Dict[str, Any]]:
    """
    Performs a DDGS text search with retry logic for rate limits.
    Uses longer delays and more retries.
    """
    delay = initial_delay
    for attempt in range(retries):
        try:
            # Add a small delay *before* each attempt after the first one, even if the previous wasn't a rate limit error,
            # to generally space out requests more.
            if attempt > 0:
                 await asyncio.sleep(delay / 2) # Add a small pre-attempt delay

            with DDGS() as ddgs:
                results = list(ddgs.text(keywords=query, max_results=max_results))
            return results # Success
        except Exception as e:
            # Check if the error message indicates a rate limit
            if "Ratelimit" in str(e) or "rate limit" in str(e).lower():
                if attempt < retries - 1:
                    print(f"Rate limit detected on attempt {attempt + 1}. Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2 # Exponential backoff (consider increasing multiplier if needed, e.g., 2.5)
                else:
                    print(f"Rate limit hit after {retries} attempts. Failing search for query: '{query}'")
                    raise e # Re-raise the exception after final attempt
            else:
                # Not a rate limit error, re-raise immediately
                print(f"Non-rate limit error during DDGS search: {e}")
                raise e
    return [] # Should not be reached if exception is always raised on failure


async def _get_imdb_id(movie_title: str) -> Optional[str]:
    """
    Searches DuckDuckGo for the IMDb title page of a movie and extracts the IMDb ID (tt...).
    Includes retry logic for rate limits.
    """
    query = f"{movie_title} site:imdb.com"
    try:
        # Use the retry wrapper for the search
        results = await _ddgs_text_search_with_retry(query=query, max_results=1)
        if results:
            url = results[0].get('href')
            if url:
                match = re.search(r'/title/(tt\d+)/', url)
                if match:
                    return match.group(1)
    except Exception as e:
        # Error already logged in _ddgs_text_search_with_retry if it's persistent
        print(f"Failed to get IMDb ID for '{movie_title}' after retries: {e}")
    return None


async def search_movie_plot_info(movie_title: str, num_results: int = 3) -> list[dict]:
    """
    Searches for movie plot summaries. Prioritizes fetching the synopsis directly from IMDb.
    Falls back to a general web search if IMDb fails or doesn't yield a plot.
    Includes retry logic for rate limits in fallback search.
    """
    # 1. Attempt to get IMDb ID and fetch plot from IMDb
    imdb_id = await _get_imdb_id(movie_title)
    if imdb_id:
        # Construct URL for the primary language plot summary page
        imdb_plot_url = f"https://www.imdb.com/title/{imdb_id}/plotsummary/"
        print(f"Attempting to fetch plot from IMDb: {imdb_plot_url}")
        html = await _fetch_html(imdb_plot_url)
        if html:
            # Use the refined extraction logic
            imdb_plot_text = _extract_imdb_plot(html)
            if imdb_plot_text:
                print("Successfully extracted plot from IMDb.")
                # Return *only* the IMDb result if successful
                return [{
                    'url': imdb_plot_url,
                    'content': imdb_plot_text
                }]
            else:
                print("Could not extract plot synopsis from IMDb page content, falling back.")
        else:
            print(f"Could not fetch IMDb plot page ({imdb_plot_url}), falling back.")
    else:
        print(f"Could not find IMDb ID for '{movie_title}', falling back to general search.")


    # 2. Fallback: General web search for plot
    print("Falling back to general web search for plot.")
    query = f"{movie_title} plot"
    extracted = []
    try:
        # Use the retry wrapper for the fallback search
        results = await _ddgs_text_search_with_retry(query=query, max_results=num_results)

        tasks = []
        urls_processed = set()
        if imdb_id: # Keep track of the main IMDb URL to avoid re-fetching
             urls_processed.add(f"imdb.com/title/{imdb_id}")

        for res in results:
            url = res.get('href')
            if not url:
                continue

            # Simple check to avoid processing the same domain multiple times if desired,
            # or specifically avoid IMDb if already tried.
            domain = re.search(r"https://(?:www\.)?([^/]+)", url)
            if domain and domain.group(1) in urls_processed:
                 # print(f"Skipping already processed domain: {domain.group(1)}")
                 continue

            # Avoid re-fetching the specific IMDb plot page if we already failed
            if imdb_id and f"imdb.com/title/{imdb_id}/plotsummary" in url:
                 continue

            tasks.append(_fetch_and_extract_plot_fallback(url))
            if domain:
                 urls_processed.add(domain.group(1)) # Mark domain as processed

        # Gather results from fallback fetches
        fallback_results = await asyncio.gather(*tasks)
        extracted = [result for result in fallback_results if result] # Filter out None results

    except Exception as e:
        # Error should have been handled or logged by the retry wrapper
        print(f"Fallback plot search failed ultimately for '{movie_title}': {e}")


    if not extracted:
        print(f"Fallback search also failed to find plot for '{movie_title}'.")

    return extracted[:num_results]

async def _fetch_and_extract_plot_fallback(url: str) -> Optional[dict]:
    """Helper coroutine for fetching and extracting plot in the fallback mechanism."""
    html = await _fetch_html(url)
    if not html:
        return None
    plot_text = _extract_plot(html) # Use the general extractor for fallbacks
    if plot_text:
        return {'url': url, 'content': plot_text}
    return None


async def search_movie_curiosities_info(movie_title: str, num_results: int = 3) -> list[dict]:
    """
    Searches DuckDuckGo for movie trivia/facts. Prioritizes fetching content directly from
    triviaforyou.com. Falls back to a general web search if the specific site fails or
    doesn't yield content.
    Includes retry logic for rate limits in both search steps.
    """
    # 1. Attempt to get trivia from triviaforyou.com
    triviaforyou_url = None
    query_specific = f'"{movie_title}" trivia site:triviaforyou.com'
    print(f"Attempting to find trivia on triviaforyou.com with query: {query_specific}")
    try:
        # Use the retry wrapper for the specific site search
        results_specific = await _ddgs_text_search_with_retry(query=query_specific, max_results=1)
        if results_specific:
            url = results_specific[0].get('href')
            # Relaxed check: Ensure it's the correct domain and looks like a trivia page.
            if url and 'triviaforyou.com' in url and '/trivia' in url.lower():
                triviaforyou_url = url
                print(f"Found potential triviaforyou.com URL: {triviaforyou_url}")
            elif url and 'triviaforyou.com' in url:
                 print(f"Found URL from site search ({url}), but path doesn't contain '/trivia'. Treating as potentially irrelevant.")
            else:
                 print("Found URL from site search, but it doesn't seem relevant (wrong domain or missing URL).")

    except Exception as e:
        # Error should have been handled or logged by the retry wrapper
        print(f"Specific trivia search failed ultimately for '{movie_title}': {e}")

    if triviaforyou_url:
        html = await _fetch_html(triviaforyou_url)
        if html:
            trivia_text = _extract_triviaforyou_trivia(html)
            if trivia_text:
                print("Successfully extracted trivia from triviaforyou.com.")
                # Return *only* the triviaforyou result if successful
                return [{
                    'url': triviaforyou_url,
                    'content': trivia_text
                }]
            else:
                print("Could not extract trivia content from triviaforyou.com page, falling back.")
        else:
            print(f"Could not fetch triviaforyou.com page ({triviaforyou_url}), falling back.")
    else:
        print(f"Could not find relevant URL on triviaforyou.com for '{movie_title}', falling back.")


    # 2. Fallback: General web search for curiosities/trivia
    print("Falling back to general web search for curiosities/trivia.")
    query_general = f"{movie_title} movie trivia facts curiosities"
    extracted = []
    try:
        # Use the retry wrapper for the general fallback search
        results_general = await _ddgs_text_search_with_retry(query=query_general, max_results=num_results)

        tasks = []
        urls_processed = set()
        if triviaforyou_url: # Avoid re-processing the specific trivia site
            urls_processed.add("triviaforyou.com")

        for res in results_general:
            url = res.get('href')
            if not url:
                continue

            domain = re.search(r"https://(?:www\.)?([^/]+)", url)
            if domain and domain.group(1) in urls_processed:
                continue

            # Use the general plot extractor as a fallback text extractor
            tasks.append(_fetch_and_extract_general_info_fallback(url))
            if domain:
                urls_processed.add(domain.group(1))

        fallback_results = await asyncio.gather(*tasks)
        extracted = [result for result in fallback_results if result]

    except Exception as e:
        # Error should have been handled or logged by the retry wrapper
        print(f"Fallback curiosity search failed ultimately for '{movie_title}': {e}")

    if not extracted:
        print(f"Fallback search also failed to find curiosities for '{movie_title}'.")

    return extracted[:num_results]

async def _fetch_and_extract_general_info_fallback(url: str) -> Optional[dict]:
    """Helper coroutine for fetching and extracting general info in the fallback mechanism."""
    html = await _fetch_html(url)
    if not html:
        return None
    # Re-using _extract_plot as a general text extractor
    info_text = _extract_plot(html)
    if info_text:
        return {'url': url, 'content': info_text}
    return None


async def search_movie_reviews(movie_title: str, num_results: int = 5) -> list[dict]:
    """
    Searches for movie reviews. Prioritizes fetching reviews directly from IMDb.
    Falls back to a general web search if IMDb fails or doesn't yield reviews.
    Includes retry logic for rate limits in fallback search.
    """
    # 1. Attempt to get IMDb ID and fetch reviews from IMDb
    imdb_id = await _get_imdb_id(movie_title)
    imdb_reviews_url = None
    if imdb_id:
        imdb_reviews_url = f"https://www.imdb.com/title/{imdb_id}/reviews/"
        print(f"Attempting to fetch reviews from IMDb: {imdb_reviews_url}")
        html = await _fetch_html(imdb_reviews_url)
        if html:
            imdb_reviews_list = _extract_imdb_reviews(html) # Use the updated extractor
            if imdb_reviews_list:
                print("Successfully extracted reviews from IMDb.")
                # Return *only* the IMDb result if successful, combining reviews into one content string
                return [{
                    'url': imdb_reviews_url,
                    'content': "\n\n---\n\n".join(imdb_reviews_list) # Join reviews with a separator
                }]
            else:
                print("Could not extract reviews from IMDb page content, falling back.")
        else:
            print(f"Could not fetch IMDb reviews page ({imdb_reviews_url}), falling back.")
    else:
        print(f"Could not find IMDb ID for '{movie_title}', falling back to general review search.")


    # 2. Fallback: General web search for reviews
    print("Falling back to general web search for reviews.")
    query = f"{movie_title} movie user reviews opinions discussion"
    extracted = []
    try:
        # Use the retry wrapper for the fallback search
        results = await _ddgs_text_search_with_retry(query=query, max_results=num_results)

        tasks = []
        urls_processed = set()
        if imdb_reviews_url: # Avoid re-processing the specific IMDb reviews page
            urls_processed.add("imdb.com") # Add the domain to avoid other imdb pages too

        for res in results:
            url = res.get('href')
            if not url:
                continue

            domain = re.search(r"https://(?:www\.)?([^/]+)", url)
            if domain and domain.group(1) in urls_processed:
                continue

            tasks.append(_fetch_and_extract_reviews_fallback(url))
            if domain:
                urls_processed.add(domain.group(1))

        fallback_results = await asyncio.gather(*tasks)
        extracted = [result for result in fallback_results if result]

    except Exception as e:
        # Error should have been handled or logged by the retry wrapper
        print(f"Fallback review search failed ultimately for '{movie_title}': {e}")

    if not extracted:
        print(f"Fallback search also failed to find reviews for '{movie_title}'.")

    return extracted[:num_results]

async def _fetch_and_extract_reviews_fallback(url: str, max_reviews_per_page: int = 5) -> Optional[dict]:
    """Helper coroutine for fetching and extracting reviews in the fallback mechanism."""
    html = await _fetch_html(url)
    if not html:
        return None
    # Use the general review extractor for fallbacks
    review_snippets = _extract_reviews(html, max_reviews=max_reviews_per_page)
    if review_snippets:
        return {
            'url': url,
            'content': "\n\n".join(review_snippets) # Join snippets from the page
        }
    return None


async def _gather_movie_data(title: str):
    """Helper to fetch plot, curiosities, and reviews concurrently using asyncio."""
    # Create tasks for fetching plot, info/curiosities, and reviews
    plot_task = asyncio.create_task(search_movie_plot_info(title))
    info_task = asyncio.create_task(search_movie_curiosities_info(title))
    rev_task  = asyncio.create_task(search_movie_reviews(title))

    # Wait for all tasks to complete and gather results
    plot_results, info_results, review_results = await asyncio.gather(plot_task, info_task, rev_task)
    return plot_results, info_results, review_results


def save_movie_to_txt(
    movie_title: str,
    filepath: str,
    overwrite: bool = False
) -> None:
    """
    Fetches movie plot, curiosities, and reviews, then writes them into a text file.

    Args:
        movie_title: The movie to search for.
        filepath:    Path where the .txt file will be created.
        overwrite:   If False and the file exists, raises FileExistsError.
    """
    path = Path(filepath)
    # Prevent accidentally overwriting existing files unless explicitly requested
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists at {filepath!r}. Use overwrite=True to replace.")

    # Run the asynchronous data gathering function using asyncio.run
    plot_results, info_results, review_results = asyncio.run(_gather_movie_data(movie_title))

    # Write the collected data to the specified file.
    with path.open('w', encoding='utf-8') as f:
        f.write(f"Movie: {movie_title}\n")
        f.write("=" * (7 + len(movie_title)) + "\n\n") # Dynamic separator line

        # Section: Movie Plot
        f.write("== Movie Plot ==\n\n")
        if plot_results:
            for i, item in enumerate(plot_results, 1): # Numbered results
                f.write(f"[Plot Source {i}]\n")
                f.write(f"Source: {item['url']}\n")
                f.write(item['content'].strip() + "\n\n")
        else:
            f.write("No plot information found.\n\n")

        # Section: Movie Curiosities & Trivia
        f.write("== Movie Curiosities & Trivia ==\n\n")
        if info_results:
            for i, item in enumerate(info_results, 1): # Numbered results
                f.write(f"[Info Source {i}]\n")
                f.write(f"Source: {item['url']}\n")
                f.write(item['content'].strip() + "\n\n")
        else:
            f.write("No curiosities/trivia information found.\n\n")

        # Section: Movie Reviews
        f.write("== Movie Reviews ==\n\n")
        if review_results:
            for i, item in enumerate(review_results, 1): # Numbered results
                f.write(f"[Review Source {i}]\n")
                f.write(f"Source: {item['url']}\n")
                f.write(item['content'].strip() + "\n\n")
        else:
            f.write("No user reviews found.\n")

    print(f"Data for '{movie_title}' saved to {filepath!r}") # Confirmation message


# # Example usage (commented out)
# async def main():
#     movie = "Inception"
#     plots = await search_movie_plot_info(movie)
#     print("Movie Plots:")
#     for item in plots:
#         print(item['url'])
#         print(item['content'], "\n---\n")

#     infos = await search_movie_curiosities_info(movie)
#     print("Movie Curiosities:")
#     for item in infos:
#         print(item['url'])
#         print(item['content'], "\n---\n")

#     reviews = await search_movie_reviews(movie)
#     print("Movie Reviews:")
#     for item in reviews:
#         print(item['url'])
#         print(item['content'], "\n---\n")

# # Entry point for running the example usage directly (commented out)
# if __name__ == '__main__':
#     asyncio.run(main())
