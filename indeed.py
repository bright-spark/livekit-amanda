@function_tool
async def indeed_job_search(
    context: RunContext,
    query: str = "customer service",
    location: str = "Johannesburg, Gauteng"
) -> str:
    """Search for jobs on Indeed using Playwright-powered scraping."""
    import urllib.parse
    from .puppeteer_crawler import crawl_page
    import logging
    try:
        base_url = "https://za.indeed.com/jobs"
        params = {
            "q": query,
            "l": location,
        }
        search_url = f"{base_url}?{urllib.parse.urlencode(params)}"
        # Use crawl_page to fetch the search results page
        listings = await crawl_page(search_url, extract_text=True)
        # Try to parse job titles and companies from the HTML/text
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(listings, "html.parser")
        jobs = []
        for div in soup.find_all('div', class_='job_seen_beacon'):
            title_elem = div.find('h2')
            company_elem = div.find('span', class_='companyName')
            location_elem = div.find('div', class_='companyLocation')
            summary_elem = div.find('div', class_='job-snippet')
            link_elem = div.find('a', href=True)
            title = title_elem.get_text(strip=True) if title_elem else None
            company = company_elem.get_text(strip=True) if company_elem else None
            location_val = location_elem.get_text(strip=True) if location_elem else None
            summary = summary_elem.get_text(strip=True) if summary_elem else None
            url = f"https://za.indeed.com{link_elem['href']}" if link_elem else None
            if title and company:
                jobs.append({
                    "title": title,
                    "company": company,
                    "location": location_val,
                    "summary": summary,
                    "url": url
                })
        if not jobs:
            return f"No jobs found for '{query}' in '{location}'."
        # Format results for output
        result = f"Here are some jobs for '{query}' in '{location}':\n\n"
        for i, job in enumerate(jobs[:5], 1):
            result += f"{i}. {job['title']} at {job['company']}\n"
            if job['location']:
                result += f"   Location: {job['location']}\n"
            if job['summary']:
                result += f"   {job['summary']}\n"
            result += "\n"
        result = sanitize_for_azure(result)
        session = getattr(context, 'session', None)
        if session:
            await handle_tool_results(session, result)
            return "Here are some jobs I found. I'll read them to you."
        return result
    except Exception as e:
        logging.error(f"[TOOL] indeed_job_search exception: {e}")
        return sanitize_for_azure(f"Sorry, I couldn't search for jobs right now: {e}")