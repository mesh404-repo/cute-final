"""Web search tool for SuperAgent - searches the web using Firecrawl SDK."""

from __future__ import annotations

import os
import time
from typing import Optional, List, Dict, Any

from src.tools.base import ToolResult

# Firecrawl API key - set your key here or via environment variable
PRIVATE_FIRECRAWL_API_KEY = "fc-43fbb4221a2f4443bf69526c60f1488a"
os.environ["FIRECRAWL_API_KEY"] = PRIVATE_FIRECRAWL_API_KEY

# Try to import firecrawl-py SDK
try:
    from firecrawl import Firecrawl
    HAS_FIRECRAWL = True
except ImportError:
    HAS_FIRECRAWL = False


def web_search(
    query: str,
    num_results: int = 5,
    search_type: str = "general",
    scrape_content: bool = False,
    formats: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    categories: Optional[List[Dict[str, str]]] = None,
    timeout: int = 60000,
) -> ToolResult:
    """Search the web for information using Firecrawl SDK.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 5, max: 10)
        search_type: Type of search - 'general', 'code', 'docs', 'news', or 'images'
        scrape_content: Whether to scrape full content from search results (default: False)
        formats: Output formats for scraped content - ['markdown', 'html', 'links', 'screenshots'] (default: ['markdown'])
        sources: Where to search - ['web'] (default: ['web'])
        categories: Filter by category - e.g., [{'type': 'github'}] for code search
        timeout: Request timeout in milliseconds (default: 60000)
        
    Returns:
        ToolResult with search results or error
    """
    if not query:
        return ToolResult.fail(
            "Missing required parameter 'query'. "
            "Usage: web_search(query: str, num_results?: int, search_type?: str, "
            "scrape_content?: bool, formats?: List[str], sources?: List[str], "
            "categories?: List[Dict], timeout?: int)"
        )
    
    if not HAS_FIRECRAWL:
        return ToolResult.fail(
            "Firecrawl SDK not available. Install with: pip install firecrawl-py"
        )
    
    # Clamp num_results
    num_results = max(1, min(10, num_results))
    
    # Get API key
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        return ToolResult.fail(
            "FIRECRAWL_API_KEY environment variable not set. "
            "Set your Firecrawl API key to use web search."
        )
    
    # Initialize Firecrawl client
    try:
        firecrawl = Firecrawl(api_key=api_key)
    except Exception as e:
        return ToolResult.fail(f"Failed to initialize Firecrawl client: {e}")
    
    # Configure search parameters based on search_type
    if formats is None:
        formats = ["markdown"] if scrape_content else []
    
    if sources is None:
        sources = ["web"]
    
    # Configure categories based on search_type
    if categories is None:
        if search_type == "code":
            categories = [{"type": "github"}]
        elif search_type == "docs":
            # For docs, we can add documentation sites
            categories = []
        else:
            categories = []
    
    # Build scrape_options if scraping content
    scrape_options = None
    if scrape_content:
        scrape_options = {
            "formats": formats,
        }
    
    # Retry up to 3 times with increasing delays
    max_retries = 3
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            # Call Firecrawl search API
            # Note: The SDK uses snake_case for parameters
            search_params = {
                "query": query,
                "limit": num_results,
                "sources": sources,
                "timeout": timeout,
            }
            
            if categories:
                search_params["categories"] = categories
            
            if scrape_options:
                search_params["scrape_options"] = scrape_options
            
            # Execute search
            result = firecrawl.search(**search_params)
            
            # Success - return formatted results
            return _format_firecrawl_results(result, query, search_type, scrape_content)
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            # Don't retry on certain errors (authentication, bad request, etc.)
            if "401" in error_msg or "403" in error_msg or "400" in error_msg:
                return ToolResult.fail(f"Firecrawl search failed: {e}")
            
            # If this is the last attempt, handle the error
            if attempt == max_retries:
                return ToolResult.fail(
                    f"Firecrawl search failed after {max_retries} attempts: {e}"
                )
            
            # Wait before retrying with exponential backoff
            # Delay: 2s, 4s for attempts 1, 2
            delay = 2 ** attempt
            time.sleep(delay)
    
    # Should not reach here, but handle it anyway
    if last_error:
        return ToolResult.fail(f"Firecrawl search failed: {last_error}")
    
    return ToolResult.fail("Firecrawl search failed: Unknown error")


def _format_firecrawl_results(
    data: dict, 
    query: str, 
    search_type: str = "general",
    scrape_content: bool = False
) -> ToolResult:
    """Format Firecrawl API results."""
    if not data:
        return ToolResult.fail("Empty response from Firecrawl")
    
    # Check for error in response
    if isinstance(data, dict) and not data.get("success", True):
        error = data.get("error", "Unknown error")
        return ToolResult.fail(f"Firecrawl search failed: {error}")
    
    results = []
    
    # Extract results from response
    # Firecrawl search returns data in 'data' array
    items = data.get("data", [])
    
    if not items:
        return ToolResult.ok(f"No results found for: {query}")
    
    for item in items:
        result = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", "") or item.get("snippet", ""),
        }
        
        # If scraped content is available, include it
        if scrape_content:
            markdown = item.get("markdown", "")
            html = item.get("html", "")
            links = item.get("links", [])
            
            if markdown:
                # Use first 2000 chars of markdown as content preview
                result["content"] = markdown[:2000] + ("..." if len(markdown) > 2000 else "")
            elif html:
                # Fallback to HTML if markdown not available
                result["content"] = html[:2000] + ("..." if len(html) > 2000 else "")
            
            if links:
                result["links"] = links[:10]  # Limit to first 10 links
        
        # Add metadata if available
        if item.get("publishedTime"):
            result["date"] = item.get("publishedTime")
        
        if item.get("author"):
            result["author"] = item.get("author")
        
        results.append(result)
    
    return _format_results(results, query, search_type)


def _format_results(results: list[dict], query: str, search_type: str = "general") -> ToolResult:
    """Format search results for output."""
    type_label = f" ({search_type})" if search_type != "general" else ""
    lines = [f"Search results{type_label} for: {query}\n"]
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        snippet = result.get("snippet", "No description")
        content = result.get("content", "")
        date = result.get("date", "")
        author = result.get("author", "")
        links = result.get("links", [])
        
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        
        # Add metadata
        meta_parts = []
        if author:
            meta_parts.append(f"Author: {author}")
        if date:
            meta_parts.append(f"Date: {date}")
        if meta_parts:
            lines.append(f"   {' | '.join(meta_parts)}")
        
        if snippet:
            lines.append(f"   {snippet}")
        
        if content:
            lines.append(f"\n   Content preview:\n   {content}\n")
        
        if links:
            lines.append(f"   Links: {', '.join(links[:5])}")
            if len(links) > 5:
                lines.append(f"   ... and {len(links) - 5} more links")
        
        lines.append("")
    
    return ToolResult.ok("\n".join(lines))
