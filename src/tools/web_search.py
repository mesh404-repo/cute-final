"""Web search tool for SuperAgent - searches the web using Firecrawl SDK."""

from __future__ import annotations

import os
import time
from typing import Optional, List, Dict, Any, Union

from src.tools.base import ToolResult

# Firecrawl API key - set your key here or via environment variable
PRIVATE_FIRECRAWL_API_KEY = "fc-2d8c678d51274347b9b38637d59299af"
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
    categories: Optional[Union[List[str], List[Dict[str, str]]]] = None,
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
        categories: Filter by category - list of 'github' | 'research' | 'pdf'. github: GitHub repos, code, issues, docs; research: arXiv, Nature, IEEE, PubMed, etc.; pdf: PDF files.
        timeout: Request timeout in milliseconds (default: 60000)
        
    Returns:
        ToolResult with search results or error
    """
    if not query:
        return ToolResult.fail(
            "Missing required parameter 'query'. "
            "Usage: web_search(query: str, num_results?: int, search_type?: str, "
            "categories?: List[str], scrape_content?: bool, formats?: List[str], "
            "sources?: List[str], timeout?: int)"
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
    
    # Normalize categories: accept list of strings from spec (e.g. ["github", "research"])
    # and convert to Firecrawl format [{"type": "github"}, {"type": "research"}]
    if categories is not None:
        normalized = []
        for c in categories:
            if isinstance(c, dict) and "type" in c:
                normalized.append(c)
            elif isinstance(c, str) and c in ("github", "research", "pdf"):
                normalized.append({"type": c})
        categories = normalized if normalized else None
    # Configure categories based on search_type when not explicitly provided
    if categories is None:
        if search_type == "code":
            categories = [{"type": "github"}]
        elif search_type == "docs":
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

    timeout = min(max(timeout, 1000), 300000)
    
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
    data: Any, 
    query: str, 
    search_type: str = "general",
    scrape_content: bool = False
) -> ToolResult:
    """Format Firecrawl API results.
    
    The SearchData object/response has:
    - web: List[dict] - web search results with url, title, description, position
    - news: List[dict] - news search results with title, url, snippet, date, position
    - images: List[dict] - image search results with title, imageUrl, imageWidth, imageHeight, url, position
    """
    if not data:
        return ToolResult.fail("Empty response from Firecrawl")
    
    # Extract items based on search type
    items = []
    result_type = search_type
    
    # Handle SearchData object - it has web, news, images attributes
    if hasattr(data, 'web') or hasattr(data, 'news') or hasattr(data, 'images'):
        # It's a SearchData object - access attributes
        if search_type == "news" and hasattr(data, 'news') and data.news:
            items = data.news
            result_type = "news"
        elif search_type == "images" and hasattr(data, 'images') and data.images:
            items = data.images
            result_type = "images"
        elif hasattr(data, 'web') and data.web:
            # Default to web results
            items = data.web
            result_type = "web"
        else:
            # Try to get any available results
            if hasattr(data, 'web') and data.web:
                items = data.web
                result_type = "web"
            elif hasattr(data, 'news') and data.news:
                items = data.news
                result_type = "news"
            elif hasattr(data, 'images') and data.images:
                items = data.images
                result_type = "images"
    elif isinstance(data, dict):
        # Check for error in response
        if not data.get("success", True) if "success" in data else True:
            error = data.get("error", "Unknown error")
            return ToolResult.fail(f"Firecrawl search failed: {error}")
        # Extract from dict format
        if search_type == "news" and "news" in data and data["news"]:
            items = data["news"]
            result_type = "news"
        elif search_type == "images" and "images" in data and data["images"]:
            items = data["images"]
            result_type = "images"
        elif "web" in data and data["web"]:
            items = data["web"]
            result_type = "web"
        else:
            # Fallback: try any available
            items = data.get("web", []) or data.get("news", []) or data.get("images", [])
            if data.get("web"):
                result_type = "web"
            elif data.get("news"):
                result_type = "news"
            elif data.get("images"):
                result_type = "images"
    else:
        # Try to access as attribute or convert to dict
        try:
            if hasattr(data, '__dict__'):
                data_dict = data.__dict__
            else:
                # Try to get as dict directly
                data_dict = dict(data) if hasattr(data, '__iter__') and not isinstance(data, str) else {}
            
            if search_type == "news" and "news" in data_dict and data_dict["news"]:
                items = data_dict["news"]
                result_type = "news"
            elif search_type == "images" and "images" in data_dict and data_dict["images"]:
                items = data_dict["images"]
                result_type = "images"
            elif "web" in data_dict and data_dict["web"]:
                items = data_dict["web"]
                result_type = "web"
            else:
                items = data_dict.get("web", []) or data_dict.get("news", []) or data_dict.get("images", [])
        except Exception as e:
            return ToolResult.fail(f"Unexpected response format from Firecrawl: {e}")
    
    if not items:
        return ToolResult.ok(f"No results found for: {query}")
    
    results = []
    
    for item in items:
        # Handle both dict and object formats
        if isinstance(item, dict):
            # Dictionary format (most common)
            if result_type == "images":
                # Image results have: title, imageUrl, imageWidth, imageHeight, url, position
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": f"Image: {item.get('imageUrl', '')} ({item.get('imageWidth', 0)}x{item.get('imageHeight', 0)})",
                    "imageUrl": item.get("imageUrl", ""),
                    "imageWidth": item.get("imageWidth"),
                    "imageHeight": item.get("imageHeight"),
                    "position": item.get("position"),
                }
            elif result_type == "news":
                # News results have: title, url, snippet, date, position
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                    "date": item.get("date", ""),
                    "position": item.get("position"),
                }
            else:
                # Web results have: url, title, description, position
                result = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                    "position": item.get("position"),
                    "category": item.get("category"),  # May be present for GitHub results
                }
        elif hasattr(item, 'url') or hasattr(item, '__dict__'):
            # Object format - convert to dict
            try:
                if hasattr(item, '__dict__'):
                    item_dict = item.__dict__
                else:
                    # Access attributes directly
                    item_dict = {
                        "title": getattr(item, "title", ""),
                        "url": getattr(item, "url", ""),
                        "description": getattr(item, "description", ""),
                        "snippet": getattr(item, "snippet", ""),
                        "position": getattr(item, "position", None),
                        "category": getattr(item, "category", None),
                        "date": getattr(item, "date", None),
                        "imageUrl": getattr(item, "imageUrl", None),
                    }
                
                if result_type == "images":
                    result = {
                        "title": item_dict.get("title", ""),
                        "url": item_dict.get("url", ""),
                        "snippet": f"Image: {item_dict.get('imageUrl', '')} ({item_dict.get('imageWidth', 0)}x{item_dict.get('imageHeight', 0)})",
                        "imageUrl": item_dict.get("imageUrl", ""),
                        "imageWidth": item_dict.get("imageWidth"),
                        "imageHeight": item_dict.get("imageHeight"),
                        "position": item_dict.get("position"),
                    }
                elif result_type == "news":
                    result = {
                        "title": item_dict.get("title", ""),
                        "url": item_dict.get("url", ""),
                        "snippet": item_dict.get("snippet", ""),
                        "date": item_dict.get("date", ""),
                        "position": item_dict.get("position"),
                    }
                else:
                    result = {
                        "title": item_dict.get("title", ""),
                        "url": item_dict.get("url", ""),
                        "snippet": item_dict.get("description", "") or item_dict.get("snippet", ""),
                        "position": item_dict.get("position"),
                        "category": item_dict.get("category"),
                    }
            except Exception:
                continue  # Skip items we can't process
        else:
            continue  # Skip unknown formats
        
        # If scraped content is available, include it
        if scrape_content:
            markdown = None
            html = None
            links = None
            
            if isinstance(item, dict):
                markdown = item.get("markdown")
                html = item.get("html")
                links = item.get("links")
            else:
                markdown = getattr(item, "markdown", None) if hasattr(item, "markdown") else None
                html = getattr(item, "html", None) if hasattr(item, "html") else None
                links = getattr(item, "links", None) if hasattr(item, "links") else None
            
            if markdown:
                result["content"] = markdown[:2000] + ("..." if len(markdown) > 2000 else "")
            elif html:
                result["content"] = html[:2000] + ("..." if len(html) > 2000 else "")
            
            if links and isinstance(links, list):
                result["links"] = links[:10]
        
        results.append(result)
    
    return _format_results(results, query, result_type)


def _format_results(results: list[dict], query: str, search_type: str = "general") -> ToolResult:
    """Format search results for output.
    
    Handles different result types:
    - web: url, title, description, position, category
    - news: title, url, snippet, date, position
    - images: title, imageUrl, imageWidth, imageHeight, url, position
    """
    type_label = f" ({search_type})" if search_type != "general" else ""
    lines = [f"Search results{type_label} for: {query}\n"]
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        snippet = result.get("snippet", "No description")
        content = result.get("content", "")
        date = result.get("date", "")
        author = result.get("author", "")
        category = result.get("category")
        position = result.get("position")
        links = result.get("links", [])
        
        # Handle image results
        imageUrl = result.get("imageUrl")
        imageWidth = result.get("imageWidth")
        imageHeight = result.get("imageHeight")
        
        lines.append(f"{i}. {title}")
        
        if url:
            lines.append(f"   URL: {url}")
        
        # For images, show image URL and dimensions
        if search_type == "images" and imageUrl:
            image_info = f"Image: {imageUrl}"
            if imageWidth and imageHeight:
                image_info += f" ({imageWidth}x{imageHeight})"
            lines.append(f"   {image_info}")
        
        # Add metadata
        meta_parts = []
        if position is not None:
            meta_parts.append(f"Position: {position}")
        if category:
            meta_parts.append(f"Category: {category}")
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
