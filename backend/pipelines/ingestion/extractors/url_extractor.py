from __future__ import annotations

import logging
from typing import Any, List, Optional
import urllib.parse
import urllib.robotparser

import httpx
import trafilatura

from .base import BaseExtractor, ExtractedPage

logger = logging.getLogger("neuroflow-url-extractor")


class URLExtractor(BaseExtractor):
    """
    URL Content Extractor:
    - Fetches web pages asynchronously using httpx
    - Validates robots.txt permissions using RobotFileParser before fetching
    - Extracts main content and tables using trafilatura with include_tables=True
    - Extracts metadata: title, author, canonical URL, publish date
    """

    async def _check_robots_permission(self, url: str, client: httpx.AsyncClient) -> bool:
        """Check if robots.txt permits crawling the URL."""
        parsed = urllib.parse.urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        try:
            res = await client.get(robots_url, timeout=5.0)
            if res.status_code == 200:
                rp.parse(res.text.splitlines())
                return rp.can_fetch("*", url)
            # If robots.txt returns 404 or fails, allowed by default
            return True
        except Exception as err:
            logger.warning(f"Could not check robots.txt at {robots_url}: {err}")
            return True

    async def extract(self, source: str, **kwargs) -> List[ExtractedPage]:
        url = str(source)
        async with httpx.AsyncClient(headers={"User-Agent": "NeuroFlowBot/1.0"}, follow_redirects=True) as client:
            allowed = await self._check_robots_permission(url, client)
            if not allowed:
                raise PermissionError(f"Access to URL '{url}' is restricted by robots.txt")

            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            html_content = response.text

        # Extract main body text with tables
        extracted_content = trafilatura.extract(
            html_content,
            include_tables=True,
            output_format="txt",
        )

        if not extracted_content:
            extracted_content = html_content

        # Extract structured metadata
        metadata_dict = {"url": url}
        try:
            meta = trafilatura.extract_metadata(html_content)
            if meta:
                if meta.title:
                    metadata_dict["title"] = meta.title
                if meta.author:
                    metadata_dict["author"] = meta.author
                if meta.url:
                    metadata_dict["canonical_url"] = meta.url
                if meta.date:
                    metadata_dict["publish_date"] = meta.date
        except Exception as err:
            logger.warning(f"Metadata extraction failed for {url}: {err}")

        return [
            ExtractedPage(
                page_number=1,
                content=extracted_content,
                content_type="text",
                metadata=metadata_dict,
            )
        ]
