from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExtractedPage:
    page_number: int
    content: str
    content_type: str  # "text" | "table" | "image_description"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    """Abstract base class for all file and content extractors."""

    @abstractmethod
    async def extract(self, source: Any, **kwargs) -> List[ExtractedPage]:
        """Extract content from file path, bytes, or URL into a list of ExtractedPage objects."""
        ...
