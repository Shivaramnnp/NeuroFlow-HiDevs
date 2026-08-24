from __future__ import annotations

import base64
import io
import logging
from typing import Any, List, Optional, Union

from PIL import Image
import pytesseract

from .base import BaseExtractor, ExtractedPage
try:
    from backend.providers.base import ChatMessage
    from backend.providers.router import RoutingCriteria
    from backend.providers.client import NeuroFlowClient
except ImportError:
    from providers.base import ChatMessage
    from providers.router import RoutingCriteria
    from providers.client import NeuroFlowClient

logger = logging.getLogger("neuroflow-image-extractor")


class ImageExtractor(BaseExtractor):
    """
    Image Extractor:
    - Accepts JPEG, PNG, WEBP
    - Resizes to max 1024px on longest side before sending to Vision LLM
    - Uses Vision LLM (routed with require_vision=True) to generate a detailed description
    - Runs pytesseract OCR for textual content in image
    - Combines: description + "\n\nText found in image: " + ocr_text
    """

    def __init__(self, client: Optional[NeuroFlowClient] = None):
        self.client = client

    async def extract(self, source: Union[str, bytes, io.BytesIO], **kwargs) -> List[ExtractedPage]:
        if isinstance(source, bytes):
            img_file = io.BytesIO(source)
        elif isinstance(source, io.BytesIO):
            img_file = source
        else:
            img_file = source

        image = Image.open(img_file)
        orig_format = image.format or "PNG"
        orig_size = image.size

        # 1. Resize to max 1024px on longest side
        max_dim = max(image.width, image.height)
        if max_dim > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        # 2. Run OCR via pytesseract
        ocr_text = ""
        try:
            ocr_text = pytesseract.image_to_string(image).strip()
        except Exception as err:
            logger.warning(f"pytesseract OCR extraction failed: {err}")

        # 3. Encode resized image to base64 for Vision LLM
        buffered = io.BytesIO()
        save_format = orig_format if orig_format.upper() in ["JPEG", "PNG", "WEBP"] else "PNG"
        image.save(buffered, format=save_format)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mime_type = f"image/{save_format.lower()}"

        # 4. Generate vision description via LLM
        description = ""
        client = self.client or kwargs.get("client") or NeuroFlowClient.get_instance()
        if client is not None:
            try:
                # Multi-modal ChatMessage with vision
                messages = [
                    ChatMessage(
                        role="user",
                        content=[
                            {
                                "type": "text",
                                "text": "Describe this image in detail, explaining all diagrams, objects, charts, and visual structure.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{img_b64}"},
                            },
                        ],
                    )
                ]
                criteria = RoutingCriteria(task_type="rag_generation", require_vision=True)
                res = await client.chat(messages, criteria=criteria)
                description = res.content.strip()
            except Exception as err:
                logger.warning(f"Vision LLM description failed: {err}")
                description = f"Image ({save_format}, {image.width}x{image.height})"

        if not description:
            description = f"Image ({save_format}, {image.width}x{image.height})"

        # 5. Combine: description + "\n\nText found in image: " + ocr_text
        if ocr_text:
            combined = f"{description}\n\nText found in image: {ocr_text}"
        else:
            combined = description

        return [
            ExtractedPage(
                page_number=1,
                content=combined,
                content_type="image_description",
                metadata={
                    "original_size": orig_size,
                    "processed_size": image.size,
                    "format": save_format,
                    "has_ocr_text": bool(ocr_text),
                },
            )
        ]
