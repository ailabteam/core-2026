from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Tuple

import google.api_core.exceptions
import google.generativeai as genai

from src.config import get_settings

logger = logging.getLogger(__name__)


PROMPT = """You are an OCR parser. Analyze the image and return ONLY valid JSON, no markdown, no code blocks, no explanations.

Required JSON schema:
{
  "page": 0,
  "width": 1200,
  "height": 1600,
  "blocks": [
    {
      "type": "text",
      "bbox": [x0, y0, x1, y1],
      "lines": [
        {
          "text": "extracted text here",
          "bbox": [x0, y0, x1, y1]
        }
      ]
    },
    {
      "type": "table",
      "bbox": [x0, y0, x1, y1],
      "table": {
        "rows": [
          [{"text": "cell text", "bbox": [x0,y0,x1,y1]}]
        ]
      }
    }
  ],
  "full_text": "all text concatenated in reading order"
}

Return ONLY the JSON object, nothing else."""


class GeminiClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        genai.configure(api_key=self.settings.gemini_api_key)
        self.model = genai.GenerativeModel(self.settings.model_name)
        self.request_count = 0  # Counter for API requests
    
    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract JSON from text, handling markdown code blocks and surrounding text.
        """
        text = text.strip()
        
        # Try direct JSON parse first
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks (```json ... ```)
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try finding JSON object boundaries
        # Look for first { and last }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_candidate = text[start_idx:end_idx + 1]
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON array boundaries
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_candidate = text[start_idx:end_idx + 1]
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass
        
        return ""

    def ocr_page(self, page_index: int, image_bytes: bytes, image_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Call Gemini with retry/backoff. Returns dict.
        Handles DeadlineExceeded with progressive timeout increase.
        """
        delay = 1.0
        base_timeout = self.settings.api_timeout
        image_size_mb = len(image_bytes) / (1024 * 1024)
        logger.info(f"Page {page_index}: {image_size[0]}x{image_size[1]}, {image_size_mb:.2f}MB")
        
        for attempt in range(1, self.settings.retry_times + 1):
            # Progressive timeout: increase for each retry
            timeout = int(base_timeout * (1.5 ** (attempt - 1)))
            try:
                logger.debug(f"Attempt {attempt}/{self.settings.retry_times}, timeout={timeout}s")
                # Increment request counter
                self.request_count += 1
                logger.debug(f"API request #{self.request_count} for page {page_index}")
                
                resp = self.model.generate_content(
                    [PROMPT, {"mime_type": "image/png", "data": image_bytes}],
                    request_options={"timeout": timeout},
                )
                text = resp.text.strip()
                
                # Log first 500 chars for debugging
                logger.debug(f"Response preview (first 500 chars): {text[:500]}")
                
                # Try to extract JSON from markdown code blocks
                json_text = self._extract_json(text)
                if not json_text:
                    logger.warning(f"Could not extract JSON from response. Full text: {text[:1000]}")
                    raise json.JSONDecodeError("No JSON found in response", text, 0)
                
                data = json.loads(json_text)
                # ensure page metadata present
                data.setdefault("page", page_index)
                data.setdefault("width", image_size[0])
                data.setdefault("height", image_size[1])
                logger.info(
                    f"Page {page_index} OCR completed successfully "
                    f"(used {attempt} request{'s' if attempt > 1 else ''})"
                )
                return data
            except google.api_core.exceptions.DeadlineExceeded as exc:
                logger.warning(f"Page {page_index} attempt {attempt}: DeadlineExceeded (timeout={timeout}s)")
                if attempt == self.settings.retry_times:
                    logger.error(f"Page {page_index} failed after {self.settings.retry_times} attempts")
                    raise RuntimeError(f"OCR timeout after {self.settings.retry_times} attempts. Image may be too large or complex.") from exc
                time.sleep(delay)
                delay *= self.settings.retry_backoff
                continue
            except json.JSONDecodeError as exc:
                logger.error(f"Page {page_index} attempt {attempt}: Invalid JSON response: {exc}")
                logger.error(f"Response text (first 1000 chars): {text[:1000] if 'text' in locals() else 'N/A'}")
                if attempt == self.settings.retry_times:
                    raise RuntimeError(
                        f"Invalid JSON response from Gemini API after {self.settings.retry_times} attempts. "
                        f"Response preview: {text[:500] if 'text' in locals() else 'N/A'}"
                    ) from exc
                time.sleep(delay)
                delay *= self.settings.retry_backoff
                continue
            except Exception as exc:  # broad catch for other API errors
                logger.warning(f"Page {page_index} attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt == self.settings.retry_times:
                    raise
                time.sleep(delay)
                delay *= self.settings.retry_backoff
                continue
        return {}
    
    def get_request_count(self) -> int:
        """Get total number of API requests made."""
        return self.request_count
    
    def reset_counter(self) -> None:
        """Reset the request counter."""
        self.request_count = 0

