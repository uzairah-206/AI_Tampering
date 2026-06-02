"""
SMARTZI — Gemini Semantic Initial Analysis & Key Rotation
Acts as the first line of defense, analyzing visual artifacts for AI generation.
Automatically falls back to alternate API keys if quota is exceeded.
"""

import logging
import os
import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from PIL import Image

logger = logging.getLogger("smartzi.gemini")

class GeminiInitialAnalysis(BaseModel):
    ai_probability: float = Field(
        description="Confidence score from 0.0 to 1.0 indicating how likely the image is AI-generated based on visual artifacts."
    )
    explanation: str = Field(
        description="Strictly 2 to 3 sentences explaining the visual reasoning for the score. Do not exceed 3 sentences."
    )

def run_gemini_initial_analysis(
    image_path: str,
    api_keys: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """
    Evaluates the image visually before forensics run. 
    Rotates through api_keys automatically if one fails/hits quota.
    """
    result: Dict[str, Any] = {
        "available": False,
        "ai_probability": 0.0,
        "explanation": "",
    }

    if not api_keys or not any(api_keys):
        logger.warning("Gemini: No API keys provided for rotation.")
        return result

    from google import genai
    from google.genai.errors import APIError

    image_pil = Image.open(image_path).convert("RGB")
    
    prompt = """Forensic AI analyst. Score this image 0.0-1.0 for AI generation probability. Check: structural inconsistencies, lighting errors, impossible anatomy, synthetic textures, frequency artifacts. Return ai_probability and 2-3 sentence explanation. No filler."""
    
    parts = [image_pil, prompt]

    for attempt, current_key in enumerate(api_keys):
        key = current_key.strip()
        if not key:
            continue
            
        try:
            logger.info("Attempting Gemini analysis with API Key slot %d...", attempt + 1)
            client = genai.Client(api_key=key)
            
            response = client.models.generate_content(
                model=model_name,
                contents=parts,
                config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                    "response_schema": GeminiInitialAnalysis,
                },
            )

            if response and response.text:
                data = json.loads(response.text)
                result["available"] = True
                result["ai_probability"] = float(data.get("ai_probability", 0.0))
                result["explanation"] = str(data.get("explanation", "")).strip()
                return result

        except Exception as exc:
            err_msg = str(exc)
            logger.error("Gemini attempt %d failed: %s", attempt + 1, err_msg)
            
            # If it's a quota/rate limit error, continue to the next key. Otherwise, break.
            if "429" in err_msg or "quota" in err_msg.lower() or "503" in err_msg:
                logger.info("Quota exceeded or service unavailable. Rotating to next key...")
                continue
            else:
                logger.error("Non-recoverable error encountered. Aborting Gemini stage.")
                break

    logger.warning("All Gemini API keys exhausted or failed.")
    return result

class GeminiService:
    def __init__(self):
        from app.core.config import settings
        # Load keys from .env as a comma-separated string, split into a list
        raw_keys = os.environ.get("GEMINI_API_KEYS", settings.GEMINI_API_KEY)
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.model_name = settings.GEMINI_MODEL or "gemini-2.5-flash"
        
    async def chat(self, prompt: str) -> str:
        if not self.api_keys:
            raise ConnectionError("No GEMINI_API_KEYS set.")
            
        from google import genai
        from google.genai.errors import APIError
        import asyncio
        
        full_prompt = f"You are SMARTZI, a concise AI forensics copilot. Answer questions about image tampering, ELA, EXIF, and AI detection. Be direct and technical.\nUser: {prompt}"
        
        for attempt, key in enumerate(self.api_keys):
            key = key.strip()
            if not key:
                continue
            try:
                client = genai.Client(api_key=key)
                
                def _generate(c=client):
                    resp = c.models.generate_content(
                        model=self.model_name,
                        contents=full_prompt,
                        config={"temperature": 0.4}
                    )
                    return resp.text if resp and resp.text else ""
                    
                return await asyncio.to_thread(_generate)
            except Exception as exc:
                err_msg = str(exc)
                logger.error("Chat attempt %d failed: %s", attempt + 1, err_msg)
                if "429" in err_msg or "quota" in err_msg.lower() or "503" in err_msg:
                    continue
                raise ConnectionError(f"Gemini chat failed: {err_msg}")
        
        raise ConnectionError("All Gemini API keys exhausted for chat.")
        
gemini_service = GeminiService()