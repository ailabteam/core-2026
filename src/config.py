from __future__ import annotations

from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    model_name: str = Field("gemini-2.5-flash", env="GEMINI_MODEL_NAME")
    max_image_dim: int = Field(2000, env="OCR_MAX_IMAGE_DIM")
    pdf_dpi: int = Field(300, env="OCR_PDF_DPI")
    enable_deskew: bool = Field(True, env="OCR_ENABLE_DESKEW")
    deskew_threshold: float = Field(2.0, env="OCR_DESKEW_THRESHOLD")
    retry_times: int = Field(3, env="OCR_RETRY_TIMES")
    retry_backoff: float = Field(2.0, env="OCR_RETRY_BACKOFF")
    api_timeout: int = Field(300, env="OCR_API_TIMEOUT")
    output_dir: str = Field("outputs", env="OCR_OUTPUT_DIR")
    allowed_exts: List[str] = Field(
        default_factory=lambda: [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    )

    # Print the environment variables

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unexpected envs to avoid ValidationError
    )

    @field_validator("gemini_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("GEMINI_API_KEY is required")
        return v


@lru_cache()
def get_settings() -> Settings:
    load_dotenv(override=False)
    return Settings()

