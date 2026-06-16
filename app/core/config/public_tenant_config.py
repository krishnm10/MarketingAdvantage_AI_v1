"""
Frontend-safe projection of tenant configuration.

Strips secrets backends, secret references, and internal credentials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from app.core.config.client_config_schema import FeatureFlags, ParserConfig
from app.core.config.secret_ref import SecretsBackendConfig

if TYPE_CHECKING:
    from app.core.config.client_config_schema import ClientConfig


class PublicVisionConfig(BaseModel):
    """Multimodal / vision settings safe for admin UI (no secret refs)."""

    ai_profile: str = "cpu"
    vision_model_cpu: str
    vision_model_gpu: str
    vision_api_provider: str = "openai"
    vision_api_model: str = "gpt-4o"
    vision_api_configured: bool = False
    vision_quantize: str = "4bit"
    vision_flash_attention: bool = True
    enable_visual_explanation: bool = True
    enable_audio_language_detection: bool = True
    video_vision_frames: int = 8


class PublicIngestionConfig(BaseModel):
    enable_visual_llm_explanation: bool = True
    visual_llm_concurrency: int = 4
    vision: PublicVisionConfig


class PublicProviderFlags(BaseModel):
    embedder_type: str
    embedder_configured: bool
    llm_configured: bool
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    vectordb_type: str
    reranker_enabled: bool = False
    reranker_type: Optional[str] = None
    secret_store_provider: Optional[str] = Field(
        None,
        description="Secrets store provider id only — connection details are never exposed.",
    )


class PublicTenantConfig(BaseModel):
    """
    Safe tenant config for browser / public API surfaces.

    MUST NOT contain ``secrets_backend``, ``secret_ref``, database passwords,
    or raw credential material.
    """

    client_id: str
    client_name: str = ""
    description: str = ""
    version: str = "1.0"
    providers: PublicProviderFlags
    features: FeatureFlags
    parsers: ParserConfig
    ingestion: PublicIngestionConfig

    @classmethod
    def from_client_config(cls, config: "ClientConfig") -> "PublicTenantConfig":
        return cls.build_from_client_config(config)

    @staticmethod
    def _embedder_has_secret(config: "ClientConfig") -> bool:
        et = config.embedder.type.value
        sub = getattr(config.embedder, et, None)
        if sub is None:
            return False
        return getattr(sub, "secret_ref", None) is not None

    @staticmethod
    def _llm_summary(config: "ClientConfig") -> tuple[bool, Optional[str], Optional[str]]:
        if not config.llm or not config.llm.single:
            return False, None, None
        s = config.llm.single
        return True, s.type.value, s.model

    @staticmethod
    def _vision_public(config: "ClientConfig") -> PublicVisionConfig:
        v = config.ingestion.vision
        return PublicVisionConfig(
            ai_profile=v.ai_profile,
            vision_model_cpu=v.vision_model_cpu,
            vision_model_gpu=v.vision_model_gpu,
            vision_api_provider=v.vision_api_provider,
            vision_api_model=v.vision_api_model,
            vision_api_configured=v.vision_api_secret_ref is not None,
            vision_quantize=v.vision_quantize,
            vision_flash_attention=v.vision_flash_attention,
            enable_visual_explanation=v.enable_visual_explanation,
            enable_audio_language_detection=v.enable_audio_language_detection,
            video_vision_frames=v.video_vision_frames,
        )

    @classmethod
    def build_from_client_config(cls, config: "ClientConfig") -> "PublicTenantConfig":
        llm_ok, llm_provider, llm_model = cls._llm_summary(config)
        sb: Optional[SecretsBackendConfig] = config.secrets_backend
        provider_only = sb.provider.value if sb is not None else None

        return cls(
            client_id=config.client_id,
            client_name=config.client_name,
            description=config.description,
            version=config.version,
            providers=PublicProviderFlags(
                embedder_type=config.embedder.type.value,
                embedder_configured=cls._embedder_has_secret(config),
                llm_configured=llm_ok,
                llm_provider=llm_provider,
                llm_model=llm_model,
                vectordb_type=config.vectordb.type.value,
                reranker_enabled=config.is_reranking_enabled(),
                reranker_type=(
                    config.reranker.type.value if config.reranker is not None else None
                ),
                secret_store_provider=provider_only,
            ),
            features=config.features.model_copy(deep=True),
            parsers=config.parsers.model_copy(deep=True),
            ingestion=PublicIngestionConfig(
                enable_visual_llm_explanation=config.ingestion.enable_visual_llm_explanation,
                visual_llm_concurrency=config.ingestion.visual_llm_concurrency,
                vision=cls._vision_public(config),
            ),
        )

    def model_dump_public(self) -> dict:
        """Serialize for API responses; double-check no secret keys leak."""
        data = self.model_dump(mode="json")
        forbidden = ("secret_ref", "secrets_backend", "api_key_env", "password_env", "token_env")
        dumped = str(data).lower()
        for key in forbidden:
            if key in dumped:
                raise RuntimeError(f"PublicTenantConfig leak detected: {key}")
        return data
