"""Pipeline recommendation engine — catalog-driven, pure Python."""

from .pipeline_recommender import (
    generate_recommendation,
    list_catalog_models,
    PipelineRecommendationError,
)

__all__ = [
    "generate_recommendation",
    "list_catalog_models",
    "PipelineRecommendationError",
]
