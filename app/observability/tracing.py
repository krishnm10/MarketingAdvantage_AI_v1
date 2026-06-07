from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)

_TRACING_INITIALIZED: bool = False


def init_tracing(service_name: str = "marketing_advantage_ai") -> None:
    """
    Initialize OpenTelemetry tracing provider and optional OTLP exporter.

    Safe to call multiple times; only the first call wins.
    """
    global _TRACING_INITIALIZED
    if _TRACING_INITIALIZED:
        return

    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        exporter: Optional[OTLPSpanExporter] = None

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        )
        if endpoint:
            try:
                exporter = OTLPSpanExporter(endpoint=endpoint)
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)
                logger.info("[Tracing] OTLP exporter configured for endpoint=%s", endpoint)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning(
                    "[Tracing] Failed to initialize OTLP exporter (%s); continuing without remote export",
                    exc,
                )

        trace.set_tracer_provider(provider)
        _TRACING_INITIALIZED = True
        logger.info("[Tracing] Tracer provider initialized (service.name=%s)", service_name)
    except Exception as exc:  # pragma: no cover - tracing must not break startup
        logger.warning(
            "[Tracing] Tracing initialization failed (%s); proceeding without tracing", exc
        )

