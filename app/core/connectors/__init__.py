"""
Connectors — public API for the connectors package.
File: app/core/connectors/__init__.py

Import from here, never from individual files directly:
    from app.core.connectors import WebConnector, APIConnector, RSSConnector
    from app.core.connectors import BaseConnector, ConnectorResult
"""
from app.core.connectors.base import BaseConnector, ConnectorResult
from app.core.connectors.web_connector import WebConnector
from app.core.connectors.rss_connector import RSSConnector
from app.core.connectors.api_connector import APIConnector
from app.core.connectors.kafka_connector import KafkaConnector

__all__ = [
    "BaseConnector",
    "ConnectorResult",
    "WebConnector",
    "RSSConnector",
    "APIConnector",
    "KafkaConnector",
]
