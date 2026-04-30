"""
================================================================================
Marketing Advantage AI — Runtime Configuration
File: app/core/runtime/runtime_flags.py

DESIGN: Config-driven architecture with ZERO migration flags.

This module provides runtime configuration constants. All behavior is
controlled via ClientConfig, not environment variables.

The shared retrieval and generation executors are ALWAYS enabled.
There are no dual-path or shadow execution modes.
================================================================================
"""

# Shared executors are the ONLY execution path.
# These are not feature flags - they are constants that confirm
# the architecture uses unified execution.
ENABLE_SHARED_RETRIEVAL: bool = True
ENABLE_SHARED_GENERATION: bool = True
ENABLE_SHARED_CACHE: bool = True

# Runtime context is always enabled for proper telemetry and tracing.
ENABLE_RUNTIME_CONTEXT: bool = True
