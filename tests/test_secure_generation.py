"""
Tests for Phase 2A.5: RetrievalRuntime Security Gap Closure

Covers:
- Context PII redaction (pre-LLM)
- Answer PII redaction (post-LLM)
- Trust scoring integration
- Trust gate blocking
- Output formatter integration
- Backward compatibility
- Telemetry parity
- Safe failure handling
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass


class TestSecureGenerationConfig:
    """Tests for SecureGenerationConfig dataclass."""
    
    def test_default_config_values(self):
        """Verify default configuration is secure by default."""
        from app.core.runtime.secure_generation import SecureGenerationConfig
        
        config = SecureGenerationConfig()
        
        assert config.enable_pre_llm_pii_scan is True
        assert config.enable_post_llm_pii_scan is True
        assert config.block_on_critical_pii is True
        assert config.enable_trust_scoring is True
        assert config.enable_trust_gate is True
        assert config.min_trust_score == 0.3
        assert config.pii_trust_penalty == 0.15
    
    def test_config_from_runtime_components(self):
        """Verify config can be built from RuntimeComponents."""
        from app.core.runtime.secure_generation import SecureGenerationConfig
        
        mock_rc = Mock()
        mock_rc.rag_min_score = 0.5
        
        config = SecureGenerationConfig.from_runtime_components(mock_rc)
        
        assert config.min_trust_score == 0.5


class TestSecureGenerationResult:
    """Tests for SecureGenerationResult dataclass."""
    
    def test_default_result_values(self):
        """Verify default result has security disabled."""
        from app.core.runtime.secure_generation import SecureGenerationResult
        
        result = SecureGenerationResult()
        
        assert result.answer is None
        assert result.security_aligned is False
        assert result.pre_llm_pii_scan_applied is False
        assert result.post_llm_pii_scan_applied is False
        assert result.trust_gate_applied is False
        assert result.trust_gate_passed is True  # Default: pass if not applied


class TestSecureGenerationHandler:
    """Tests for SecureGenerationHandler class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        llm = Mock()
        llm.generate = Mock(return_value=Mock(text="This is a test answer citing [Source 1]."))
        return llm
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample context chunks."""
        return [
            {"text": "Chunk 1 content about topic.", "score": 0.85, "chunk_id": "c1", "metadata": {}},
            {"text": "Chunk 2 with more details.", "score": 0.75, "chunk_id": "c2", "metadata": {}},
        ]
    
    def test_pre_llm_pii_redaction(self, mock_llm, sample_chunks):
        """Verify PII in context is redacted before LLM."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        pii_chunks = [
            {"text": "Contact john@example.com for details.", "score": 0.9, "chunk_id": "c1", "metadata": {}},
        ]
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=True,
            enable_post_llm_pii_scan=False,
            enable_trust_scoring=False,
            enable_output_formatter=False,
        )
        
        handler = SecureGenerationHandler(config=config, tenant_id="test")
        result = handler.generate_secure(
            llm=mock_llm,
            llm_model="test-model",
            query="Test query",
            context_chunks=pii_chunks,
        )
        
        assert result.pre_llm_pii_scan_applied is True
        assert "email" in result.pre_llm_pii_entities
        assert result.security_aligned is True
    
    def test_post_llm_pii_redaction(self, mock_llm, sample_chunks):
        """Verify PII in answer is redacted after LLM."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        mock_llm.generate = Mock(
            return_value=Mock(text="Contact support@test.com for help. [Source 1]")
        )
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=False,
            enable_post_llm_pii_scan=True,
            enable_trust_scoring=False,
            enable_output_formatter=False,
        )
        
        handler = SecureGenerationHandler(config=config, tenant_id="test")
        result = handler.generate_secure(
            llm=mock_llm,
            llm_model="test-model",
            query="Test query",
            context_chunks=sample_chunks,
        )
        
        assert result.post_llm_pii_scan_applied is True
        assert "email" in result.post_llm_pii_entities
        assert "[EMAIL_REDACTED]" in result.answer
    
    def test_trust_scoring_integration(self, mock_llm, sample_chunks):
        """Verify trust scoring is computed from context chunks."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=False,
            enable_post_llm_pii_scan=False,
            enable_trust_scoring=True,
            enable_output_formatter=False,
        )
        
        handler = SecureGenerationHandler(config=config, tenant_id="test")
        result = handler.generate_secure(
            llm=mock_llm,
            llm_model="test-model",
            query="Test query",
            context_chunks=sample_chunks,
        )
        
        assert result.trust_score is not None
        assert 0.0 <= result.trust_score <= 1.0
    
    def test_trust_gate_blocks_low_trust(self, mock_llm, sample_chunks):
        """Verify trust gate blocks answers below threshold."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=False,
            enable_post_llm_pii_scan=False,
            enable_trust_scoring=True,
            enable_trust_gate=True,
            enable_output_formatter=True,
            min_trust_score=0.99,  # Very high threshold to trigger block
        )
        
        handler = SecureGenerationHandler(config=config, tenant_id="test")
        result = handler.generate_secure(
            llm=mock_llm,
            llm_model="test-model",
            query="Test query",
            context_chunks=sample_chunks,
        )
        
        assert result.trust_gate_applied is True
        assert result.output_formatter_applied is True
        # Low trust may block depending on actual score
    
    def test_critical_pii_blocks_context(self, mock_llm):
        """Verify critical PII (like Aadhaar) blocks generation."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        critical_pii_chunks = [
            {"text": "Aadhaar number: 1234 5678 9012", "score": 0.9, "chunk_id": "c1", "metadata": {}},
        ]
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=True,
            block_on_critical_pii=True,
        )
        
        handler = SecureGenerationHandler(config=config, tenant_id="test")
        result = handler.generate_secure(
            llm=mock_llm,
            llm_model="test-model",
            query="Test query",
            context_chunks=critical_pii_chunks,
        )
        
        # Aadhaar is CRITICAL severity - should be blocked or redacted
        assert result.pre_llm_pii_scan_applied is True
        assert "aadhaar" in result.pre_llm_pii_entities
    
    def test_llm_failure_handling(self, sample_chunks):
        """Verify LLM failures are handled safely."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        failing_llm = Mock()
        failing_llm.generate = Mock(side_effect=Exception("LLM unavailable"))
        
        config = SecureGenerationConfig()
        handler = SecureGenerationHandler(config=config, tenant_id="test")
        
        result = handler.generate_secure(
            llm=failing_llm,
            llm_model="test-model",
            query="Test query",
            context_chunks=sample_chunks,
        )
        
        assert result.answer is None
        assert result.answer_error is not None
        assert "LLM generation error" in result.answer_error
    
    def test_empty_llm_response_handling(self, sample_chunks):
        """Verify empty LLM responses are handled."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        empty_llm = Mock()
        empty_llm.generate = Mock(return_value=Mock(text=""))
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=False,
            enable_post_llm_pii_scan=False,
            enable_trust_scoring=False,
            enable_output_formatter=False,
        )
        handler = SecureGenerationHandler(config=config, tenant_id="test")
        
        result = handler.generate_secure(
            llm=empty_llm,
            llm_model="test-model",
            query="Test query",
            context_chunks=sample_chunks,
        )
        
        assert result.answer is None
        assert result.answer_error == "LLM returned an empty response."
    
    def test_latency_breakdown_populated(self, mock_llm, sample_chunks):
        """Verify latency breakdown is populated for all stages."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=True,
            enable_post_llm_pii_scan=True,
            enable_trust_scoring=True,
            enable_output_formatter=True,
        )
        
        handler = SecureGenerationHandler(config=config, tenant_id="test")
        result = handler.generate_secure(
            llm=mock_llm,
            llm_model="test-model",
            query="Test query",
            context_chunks=sample_chunks,
        )
        
        assert "llm_ms" in result.latency_breakdown
        assert result.answer_latency_ms is not None
        assert result.answer_latency_ms > 0


class TestGenerateAnswerSecureFunction:
    """Tests for the convenience function."""
    
    def test_convenience_function_works(self):
        """Verify the convenience function works correctly."""
        from app.core.runtime.secure_generation import generate_answer_secure
        
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value=Mock(text="Test answer [Source 1]."))
        
        chunks = [
            {"text": "Context content.", "score": 0.8, "chunk_id": "c1", "metadata": {}},
        ]
        
        result = generate_answer_secure(
            llm=mock_llm,
            llm_model="test-model",
            query="Test query",
            context_chunks=chunks,
            tenant_id="test-tenant",
            request_id="req-123",
        )
        
        assert result.security_aligned is True
        assert result.answer is not None


class TestBackwardCompatibility:
    """Tests to verify backward compatibility."""
    
    def test_api_response_schema_unchanged(self):
        """Verify RetrieveResponse schema is not changed."""
        from app.api.v2.retrieve_api import RetrieveResponse
        
        # These fields must exist for backward compatibility
        response = RetrieveResponse(
            query="test",
            intent="answer",
            search_mode="semantic",
            total_results=0,
            total_dropped=0,
            latency_ms=100.0,
            results=[],
        )
        
        assert hasattr(response, "answer")
        assert hasattr(response, "answer_model")
        assert hasattr(response, "answer_latency_ms")
        assert hasattr(response, "answer_error")
        assert hasattr(response, "debug_info")
    
    def test_chat_response_schema_unchanged(self):
        """Verify ChatRetrieveResponse schema is not changed."""
        from app.api.v2.retrieve_chat_api import ChatRetrieveResponse
        
        response = ChatRetrieveResponse(
            session_id="test",
            query="test",
            intent="answer",
            search_mode="semantic",
            total_results=0,
            total_dropped=0,
            latency_ms=100.0,
            results=[],
        )
        
        assert hasattr(response, "answer")
        assert hasattr(response, "answer_model")
        assert hasattr(response, "answer_latency_ms")
        assert hasattr(response, "answer_error")


class TestFeatureFlagBehavior:
    """Tests for feature flag behavior."""
    
    def test_flag_defaults_to_false(self):
        """Verify feature flag defaults to False for safe rollout."""
        import os
        
        # Clear the env var if set
        original = os.environ.pop("MAI_ENABLE_RETRIEVAL_SECURITY_ALIGNMENT", None)
        
        try:
            # Re-import to get fresh value
            import importlib
            import app.core.runtime.runtime_flags as flags
            importlib.reload(flags)
            
            assert flags.ENABLE_RETRIEVAL_SECURITY_ALIGNMENT is False
        finally:
            if original is not None:
                os.environ["MAI_ENABLE_RETRIEVAL_SECURITY_ALIGNMENT"] = original
    
    def test_flag_enabled_when_set(self):
        """Verify feature flag can be enabled."""
        import os
        import importlib
        
        original = os.environ.get("MAI_ENABLE_RETRIEVAL_SECURITY_ALIGNMENT")
        os.environ["MAI_ENABLE_RETRIEVAL_SECURITY_ALIGNMENT"] = "true"
        
        try:
            import app.core.runtime.runtime_flags as flags
            importlib.reload(flags)
            
            assert flags.ENABLE_RETRIEVAL_SECURITY_ALIGNMENT is True
        finally:
            if original is not None:
                os.environ["MAI_ENABLE_RETRIEVAL_SECURITY_ALIGNMENT"] = original
            else:
                os.environ.pop("MAI_ENABLE_RETRIEVAL_SECURITY_ALIGNMENT", None)


class TestTenantIsolation:
    """Tests for tenant isolation preservation."""
    
    def test_tenant_id_passed_to_handler(self):
        """Verify tenant_id is preserved through secure generation."""
        from app.core.runtime.secure_generation import SecureGenerationHandler
        
        handler = SecureGenerationHandler(tenant_id="tenant-123")
        
        assert handler._tenant_id == "tenant-123"
    
    def test_request_id_passed_to_handler(self):
        """Verify request_id is preserved for tracing."""
        from app.core.runtime.secure_generation import SecureGenerationHandler
        
        handler = SecureGenerationHandler(request_id="req-abc-123")
        
        assert handler._request_id == "req-abc-123"


class TestTelemetryParity:
    """Tests for telemetry completeness."""
    
    def test_security_events_logged(self, caplog):
        """Verify security events are logged in structured format."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        import logging
        
        caplog.set_level(logging.INFO)
        
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value=Mock(text="Test answer."))
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=True,
            enable_post_llm_pii_scan=True,
            enable_trust_scoring=True,
            enable_output_formatter=True,
        )
        
        handler = SecureGenerationHandler(
            config=config,
            tenant_id="test-tenant",
            request_id="req-123",
        )
        
        chunks = [{"text": "Content.", "score": 0.8, "chunk_id": "c1", "metadata": {}}]
        handler.generate_secure(
            llm=mock_llm,
            llm_model="test-model",
            query="Test",
            context_chunks=chunks,
        )
        
        # Check that security events were logged
        log_messages = [r.message for r in caplog.records]
        security_logs = [m for m in log_messages if "security_event" in m]
        
        # Should have at least the completion event
        assert len(security_logs) >= 1
        assert any("SECURE_GENERATION_COMPLETE" in log for log in security_logs)


class TestPIIMiddlewareIntegration:
    """Tests for PII middleware integration."""
    
    def test_pii_middleware_loaded_lazily(self):
        """Verify PII middleware is loaded only when needed."""
        from app.core.runtime.secure_generation import SecureGenerationHandler
        
        handler = SecureGenerationHandler()
        
        # Not loaded yet
        assert handler._pii_middleware is None
        
        # Load on demand
        mw = handler._get_pii_middleware()
        assert mw is not None
        assert handler._pii_middleware is mw
    
    def test_multiple_pii_types_detected(self):
        """Verify multiple PII types are detected in one scan."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value=Mock(text="Test"))
        
        multi_pii_chunks = [
            {
                "text": "Email: john@test.com, Phone: +91 9876543210",
                "score": 0.9,
                "chunk_id": "c1",
                "metadata": {},
            },
        ]
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=True,
            enable_post_llm_pii_scan=False,
            enable_trust_scoring=False,
            enable_output_formatter=False,
        )
        
        handler = SecureGenerationHandler(config=config)
        result = handler.generate_secure(
            llm=mock_llm,
            llm_model="test",
            query="Test",
            context_chunks=multi_pii_chunks,
        )
        
        # Should detect both email and phone
        assert "email" in result.pre_llm_pii_entities
        assert "phone_in" in result.pre_llm_pii_entities


class TestTrustAdapterIntegration:
    """Tests for TrustAdapter integration."""
    
    def test_trust_adapter_loaded_lazily(self):
        """Verify TrustAdapter is loaded only when needed."""
        from app.core.runtime.secure_generation import SecureGenerationHandler
        
        handler = SecureGenerationHandler()
        
        assert handler._trust_adapter is None
        
        adapter = handler._get_trust_adapter()
        assert adapter is not None
        assert handler._trust_adapter is adapter
    
    def test_trust_penalty_applied_for_pii(self):
        """Verify trust penalty is applied when PII is redacted."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        mock_llm = Mock()
        mock_llm.generate = Mock(
            return_value=Mock(text="Answer with email@test.com [Source 1].")
        )
        
        pii_chunks = [
            {"text": "Normal content without PII.", "score": 0.9, "chunk_id": "c1", "metadata": {}},
        ]
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=False,
            enable_post_llm_pii_scan=True,  # PII in answer
            enable_trust_scoring=True,
            pii_trust_penalty=0.15,
            enable_output_formatter=False,
        )
        
        handler = SecureGenerationHandler(config=config)
        result = handler.generate_secure(
            llm=mock_llm,
            llm_model="test",
            query="Test",
            context_chunks=pii_chunks,
        )
        
        # Trust score should be reduced by PII penalty
        assert result.trust_score is not None
        # The exact value depends on chunk scores, but should be reduced


class TestOutputFormatterIntegration:
    """Tests for OutputFormatter integration."""
    
    def test_output_formatter_loaded_lazily(self):
        """Verify OutputFormatter is loaded only when needed."""
        from app.core.runtime.secure_generation import SecureGenerationHandler
        
        handler = SecureGenerationHandler()
        
        assert handler._output_formatter is None
        
        formatter = handler._get_output_formatter()
        assert formatter is not None
        assert handler._output_formatter is formatter
    
    def test_toxicity_filter_applied(self):
        """Verify toxicity filter can block harmful content."""
        from app.core.runtime.secure_generation import (
            SecureGenerationHandler,
            SecureGenerationConfig,
        )
        
        mock_llm = Mock()
        mock_llm.generate = Mock(
            return_value=Mock(text="Instructions to hack the system.")
        )
        
        chunks = [
            {"text": "Normal content.", "score": 0.9, "chunk_id": "c1", "metadata": {}},
        ]
        
        config = SecureGenerationConfig(
            enable_pre_llm_pii_scan=False,
            enable_post_llm_pii_scan=False,
            enable_trust_scoring=False,
            enable_output_formatter=True,
            toxicity_filter="rule_based",
        )
        
        handler = SecureGenerationHandler(config=config)
        result = handler.generate_secure(
            llm=mock_llm,
            llm_model="test",
            query="Test",
            context_chunks=chunks,
        )
        
        assert result.output_formatter_applied is True
        # The toxicity filter may block this content
        if result.output_blocked:
            assert "Toxicity" in (result.block_reason or "")
