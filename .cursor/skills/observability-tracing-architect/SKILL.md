---
name: observability-tracing-architect
description: Design and review observability, logging, metrics, and distributed tracing architectures for RAG and LLM-powered systems. Use when the user asks to instrument pipelines, define traces/metrics/logs, set up telemetry backends, or debug latency and errors across services and microservices.
---

# Observability & Tracing Architect

## Instructions

Use this skill to design or refine observability for AI, RAG, and microservice-based systems. Focus on:

- End-to-end visibility for each request
- Fast root-cause analysis
- SLIs/SLOs that reflect real user experience
- Cost-aware telemetry (avoid noisy, high-cardinality data)

Always think in terms of **logs + metrics + traces**, with tracing as the backbone that ties everything together.

---

## When to Apply This Skill

Use this skill when:

- The user wants to add or improve logging, metrics, or tracing
- The user is debugging latency, errors, or flaky behavior in a distributed system
- A RAG or LLM pipeline needs stage-level visibility (retrieval vs rerank vs generation)
- The team is standardizing observability across services
- The user is unsure how to structure tracing (span boundaries, attributes, sampling)

---

## High-Level Workflow

Follow this workflow:

1. **Map the system and critical paths**
2. **Define observability goals and SLIs/SLOs**
3. **Design a tracing model (spans, attributes, links)**
4. **Define metrics and logging standards**
5. **Integrate with telemetry backends**
6. **Design dashboards and alerts**
7. **Validate with real failure and load scenarios**

Keep the design **implementation-agnostic but concrete** enough to be translated into code and configuration.

---

## 1. Understand the System & Requirements

First, build a mental model of the system:

- **Architecture**
  - Entry points: HTTP APIs, message queues, scheduled jobs
  - Services: API gateway, orchestrators, retrievers, rerankers, LLM gateways, workers
  - Data stores: vector DB, relational DB, caches, object storage
- **Workflows**
  - RAG / LLM request flow: input → safety / validation → retrieval → reranking → prompt assembly → LLM call → post-processing → output
- **Non-functional requirements**
  - Latency targets (P50, P95, P99)
  - Error budgets and availability
  - Throughput expectations
  - Compliance / PII constraints

Clarify target questions observability must answer, for example:

- "Why is P95 latency spiking?"
- "Which stage is slow for this tenant?"
- "Which vector DB queries or LLM calls are causing errors?"
- "Is retrieval recall degrading over time?"

---

## 2. Core Observability Principles

Apply these principles:

- **Request-centric view**
  - Every external request should map to a **single trace** with a stable `trace_id`
  - All internal calls should propagate context (traceparent headers or equivalent)

- **Structured, minimal logs**
  - Use JSON logs with consistent keys (`timestamp`, `level`, `service`, `request_id`, `trace_id`, `span_id`, `tenant_id`, `message`, `context`)
  - Avoid logging PII; redact or hash sensitive fields
  - Prefer **few, meaningful logs** over chatty line-by-line logging

- **Metrics over logs for aggregates**
  - Use metrics for rates, counts, durations, and resource usage
  - Reserve logs for debugging, audits, and unusual events

- **Traces for causality**
  - Use spans to represent meaningful operations, not individual lines of code
  - Use attributes to capture parameters that matter for debugging and analysis

- **Cost-aware design**
  - Avoid unbounded cardinality labels (e.g., `user_id` in metrics)
  - Use sampling for traces (head-based or tail-based) with higher sampling for errors and slow requests

---
## Hard Observability Failure Conditions

If any of the following are detected, mark system as **"Observability Insufficient"**:

- Missing trace coverage for critical request paths
- No correlation between logs, metrics, and traces
- No visibility into key RAG stages (retrieval, rerank, LLM)
- High error rates with no traceability
- Missing request_id / trace_id propagation

When triggered:
- Explicitly state: "System lacks sufficient observability for production debugging"
- Prioritize instrumentation before optimization

## 3. RAG & LLM Pipeline Tracing Model

Design traces so that each user request becomes a **single trace** with well-defined spans.

### 3.1 Recommended Span Structure

Root span (kind=SERVER):

- `rag_request` or `llm_request`

Child spans (examples):

- `validate_input`
- `safety_checks` (prompt / content filtering)
- `retrieve_documents`
- `vector_db_query` (one per underlying query or batch)
- `rerank_documents`
- `build_context` (prompt assembly / context construction)
- `llm_call` (one per model invocation)
- `post_process_output`
- `write_audit_log` / `store_conversation`

For streaming responses, consider additional spans such as:

- `llm_stream_setup`
- `llm_stream_tokens`

### 3.2 Key Span Attributes

Capture attributes that are:

- Safe (no PII)
- Stable and useful for debugging, routing, and analysis

Examples:

- Request-level
  - `request.id` (internal request ID, not user PII)
  - `tenant.id`
  - `user.segment` (coarse-grained, e.g., "free", "pro", "enterprise")
  - `channel` (web, api, batch)

- Retrieval
  - `retrieval.top_k`
  - `retrieval.strategy` (dense, sparse, hybrid)
  - `retrieval.vector_db` (e.g., qdrant, pinecone, milvus)
  - `retrieval.latency_ms`
  - `retrieval.result_count`

- Reranking
  - `reranker.model`
  - `reranker.latency_ms`
  - `reranker.result_count`

- LLM calls
  - `llm.provider`
  - `llm.model`
  - `llm.temperature`
  - `llm.max_tokens`
  - `llm.tokens.prompt`
  - `llm.tokens.completion`
  - `llm.latency_ms`
  - `llm.cache_hit` (true/false)

- Context
  - `context.tokens.total`
  - `context.tokens.truncated` (if context overflow handling is in place)


## Trace Completeness Validation

Check:

- All critical pipeline stages have spans
- No missing segments in request flow
- Parent-child relationships are intact

Flag:
- Partial traces
- Missing spans for key operations

---

## 4. Metrics Design

Define metrics that align with RAG/LLM behavior and user-facing SLIs.

### 4.1 Core Metrics

At minimum:

- **Latency**
  - `request_latency_seconds` (histogram): per API, per tenant or segment
  - Stage-specific: `retrieval_latency_seconds`, `rerank_latency_seconds`, `llm_latency_seconds`

- **Errors**
  - `request_errors_total`: by error type (validation, retrieval, ranking, generation, timeout)
  - `external_errors_total`: upstream provider failures (LLM, vector DB, storage)

- **Throughput**
  - `requests_total`
  - `llm_calls_total`
  - `retrieval_queries_total`

- **Tokens & context**
  - `llm_prompt_tokens_total`
  - `llm_completion_tokens_total`
  - `context_tokens_total`
  - `context_truncation_events_total`

### 4.2 RAG Quality Signals (Optional)

Include metrics that help correlate observability with retrieval quality:

- `retrieval_docs_returned` (distribution)
- `retrieval_filter_miss_total` (when filters return zero docs)
- `reranker_reordered_fraction` (fraction of results whose position changed)

These metrics help differentiate **retrieval failure**, **ranking failure**, and **context failure**.

---

## 5. Logging Standards

Define clear logging rules:

- **Levels**
  - `DEBUG`: detailed diagnostic data, disabled in production by default
  - `INFO`: high-level lifecycle events (request started/finished, stage completed)
  - `WARN`: unusual but recoverable conditions (fallbacks, degraded mode)
  - `ERROR`: failed operations impacting the request

- **Structure**
  - Always include `trace_id`, and if possible `span_id` and `request_id`
  - Use `error.type`, `error.message`, `error.stack` (where appropriate)
  - Tag logs with stage identifiers (`stage: retrieval`, `stage: llm`, etc.)

- **Privacy**
  - Do not log raw prompts, user inputs, or document contents unless explicitly allowed and redacted
  - Prefer hashes or sampled redacted examples for debugging

- **PII and Data Leak Protection**

 Enforce:

   - Automatic redaction of sensitive fields
   - Hashing of identifiers where needed
   - No raw prompts or documents in logs by default

 Flag:
   - Any logging that exposes user data, prompts, or internal context

---

## 6. Distributed Tracing Implementation Blueprint

When advising on implementation:

- **Use OpenTelemetry (OTel) where possible**
  - Adopt OTel SDKs for supported languages
  - Enable auto-instrumentation for HTTP/gRPC servers, database clients, and messaging libraries
  - Add manual spans around RAG-specific stages

- **Context propagation**
  - Enforce propagation via HTTP headers (`traceparent`), gRPC metadata, or message headers
  - Ensure background jobs started from a request copy or link to the original trace context

- **Sampling**
  - Start with conservative head-based sampling (for example 5–20%)
  - Increase sampling for:
    - Error traces (`status.error == true`)
    - High-latency traces (e.g., `latency_ms > P95`)
  - For high-traffic systems, consider tail-based sampling in the backend
   
- **Adaptive Sampling Strategy**

  Define:

    - Baseline sampling rate (e.g., 5–10%)
    - Always sample:
    - Errors
    - High-latency requests (P95+)
    - Dynamically reduce sampling under high load
    - Increase sampling for specific tenants or endpoints during debugging

  Ensure:
   - Critical traces are never dropped
   - Sampling does not hide systemic issues

- **Backends**
  - Keep recommendations backend-agnostic (e.g., "traces backend", "metrics backend", "logging sink")
  - Mention typical stacks (Prometheus, Grafana, Tempo, OpenSearch) as examples, not hard requirements

- **Context Propagation Enforcement**

 Ensure:

 - trace_id is propagated across all services
 - Background jobs maintain linkage to parent traces
 - No orphan spans

 Flag:
   - Missing or broken trace context

---

## 7. Dashboards and Alerts

Help the user design observability assets that answer real questions.

### 7.1 Dashboards

Recommend dashboards such as:

- **Request overview**
  - Request rate, error rate, latency percentiles
  - Split by API route, tenant, or feature flag

- **RAG pipeline**
  - Stage latencies (retrieval, rerank, LLM)
  - Context size distributions and truncation events
  - Vector DB query rate and errors

- **LLM behavior**
  - Token usage over time by model and tenant
  - Cache hit rates (if using an LLM cache)
  - Response time distributions per model

### 7.2 Alerts

Define SLO-backed alerts:

- High-level:
  - Error rate above threshold for sustained window
  - P95 or P99 latency above SLO
  - Sudden drop in traffic or spikes indicating incidents

- RAG-specific:
  - Retrieval zero-result rate exceeds threshold
  - Context truncation events spike
  - Specific upstream dependency (vector DB, LLM provider) latency or error spikes

Always require an **actionable runbook** for each alert.

---

## 8. Failure Analysis & Tuning Workflow

When the user is debugging an issue:

1. **Classify failure type**
   - Retrieval failure (missing relevant docs)
   - Ranking failure (relevant docs but poor ordering)
   - Context failure (overflow/truncation)
   - Generation failure (hallucination, formatting, policy violations)
2. **Use traces and metrics to localize**
   - Look for slow or error spans
   - Correlate with metrics (spikes in latency or errors)
3. **Inspect logs for context**
   - Filter logs by `trace_id` or `request_id`
   - Look for warnings indicating fallbacks or degraded modes
4. **Recommend instrumentation gaps to close**
   - Missing attributes
   - Missing spans around critical stages
   - Lack of stage-specific metrics

Propose concrete changes to instrumentation and configuration based on what is missing.

---

## 9. Deliverables Checklist

When using this skill to design or refine observability, aim to produce:

- A clear **tracing model** (span hierarchy, key attributes, sampling strategy)
- A **metrics specification** (names, types, labels, and SLIs/SLOs)
- A **logging standard** (structure, levels, redaction rules)
- A list of **dashboards and alerts** mapped to use cases and failure modes
- A prioritized list of **implementation steps** for the engineering team

Keep the outputs concise, strongly structured, and directly actionable.

