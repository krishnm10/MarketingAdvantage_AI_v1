---
name: architecture-reviewer
description: >
  Review software and AI system architectures for modularity, service separation,
  API boundaries, storage–compute separation, and migration readiness. Use when
  the user asks for an architecture review, system design critique, refactor
  proposal, or scalability and deployment strategy.
---

# Architecture Reviewer

## Purpose

This skill helps the agent act as a senior architecture reviewer for software and AI/RAG systems.
It focuses on system decomposition, service boundaries, data and storage design, deployment
topology, and readiness for change and scaling.

Use this skill when:
- The user asks for **system design** or **architecture review**
- The user shares an **architecture diagram**, **RAG pipeline**, or **service layout**
- The user wants to **refactor a monolith**, **separate services**, or **plan a migration**
- The user asks about **scalability**, **multi-region**, or **cloud-native deployment**

## Review Workflow

When using this skill, follow this workflow:

1. **Clarify the scope**
   - Identify the core business/domain problem the system is solving.
   - List the main capabilities (e.g., ingestion, retrieval, generation, analytics, billing).
   - Note hard constraints (latency, cost, compliance, data residency, SLOs).

2. **Map the current architecture**
   - Identify all major components: services, databases, queues, caches, file/object stores,
     third-party APIs, LLMs, vector DBs, and batch/stream processors.
   - Describe how data flows end-to-end for the key workflows.
   - Call out any tight coupling (shared databases, hidden back channels, shared mutable state).

3. **Assess modularity and coupling**
   - Check if responsibilities are clearly separated (single responsibility for each service/module).
   - Look for feature or data ownership (which service owns which tables/indexes/queues).
   - Flag patterns like shared “god” services, overgrown gateways, or cross-cutting logic spread
     across many places.

4. **Evaluate service separation**
   - Distinguish between:
     - **User-facing gateways/APIs**
     - **Domain services** (business logic)
     - **Infrastructure services** (ingestion, vector DB, cache, queue, object store, search)
   - Check that long-running or heavy ingestion is **decoupled** from online request paths.
   - Verify that services communicate via clear APIs or messages, not shared DB reads/writes.

5. **Evaluate API boundaries**
   - Check API contracts: request/response shapes, versioning, and error semantics.
   - Look for clear separation between **external APIs** and **internal service interfaces**.
   - Ensure idempotency and retry-safety where needed (especially for write operations).
   - Flag any APIs that leak internal storage details (e.g., table/column names, low-level schemas).

6. **Check storage vs compute separation**
   - List all storage systems: relational DBs, document stores, vector DBs, caches, object storage,
     logs, analytics stores.
   - Ensure compute (services, workers, functions) is **stateless** and can be horizontally scaled;
     avoid relying on local disk for durable state.
   - Verify that ingestion, indexing, and querying are **decoupled** from storage implementation
     (abstractions over DB/query engines, pluggable vector DBs, etc.).
   - Flag tight coupling where business logic depends heavily on vendor-specific features
     (e.g., DB-specific stored procedures, proprietary search syntax) without an abstraction layer.

7. **Assess migration readiness**
   - Identify how easy it is to:
     - Change schemas, move tables, or re-index vectors
     - Swap out infrastructure (vector DB, LLM provider, cache)
     - Introduce new services or split existing ones
   - Look for:
     - Backward-compatible schema patterns (nullable columns, expand-then-contract)
     - Feature flags and configuration-driven behavior
     - Use of metadata and versioning for data and embeddings
   - Flag risks where migrations require big-bang changes or coordinated multi-service downtime.

8. **RAG and AI-specific checks (if applicable)**
   - Verify clear separation between:
     - **Ingestion/indexing** (load, chunk, embed, store)
     - **Retrieval** (hybrid search, filters, ranking)
     - **Generation** (prompting, grounding, post-processing)
   - Ensure vector DB, object storage, and metadata DB are treated as external infrastructure
     services, not local files or in-process stores.
   - Check for:
     - Proper chunking strategy (semantic/recursive, overlap, per-source metadata)
     - Embedding consistency (single embedding space per index, documented dimensions/metric)
     - Hybrid retrieval plus reranking for most knowledge-heavy queries.

9. **Evaluate observability, resilience, and deployment**
   - Check for logging, metrics, and tracing across services (including correlation IDs).
   - Identify single points of failure, retry/backoff patterns, and graceful degradation plans.
   - Review deployment topology: containers, orchestration (e.g., Kubernetes), health checks,
     and resource isolation for heavy workloads.

10. **Synthesize findings and propose improvements**
   - Summarize the main structural issues and anti-patterns.
   - Propose a more modular, decoupled architecture that can evolve over time.
   - Outline migration paths that minimize risk and allow incremental rollout.

## Hard Architecture Failure Conditions

If any of the following are detected, mark system as **"Architecture Unsafe"**:

- Tight coupling between services via shared databases
- Ingestion and online request paths not decoupled
- No clear service boundaries (monolithic or god service patterns)
- Vendor-locked implementations with no abstraction layer
- No migration or versioning strategy for data or embeddings
- Missing separation between ingestion, retrieval, and generation in RAG systems

When triggered:
- Explicitly state: "This architecture is not safe for production"
- Prioritize structural fixes before optimization

## Output Format

Always structure your final response to the user using these top-level sections:

### Problem Breakdown
- Briefly restate the system’s purpose and primary workflows in your own words.

### Issues & Risk Analysis
- Identify and group **design flaws** and risks by area:
  - Modularity and coupling
  - Service separation
  - API boundaries
  - Storage vs compute separation
  - Scalability and performance
  - Migration readiness
  - (If applicable) RAG/AI-specific concerns

### Structured Solution
- Propose an improved target architecture:
  - Key services and their responsibilities
  - Data and storage strategy
  - Service and API boundaries
  - RAG/AI pipeline structure (if relevant)

### Optimizations
- Call out explicit **scaling strategies**, such as:
  - Horizontal scaling of stateless services
  - Dedicated ingestion/indexing workers
  - Caching, precomputation, and denormalization strategies
  - Multi-region or multi-tenant patterns, if needed

### Risks & Edge Cases
- Highlight migration risks, operational hazards, and edge cases:
  - Data migrations and backward compatibility
  - Failure modes and graceful degradation
  - Vendor lock-in and portability concerns

Additionally, to match concise user-facing expectations, include these three summary subsections
near the end of the response:

#### Design flaws
- Bullet list of the most critical architectural flaws, prioritized.

#### Improved architecture
- Short description of the recommended architecture, focusing on service and data boundaries.

#### Scalability suggestions
- Concrete scaling and resilience approaches (how to scale each major component and handle load).

Keep the overall response concise but specific, favoring clear trade-offs and actionable next steps
over exhaustive theoretical discussion.

## Data Flow Validation

Verify:

- End-to-end data flow is clearly defined and traceable
- No hidden data paths or implicit dependencies
- Data transformations are explicit and reproducible

Flag:
- Unclear or implicit data flows
- Data duplication or inconsistency across services

## Failure Domain Isolation

Ensure:

- Failures in one service do not cascade across the system
- Critical services are isolated (retrieval, LLM, ingestion)
- Circuit breakers and fallback paths exist

Flag:
- Shared dependencies causing cascading failures
- Lack of isolation boundaries

## Cost–Architecture Trade-offs

Evaluate:

- Infrastructure cost per service (LLM, vector DB, storage)
- Over-provisioned components
- Inefficient data movement between services

Flag:
- Architectures that scale cost faster than throughput
- Redundant services or pipelines


## Deployment and Infrastructure Validation

Check:

- Containerization and orchestration readiness (Docker, Kubernetes)
- Stateless service design for horizontal scaling
- Proper resource isolation (CPU, memory, GPU workloads)

Flag:
- Services that cannot scale independently
- Tight coupling between deployment units

## Versioning and Compatibility

Ensure:

- APIs are versioned
- Data schemas are backward-compatible
- Embedding models and indexes are versioned

Flag:
- Breaking changes without versioning
- Systems requiring coordinated updates across services


## Security and Access Boundaries

Check:

- Authentication and authorization at service boundaries
- Least-privilege access between services
- Secure handling of API keys, tokens, and secrets

Flag:
- Over-permissive access
- Missing auth between internal services

## Data Consistency Model

Evaluate:

- Consistency model (strong vs eventual)
- Synchronization between systems (DB, vector DB, cache)
- Handling of partial updates or failures

Flag:
- Inconsistent views of data across services
- Missing reconciliation mechanisms

## Multi-Tenancy Isolation

Ensure:

- Data isolation between tenants (metadata filters, DB separation)
- Query-level isolation in retrieval and LLM pipelines
- No cross-tenant leakage in caching or logs

Flag:
- Shared data paths without isolation
- Missing tenant identifiers in queries or storage

## Design vs Runtime Validation

Verify:

- Actual runtime behavior matches documented architecture
- No hidden service dependencies or undocumented communication paths
- Observability data (traces/metrics) aligns with intended architecture

Flag:
- Drift between architecture design and production behavior
- Shadow dependencies not captured in design