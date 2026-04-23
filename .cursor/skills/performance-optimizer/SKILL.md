---
name: performance-optimizer
description: Analyze system performance by identifying latency bottlenecks, throughput limits, and blocking operations across services and pipelines. Use when the user asks to profile or optimize performance, latency, throughput, concurrency, or caching for an application or RAG/LLM system.
---

# Performance Optimizer

This skill guides the agent to analyze and optimize system performance end-to-end and propose concrete, actionable improvements.

The agent should use this skill when:
- The user asks to **analyze, profile, or optimize performance**
- The user reports **high latency, low throughput, or timeouts**
- The user suspects **blocking operations, poor parallelization, or missing caching**

Always aim for **production-grade**, **cost-aware**, and **observable** designs.

---

## Required Inputs

Before analyzing performance, try to gather or infer:

- **High-level system description**
  - Architecture diagram or prose: services, queues, databases, caches, external APIs, LLM/RAG components.
  - Main user flows (e.g., request/response APIs, batch jobs, ingestion pipelines, RAG query flows).
- **Current performance signals**
  - Latency metrics (P50/P95/P99) per endpoint or job.
  - Throughput metrics (RPS, jobs/min, QPS) and known limits.
  - Error/timeout rates and any recent incidents.
- **Runtime artifacts (as available)**
  - Traces (e.g., OpenTelemetry, APM tools).
  - Logs with timing information.
  - Profiler output (CPU, memory, async-blocking, DB query timings).
- **Constraints and SLOs**
  - Target latency and throughput goals.
  - Cost constraints (e.g., LLM budget, infra budget).

If code or infrastructure is in a repo, **inspect it directly** (using search/semantic tools) rather than relying only on prose.

---

## Core Classification Axes

When reasoning about performance, always classify issues along these axes:

- **Latency bottlenecks**: Single requests are too slow.
- **Throughput limits**: The system cannot handle enough concurrent work.
- **Blocking operations**: Critical threads or event loops are blocked by slow I/O or heavy CPU.

For each issue, identify:
- **Location** (component, endpoint, job, function).
- **Root cause type** (I/O-bound, CPU-bound, lock contention, external dependency, configuration).
- **Evidence** (metrics, traces, code patterns).

---

## Hard Performance Failure Conditions

If any of the following are detected, mark system as **"Performance Unsafe"**:

- P95/P99 latency exceeds defined SLO
- System cannot sustain expected throughput under load
- Blocking operations on main request path
- No caching for high-frequency expensive operations
- No backpressure or overload handling

When triggered:
- Explicitly state: "This system is performance unsafe"
- Prioritize fixes before further optimization


## Analysis Workflow

When applying this skill, follow these steps:

1. **Map critical paths**
   - For each key user flow, map the full path: `Client → Gateway/API → Services → DB/Cache/Vector DB → External APIs/LLMs → Response`.
   - Note where synchronous calls, transactions, and cross-service calls occur.
  
  **Hot Path Identification**

    Identify:

  - Top 5–10% most frequently executed paths
  - Highest latency contributors
  - Most expensive operations

  Prioritize optimization for these paths first for maximum impact


2. **Baseline current performance**
   - Gather current latency and throughput metrics for each critical path.
   - Focus on P95/P99 latency, not just averages.
   - Identify endpoints or jobs that are clearly out of SLO or near capacity.
3. **Identify latency bottlenecks**
   - Use traces/profilers to break down latency by segment (network, DB, cache, external API, LLM call, CPU).
   - Look for:
     - N+1 queries or repeated DB calls in loops.
     - Chained external requests executed sequentially.
     - Oversized payloads or over-fetching data.
     - Long-running synchronous work in request handlers or event loops.
4. **Analyze throughput and concurrency**
   - Identify where work is serialized:
     - Single-threaded workers doing I/O-bound work.
     - Insufficient DB connection pool or thread pool sizes.
     - Long transactions holding locks and blocking others.
     - Shared global locks or critical sections in application code.
   - Check queue depths, backpressure behavior, and retry storms.
5. **Detect blocking operations**
   - In async or event-loop based services, flag:
     - Blocking I/O (file, network, DB) executed without `await` or offloading.
     - Heavy CPU work (parsing, compression, large JSON/CSV processing) on the main loop.
   - In threaded/process-based systems, flag:
     - Long critical sections holding locks.
     - Contended mutexes or hot shared state.
6. **Evaluate opportunities for parallelization and batching**
   - Identify independent I/O calls that can be run in parallel.
   - Consider:
     - Async/await with concurrent tasks (e.g., `asyncio.gather`).
     - Batched DB queries instead of per-item queries.
     - Bulk external API calls where supported.
     - Background job queues for work not required in the synchronous response.
7. **Design caching strategy**
   - Determine what is safe and valuable to cache:
     - Read-heavy, infrequently changing data (configs, reference data, feature flags).
     - Expensive computations (aggregations, feature generation, embedding or LLM outputs when appropriate).
   - Choose cache layers:
     - In-process (L1) cache for ultra-low-latency reads.
     - Distributed cache (e.g., Redis) as L2 for sharing across instances.
   - Define keys, TTLs, invalidation rules, and consistency requirements.
8. **Validate observability and safety**
   - Ensure metrics exist for latency, throughput, errors, and saturation (e.g., CPU, memory, connection pools).
   - Recommend tracing for slow paths and heavy external dependencies.
   - Confirm that changes (parallelization, caching) preserve correctness and isolation guarantees.

---

## Cost–Performance Trade-offs

For each optimization, evaluate:

- Cost increase vs latency reduction
- Throughput gain vs infrastructure cost
- Parallelization vs resource usage

Flag:
- Optimizations that significantly increase cost
- Inefficient scaling (cost grows faster than throughput)

Always balance:
Latency vs Throughput vs Cost


## Detailed Checklists

### 1. Latency Bottleneck Checklist

For each slow path, check:

- **Network and API boundaries**
  - Excessive serialization/deserialization or middleware layers.
  - Large payload sizes; opportunities for pagination or projection (select only needed fields).
- **Database and storage**
  - Missing indexes or non-selective indexes on hot queries.
  - N+1 query patterns and per-row lookups.
  - Expensive joins or unbounded scans.
- **External services and LLM/RAG components**
  - Multiple sequential external calls that could be parallelized.
  - Overly large LLM prompts (context bloat) and unnecessary reruns of identical queries.
  - Vector DB queries with too high top-k or slow index settings.
- **Application CPU work**
  - Heavy JSON/XML parsing or transformation on the critical path.
  - Complex in-memory aggregation in request handlers.
  - Compression, encryption, or image processing done synchronously.

Flag as latency bottlenecks any segments that dominate P95/P99 for that path.

### 2. Throughput Limit Checklist

Check for:

- **Worker and pool sizing**
  - Too few workers, threads, or async tasks for I/O-bound workloads.
  - DB connection pools or HTTP client pools that cap concurrency.
- **Locking and contention**
  - Global locks or shared resources used in hot paths.
  - Long-running transactions blocking others.
- **Resource saturation**
  - High CPU or memory utilization on critical services.
  - Disk or network bandwidth saturation.
- **Queue behavior**
  - Unbounded queues leading to latency spikes.
  - Backpressure or dropping behavior when overload occurs.

   ## Backpressure and Overload Handling

Ensure system handles overload safely:

- Rate limiting at API boundaries
- Queue length limits and backpressure signals
- Circuit breakers for failing dependencies
- Graceful degradation (fallback responses, partial results)

Flag:
- Systems that crash or degrade unpredictably under load
Treat the narrowest capacity (workers, DB pool, external rate limits) as the effective throughput limit and design around it.

### 3. Blocking Operation Checklist

Look for:

- Blocking I/O on main threads or event loops (file, DB, network).
- Synchronous LLM or vector DB calls inside otherwise async handlers without concurrency.
- Long-running CPU-bound tasks (e.g., ML inference, heavy parsing) not offloaded to worker pools or separate services.
- Tight loops that perform remote calls or disk access without yielding.

For each, propose either:
- Moving work off the critical path (background jobs).
- Making it truly asynchronous and concurrently executed.
- Breaking it into smaller, non-blocking steps.

---

## Output Format

Always structure the final answer using this template (adapt bullet counts as needed, but keep the sections):

```markdown
## Bottlenecks

- **[Component / path] – [Type: Latency / Throughput / Blocking]**: [Short description of the issue, evidence (metrics/traces), and why it matters].
- ...

## Parallelization plan

- **[Area or flow]**: [Concrete change to increase parallelism or reduce serialization, e.g., "Run these DB or HTTP calls in parallel using async tasks / worker pool / batched queries"; include expected impact].
- ...

## Caching strategy

- **[Layer: L1 / L2 / DB / RAG-specific]**: [What to cache, cache key, TTL/invalidation approach, and expected latency/throughput gains].
- ...

## Additional recommendations (optional)

- [Any further infra, observability, or architectural improvements relevant to performance].
```

Ensure each recommendation is **specific**, **actionable**, and tied to a measurable impact (latency reduction, throughput increase, or cost savings).

---

## Examples

### Example: Web API with Slow Responses

- **Bottlenecks**
  - API handler performs 5 sequential DB queries and 2 sequential external API calls; DB queries lack appropriate indexes.
- **Parallelization plan**
  - Run the 2 external API calls and 3 of the DB queries in parallel using async HTTP client and batched DB queries; offload heavy report generation to a background job.
- **Caching strategy**
  - Cache read-only configuration data and aggregated report summaries in a distributed cache with a 5–15 minute TTL; use in-process L1 cache for per-process hot keys.

### Example: RAG/LLM Workflow

- **Bottlenecks**
  - Sequential embedding and vector DB calls for each user turn; top-k set too high, leading to slow vector queries and large prompts.
- **Parallelization plan**
  - Batch embeddings and vector DB queries where possible; parallelize retrieval across collections or indices; pre-compute and cache embeddings for frequently accessed documents.
- **Caching strategy**
  - Cache embeddings and frequently repeated retrieval results; cache LLM responses for idempotent queries with stable inputs; store results in a shared cache keyed by normalized query and filters.

## RAG-Specific Performance Considerations

Evaluate:

- Embedding latency and batching efficiency
- Vector DB query latency (top-k, indexing strategy)
- Reranker latency and scalability
- LLM response time and context size impact

Flag:
- High top-k causing slow retrieval
- Large prompts causing latency spikes
- Reranker bottlenecks under load
