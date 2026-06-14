# webAI — Data Platform Engineer: Deep Technical Reference

> **What this is.** A book-length reference covering *every* topic in the JD to interview depth. For each major area you get three layers:
> **① Concepts** (what you must be able to explain on a whiteboard) → **② The market** (real products/tools as of 2026, so you can name-drop and compare credibly) → **③ How to design** (the architecture answer + tradeoffs).
>
> Use this *with* the first study guide (`webAI_Data_Platform_Engineer_Interview_Prep.md`), which has the answer-banks and behavioral prep. This file is the depth behind it.
>
> **Reading order if short on time:** Part 1 (Edge AI) → Part 2 (Connectors) → Part 8 (Privacy) → Part 9 (worked designs). Those four are where this role is won.

---

## Table of contents
- **Part 0** — Orientation: webAI's stack and how your role plugs in
- **Part 1** — Edge AI (concepts, market, design) ← *they named this; go deep*
- **Part 2** — Data connectors (SQL/NoSQL/REST/GraphQL/Salesforce/S3/SaaS)
- **Part 3** — Schema interpretation & modeling heterogeneous sources
- **Part 4** — Transforming data for AI inference (RAG, embeddings, features)
- **Part 5** — Distributed systems: latency, consistency, edge constraints
- **Part 6** — Resilient ingestion (unreliable / rate-limited systems)
- **Part 7** — Observability, logging, lineage, debuggability
- **Part 8** — Privacy, security & compliance (the deep one)
- **Part 9** — End-to-end system designs (worked, including Airbus)
- **Part 10** — Design methodology + how to run a whiteboard
- **Appendix A** — One-page market cheat-sheet
- **Appendix B** — Glossary of terms they may throw at you

---

## Part 0 — Orientation: webAI's stack and where you fit

You're building the **data on-ramp** to a private, edge-deployed inference platform. Map the pieces so every answer connects to their world:

```
   ENTERPRISE DATA              YOUR ROLE                    webAI PLATFORM
 ┌──────────────────┐   ┌───────────────────────┐    ┌───────────────────────┐
 │ SQL / NoSQL DBs  │   │  Connector framework  │    │ Navigator (build/train)│
 │ REST / GraphQL   │──▶│  + schema modeling    │──▶ │ webFrame (inference IR,│
 │ Salesforce / CRM │   │  + transform-for-     │    │   quantization, shard) │
 │ S3 / object store│   │    inference          │    │ Runtime (orchestrate)  │
 │ SaaS apps        │   │  + privacy controls   │    │ Network (local fabric) │
 │ files / streams  │   │  + resilience + obs   │    │ Companion (on-device   │
 └──────────────────┘   └───────────────────────┘    │   assistant / agentic) │
                                                      │ Infrastructure (govern)│
                                                      └───────────────────────┘
```

Key reusable phrases (they signal you "get it"):
- webFrame parses models into an **intermediate representation (IR)** then maps to backend modules and builds a **compute plan** across nodes. → You can frame your *data* layer the same way: sources → **canonical IR** → delivery plan per device.
- Their pillars are **privacy, energy, cost**. Tie answers back to these.
- Their values: **Truth, Ownership, Tenacity, Humility.**
- The role mandate: **"platform primitives, not pipelines."** Generalize, don't one-off.

---

# Part 1 — Edge AI (deep dive)

> This is webAI's core domain and the thing you flagged. Goal: be able to talk fluently about *what* edge AI is, *every* sub-concept, the *real products*, and *how to design* an edge data+inference system.

## 1.1 Concepts

### 1.1.1 The deployment spectrum (define each, know the tradeoffs)
- **Cloud AI** — model runs in a centralized data center; data is shipped to it. Pros: unlimited compute, easy ops. Cons: data leaves the premises (privacy/residency/ITAR risk), latency, network dependence, recurring cost, energy.
- **Edge AI** — computation pushed to the network edge / on the device that generates the data. Data ideally **never leaves** the device. Pros: privacy, low latency, offline capability, data sovereignty. Cons: constrained compute/memory/power; harder to update; harder to observe.
- **On-device AI** — strictest form of edge: inference fully local on a phone/tablet/laptop. This is webAI's sweet spot (iPad, Mac).
- **Fog computing** — an intermediate layer between device and cloud (e.g., an on-prem gateway/server). Useful when a single device is too weak but you still won't touch the public cloud. webAI's "cluster of Macs" is effectively a private fog/edge cluster.
- **Hybrid / edge-cloud** — split work: sensitive or latency-critical inference on the edge, heavy/batch training in a controlled environment. Many real deployments are hybrid.

> **One-liner:** "Edge AI moves computation to where data is generated so the data never has to move — trading raw compute for privacy, latency, sovereignty, and offline resilience."

### 1.1.2 Why edge is hard — the constraint set
- **Memory** — a 70B model in FP16 is ~140 GB; a device has single-digit-to-tens of GB. Forces quantization and/or sharding.
- **Compute** — no datacenter GPUs; you use CPU + integrated GPU + NPU.
- **Power / thermal** — battery and heat budgets; you can't run flat-out continuously.
- **Storage** — can't store the whole enterprise corpus on an iPad.
- **Connectivity** — intermittent or absent; must work offline (the Airbus case).
- **Updatability** — pushing new models/data to fleets of devices is a distribution problem.
- **Physical security** — a device can be lost/stolen → encryption at rest + remote wipe.
- **Observability** — you can't tail logs on a thousand offline iPads in real time.

### 1.1.3 Making big models fit — model compression (know each precisely)
- **Quantization** — store/compute weights at lower precision (FP16 → INT8 → INT4 → even sub-4-bit). Reduces memory ~2–8× and speeds inference. Types:
  - *Post-training quantization (PTQ)* — quantize a trained model directly; fast, slight accuracy loss.
  - *Quantization-aware training (QAT)* — simulate quantization during training; better accuracy, more work.
  - *Weight-only vs. weight+activation* (e.g., W4A16 = 4-bit weights, 16-bit activations).
  - Formats/schemes you can name: **GGUF** (llama.cpp's format, 1.5–8-bit), GPTQ, AWQ, INT8, NVFP4/FP8 (newer NVIDIA). webAI claims adaptive quantization retaining ~99.5% accuracy.
  - *Tradeoff:* lower bits = smaller/faster but accuracy degrades; the art is finding the knee of the curve (often INT4 for 7B-class models).
- **Pruning** — remove redundant weights/neurons (structured vs. unstructured). Smaller model, needs care to keep accuracy.
- **Knowledge distillation** — train a small "student" model to mimic a large "teacher." webAI's "networks of smaller specialized models" philosophy is distillation-adjacent.
- **LoRA / QLoRA (parameter-efficient fine-tuning)** — adapt a base model with small low-rank adapters instead of retraining all weights; cheap to fine-tune and to hot-swap on device. *Very* relevant for per-customer personalization on the edge.
- **Speculative decoding** — a small draft model proposes tokens a big model verifies, speeding generation (EAGLE-3 etc.).

### 1.1.4 Distributed edge inference (webFrame's territory)
When one device can't hold the model, split it across several:
- **Tensor parallelism** — split individual layers'/matrices' math across devices (chatty, needs fast interconnect).
- **Pipeline parallelism** — assign different layers to different devices; data flows like an assembly line (what webFrame-style sharding across a Mac cluster does; "shards that collaborate as one").
- **KV-cache management** — the attention cache dominates memory during generation; **PagedAttention** (from vLLM) treats it like virtual-memory pages to cut waste and boost throughput.
- **Compute-plan selection** — read available hardware at runtime, decide single-node vs. multi-node split (webFrame does exactly this). Your data layer's analog: decide *what data goes to which device*.

### 1.1.5 Hardware you should be able to name
- **Apple Silicon (M-series, A-series)** — the key one for webAI. **Unified Memory Architecture (UMA):** CPU + GPU + Neural Engine share one physical memory pool → no costly CPU↔GPU copies, which is *why* on-device LLMs are viable on Macs/iPads. The **Apple Neural Engine (ANE)** is the NPU.
- **NPUs / AI accelerators** — Apple Neural Engine, Qualcomm Hexagon (Snapdragon), Intel Core Ultra NPU. Low-bit inference at low power.
- **NVIDIA Jetson (Thor)** — edge GPU for robotics/heavier edge servers.
- **Memory bandwidth is often the real bottleneck** for LLM inference (not FLOPs) — a good point to make.

### 1.1.6 Edge data lifecycle (your part!)
1. **Select & minimize** what data a device needs (you can't ship everything).
2. **Transform** to inference-ready form (chunk + embed; build a *local* index).
3. **Pre-position** (sync) the minimized, encrypted subset to the device.
4. **Serve** locally/offline (retrieval + inference on device).
5. **Reconcile** on reconnect: pull deltas, push audit logs, resolve conflicts.
6. **Govern**: enforce access, residency, retention, deletion — on device.

## 1.2 The market (name these; show you know the landscape)

### On-device / edge inference runtimes (2026)
| Runtime | What it is / when to pick | Notes |
|---|---|---|
| **MLX / MLX-LM** | Apple's array framework optimized for Apple Silicon **unified memory**; PyTorch-like. **Leads on Macs** for models <14B (20–87% faster); supports fine-tuning + distributed inference. | *Most relevant to webAI's Apple-first stack.* Mac-only. |
| **llama.cpp** | C/C++, **GGUF** format, 1.5–8-bit quant, runs on virtually any hardware (Metal/CUDA/Vulkan), 100+ architectures, many language bindings. Most portable. | Industry default for local LLMs; low-level. |
| **Ollama** | Convenience HTTP-API wrapper over llama.cpp; v0.19 (Mar 2026) added an **MLX backend**. | Great for prototypes/dev; not heavy production serving. |
| **ExecuTorch** | Meta's on-device PyTorch runtime; **50 KB base footprint**, microcontrollers → flagship phones; 12 hardware backends (CPU/GPU/NPU/DSP); hit v1.0 late 2025; powers AI in Instagram/WhatsApp/Messenger. | The "tiny device → big device" production path. |
| **Core ML** | Apple's framework; can target CPU/GPU/**ANE**. | Native Apple deployment. |
| **ONNX Runtime** | Cross-platform via **Execution Providers**; unified backend across mobile/desktop/cloud. | Best when you need one backend everywhere. |
| **TensorRT (Edge-LLM) / vLLM** | NVIDIA datacenter/edge-GPU serving; vLLM's **PagedAttention** for throughput. | For GPU/Jetson, not Apple edge. |
| **LiteRT** (formerly TensorFlow Lite) **/ MediaPipe** | Mature mobile/embedded ML; large ecosystem. | Broad mobile ML beyond LLMs. |
| **MLC-LLM** | TVM-compiled GPU kernels; WebGPU/browser via WebLLM. | Compiled native perf, browser inference. |

**The architectural lesson to voice:** runtimes churn fast (three major shifts in 18 months), so a mature platform puts a **thin inference-engine abstraction** behind a stable interface rather than hard-coding one runtime. This mirrors the "platform primitive" instinct — and it's exactly what webFrame is.

## 1.3 How to design an edge AI data system (template answer)

**Prompt they might give:** *"Design how data gets onto and stays current on an offline edge device running our models."* (This generalizes the Airbus case.)

1. **Clarify constraints** — device type & memory, offline duration, data sensitivity/classification, freshness SLA, fleet size, update cadence.
2. **Source & ingest** — your connector framework pulls from enterprise systems (Part 2), CDC-driven for freshness.
3. **Minimize & classify** — pull only contract-required fields; tag PII/classification at ingestion (Part 8); scope by device role/program.
4. **Transform to inference-ready** — clean → chunk → embed with an **on-device-compatible embedding model** → build a **local vector index** (e.g., an embedded store like LanceDB-style, see Part 4); attach permission + version metadata to every chunk.
5. **Selective sync / pre-position** — compute a per-device data plan (analogous to webFrame's compute plan): which subset, how compressed, encrypted at rest in the **Secure Enclave**.
6. **Serve offline** — retrieval filtered by the user's clearance + on-device inference (webFrame/Companion). Surface **staleness/version** so users never trust outdated content.
7. **Reconcile on reconnect** — delta sync (re-embed only changed source sections via CDC), conflict resolution (CRDT/LWW), push **tamper-evident hash-chained audit logs** that couldn't leave while offline.
8. **Govern** — deletion/revocation propagates to the on-device index on next sync; retention enforced locally; remote wipe on loss.
9. **Observe** — per-device freshness, index staleness, sync success, DLQ, lineage (Part 7).

**Tradeoffs to call out:** freshness vs. battery/bandwidth (batch sync windows); index size vs. device storage (more aggressive minimization/summarization); accuracy vs. footprint (quantization level); strong consistency is off the table offline → eventual + staleness signals.

---

# Part 2 — Data connectors

> The literal core of the job. Be able to integrate each named source *and* describe the framework that makes them all reusable primitives.

## 2.1 Concepts

### 2.1.1 The anatomy of any connector
Every connector, regardless of source, must handle: **auth** (and token refresh), **discovery** (schema), **extraction** (read strategy), **pagination**, **rate-limit compliance**, **incremental sync** (state/cursor), **type normalization** (→ IR), **error handling & retries**, **idempotency**, **observability**, and **privacy controls**. In a *framework*, all of these are cross-cutting and inherited; the per-source adapter only describes what's unique.

### 2.1.2 Extraction strategies (universal)
- **Full snapshot** — read everything each run. Simple; expensive; fine for small/cold data.
- **Incremental by watermark** — `WHERE updated_at > :cursor`. Needs a reliable monotonic column; misses hard deletes; watch clock skew and rows updated without bumping the column.
- **Change Data Capture (CDC)** — consume the DB's change log (WAL/binlog). Captures inserts/updates/**deletes**, low source impact, near-real-time. Heavier ops (replication privileges, log retention). The gold standard for keeping downstream stores fresh.
- **Event/push** — source emits events (webhooks, streams) you subscribe to.

### 2.1.3 Per-source depth

**SQL / relational (Postgres, MySQL, SQL Server, Oracle)**
- Connection pooling, TLS, read replicas (never melt prod).
- Discovery via `information_schema`/catalog (tables, columns, types, PK/FK, nullability).
- Large tables: **keyset/seek pagination by PK**, never `OFFSET` on big tables.
- CDC: Postgres logical replication / `wal2json`; MySQL binlog. Debezium is the common engine.
- Type mapping into IR: NUMERIC/DECIMAL precision, timestamp timezones, JSONB, arrays, enums.

**NoSQL (MongoDB, DynamoDB, Cassandra, Redis)**
- Schema-on-read → infer by sampling, handle **drift** as a first-class event.
- Mongo: change streams (CDC), `_id` key, nested docs (flatten vs. preserve).
- DynamoDB: partition/sort keys dictate access; **Streams** for CDC; scans burn RCUs → segmented + throttled.
- Cassandra: tunable consistency (`ONE`/`QUORUM`/`ALL`) — consistency vs. latency per query.

**REST APIs**
- Pagination: offset/limit, page-number, **cursor/keyset**, link-header. Abstract behind one paginator.
- Auth: API key, Basic, **OAuth 2.0** (auth-code, client-credentials, refresh rotation), JWT.
- Rate limits: honor `Retry-After`, `X-RateLimit-*`; client-side token bucket.
- Incremental via filter params, but APIs lie/are eventually consistent → overlapping windows + dedupe.

**GraphQL APIs**
- Single endpoint; client selects fields → **data minimization by construction**.
- **Introspection** query returns the schema → connector self-configures discovery.
- Watch the **N+1** problem, server depth/complexity limits, Relay cursor pagination (`edges`/`node`/`pageInfo`), and partial responses (`data` + `errors` together).

**Salesforce / CRM (JD names it — know specifics)**
- APIs: REST, SOAP, **Bulk API 2.0** (async large extracts, CSV batches), **CDC / Platform Events / Streaming** (near-real-time), **SOQL** query language.
- **Governor limits** — hard API/query caps per 24h → connector must budget them; Bulk for backfills, incremental (`SystemModstamp`) otherwise.
- **Describe API** (`sObject describe`) → discover custom objects (`__c`), custom fields, picklists, record types dynamically; never hardcode schema.
- Auth: OAuth 2.0 **JWT bearer flow** (server-to-server), connected apps, scopes.

**S3 / object storage (JD names it)**
- List (paginated), get, multipart for large objects, byte-range reads, ETag/`If-Modified-Since` for incremental, **S3 events / EventBridge** to trigger on new objects.
- Formats: CSV (delimiter/quoting hell), JSON/JSONL, **Parquet/ORC** (columnar, embedded schema, predicate pushdown — prefer), Avro (schema evolution), unstructured (PDF/image/audio → multimodal).
- Security: IAM **least-privilege scoped** policies, SSE-KMS encryption, VPC endpoints, short-lived rotating creds — never long-lived root keys.

**SaaS generally (CRM, storage, ticketing, HRIS, etc.)**
- Almost all are OAuth + REST/GraphQL + webhooks + per-tenant rate limits + idiosyncratic pagination and schemas. The framework's job is to absorb that variance so adding one is config + a thin adapter.

**Files & streaming**
- File: CSV/Parquet/Excel/PDF; schema inference + user override; encoding/locale.
- Streaming: Kafka/Kinesis/Pulsar topics; offsets as cursors; exactly-once semantics via idempotent consumers.

## 2.2 The market

| Tool | Model | Notes (2026) |
|---|---|---|
| **Airbyte** | Open-source + cloud; own connector framework; **600+ connectors**; low-code connector **builder**; self-host or cloud; native **Debezium CDC**; vector-DB destinations; SOC2/HIPAA/GDPR. | Closest reference to "a connector platform." Some community connectors are not enterprise-grade. |
| **Singer** | Open **spec** (taps=sources, targets=destinations). | The *protocol* idea worth borrowing. |
| **Meltano** | Singer-based, **CLI-first, GitOps**, treats pipelines as software (version control, CI/CD, tests). | Engineering-led, code-centric — matches a primitives mindset. |
| **Fivetran** | Fully managed ELT, largest catalog, MAR pricing; limited self-host/custom. | Managed, low-ops, less flexible. |
| **Estuary** | CDC + streaming + batch in one; low latency. | Strong real-time alternative. |
| **Debezium** | Open-source **log-based CDC** engine (usually with Kafka). | The de-facto OSS CDC; sub-second possible. |
| **Kafka Connect** | Connector framework around Kafka (source/sink connectors). | Pattern reference for a pluggable connector runtime. |
| **dlt** | Python library for building pipelines as code. | Lightweight, code-first. |
| Others | AWS DMS/Glue, Striim, HVR/Qlik, Hevo, Matillion, NiFi. | DMS = cloud CDC; NiFi = visual dataflow w/ strong security. |

**Borrow, don't copy (say this):** "I'd take Singer/Airbyte's idea of a *standard connector contract* every adapter implements, Debezium's log-based CDC for freshness, and Meltano's software-engineering discipline — but make **privacy controls and edge-awareness first-class**, which none of these do out of the box."

## 2.3 How to design the connector framework (the headline answer)

```
            ┌──────────────────────── Connector Framework (SDK) ────────────────────────┐
 declare    │  Cross-cutting services every adapter inherits:                            │
 capabilities│  auth/secrets · pagination · rate-limit/backoff · retry/circuit-breaker ·  │
   │         │  schema discovery → IR · incremental state/checkpoint · type normalize ·   │
   ▼         │  idempotency · privacy (classify/minimize/residency/consent) ·             │
 ┌────────┐  │  observability (metrics/logs/traces) · lineage emission · DLQ · contract   │
 │Adapter │──▶  tests / conformance suite                                                  │
 │(thin)  │  └───────────────────────────────────────────────────────────────────────────┘
 └────────┘   Adapter only supplies: how to auth, how to discover, how to paginate,
              which extraction modes it supports, source→IR type map.
```

Design principles to articulate:
- **Capability interface** — each source declares `supportsCDC`, `supportsIncremental`, `supportsFieldSelection`, etc.; the framework adapts behavior.
- **Canonical IR** — one internal record/schema model all sources normalize into (echo webFrame's IR vocabulary).
- **Declarative adapters** — a manifest + thin code; adding a source ≈ config, not a new codebase.
- **Durable state store** — per-source cursor/checkpoint so syncs resume, not restart.
- **Conformance suite** — every adapter must pass identical tests (pagination correctness, incremental correctness, drift handling, rate-limit compliance, PII tagging).
- **Versioned contracts** — connector output is a versioned interface to downstream ML.
- **Self-service for forward-deployed teams** — good defaults, great errors, dry-run/replay.

**This is the question that most directly tests "primitives, not pipelines." Nail this diagram and you've answered the central theme of the role.**

---

# Part 3 — Schema interpretation & modeling heterogeneous sources

## 3.1 Concepts
- **Discovery first** — auto-introspect every source (SQL catalog, Salesforce Describe, GraphQL introspection, Parquet metadata, document sampling).
- **Canonical / unified model** — map disparate schemas into shared entities (Salesforce `Account` + Postgres `users` + Mongo `profiles` → one `Customer`). This is **schema mapping**.
- **Entity resolution** — same real-world entity across sources: deterministic (shared ID) vs. probabilistic/fuzzy matching; dedup.
- **Type reconciliation** — unify into the IR; nulls, units, timezones, encodings, currency, precision.
- **Semantic modeling** — meaning beyond type: decode `status=3`, capture data dictionaries/metadata.
- **Schema drift & evolution** — sources change. Strategy: additive-by-default, **versioned schemas**, surface breaking changes as alerts, **never silently drop** fields. (Tie to OpenLineage emitting SchemaChange events, Part 7.)
- **Data quality gates** — completeness, validity, freshness, uniqueness checks at ingestion so garbage never reaches the model.
- **Data contracts** — a formal, versioned agreement (schema + semantics + SLAs + ownership) between producer and consumer. This *is* the "define feature expectations with ML" bullet, formalized.

## 3.2 The market
- **dbt** — transformation + tests + docs; de-facto for modeling/contracts in the warehouse.
- **Schema registries** — Confluent Schema Registry (Avro/Protobuf/JSON) for streaming schema evolution & compatibility rules.
- **Data catalogs** — DataHub (LinkedIn, strong momentum), OpenMetadata, Amundsen, Unity Catalog (Databricks), Atlan, Collibra — discovery, lineage, governance, business glossary.
- **Data quality** — Great Expectations, Soda (Soda Core OSS), Anomalo, Bigeye, Elementary.
- **Table formats** — Apache Iceberg / Delta / Hudi: schema evolution + time travel at the storage layer (worth knowing as where modern lakes are going).

## 3.3 How to design
Given N messy sources:
1. **Introspect** each → raw schemas.
2. **Map** to a canonical entity model; record the mapping as metadata.
3. **Resolve entities** (keys + fuzzy match) and dedupe.
4. **Reconcile types** into IR; capture semantics/units.
5. **Gate quality** at ingestion (assertions); route failures to DLQ, not downstream.
6. **Version** the canonical schema; emit drift as monitored events; coordinate breaking changes with ML via the data contract.
7. **Lineage** every field source→IR→destination for debugging *and* privacy (DSAR).

> **One-liner:** "Heterogeneous modeling = entity resolution + type/semantic reconciliation into a canonical IR, with drift treated as a versioned, monitored event and quality enforced at the gate — not a one-off mapping per customer."

---

# Part 4 — Transforming raw data into AI-inference-ready formats

> The "applied AI" half. webAI runs LLMs *and* vision models locally, so "inference-ready" spans **RAG/LLM** and **structured features**.

## 4.1 Concepts — RAG (retrieval-augmented generation) pipeline

The canonical flow: **collect → clean → chunk → embed → store/index → retrieve → (rerank) → generate.**

- **Clean / normalize** — strip boilerplate, fix encoding, convert PDFs/HTML/tables → text; OCR scanned docs; extract structure.
- **Chunking** — split documents into retrievable units. Strategies:
  - *Fixed-size with overlap* — simple; overlap preserves context across boundaries.
  - *Recursive / structural* — split on headers/paragraphs/sentences, respecting document structure.
  - *Semantic chunking* — split where meaning shifts (embedding-distance-based).
  - *Sentence-window / parent-document* — retrieve a small precise unit but feed surrounding context to the model.
  - **Tradeoff:** chunks too big → imprecise retrieval + context bloat; too small → lost context. **Carry metadata on every chunk** (source, section, timestamp, **permissions/classification**).
- **Embeddings** — turn chunks into vectors via an embedding model. *The same model must embed both documents and queries.* On webAI's edge: the embedding model itself can run on-device (privacy win). Know dimensionality, normalization, and that embeddings of personal data **are** personal data (deletion implications, Part 8).
- **Vector index / ANN** — approximate nearest-neighbor search:
  - **HNSW** (graph-based) — high recall, fast queries, more memory; the common default.
  - **IVF** (inverted-file/clustering) — partitions space; tune `nprobe` for recall/latency.
  - **PQ** (product quantization) — compress vectors to cut memory (great for edge).
  - **Metadata filtering** — filter by permissions/source before/with the vector search (filtered-HNSW performance matters — Qdrant's ACORN, etc.).
- **Retrieval** — top-k similarity; **hybrid search** combines dense vectors + sparse/keyword (BM25/SPLADE) for better recall on names/IDs.
- **Reranking** — a cross-encoder reorders candidates for precision before generation.
- **Freshness** — on source change (CDC) → re-chunk/re-embed **only the delta** → upsert into the index; track which source version produced which vectors (lineage).
- **Evaluation** — retrieval metrics (recall@k, MRR, nDCG) + answer quality; guard against retrieval of stale/unauthorized chunks.

## 4.2 Concepts — structured / feature prep
- **Feature contract** with ML: name, type, semantics, freshness SLA, null policy, allowed ranges, version. This formalizes "define feature expectations."
- **Transformations** — joins, aggregations, encodings (categorical→numeric), scaling/normalization, time-windowing.
- **Point-in-time correctness / no leakage** — features must reflect only what was known at inference time (also a privacy/correctness concern).
- **Online/offline skew** — training-time and serving-time features must match; a **feature store** enforces this.

## 4.3 The market
- **Vector databases (2026):**
  - **pgvector / pgvectorscale** — Postgres extension; ACID, full SQL, easy ops; great up to ~10–50M vectors; v0.9 added sparse vectors. *Default if you're already on Postgres.*
  - **Qdrant** — Rust; excellent **filtered** search (ACORN), int8 quantization, hybrid search, strong price/performance; popular self-host pick (good for **data-residency** needs).
  - **Weaviate** — Go; built-in vectorization modules, hybrid search, GraphQL API.
  - **Milvus / Zilliz** — largest scale, billion-vector, hybrid (Sparse-BM25); operationally heavier (Zilliz = managed).
  - **Pinecone** — managed, zero-ops, fastest path to production; vendor lock-in/cost.
  - **Chroma** — simple, OSS, great for prototyping/local.
  - **LanceDB** — **embedded, in-process, zero-copy on the Lance columnar format; built for edge/local-first/desktop apps.** ← *Most relevant to webAI's on-device case — name this specifically.*
  - **FAISS** — a vector *library* (no persistence/replication), not a database; building block.
- **Embedding models** — sentence-transformers family, BGE, E5, GTE, Nomic, plus provider APIs; on edge you want a small, quantizable model that runs locally.
- **RAG / orchestration frameworks** — LangChain, LlamaIndex, Haystack (chunking, retrieval, pipelines). Useful vocabulary; not required at infra layer.
- **Feature stores** — Feast (OSS), Tecton (managed); concept matters even if webAI has none yet.

## 4.4 How to design (RAG ingestion as a primitive)
1. Connector framework lands raw docs/records in IR (Part 2).
2. Normalize/clean per type (PDF/HTML/table extractors as pluggable steps).
3. Chunk with a strategy chosen per content type; attach **permission + classification + source-version** metadata to each chunk.
4. Embed with an on-device-compatible model; normalize vectors.
5. Upsert into a **local, embedded** vector index (LanceDB-style for edge) with metadata filtering.
6. Retrieval **filters by the requesting user's clearance** (access control at retrieval, not in the prompt — Part 8).
7. CDC-driven delta re-embedding keeps the edge index fresh; surface staleness.
8. Version the whole transform so model + data evolve together (data contract).

> **One-liner:** "'Inference-ready' for LLMs means clean → chunk → embed → index locally, with permissions and source-version carried as chunk metadata so retrieval itself is access-controlled and auditable; for features it means a versioned, point-in-time-correct contract my connectors guarantee and monitor."

---

# Part 5 — Distributed systems: latency, consistency, edge constraints

## 5.1 Concepts
- **CAP theorem** — under a network **P**artition you must choose **C**onsistency or **A**vailability. On the edge, partitions are *normal*, so you typically design **AP** with eventual reconciliation.
- **PACELC** — extends CAP: even when there's no partition (**E**lse), you trade **L**atency vs. **C**onsistency. Edge serving favors local low-latency reads over global consistency.
- **Consistency models** (define each): **strong/linearizable**, **sequential**, **causal**, **read-your-writes**, **eventual**. Know when each is acceptable; an offline iPad can only offer local + eventual.
- **Replication** — leader/follower vs. multi-leader vs. leaderless (quorum, `R+W>N`). Sync vs. async (async = possible data loss on failover).
- **Partitioning/sharding** — hash vs. range; hotspots; rebalancing.
- **Conflict resolution** on reconnect — **last-writer-wins** (needs synchronized clocks; can lose data), **vector clocks** (detect concurrency), **CRDTs** (conflict-free replicated data types that merge deterministically), or domain-specific merge.
- **Local-first software** — the device is the source of truth for its own writes; sync is best-effort; UX never blocks on the network.
- **Idempotency & exactly-once** — true exactly-once end-to-end is usually **at-least-once delivery + idempotent consumer**.
- **Latency optimization** — caching, locality (process near the data — the whole edge thesis), batching, async, connection reuse, precomputation (pre-embed before the user asks).

## 5.2 The market / patterns
- **CRDT libraries / local-first**: Automerge, Yjs (the local-first sync space).
- **Distributed stores**: Cassandra/Scylla (leaderless, tunable consistency), DynamoDB (managed), CockroachDB/Spanner (distributed SQL, strong consistency), Redis (cache/low-latency).
- **Consensus**: Raft/Paxos (etcd, ZooKeeper) — for coordination/metadata, not edge data paths.

## 5.3 How to design for the edge
- Treat partition as the default → **local-first reads, eventual reconciliation**.
- Ship only the **minimized** subset each device needs; compress via embeddings/summarization; PQ-quantize the index.
- **Encrypt at rest** (Secure Enclave); remote wipe; data leases that expire.
- Design **reconnect-time conflict resolution** explicitly (CRDT/LWW + audit).
- **Surface staleness** to the user instead of pretending freshness.

> **One-liner:** "On the edge I design for partitions as the normal case: local-first, eventual reconciliation, minimized encrypted pre-positioning, and explicit conflict resolution on reconnect — with staleness made visible rather than hidden."

---

# Part 6 — Resilient ingestion (unreliable / rate-limited systems)

## 6.1 Concepts (each is a likely drill)
- **Retries** — only for *retryable* errors (5xx, 429, timeouts), never blind retries on 4xx auth/validation. Cap attempts.
- **Exponential backoff + jitter** — increasing delays with randomization to avoid synchronized **thundering-herd** retries.
- **Rate-limit compliance** — honor `Retry-After` and `X-RateLimit-*`; client-side **token-bucket** (burst + steady rate) or **leaky-bucket** limiter; **adaptive throttling** as you near the limit. Salesforce **governor limits** are the canonical case.
- **Idempotency** — idempotency keys for writes; dedupe on reads; whole sync must be **replay-safe**.
- **Checkpointing / resumability** — durable cursor so a crash resumes mid-sync (and doesn't re-hammer the source).
- **Backpressure** — if downstream (embedding/edge sync) can't keep up, signal upstream to slow; **bounded queues** (never unbounded buffering).
- **Dead-letter queue (DLQ)** — poison records go to a DLQ with full context for inspection; never silently dropped, never block the stream.
- **Circuit breaker** — stop calling a failing source (open), fail fast, probe to recover (half-open → closed).
- **Bulkhead isolation** — one source's failure can't starve resources for the others.
- **Timeouts everywhere** — connect, read, total; no unbounded waits.
- **Delivery semantics** — at-most-once / at-least-once / exactly-once; be honest that exactly-once = at-least-once + idempotency.
- **Batch vs. streaming** — batch (windows, simpler, higher latency) vs. streaming (continuous, lower latency, harder semantics); micro-batching as a middle ground.

## 6.2 The market
- **Queues/streams**: Apache Kafka (log/streaming backbone), Apache Pulsar, AWS Kinesis, AWS SQS (DLQ built in), RabbitMQ, Redpanda.
- **Stream processing**: Apache Flink, Spark Structured Streaming, Kafka Streams (windowing, joins, enrichment in-flight).
- **Workflow/orchestration**: Apache Airflow, Dagster, Prefect (DAGs, retries, scheduling); **Temporal** (durable execution — built-in retries/idempotency/state, excellent for resilient long-running connector workflows).
- **Resilience libs**: resilience4j / Polly (circuit breakers, retries, bulkheads).

## 6.3 How to design
- The framework gives every connector: backoff+jitter, header-aware token-bucket throttling, durable checkpoints, idempotent replay, bounded queues + backpressure, a DLQ, and a circuit breaker per source.
- Distinguish retryable vs. fatal centrally (error taxonomy).
- For long-running syncs, a durable-execution engine (Temporal-style) so state survives crashes.
- Per-source **budgets** (rate-limit-aware scheduler) so one greedy sync doesn't exhaust a shared quota.

> **One-liner:** "I assume every source is flaky and rate-limited. The framework standardizes backoff+jitter, header-aware throttling, durable checkpoints, idempotent replay, bounded-queue backpressure, a DLQ for poison records, and per-source circuit breakers — so a bad source degrades gracefully instead of cascading."

---

# Part 7 — Observability, logging, lineage, debuggability

## 7.1 Concepts
- **Three pillars** — **metrics** (rates, latencies, error %, records synced, freshness/lag), **logs** (structured, correlation IDs, **never PII in plaintext**), **traces** (distributed tracing across connector→transform→embed→sync).
- **Connector-specific signals** — sync success/failure, records processed, schema-drift events, rate-limit hits, retry counts, DLQ depth, freshness/lag, auth/token-refresh failures.
- **Data observability "five pillars"** (Monte Carlo's framing) — **freshness, volume, schema, distribution, lineage.**
- **Data lineage** — where each record came from, what transformed it, where it landed (and **column-level** lineage). Does double duty: debugging *and* privacy/compliance (DSAR, audit, proving non-export).
- **SLOs & alerting** — define freshness/reliability SLOs per source; alert on breach, not every blip; route to owners.
- **Debuggability for forward-deployed teams** — clear error taxonomy ("Salesforce 401: token expired — re-auth connector X"), dry-run/replay mode, single-record trace ("show this record's journey").
- **PII-safe logging** — redact/tokenize before write; log *references*, not values.

## 7.2 The market
- **Telemetry standard**: **OpenTelemetry** (vendor-neutral metrics/logs/traces).
- **Metrics/dashboards**: Prometheus + Grafana; Datadog; Dynatrace.
- **Data lineage**: **OpenLineage** (open standard; emits Job/Run/Dataset events with facets, incl. SchemaChange) + **Marquez** (reference backend/UI); DataHub; Apache Atlas (Hadoop).
- **Data observability platforms**: Monte Carlo, Acceldata, Bigeye, Soda, Anomalo, Sifflet, Metaplane, Elementary.
- **Quality assertions**: Great Expectations, Soda Core.

## 7.3 How to design
- Observability is **emitted by the framework**, not bolted on per connector: standard structured PII-safe logs, per-source metrics, OTel traces, and **OpenLineage events** for lineage.
- Lineage graph powers both root-cause debugging and "where is this data subject's data?" for DSAR/erasure.
- SLOs per source; staleness surfaced to edge users; DLQ depth alerts.
- A "single record journey" debug view for forward-deployed engineers.

> **One-liner:** "Observability ships with the framework — PII-safe structured logs, per-source freshness/error/rate-limit metrics, OpenTelemetry traces, and OpenLineage lineage. Lineage is the bridge between debuggability and compliance: it's how a field engineer roots out a bug *and* how we answer an audit or erasure request."

---

# Part 8 — Privacy, security & compliance (the deep one)

> Your home turf and webAI's whole reason to exist. Goal: make privacy a **property of the platform, enforced by construction** — not a checklist. Structured by regulation → technique → architecture → AI-specific threats.

## 8.1 The regulatory landscape (definition + the "so what" for a connector)

- **GDPR (EU)** — lawful basis, **data minimization**, purpose & storage limitation, **data-subject rights** (access/DSAR, erasure/"right to be forgotten," portability, rectification, object), **DPIA** for high-risk processing, 72h breach notice, international-transfer rules (SCCs/adequacy). Fines up to **4% global revenue or €20M**. *So what:* every connector must support minimization, deletion-propagation, lineage, and purpose/consent metadata.
- **CCPA / CPRA (California)** — rights to know/delete/correct, opt-out of "sale/sharing," sensitive-PI category. *So what:* opt-out + deletion must reach **derived data including embeddings**.
- **HIPAA (US health)** — PHI, covered entities/business associates (**BAA**), minimum-necessary, Security Rule (admin/physical/technical safeguards). *So what:* healthcare customers → encryption + access control + audit mandatory; **on-device is a selling point** (PHI never leaves premises).
- **SOC 2 (Type II)** — auditor attestation across Security/Availability/Processing-Integrity/Confidentiality/Privacy. *So what:* your access control, change mgmt, and logging feed the audit.
- **ISO 27001 (ISMS) / ISO 27701 (privacy) / NIST CSF / NIST 800-53** — security & privacy management frameworks. Name 27701 to show privacy-framework awareness.
- **ITAR / EAR (US export control)** — defense articles on the **US Munitions List**; technical data can't be "exported," and an export includes **sharing a file, granting cloud access, or even an AI prompt** to a foreign person/region. USML scope revised Sept 2025. *So what:* **the killer argument for webAI's edge model — if data never leaves the controlled device/environment, you collapse ITAR/EAR export exposure.** (Central to the Airbus case.)
- **EU AI Act** — risk tiers; transparency, data-governance, and logging duties for high-risk systems.
- **Data residency / sovereignty** — data must physically stay in a jurisdiction. *So what:* edge/on-prem inherently satisfies residency — no exfiltration.

## 8.2 PII handling — be precise about the differences (interviewers test this)
- **Classification / discovery** — auto-detect & tag PII/PHI/sensitive at ingestion (pattern + ML detection). Tag once at the connector, propagate via lineage. *You can't protect what you haven't classified.*
- **Data minimization** — pull/retain only contract-required fields. GraphQL field selection and column projection are minimization *mechanisms*.
- **Pseudonymization** — replace identifiers with **reversible** tokens (mapping held separately/secured). Still personal data under GDPR; a recognized safeguard.
- **Anonymization** — **irreversible** removal of identifiability; takes data out of GDPR scope *but is hard* — beware re-identification via **quasi-identifiers**.
- **Tokenization** — swap sensitive values for tokens via a **vault** (common for PCI/PII).
- **Masking / redaction** — hide or partially obscure values (display/logging).
- **k-anonymity / l-diversity / t-closeness** — each record indistinguishable within a group of k; guard against attribute disclosure when sharing aggregates.
- **Differential privacy (DP)** — add **calibrated noise** so individuals can't be inferred from outputs/aggregates; quantified by **ε (privacy budget)**: smaller ε = more private, less accurate. Global vs. local DP.
- **Zero-knowledge proofs** — prove a statement without revealing the data (identity/eligibility use cases).

## 8.3 Encryption & key management
- **In transit** — TLS 1.2+/**mTLS** service-to-service; validate certs (never disable verification "to make it work").
- **At rest** — disk/volume encryption + **field/column-level** encryption for sensitive fields; **envelope encryption** (data keys wrapped by a master key in a **KMS/HSM**).
- **Key management** — rotation, separation of duties, KMS/HSM; never hardcode; on the edge, keys live in the **Secure Enclave**.
- **Secrets management for connectors** (the #1 practical failure mode) — source credentials in a **vault** (HashiCorp Vault / cloud KMS / secrets manager), **short-lived scoped tokens**, auto-rotation; never in code, config, or logs.

## 8.4 Access control & authorization
- **Least privilege / need-to-know** — connector creds scoped to exactly the data needed.
- **RBAC vs. ABAC** — role-based vs. attribute-based (clearance, classification, location, time). **Aerospace/ITAR usually needs ABAC** with clearance attributes RBAC can't express.
- **Authorization carried into retrieval** — for RAG, each chunk carries permission metadata so retrieval returns only authorized content. **Filter at retrieval — never rely on the model to "decline."** (Sophisticated point; say it.)
- **Zero-trust** — authenticate/authorize every request; no implicit trust from network location.
- **JIT access + separation of duties** for operators.

## 8.5 Edge / on-device privacy techniques (webAI-native — emphasize)
- **On-device processing = the core privacy primitive** — data processed where generated; never traverses network/third-party cloud. Collapses residency, ITAR-export, and third-party-processor risk. *This is the thesis.*
- **Trusted Execution Environments (TEEs) / Secure Enclave / confidential computing** — isolated processing region protected even from a compromised OS; **remote attestation** proves the right code is running. (Azure Confidential Computing, AWS Nitro Enclaves, Intel SGX/TDX, Apple Secure Enclave.)
- **Federated learning (FL)** — train across devices; only **model updates (gradients)** leave, never raw data. Pair with **secure aggregation** (coordinator sees only combined updates) and **DP** (noised updates) for provable bounds. Tradeoff: FL models can underperform centralized; communication overhead. Frameworks: **Flower**, NVIDIA FLARE, TensorFlow Federated.
- **Homomorphic encryption (HE)** — compute on **encrypted** data without decrypting. Powerful but compute-heavy → narrow use cases. Libraries: Microsoft SEAL, OpenFHE.
- **Secure multi-party computation (SMPC)** — multiple parties jointly compute over private inputs without revealing them.
- **Maturity signal:** the strongest enterprise stack is usually **hybrid — FL + DP for privacy, TEEs/confidential computing for system-level protection** — and knowing *when not* to reach for HE/SMPC (too heavy) is itself a sign of judgment.
- **DP tooling** — OpenDP, Google DP, Opacus (PyTorch), TensorFlow Privacy.

## 8.6 Audit, lineage & forensic logging on the edge (the genuinely hard part)
- **Audit logging** — who accessed what, when, what flowed where (SOC2/HIPAA/ITAR).
- **The edge problem** — air-gapped/offline devices can't stream logs to a central SIEM, yet logs must be tamper-evident. **Solution: append-only / WORM local storage with cryptographic hash chains** — prove integrity offline, reconcile on reconnect.
- **Lineage for compliance** — the same lineage that aids debugging answers "where is this subject's data?" and **proves data didn't cross a boundary** (ITAR).
- **Deletion propagation** — "right to be forgotten" must reach **derived artifacts: embeddings, vector indexes, on-device caches** — not just the source row. *Embeddings of personal data are personal data.* (Impressive, frequently-missed point.)

## 8.7 AI/ML-specific security threats (expect "what's unique about securing an AI data platform?")
- **Training/fine-tuning data leakage & membership inference** — models memorize and can regurgitate training data. Mitigate: minimization, DP training, eval for memorization.
- **Prompt injection & data poisoning (RAG/agentic)** — malicious content in ingested docs can hijack the model or poison retrieval. Mitigate: **treat retrieved/ingested content as untrusted**, sanitize, provenance/signature checks, content filtering at ingestion, output constraints. (Highly relevant to Companion/agentic + adjacent to your TikTok agentic-data work.)
- **Model inversion / extraction** — reconstruct inputs or steal the model via queries.
- **Supply chain** — you download a model from HuggingFace and deploy it → **sign/verify artifacts, scan, pin versions**; same scrutiny for connector dependencies. (Commonly skipped, per the security literature.)
- **Synthesis** — a "secure AI data platform" = untrusted-input handling + minimization + access-controlled retrieval + lineage + on-device isolation. Deliver this as one coherent architecture.

## 8.8 Privacy-/security-by-design (your thesis statement — memorize)
> "I don't bolt privacy on. The connector framework enforces it by construction: PII is **classified and tagged at ingestion**; **minimization** is the default via field/column selection; **residency and consent travel as metadata** through lineage; **deletion propagates to embeddings and edge caches**; **logs are PII-safe and tamper-evident**; **access control lives in retrieval**; and **on-device processing means most data never leaves the boundary at all**. That's compliance as a platform property — which is the only thing that scales across customers — and it's exactly the work I did at TikTok PDPO, now applied to an edge AI platform."

## 8.9 The market (privacy/security tooling to name)
- **Secrets/keys**: HashiCorp Vault, AWS KMS/Secrets Manager, GCP KMS, HSMs.
- **Confidential computing**: Azure Confidential Computing, AWS Nitro Enclaves, Intel SGX/TDX, Apple Secure Enclave.
- **Federated learning**: Flower, NVIDIA FLARE, TensorFlow Federated.
- **Differential privacy**: OpenDP, Google DP, Opacus, TF Privacy.
- **HE / SMPC**: Microsoft SEAL, OpenFHE.
- **Data governance/classification**: BigID, Collibra, OneTrust, Immuta (policy-based access), Concentric (ITAR classification).
- **Secure RAG patterns**: per-chunk ACLs + retrieval-time filtering + prompt-injection defense (an emerging best-practice category).

---

# Part 9 — End-to-end system designs (worked)

> Practice these **out loud**. Structure every answer: **clarify → constraints (edge/privacy) → components → name the primitive → resilience → observability → privacy → tradeoffs.**

## 9.1 Design a connector *framework* (the platform-primitive question)
Use the Part 2.3 diagram. Capability interface; canonical IR; declarative thin adapters; durable state store; cross-cutting auth/rate-limit/retry/observability/privacy inherited; conformance test suite; versioned output contracts. Borrow Singer/Airbyte's connector-contract idea + Debezium CDC + Meltano's SW discipline, **adding privacy + edge-awareness as first-class.** *This directly answers the role's headline — over-prepare it.*

## 9.2 Design a Salesforce connector
Clarify (volume? real-time? custom schema?). Bulk API 2.0 for backfill, **CDC/Platform Events** for incremental, **Describe API** for dynamic schema, OAuth JWT-bearer auth, **governor-limit-aware scheduler**, `SystemModstamp` watermark, checkpointed cursor, normalize to IR. It's an *adapter* on the framework — auth/pagination/rate-limit/obs/privacy inherited. Privacy: classify PII at ingestion, minimize to the feature contract, scoped connected-app permissions, PII-safe logs. Resilience/obs: 429 backoff, DLQ, lineage, freshness metric.

## 9.3 Design ingestion for an unreliable, rate-limited REST API
Token-bucket honoring rate headers; backoff+jitter; circuit breaker; idempotent + checkpointed + replay-safe; overlapping incremental windows + dedupe (APIs are eventually consistent); DLQ for poison records; bounded queues for backpressure; metrics on rate-limit-hits and lag.

## 9.4 Design the data layer for the offline edge device (Airbus iPad) — full
**Scenario:** Airbus floor/maintenance workers get safe, secure AI answers **offline**, keeping aerospace IP private.

**Why webAI fits (lead with this):** aerospace technical data is often **ITAR/EAR-controlled** and proprietary → can't go to a third-party cloud or cross borders; **on-device inference means data never leaves the controlled environment**, collapsing export-control + residency. Factory floors have poor connectivity → **offline-first** is mandatory. Lost/stolen iPad → enclave encryption + remote wipe.

**Your design (the data half):**
1. **Sources** — maintenance manuals & service bulletins (PDF/docs), parts catalogs (SQL/PLM), work-order systems, telemetry, SaaS asset systems, S3. Heterogeneous → connector framework.
2. **Classify & access-control at ingestion** — tag ITAR-controlled vs. general, by program, by clearance; attach permission + classification metadata to every chunk (ABAC).
3. **Transform** — clean → chunk → embed (on-device-compatible model) → **local vector index** (embedded, LanceDB-style) with metadata.
4. **Selective minimized pre-positioning** — sync only the subset for that worker's role/aircraft/program; minimized + encrypted at rest in the Secure Enclave (can't ship the whole corpus to an iPad).
5. **Serve offline** — Companion/webFrame answers locally; retrieval **filters by clearance** so a worker only retrieves authorized content.
6. **Reconcile on reconnect** — CDC-driven minimized deltas pulled; **tamper-evident hash-chained audit logs** (that couldn't be exfiltrated offline) pushed; conflict resolution for local writes.
7. **Compliance properties you can claim** — residency satisfied; ITAR exposure minimized (no export); audit preserved offline; deletion/revocation propagates to the on-device index next sync; least-privilege scoped creds; encryption in transit + at rest.
8. **Observability** — per-device sync freshness, index staleness surfaced to the user, DLQ, lineage source-doc→chunk→answer.

**Curveballs & answers:**
- *Safety-critical manual revised?* CDC on source → re-chunk/re-embed only changed sections → push delta → surface version/staleness on device; **gate answers from superseded bulletins** (version pinning + supersession flags).
- *Guarantee ITAR never leaks?* Classification + ABAC retrieval filtering + on-device-only + lineage proof-of-non-export + audit. Strongest guarantee is **architectural**: it can't be exported if it never leaves the device.
- *Question the index doesn't cover, offline?* Honest "no answer / escalate" beats hallucination in a safety context; queue for reconnect; never fabricate maintenance guidance.
- *Lost iPad?* Enclave-held keys, remote wipe on next check-in, expiring local data leases.

## 9.5 "A model needs feature X; the source is messy and slow."
Negotiate the **feature contract** (semantics/format/freshness/null policy) → pick extraction (CDC vs. incremental) → transform with quality gates + point-in-time correctness → version the contract → monitor freshness/drift → coordinate version bump with ML so model + data evolve together.

---

# Part 10 — Design methodology + running a whiteboard

A repeatable frame for *any* design prompt (use it visibly — interviewers grade your process):

1. **Clarify & scope** — ask 2–4 sharp questions (volume, latency/freshness, real-time vs. batch, sensitivity/classification, offline?, fleet size). Don't design before you know the constraints. (Matches "comfortable with ambiguity.")
2. **State assumptions** — out loud, so they can correct you.
3. **Sketch components** — boxes & arrows; sources → connector framework → IR → transform → index/store → delivery → inference.
4. **Name the primitive** — explicitly say "the reusable piece here is …" (the role's whole theme).
5. **Resilience** — backoff, idempotency, checkpoints, DLQ, circuit breaker.
6. **Observability** — metrics/logs/traces/lineage; how you'd debug it.
7. **Privacy/security** — classification, minimization, access control, encryption, residency/ITAR, deletion propagation.
8. **Edge constraints** — offline, minimized sync, conflict resolution, staleness.
9. **Tradeoffs** — articulate at least two (consistency vs. latency, freshness vs. battery, accuracy vs. footprint, managed vs. self-host). *Naming tradeoffs is what separates senior from mid.*
10. **Calibrated honesty** — if pushed past your knowledge: state the principle, state the tradeoff, say how you'd validate ("I'd confirm exact ITAR scope with compliance, but architecturally the control is X"). Matches **Truth** + **Humility**.

---

# Appendix A — One-page market cheat-sheet

- **Edge inference runtimes:** MLX/MLX-LM (Apple, fastest <14B), llama.cpp (portable, GGUF), Ollama (dev wrapper, now MLX backend), ExecuTorch (tiny→phone, Meta prod), Core ML/ANE, ONNX Runtime (cross-platform), TensorRT/vLLM (NVIDIA, PagedAttention), LiteRT/MediaPipe (mobile), MLC-LLM (compiled/WebGPU).
- **Connectors/ETL/CDC:** Airbyte (600+ connectors, builder, Debezium CDC), Singer (spec), Meltano (CLI/GitOps), Fivetran (managed), Estuary (CDC+stream), **Debezium** (OSS log-CDC), Kafka Connect, dlt, AWS DMS.
- **Schema/modeling:** dbt, Confluent Schema Registry, DataHub/OpenMetadata/Unity Catalog, Great Expectations/Soda, Iceberg/Delta/Hudi.
- **Vector DBs:** **pgvector** (Postgres default), **Qdrant** (filtered/hybrid, residency), Weaviate (modules/hybrid), Milvus/Zilliz (scale), Pinecone (managed), Chroma (proto), **LanceDB** (embedded/edge ← webAI-relevant), FAISS (library).
- **RAG frameworks:** LangChain, LlamaIndex, Haystack. **Feature stores:** Feast, Tecton.
- **Queues/stream/orchestration:** Kafka, Pulsar, Kinesis, SQS; Flink, Spark Streaming; Airflow, Dagster, Prefect, **Temporal** (durable execution).
- **Observability/lineage:** OpenTelemetry, Prometheus+Grafana, Datadog; **OpenLineage**+Marquez, DataHub, Atlas; Monte Carlo, Soda, Bigeye.
- **Privacy/security:** Vault/KMS/HSM; Azure Confidential Computing / AWS Nitro / Intel SGX-TDX / Apple Secure Enclave; **Flower**/NVFLARE/TFF (FL); OpenDP/Opacus/TF-Privacy (DP); SEAL/OpenFHE (HE); BigID/Immuta/OneTrust (governance).

# Appendix B — Glossary (terms they may throw at you)
- **IR (intermediate representation):** a normalized internal model (webFrame uses one for models; you use one for data).
- **CDC:** change data capture — reading a DB's change log to capture inserts/updates/deletes.
- **ANN / HNSW / IVF / PQ:** approximate nearest neighbor; graph index / inverted-file index / product quantization.
- **Hybrid search:** dense (vector) + sparse (BM25/keyword) retrieval combined.
- **RAG:** retrieval-augmented generation — retrieve relevant chunks, feed to the model.
- **UMA:** Apple unified memory architecture — shared CPU/GPU/NPU memory (why edge inference works).
- **Quantization (PTQ/QAT, INT8/INT4, GGUF, AWQ/GPTQ):** lower-precision weights to shrink/speed models.
- **LoRA/QLoRA:** low-rank adapters for cheap fine-tuning / per-customer personalization.
- **PagedAttention:** KV-cache-as-virtual-memory technique (vLLM) that boosts throughput.
- **CRDT / vector clock / LWW:** conflict-resolution mechanisms for distributed/offline writes.
- **CAP / PACELC:** consistency-vs-availability (under partition) / -vs-latency (else) tradeoff frameworks.
- **Backpressure / DLQ / circuit breaker / token bucket:** resilience patterns (Part 6).
- **DSAR:** data-subject access request (GDPR/CCPA).
- **DP / FL / SMPC / HE / TEE:** differential privacy / federated learning / secure multi-party computation / homomorphic encryption / trusted execution environment.
- **ABAC / RBAC:** attribute-based / role-based access control.
- **ITAR / EAR / USML:** US export-control regimes / the munitions list; an "export" can include cloud access or a prompt.
- **WORM:** write-once-read-many storage (tamper-evident audit logs).
- **PETs:** privacy-enhancing technologies (umbrella for DP/FL/SMPC/HE/tokenization/etc.).

---

*Pair this with the first study guide for behavioral prep, STAR stories, and questions to ask them. Lead with privacy — it's your edge. Bend every design toward reusable primitives. Be precise where strong, calibrated where stretching.*
