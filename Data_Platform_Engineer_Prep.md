# webAI — Data Platform Engineer: Complete Interview Prep

> Built for someone coming from **TikTok PDPO (Data Scientist)** with deep privacy-workflow / data-infra experience, interviewing for a role that lives at the intersection of **data connectors + distributed systems + applied AI**, with a heavy expected drill on **privacy, security, and compliance**.

---

## 0. How to use this guide

- **Sections 1–3**: Know the company, the product, and what they *actually* want. Memorize the "platform primitive vs. pipeline" framing — it is the spine of this role.
- **Sections 4–9**: Core technical depth (connectors, transformation, distributed systems, ingestion resilience, observability).
- **Section 10**: Privacy / security / compliance — *the area you flagged as your worry.* This is the longest section on purpose. This is also your **home turf** given PDPO. Treat it as where you win, not where you lose.
- **Section 11**: The Airbus edge use case, fully walked through.
- **Section 12**: System-design exercises with worked answers.
- **Sections 13–14**: Rapid-fire Q&A banks (technical + privacy).
- **Sections 15–17**: Behavioral (mapped to their values), positioning your TikTok story, and questions to ask them.
- **Section 18**: 48-hour checklist.

**One-line mental model for the whole interview:** *"I build reusable data primitives that pull heterogeneous enterprise data into a private, edge-deployed AI platform — and I make privacy, resilience, and observability properties of the platform itself, not afterthoughts."*

---

## 1. Company snapshot — webAI

**What they are:** A privacy-first / "sovereign AI" company building infrastructure to run large AI models **locally on consumer/enterprise hardware (especially Apple Silicon)** instead of the cloud. Founded 2019, HQ Austin TX. Founders: **David Stout (CEO)**, Tyler Mauer, Ethan Baird. Series A brought total raised to ~$60M at a ~$700M valuation. ~120+ employees.

**Core thesis (say this back to them in your own words):** AI is valuable only if a company's data and IP stay private. Sending everything to a centralized cloud creates data-protection risk, energy cost, and runaway compute cost. The future is **distributed processing at the edge** — computation close to where data is generated — using networks of smaller specialized models that collaborate, rather than one giant cloud model.

**Product suite (know these names cold):**

| Product | What it does |
|---|---|
| **Navigator** | Local AI development workspace — drag-and-drop canvas to connect inputs (cameras, datasets, media), build/train/fine-tune, and deploy models to a device or a local network of devices. No cloud dependency. |
| **Companion** | On-device AI assistant embedded in employee workflows; taps a marketplace of private enterprise AI solutions; runs across devices. |
| **webFrame** | The inference/optimization engine. Parses HuggingFace models into an **intermediate representation** mapped to backend-specific modules; reads available compute at runtime; builds a **compute plan** that can run on one node or shard across a cluster. Uses **adaptive quantization + intelligent batching** (claims ~99.5% accuracy retained, up to ~30% size reduction, <0.5% accuracy loss). Demonstrated running Llama 3.1 405B across a cluster of consumer devices. |
| **Runtime** | Orchestrates distributed AI workloads across heterogeneous hardware — one device up to clusters. |
| **Network** | A secure *local* fabric connecting models, datasets, and devices for high-speed collaboration without the cloud. |
| **Infrastructure** | The governance/control layer for IT + security teams to manage company-wide AI deployment. |
| **CLI** | Command-line control to build/extend/manage AI environments at scale. |

**Why this matters for *your* role:** All of webFrame/Runtime/Network is about *running models* privately on the edge. **Your role is the missing front half** — getting the *data* into that world. You build "the connective tissue between real-world data systems and the AI platform." The models are useless without high-quality, well-modeled, privacy-respecting data flowing in. You are the on-ramp.

**Apple Silicon angle to know:** Apple's **Unified Memory Architecture** means CPU and GPU share the same physical memory (no PCIe bus copy), which is *why* on-device inference is performant. They've partnered with **MacStadium** to host on Apple Silicon in a cloud-like way. You don't need to be an Apple internals expert, but referencing "unified memory is why edge inference is viable here" shows you get the thesis.

**Their stated values:** **Truth, Ownership, Tenacity, Humility.** Memorize these — behavioral questions and "why us" answers should echo them naturally (Section 15).

---

## 2. The role, decoded

**Title:** Data Platform Engineer — "design and build the connective tissue between real-world data systems and our AI platform."

**The literal responsibilities, translated into what they're testing:**

| JD line | What they're really probing |
|---|---|
| Build connectors across SQL/NoSQL, REST/GraphQL, SaaS (CRM, storage) | Breadth + depth: can you actually integrate Salesforce, S3, a Postgres, a Mongo, a rate-limited REST API? |
| Interpret and model heterogeneous source schemas | Can you reason about messy real-world schemas and normalize them into something a model can consume? |
| Transform raw source data into formats optimized for AI inference | Do you understand RAG / embeddings / chunking / feature prep — the *AI-adjacent* part? |
| Work with ML / applied AI / forward-deployed teams on feature expectations | Can you collaborate across the inference boundary and translate "the model needs X" into a data contract? |
| Collaborate with infra to ship hosted data pipelines | Can you operate, not just prototype? |
| Optimize for latency, consistency, and **edge constraints** | Do you understand the constraints of intermittent/offline edge environments? |
| Design resilient ingestion for unreliable / rate-limited systems | Backoff, retries, idempotency, CDC, backpressure, DLQs. |
| Build logging, monitoring, debuggability into all integrations | Observability is a first-class requirement, not optional. |

**The single most important framing in the JD:** *"This role will act in terms of platform primitives, not pipelines."*

- A **pipeline** is a one-off: "ingest Acme Corp's Salesforce into our system." Bespoke, brittle, doesn't generalize.
- A **platform primitive** is a *reusable building block*: a connector framework where adding a new SaaS source is configuration + a thin adapter, with auth, pagination, rate-limiting, schema discovery, incremental sync, observability, and privacy controls **built into the framework**, inherited by every connector.

> **If you take one thing into this interview:** every design answer should bend toward *generalization, reuse, and abstraction*. When they describe a specific integration, your instinct should be "what's the primitive underneath this?"

**"Forward-deployed teams"** = engineers who embed with customers (à la Palantir). Implication: your connectors get used in messy, real customer environments by people under deadline pressure. So **self-service, good defaults, good errors, and debuggability** matter enormously.

---

## 3. Your positioning (read this before anything else)

You are afraid they'll "drill you in the secure AI domain." Flip it: **PDPO privacy data-infra is the rarest and most valuable thing you bring.** Most strong data-platform engineers can build a Salesforce connector. Very few can build one that is correct *and* privacy-compliant by construction, with consent propagation, data minimization, residency enforcement, and audit lineage. webAI's entire pitch is privacy. **You are unusually well-matched.**

Your job in the interview is to *connect* your existing skill to their stack:
- "At TikTok PDPO I built data infrastructure for privacy workflows — consent enforcement, PII handling, data subject requests, lineage. That's exactly the property webAI needs baked into every connector."
- "I've operated where privacy is a hard requirement, not a nice-to-have. I default to data minimization and provable controls."

Where you'll need to stretch (study these): the **AI-inference-specific** data prep (RAG/embeddings/chunking), **distributed-systems edge constraints**, and the **connector-framework-as-product** mindset. Sections 6–9 and 12 cover these.

---

## 4. Connectors — the core craft

### 4.1 Relational / SQL databases (Postgres, MySQL, SQL Server, Oracle)

Be ready to talk through:
- **Connection management:** pooling (PgBouncer-style), connection limits, TLS to the DB, read replicas to avoid loading production.
- **Extraction strategies:**
  - **Full snapshot** (simple, expensive, fine for small/cold tables).
  - **Incremental by watermark** (`updated_at > last_seen`) — requires a reliable monotonic column; beware clock skew and rows updated without bumping the column.
  - **Change Data Capture (CDC):** read the WAL/binlog (Debezium, logical replication). Captures *deletes* and is low-impact on the source. The gold standard for keeping a downstream store fresh. Know the tradeoff: operationally heavier, needs replication privileges.
- **Schema discovery:** `information_schema` / catalog tables to auto-discover tables, columns, types, PKs, FKs, nullability. This is how you make the connector *generic*.
- **Type mapping:** DB types → a normalized internal type system (your IR). Handle `NUMERIC/DECIMAL` precision, timezones in `TIMESTAMP`, `JSONB`, arrays, enums.
- **Gotchas:** large tables (chunk by PK ranges / keyset pagination, never `OFFSET` on huge tables), long-running transactions, isolation level (read snapshot consistency), and not melting the source DB.

### 4.2 NoSQL (MongoDB, DynamoDB, Cassandra, Redis)

- **No fixed schema** → you must *infer* schema by sampling, and handle schema drift (documents that don't match). Decide: enforce a schema, or carry a flexible/union type forward?
- **MongoDB:** change streams (CDC equivalent), `_id` as the natural key, nested documents → flatten or preserve nesting depending on downstream need.
- **DynamoDB:** partition/sort key design dictates access patterns; **DynamoDB Streams** for CDC; scans are expensive (consume RCUs) — prefer query or parallel segmented scans with rate control.
- **Cassandra:** wide rows, eventual consistency, tunable consistency levels (`ONE`/`QUORUM`/`ALL`) — know that you choose consistency vs. latency per query.
- **Key point to make:** "NoSQL forces schema-on-read; my connector treats schema inference + drift handling as a first-class feature, surfacing drift as an observable event rather than silently dropping fields."

### 4.3 REST APIs

- **Pagination styles:** offset/limit, page-number, cursor/keyset, link-header (`rel="next"`). A good connector framework abstracts all of these behind one "paginator" interface.
- **Auth:** API key, Basic, **OAuth 2.0** (authorization code, client credentials, refresh-token rotation), JWT. Token refresh + secure storage is a recurring theme.
- **Rate limits:** respect `Retry-After`, `X-RateLimit-Remaining/Reset`. Token-bucket client-side limiter. (See Section 8.)
- **Resilience:** timeouts, retries with exponential backoff + jitter, circuit breakers, idempotency keys for writes.
- **Incremental sync:** filter params (`?modified_since=`), but many APIs lie / are eventually consistent — overlap your windows and dedupe.

### 4.4 GraphQL APIs

- **Why it differs from REST:** single endpoint, *client specifies the shape* of the response → you fetch exactly the fields a feature needs (great for data minimization!).
- **Introspection:** GraphQL exposes its own schema via introspection query — perfect for *automated schema discovery* in a connector framework.
- **Pitfalls:** the **N+1 query problem**, query depth/complexity limits the server enforces, cursor-based pagination via Relay connections (`edges`/`node`/`pageInfo`), and partial responses (`data` + `errors` can both be present — handle both).
- **Strong line:** "GraphQL introspection lets the connector self-configure its schema model, and field selection lets me pull *only* the fields a given feature contract requires — minimization by design."

### 4.5 Salesforce / CRM (call this out — JD names it explicitly)

Salesforce is famously fiddly; knowing specifics signals you've actually done it:
- **APIs:** REST API, SOAP API, **Bulk API 2.0** (for large extracts — async jobs, CSV batches), **Streaming API / Change Data Capture / Platform Events** (for near-real-time), and **SOQL** (Salesforce's query language).
- **Governor limits:** Salesforce enforces hard API call limits per 24h, query row limits, etc. Your connector *must* be limit-aware → batch with Bulk API for big loads, use incremental queries (`SystemModstamp`) otherwise.
- **Schema reality:** heavily customized orgs — custom objects (`__c` suffix), custom fields, picklists, record types. Use the **Describe** (`sObject describe`) metadata API to discover schema dynamically rather than hardcoding.
- **Auth:** OAuth 2.0 (JWT bearer flow for server-to-server is common), connected apps, scopes.
- **Strong line:** "I'd build the Salesforce connector on Bulk API 2.0 for backfills and CDC/Platform Events for incremental, driven by the Describe API so it adapts to each org's custom schema — and I'd treat governor limits as a first-class budget the scheduler respects."

### 4.6 Object storage (S3 — also named in JD) & file sources

- **S3:** list (paginated), get, multipart for large objects, byte-range reads, `If-Modified-Since`/ETags for incremental, **S3 event notifications / EventBridge** to trigger ingestion on new objects.
- **Formats:** CSV (delimiter/quoting hell), JSON/JSONL, **Parquet/ORC** (columnar, predicate pushdown, schema embedded — prefer these), Avro (schema evolution), plus unstructured (PDF, images, audio → relevant for multimodal inference).
- **Auth/security:** IAM roles, **least-privilege scoped policies**, SSE-S3/SSE-KMS encryption, VPC endpoints, bucket policies. Mention you'd never use long-lived root keys — scoped, rotating credentials.
- **Schema:** for Parquet you read embedded schema; for CSV you infer + let users override.

### 4.7 The connector framework (the "platform primitive" answer)

When asked "how would you build connectors generally," describe a **framework**, not N integrations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Connector Framework (SDK)                  │
│  Common cross-cutting concerns inherited by every connector:  │
│   • Auth & secret management   • Pagination abstraction       │
│   • Rate limiting & backoff    • Retry / circuit breaking     │
│   • Schema discovery → IR      • Incremental sync / state      │
│   • Type normalization         • Observability (logs/metrics/  │
│   • Privacy controls (PII       traces) + lineage emission     │
│     tagging, minimization,     • Error taxonomy & DLQ          │
│     residency, consent)        • Idempotency / checkpointing   │
└─────────────────────────────────────────────────────────────┘
        ▲              ▲              ▲              ▲
   Postgres        Salesforce        S3        REST/GraphQL
   adapter          adapter        adapter       adapter
  (thin: just describe how to auth, paginate, discover, read)
```

Key design principles to articulate:
- **Capability interface:** each source declares capabilities (supports CDC? supports incremental? supports field selection?). The framework adapts behavior to declared capabilities.
- **Intermediate Representation (IR):** every source normalizes into one internal schema/record model (note: webAI's *webFrame* already uses an "intermediate representation" for models — using the same vocabulary for data shows you understand their design philosophy).
- **Declarative config:** adding a source = a manifest + thin adapter, not a new codebase.
- **State/checkpoint store:** durable cursor per source so syncs resume, not restart.
- **Contract tests:** every adapter passes the same conformance suite (pagination correctness, incremental correctness, schema-drift handling, rate-limit compliance).
- **Compare to prior art** (shows awareness, not that you'd copy): Airbyte/Singer/Fivetran connector models, Debezium for CDC, Meltano. You can say "I'd take the *protocol* idea from Singer/Airbyte — a standard spec every connector implements — but add privacy and edge-awareness as first-class, which off-the-shelf tools don't do."

---

## 5. Schema interpretation & modeling heterogeneous sources

This is its own JD bullet — expect a question like *"You're handed three sources with overlapping but inconsistent schemas. How do you model them?"*

- **Discovery first:** auto-introspect (catalog, Describe, GraphQL introspection, Parquet metadata, document sampling).
- **Canonical/unified model:** map disparate source schemas to a canonical entity model (e.g., a "Customer" entity that Salesforce `Account`, a Postgres `users` table, and a Mongo `profiles` collection all map into). This is classic **schema mapping / entity resolution**.
- **Entity resolution:** same real-world entity across sources → matching keys, fuzzy matching, dedup. Mention deterministic (shared ID) vs. probabilistic matching.
- **Schema drift / evolution:** sources change. Strategies: additive-only by default, versioned schemas, surface breaking changes as alerts, never silently drop.
- **Type reconciliation:** unify types across sources into your IR; handle nulls, units, timezones, encodings, currency.
- **Semantic modeling:** beyond types — what does a field *mean*? `status=3` needs the enum decoded. Capture metadata/data dictionaries.
- **Data quality gates:** completeness, validity, freshness, uniqueness checks at ingestion (Great Expectations-style assertions) so garbage doesn't reach the model.

> **Strong line:** "Heterogeneous schema modeling is fundamentally entity resolution plus type reconciliation into a canonical IR, with schema drift treated as a monitored, versioned event rather than a silent failure."

---

## 6. Transforming raw data into AI-inference-ready formats

This is the "applied AI" half — study it even if it's newer to you. webAI runs LLMs and vision models (object detection, classification) locally, so "inference-ready" spans both **LLM/RAG** and **structured feature** prep.

### 6.1 For LLMs / RAG (retrieval-augmented generation)
- **Cleaning & normalization:** strip boilerplate, normalize encoding, handle tables/PDFs/HTML → text.
- **Chunking:** split documents into retrievable units. Strategies: fixed-size with overlap, recursive/semantic (split on structure — headers, paragraphs), sentence-window. Tradeoff: too big = imprecise retrieval + context bloat; too small = lost context. Carry **metadata** (source, section, permissions, timestamp) on each chunk.
- **Embeddings:** turn chunks into vectors via an embedding model. On webAI's edge model: the embedding model itself can run on-device (privacy!). Know vector dimensionality, normalization, and that the *same* model must embed both docs and queries.
- **Vector store / index:** ANN search (HNSW, IVF). On the edge this index lives *locally*. Metadata filtering (only return chunks the user is authorized to see — ties to access control).
- **Freshness:** re-embed on source change (CDC → re-chunk → re-embed the delta, not the whole corpus).

### 6.2 For structured / feature prep
- **Feature contracts** with the ML team: name, type, semantics, freshness SLA, null policy, allowed ranges. This *is* the "define feature expectations" JD bullet — frame it as a **data contract** between you and ML.
- **Transformations:** joins, aggregations, encoding (categorical→numeric), normalization/scaling, time-windowing.
- **Point-in-time correctness / no leakage:** features must reflect what was known *at inference time* — a privacy- and correctness-sensitive topic you can speak to.
- **Feature store concept:** consistency between training-time and inference-time features (online/offline skew). Even if webAI doesn't have a formal feature store, the *concept* shows maturity.

### 6.3 Working across the inference boundary
- The deliverable of a feature/data contract is a **versioned, testable interface**. When ML says "the model needs field X in format Y with freshness Z," you encode that as a contract the connector framework enforces and monitors.
- Mention **schema/contract versioning** so a model and its data evolve together without silent breakage.

> **Strong line:** "I treat the boundary with the ML team as a versioned data contract: they specify feature semantics, freshness, and format; my connectors guarantee and monitor them. For RAG sources, 'inference-ready' means clean → chunk → embed → index locally, with permissions carried as chunk metadata so retrieval itself respects access control."

---

## 7. Distributed systems & edge constraints

You sit "at the intersection of distributed systems," and edge is webAI's whole thing. Expect conceptual depth here.

- **CAP / PACELC:** under partition, choose consistency or availability; even without partitions, latency-vs-consistency. On the edge with intermittent connectivity, **partitions are the normal case, not the exception** — design for AP with eventual reconciliation.
- **Consistency models:** strong vs. eventual vs. causal vs. read-your-writes. Know when each is acceptable. For an offline edge device, you typically accept eventual consistency + local-first reads.
- **Edge-specific constraints (this is the differentiator — emphasize):**
  - **Intermittent / no connectivity** → local-first: the device must be useful offline. Data must be *pre-positioned* on the device.
  - **Limited storage/compute/memory** → you can't ship the whole enterprise DB to an iPad. Selective sync, summarization/embedding to compress, quantized indexes.
  - **Bounded battery/thermal** → batch + schedule sync windows; avoid chatty network.
  - **Sync conflicts** when device reconnects → CRDTs, last-writer-wins with vector clocks, or domain-specific merge.
  - **Data freshness vs. staleness tradeoff** → on the edge you serve possibly-stale-but-private data; you must surface staleness.
  - **Security of data at rest on a physical device that can be lost/stolen** → full-disk + per-record encryption, remote wipe, secure enclave key storage.
- **Tie to webAI:** webFrame builds a *compute plan* across nodes and shards models; you're the data analog — deciding *what data, in what form, gets pre-positioned on which device*, and how it reconciles. "Selective, privacy-filtered, pre-embedded sync to the edge" is your headline.

> **Strong line:** "On the edge, partitions are the default. I design data delivery as local-first with eventual reconciliation, ship only the minimized/embedded subset a device needs, encrypt at rest in the secure enclave, and treat reconnection-time conflict resolution as a designed-for case."

---

## 8. Resilient ingestion patterns (unreliable / rate-limited systems)

Direct JD bullet. Have these crisp:

- **Retries with exponential backoff + jitter** (avoid thundering-herd / synchronized retries). Cap attempts; distinguish retryable (5xx, 429, timeouts) from non-retryable (4xx auth/validation).
- **Respect rate limits:** honor `Retry-After` and rate-limit headers; client-side **token-bucket / leaky-bucket** limiter; adaptive throttling (slow down as you approach the limit). Salesforce governor limits and API quotas are the canonical example.
- **Idempotency:** idempotency keys for writes; dedup on reads (records can arrive twice). Make the whole sync replay-safe.
- **Checkpointing / resumability:** durable cursor so a crash resumes mid-sync rather than restarting (and re-hammering the source).
- **Backpressure:** if downstream (embedding, the edge sync) can't keep up, signal upstream to slow rather than buffering unboundedly. Bounded queues.
- **Dead-letter queue (DLQ):** records that repeatedly fail go to a DLQ with full context for later inspection — never silently dropped, never block the stream.
- **Circuit breaker:** stop hammering a failing source, fail fast, probe periodically to recover.
- **At-least-once vs. exactly-once:** be honest — exactly-once end-to-end is usually "at-least-once + idempotent consumer." Say that; it signals maturity.
- **Poison-message handling, timeouts, bulkheads** (isolate one source's failure from others).

> **Strong line:** "I assume sources are unreliable and rate-limited by default. The framework gives every connector backoff+jitter, header-aware token-bucket throttling, durable checkpoints, idempotent replay, a DLQ for poison records, and a circuit breaker — so a flaky source degrades gracefully instead of cascading."

---

## 9. Observability — logging, monitoring, debuggability

JD: *"Build logging, monitoring and debuggability into all integrations."* Treat as first-class.

- **Three pillars:** **metrics** (rates, latencies, error rates, records synced, lag/freshness), **logs** (structured, with correlation IDs, never logging PII in plaintext — *critical given privacy*), **traces** (distributed tracing across the connector → transform → embed → sync path).
- **Connector-specific signals:** sync success/failure, records processed, schema-drift events, rate-limit hits, retry counts, DLQ depth, freshness/lag, auth/token-refresh failures.
- **Data lineage:** where did each record come from, what transforms touched it, where did it land — essential for debugging *and* for privacy/compliance (DSAR, audit). This is a natural bridge to your PDPO background.
- **Debuggability for forward-deployed teams:** good error taxonomy and messages ("Salesforce returned 401: token expired — re-auth connector X" not "request failed"), dry-run/replay mode, a way to inspect a single record's journey.
- **SLOs / alerting:** define freshness and reliability SLOs per source; alert on breach, not on every blip.
- **PII-safe logging:** redact/tokenize before logging; log *references* not values. (Say this unprompted — it's the privacy-aware instinct they want.)

> **Strong line:** "Observability isn't a dashboard I bolt on; it's emitted by the framework — structured PII-safe logs, per-source metrics for freshness/errors/rate-limits, distributed traces, and full lineage. Lineage does double duty: it's how forward-deployed engineers debug, and it's how we answer a data-subject or audit request."

---

## 10. PRIVACY, SECURITY & COMPLIANCE — the deep section

> This is where you expect to be drilled, and it's your strongest area. The goal: show you can make privacy a **property of the platform**, enforced by construction, not a manual checklist. Below is structured so you can speak fluently across regulation, technique, and architecture.

### 10.1 The regulatory landscape (know what each is + the one-line "so what")

- **GDPR (EU):** lawful basis for processing, **data minimization**, purpose limitation, storage limitation, **data subject rights** (access/DSAR, erasure/"right to be forgotten," portability, rectification), **DPIAs** for high-risk processing, breach notification (72h), **data residency / international transfer** rules (SCCs, adequacy). *So what:* every connector must support minimization, deletion propagation, and lineage to honor DSARs.
- **CCPA / CPRA (California):** consumer rights to know/delete/opt-out of "sale/sharing," sensitive PI category. *So what:* opt-out and deletion must flow through to derived data (including embeddings).
- **HIPAA (US health):** PHI, covered entities/business associates (**BAA**), minimum necessary, Security Rule (administrative/physical/technical safeguards). *So what:* if a customer is healthcare, your connectors handle PHI → encryption + access control + audit mandatory; on-device/edge is a *selling point* (PHI never leaves the premises).
- **SOC 2 (Type II):** auditor attestation on controls across Security, Availability, Processing Integrity, Confidentiality, Privacy. *So what:* webAI almost certainly is or wants to be SOC 2; your logging/access-control/change-management feed into it.
- **ISO 27001 (ISMS) / ISO 27701 (privacy extension) / NIST CSF / NIST 800-53:** security & privacy management frameworks. Name-drop ISO 27701 to show privacy-specific framework awareness.
- **ITAR / EAR (US export control):** *Directly relevant to the Airbus/aerospace use case.* ITAR governs defense articles on the **US Munitions List**; technical data can't be exported (incl. to foreign persons or foreign cloud regions) without authorization. Even **sharing a file, cloud access, or an AI prompt** can be an "export." *So what:* **this is the killer argument for webAI's edge model** — if data never leaves the controlled environment/device, you drastically shrink ITAR/EAR exposure. (See Section 11.)
- **EU AI Act:** risk tiers for AI systems; transparency, data governance, logging obligations for high-risk systems. Worth a sentence to show currency.
- **Data residency / sovereignty:** data must physically stay in a jurisdiction. *So what:* edge/on-prem inherently satisfies residency because data stays local — another core webAI advantage you can articulate.

### 10.2 PII handling techniques (your bread and butter — be precise about differences)

- **Classification / discovery:** automatically detect and tag PII/PHI/sensitive fields at ingestion (pattern + ML-based detection). You can't protect what you haven't classified. Tag once at the connector, propagate the tag through lineage.
- **Data minimization:** pull and retain only fields a feature contract needs. GraphQL field selection and column projection are minimization *mechanisms*.
- **Pseudonymization:** replace identifiers with reversible tokens (mapping held separately, secured). Reversible with the key. (GDPR-recognized safeguard, still personal data.)
- **Anonymization:** irreversibly remove identifiability (true anonymization takes data out of GDPR scope — but it's hard; beware re-identification via quasi-identifiers).
- **Tokenization:** swap sensitive values for non-sensitive tokens via a token vault (common for payment/PII).
- **k-anonymity / l-diversity / t-closeness:** make each record indistinguishable within a group of k; guard against attribute disclosure. Useful when sharing aggregates.
- **Differential privacy:** add calibrated noise so individual records can't be inferred from outputs/aggregates; quantified by ε (privacy budget). Mention you understand the **utility-vs-privacy tradeoff** (smaller ε = more private, less accurate).
- **Be precise:** interviewers love when you distinguish pseudonymization (reversible) from anonymization (irreversible) from tokenization (vault-based swap). Many candidates blur these.

### 10.3 Encryption & key management

- **In transit:** TLS 1.2+/mTLS for service-to-service; certificate validation (don't disable it "to make it work").
- **At rest:** disk/volume encryption + field/column-level encryption for sensitive fields; **envelope encryption** (data keys encrypted by a master key in a **KMS/HSM**).
- **Key management:** rotation, separation of duties, KMS/HSM (hardware security module), never hardcode keys; on the edge, keys live in the **Secure Enclave**.
- **End-to-end / advanced:** for the device case, data encrypted such that only the device can decrypt. Mention **confidential computing / TEEs** (next subsection).
- **Secrets management for connectors:** source credentials in a vault (Vault/KMS/secrets manager), short-lived scoped tokens, automatic rotation, never in code/logs/config files. This is the #1 practical security failure mode for connectors — call it out.

### 10.4 Access control & authZ

- **Least privilege / need-to-know** — the connector's source credentials are scoped to exactly the data needed.
- **RBAC vs. ABAC:** role-based vs. attribute-based (context: user clearance, data classification, location, time). For aerospace/ITAR, ABAC with clearance attributes is often necessary.
- **Authorization carried into retrieval:** for RAG, each chunk carries permissions metadata so retrieval returns only what the requesting user may see — **don't rely on the model to "not say" restricted content; filter at retrieval.** (Say this — it's a sophisticated point.)
- **Zero-trust:** authenticate/authorize every request; no implicit trust from network location.
- **Separation of duties + just-in-time access** for operators.

### 10.5 Edge / on-device privacy techniques (webAI-native — emphasize these)

- **On-device processing = the core privacy primitive:** data is processed where it's generated; it never traverses the network or hits a third-party cloud. This *is* webAI's thesis. Most compliance problems (residency, ITAR export, third-party processor risk) shrink because there's no exfiltration.
- **Trusted Execution Environments (TEEs) / Secure Enclave / confidential computing:** isolated processing region protected even from a compromised OS; remote attestation proves the right code is running.
- **Federated learning:** train across devices, only model updates (gradients) leave — raw data never does. Mention **secure aggregation** so individual updates aren't exposed.
- **Homomorphic encryption (HE):** compute on encrypted data without decrypting. Know it's powerful but computationally expensive — appropriate for narrow cases.
- **Secure multi-party computation (SMPC):** multiple parties jointly compute over private inputs without revealing them.
- **Honest tradeoffs:** HE/SMPC are heavy; federated learning + on-device + TEE + minimization is usually the pragmatic stack. Showing you know *when not* to reach for the exotic tool is a maturity signal.

### 10.6 Audit, lineage & forensic logging on the edge (the genuinely hard part)

- **Audit logging:** who accessed what, when, what flowed where — required for SOC2/HIPAA/ITAR.
- **The edge problem:** in an air-gapped/offline environment, audit logs **can't be exfiltrated** to a central SIEM in real time, yet must be tamper-evident. Solution to name: **append-only / WORM local storage with cryptographic hash chains** — you can prove logs weren't altered even if you can't export them, and reconcile when connectivity returns.
- **Lineage for compliance:** the same lineage that aids debugging answers "where is this data subject's data?" for DSAR/erasure — and proves data didn't cross a boundary it shouldn't (ITAR).
- **Deletion propagation:** "right to be forgotten" must reach **derived artifacts** — including **embeddings/vector indexes and any on-device caches**, not just the source row. This is a subtle, impressive point: *embeddings of personal data are personal data.*

### 10.7 AI/ML-specific security threats

Be ready for "what are the security risks unique to an AI data platform?":
- **Training/fine-tuning data leakage & membership inference:** models can memorize and regurgitate training data → privacy leak. Mitigate with minimization, DP training, eval.
- **Prompt injection / data poisoning (for RAG/agentic):** malicious content in ingested documents can hijack the model or poison retrieval. Mitigate: treat retrieved content as untrusted, sanitize, provenance checks, content filtering at ingestion. (webAI's agentic/Companion context makes this very relevant — and it's adjacent to your TikTok agentic-data work.)
- **Model extraction / inversion** attacks.
- **Supply chain:** you download a model from HuggingFace and deploy it — sign/verify artifacts, scan, pin versions (the article called this out as the commonly-skipped step). Connectors' dependencies need the same scrutiny.
- **The "secure AI data platform" synthesis:** untrusted input handling + minimization + access-controlled retrieval + lineage + on-device isolation. You can give this as a coherent architecture, which is exactly the level they want.

### 10.8 Privacy-by-design / compliance-by-construction (your thesis statement)

> **The line that should land:** "I don't bolt privacy on. The connector framework enforces it by construction — PII is classified and tagged at ingestion, minimization is the default via field/column selection, residency and consent travel as metadata through lineage, deletion propagates to embeddings and edge caches, logs are PII-safe and tamper-evident, and on-device processing means most data never leaves the boundary in the first place. That's compliance as a platform property, which is exactly what scales across customers — and it's the work I did at TikTok PDPO, now applied to an edge AI platform."

---

## 11. The Airbus edge use case — full walkthrough

**The scenario (as you described it):** An iPad device for Airbus floor/maintenance workers that gives AI answers **offline (no internet)**, **safely and securely**, keeping aerospace IP and data private.

Be ready to whiteboard the **data side** end to end.

**Why this is a perfect webAI fit (lead with this):**
- Aerospace technical data is often **ITAR/EAR-controlled** and highly proprietary → it cannot go to a third-party cloud or cross borders. **On-device inference means the data never leaves the controlled environment** — collapsing the export-control and residency problem.
- Factory floors / maintenance bays / remote sites have **poor or no connectivity** → offline-first is mandatory, which is webAI's edge model.
- A lost/stolen iPad is a real threat → at-rest encryption in the Secure Enclave + remote wipe.

**Your design (the data-platform engineer's part):**

1. **Sources (enterprise side):** maintenance manuals & service bulletins (PDF/docs), parts catalogs (SQL/PLM systems), work-order systems, sensor/telemetry, SaaS (e.g., a CRM/asset system), S3/object stores. Heterogeneous → your connector framework handles each.
2. **Classification & access control at ingestion:** tag ITAR-controlled vs. general, by program, by clearance. Attach permission + classification metadata to every chunk/record (ABAC with clearance attributes).
3. **Transform for inference:** clean → chunk → embed (with an on-device-compatible embedding model) → build a **local vector index**. Carry classification/permission metadata on each chunk so retrieval is access-controlled.
4. **Selective, minimized pre-positioning to the device:** you cannot sync the whole enterprise corpus to an iPad. Sync only the subset relevant to that worker's role/aircraft/program — minimized and encrypted. This is the *edge data delivery* design.
5. **On-device serving:** Companion/webFrame answers queries locally against the local index — fully offline. Retrieval filters by the user's clearance so a worker only ever retrieves authorized content.
6. **Resilience & sync:** when the device reconnects (sync window), it pulls deltas (CDC-driven, minimized) and pushes **tamper-evident, hash-chained audit logs** that couldn't be exfiltrated while offline. Conflict resolution for any local writes.
7. **Privacy/compliance properties you can claim:** data residency satisfied (data stays on-prem/on-device); ITAR exposure minimized (no export); audit trail preserved even offline; deletion/revocation propagates on next sync (including to the device's local index); least-privilege scoped source credentials; encryption in transit + at rest in Secure Enclave.
8. **Observability:** per-device sync freshness, index staleness surfaced to the user ("manual last updated X"), DLQ for failed source records, lineage from source doc → chunk → on-device answer.

**Likely curveballs & your answers:**
- *"How do you handle a manual being revised (safety-critical)?"* CDC on the source → re-chunk/re-embed only the changed sections → push delta on next sync → surface staleness + version on the device so a worker never trusts an outdated bulletin; for safety-critical updates, gate the device from answering from superseded content (version pinning + supersession flags).
- *"How do you guarantee ITAR content never leaks?"* Classification at ingestion + ABAC retrieval filtering + on-device only + lineage proof of non-export + audit. The strongest guarantee is architectural: it can't be exported if it never leaves the device.
- *"Worker asks something the index doesn't cover, offline."* Honest "no answer / escalate" beats hallucination in a safety context; queue the question for when connectivity returns; never fabricate maintenance guidance.
- *"Lost iPad?"* Encrypted at rest (enclave-held keys), remote wipe on next check-in, short-lived local data leases that expire if not renewed.

---

## 12. System-design exercises (worked)

Practice saying these out loud. Structure every answer: **clarify requirements → identify constraints (esp. edge/privacy) → sketch components → call out the "primitive" → resilience → observability → privacy → tradeoffs.**

### 12.1 "Design a Salesforce connector"
- *Clarify:* one-time vs. continuous? volume? real-time needs? custom schema?
- *Design:* Bulk API 2.0 for backfill, CDC/Platform Events for incremental, Describe API for dynamic schema, OAuth JWT bearer auth, governor-limit-aware scheduler, `SystemModstamp` watermark, checkpointed cursor, normalize to IR.
- *Primitive:* it's an *adapter* on the framework — auth/pagination/rate-limit/observability/privacy inherited; the Salesforce-specific part is just the Describe→IR mapping and the Bulk/CDC strategy.
- *Privacy:* classify PII at ingestion, minimize fields to the feature contract, scoped connected-app permissions, PII-safe logs.
- *Resilience/obs:* governor-limit budget, backoff on 429, DLQ, lineage, freshness metric.

### 12.2 "Design ingestion for an unreliable, rate-limited REST API"
- Token-bucket limiter honoring rate headers; backoff+jitter; circuit breaker; idempotent, checkpointed, replay-safe sync; overlapping incremental windows + dedupe (APIs are eventually consistent); DLQ for poison records; bounded queues for backpressure; metrics on rate-limit-hits and lag.

### 12.3 "Design the data layer for an offline edge device" (the Airbus pattern — see §11)
- Local-first, selective minimized sync, on-device vector index with permission metadata, Secure-Enclave at-rest encryption, hash-chained offline audit logs, delta sync + conflict resolution on reconnect, staleness surfaced to user.

### 12.4 "Design a connector *framework* (the platform-primitive question)"
- The §4.7 diagram: capability interface, IR, declarative adapters, cross-cutting concerns (auth/rate-limit/retry/observability/privacy) inherited, conformance test suite, state store. Compare to Singer/Airbyte protocol but with privacy + edge-awareness as first-class. **This is the question that most directly tests the JD's headline — nail it.**

### 12.5 "A model needs feature X; the source is messy/slow. Walk me through it."
- Negotiate the **feature contract** (semantics/format/freshness/null policy) → pick extraction (CDC vs. incremental) → transform with quality gates and point-in-time correctness → version the contract → monitor freshness/drift → coordinate version bump with ML so model + data evolve together.

---

## 13. Rapid-fire technical Q&A bank

- **Idempotent vs. exactly-once?** Exactly-once end-to-end is usually at-least-once delivery + an idempotent consumer (idempotency keys / dedupe). Be honest that true exactly-once is rare.
- **CDC vs. polling?** CDC captures deletes, is low-impact, near-real-time, but operationally heavier (replication access, log retention). Polling is simple but misses deletes and can hammer the source.
- **Why cursor pagination over offset?** Offset degrades on large tables and breaks under concurrent inserts/deletes; keyset/cursor is stable and efficient.
- **Backoff with vs. without jitter?** Without jitter, retries synchronize into thundering herds; jitter spreads them.
- **At-rest field encryption vs. full-disk?** Full-disk protects against device theft; field-level protects against DB compromise and limits blast radius; use both for sensitive fields (defense in depth).
- **Schema-on-read vs. schema-on-write?** NoSQL/lake = schema-on-read (flexible, drift risk); RDBMS = schema-on-write (enforced, rigid). Your framework infers + monitors drift either way.
- **How do you keep a vector index fresh?** CDC on source → re-chunk/re-embed only the delta → upsert into the index; track which source version produced which vectors (lineage).
- **Strong vs. eventual consistency on the edge?** Edge connectivity makes partitions normal → local-first + eventual reconciliation; surface staleness rather than block.
- **Parquet vs. CSV for an S3 source?** Parquet: columnar, compressed, embedded schema, predicate pushdown → cheaper/faster + schema clarity. Prefer it.
- **N+1 in GraphQL?** Batch with DataLoader-style batching / request needed fields in one query; respect server depth/complexity limits.
- **How do you not melt a production DB during extraction?** Read replica, keyset chunking, rate-limited reads, off-peak windows, snapshot/repeatable-read isolation.
- **Where do connector secrets live?** A vault/KMS with short-lived scoped tokens + rotation; never in code, config, or logs.

## 14. Rapid-fire privacy/security Q&A bank (your strong area)

- **Pseudonymization vs. anonymization?** Pseudonymization is reversible (still personal data under GDPR); anonymization is irreversible (out of GDPR scope, but hard — watch quasi-identifier re-identification).
- **How does "right to be forgotten" affect a vector store?** Embeddings of personal data are personal data; deletion must propagate to vectors, indexes, and on-device caches — not just the source row. Track via lineage.
- **Data minimization in a connector, concretely?** Column projection / GraphQL field selection to pull only contract-required fields; don't retain raw beyond need; minimize before embedding.
- **How do you keep logs compliant?** Structured logs that reference records, not values; redact/tokenize PII pre-write; access-controlled, retained per policy, tamper-evident on the edge (hash chains/WORM).
- **What's the privacy case for edge over cloud?** No exfiltration → satisfies residency/sovereignty, shrinks ITAR/EAR export exposure, removes third-party-processor risk, reduces breach surface. It's compliance by architecture.
- **How do you audit in an air-gapped device?** Append-only WORM local logs with cryptographic hash chains; prove integrity offline; reconcile to central SIEM on reconnect.
- **Differential privacy — when and what's the cost?** Add calibrated noise (budget ε) for aggregate releases/training; privacy-utility tradeoff — more privacy (smaller ε) = less accuracy.
- **Federated learning vs. central training for privacy?** FL keeps raw data on-device; only model updates leave; add secure aggregation so individual gradients aren't exposed.
- **RBAC vs. ABAC for aerospace?** ABAC — authorize on attributes (clearance, program, classification, location), which RBAC's coarse roles can't express for ITAR.
- **How do you stop restricted content reaching a user via RAG?** Filter at retrieval using per-chunk permission metadata — never rely on the model to "decline"; access control belongs in retrieval, not the prompt.
- **Top AI-platform-specific security risks?** Training-data leakage/membership inference, prompt injection & data poisoning via ingested content, model inversion/extraction, and model/dependency supply-chain. Treat retrieved content as untrusted; sign/scan model artifacts.
- **GDPR lawful basis — why does a connector care?** Purpose limitation means you ingest/retain only for a declared purpose; the connector should carry purpose/consent metadata so downstream use stays within basis.

> If they push *deeper* than any answer here, the honest move is: state the principle, state the tradeoff, and say how you'd validate it ("I'd confirm the exact ITAR classification scope with compliance, but architecturally the control is X"). **Calibrated honesty beats bluffing — and matches their value of "truth" and "humility."**

---

## 15. Behavioral — mapped to webAI's values

Their values: **Truth, Ownership, Tenacity, Humility.** Prepare one crisp **STAR** story (Situation, Task, Action, Result) per value, drawn from TikTok PDPO. Frame, don't fabricate.

- **Truth** — *intellectual honesty / data integrity.* Story idea: a time you surfaced an uncomfortable data-quality or privacy-risk finding even though it slowed a launch, because the right call mattered more than looking good.
- **Ownership** — *end-to-end accountability.* Story idea: you owned a privacy data workflow from ambiguous requirement through production, including the unglamorous monitoring/on-call, not just the build.
- **Tenacity** — *grinding through ambiguity.* Story idea: an integration/compliance problem with no clear spec and a flaky upstream that you drove to resolution through iteration.
- **Humility** — *learning, crediting others, changing your mind.* Story idea: a design you revised after an ML/legal/eng partner pushed back, and the better outcome that resulted.

**Startup-fit signals to weave in** (JD: "comfortable with ambiguity and evolving requirements"): you scope your own problems, ship pragmatically, and don't wait for perfect specs.

**Likely behavioral prompts:**
- "Tell me about a time you built something reusable instead of a one-off." → *platform-primitive instinct.*
- "A time you disagreed with an ML/eng partner." → *collaboration across the inference boundary.*
- "A privacy/compliance call you had to make under pressure." → *your wheelhouse.*
- "Most ambiguous project you've owned." → *startup fit.*
- "Why webAI?" → privacy-first thesis + you've *lived* privacy data infra + you want to build primitives, not pipelines.

## 16. Positioning your TikTok PDPO background

- **Lead with the rare overlap:** "I've built data infrastructure specifically for privacy workflows — consent, PII handling, data-subject requests, lineage. webAI needs exactly that baked into every connector, and most data engineers haven't done it."
- **Translate, don't assume they know TikTok internals:** explain *what* you built in their terms (connectors, contracts, lineage, minimization), not org-specific jargon.
- **Bridge to the gaps proactively:** "My newer surface area is the inference-specific data prep — RAG/embeddings and edge delivery — but the privacy, connector, and resilience foundations transfer directly, and here's how I'd approach the embedding side…"
- **Agentic-AI data infra:** you mentioned building data infra for agentic AI — tie it to webAI's Companion/agentic direction and to RAG/retrieval security and prompt-injection defense.
- **Caution:** don't disclose anything confidential/proprietary about TikTok. Speak in terms of patterns and principles, not internal specifics or data. (This itself demonstrates good privacy judgment.)

## 17. Questions to ask them (pick 4–5)

- "How do you currently draw the line between a reusable connector primitive and a customer-specific integration — and where does that line get blurry in practice?"
- "What does the data contract between the platform team and the ML/applied-AI team look like today? Is it formalized or ad hoc?"
- "For edge deployments like the aerospace/iPad case, how do you handle selective sync and on-device index freshness today — and what's unsolved?"
- "How do compliance requirements (ITAR, residency) currently flow into connector design — is it codified in the platform or handled case by case?"
- "What does the connector framework look like now — green-field, or evolving an existing system? What's the biggest piece of tech debt or open design question?"
- "How do forward-deployed teams give feedback to the platform team, and how fast does that loop close?"
- "What does success in this role look like at 3, 6, and 12 months?"

## 18. 48-hour prep checklist

- [ ] Can explain webAI's thesis, all products (Navigator/Companion/webFrame/Runtime/Network/Infrastructure/CLI), and the founders in 60 seconds.
- [ ] Can define **platform primitive vs. pipeline** and naturally bend every design answer toward reuse/abstraction.
- [ ] Can whiteboard the **connector framework** diagram (§4.7) from memory.
- [ ] Can speak specifically about **Salesforce** (Bulk API 2.0, CDC/Platform Events, Describe API, governor limits) and **S3** (formats, IAM least-privilege, event-driven ingestion).
- [ ] Can explain the **RAG transform path**: clean → chunk → embed → local index, with permission metadata on chunks.
- [ ] Can list **ingestion resilience** patterns crisply (backoff+jitter, token bucket, idempotency, checkpointing, DLQ, circuit breaker, backpressure).
- [ ] Can articulate **edge constraints** (offline-first, selective minimized sync, conflict resolution, Secure Enclave, staleness surfacing).
- [ ] Can deliver the **privacy-by-construction thesis** (§10.8) in one paragraph.
- [ ] Precise on **pseudonymization vs. anonymization vs. tokenization vs. differential privacy**.
- [ ] Can explain **ITAR/residency → edge** as the architectural compliance win for the Airbus case.
- [ ] Can explain **deletion propagation to embeddings/edge caches** and **tamper-evident offline audit logs (hash chains/WORM)**.
- [ ] Can name **AI-specific threats** (data leakage, membership inference, prompt injection/poisoning, supply chain) + mitigations.
- [ ] One **STAR story per value** (Truth, Ownership, Tenacity, Humility) ready.
- [ ] 4–5 **questions for them** chosen.
- [ ] Practiced saying §12 system designs **out loud**, timed.

---

### Final mindset note
You're worried about being drilled on secure AI. Reframe: this team's entire reason for existing is privacy, and you're one of the few candidates who has *built privacy data infrastructure for a living.* Lead with that. Be precise where you're strong, be calibrated and honest where you're stretching (it matches their "truth" and "humility" values), and keep returning to the one idea that defines the role: **reusable, privacy-by-construction, edge-aware data primitives — not one-off pipelines.**

Good luck.
