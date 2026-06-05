# Cheat Sheet — Data Engineering for AI & Agentic Systems

*Keep open on a second screen. Glance, don't read.*

---

## Open the show with this

> "There are two jobs hiding inside the modern data engineer title — **Data for AI** and **AI for data**. Most teams need both, most engineers can credibly do one. The ones who learn both are running platforms in three years. Apple and T. Rowe Price are both hiring 'AI Data Engineers' with that exact framing, using the same vocabulary — *context engineering, AI-ready data, grounding* — so this isn't theory anymore."

That single intro carries the whole conversation.

---

## Data for AI vs AI for data

| Data for AI | AI for data |
|---|---|
| Building the data foundation AI consumes | Using AI to build/govern/run data systems |
| RAG pipelines, vector stores, embeddings | Databricks Genie Code, Snowflake Cortex Code |
| Agent memory, retrieval, semantic layer | dbt Agents, data-quality agents, lineage agents |
| Data contracts, ground-truth, eval sets | NL-to-SQL, schema mapping, auto-docs |
| MCP endpoints on warehouses, catalogs | Agent copilots in Airflow / dbt / IDEs |
| PII redaction, retention, right-to-erasure | Auto-triage of test failures and incidents |

**Market evidence to cite:** Apple AI Data Engineer (Legal Ops, May 2026 posting) — JD literally lists MCP, vector DBs, knowledge graphs, context engineering, embedding generation, semantic layer integration. T. Rowe Price posts the same role with "AI for data / Data for AI" verbatim.

---

## Lines to land (pick 3, deliver them naturally)

- "Models got smarter faster than our data did. DE's job in 2026 is to close that gap."
- "We used to move bytes. Now we engineer context."
- "A dashboard tolerates a fuzzy column name. An agent will hallucinate a join on it."
- "RAG is a data engineering problem dressed up as an ML problem."
- "Agent memory is a feature store with worse governance and higher stakes."
- "Hallucinations are mostly an upstream data problem."
- "Genie Code and Cortex Code replace the boring 30%. The work moves up the stack."
- "Every AI demo dies in production for the same reason — bad data, ambiguous schemas, no lineage."
- "Most enterprise RAG fails because nobody owns the data. ML doesn't know SCDs, DE doesn't know embeddings."

---

## Numbers to drop (May 2026)

| Stat | Use it when… |
|---|---|
| **~90%** | of enterprise data is unstructured (IBM 2026) — for "unstructured is now in scope" |
| **72%** | of enterprises run production RAG (Q1 2026) — for "RAG isn't a prototype anymore" |
| **~40%** | retrieval failure rate of naïve RAG — for "why most RAG disappoints" |
| **20–40%** | precision lift from HyDE / query rewriting — for "fix is upstream of the model" |
| **67%** | Anthropic Contextual Retrieval reduction in top-20 retrieval failures with reranking |
| **2M → 97M** | MCP monthly SDK downloads, Nov '24 → Mar '26 (~4,750%) — for "MCP is real, fast" |
| **500+** | public MCP servers, early 2026 — for "ecosystem maturity" |
| **18% → 60%** | eng teams on AI eval/obs, 2025 → 2028 (Gartner) — for "evals are the next big skill" |
| **72.9% / 17s** | full-context in-process agent memory accuracy / p95 latency (Mem0 benchmarks) |
| **66.9% / 1.4s** | flat vector memory — for "memory architecture choice is a real tradeoff" |

---

## Tool name-drops (examples, not endorsements)

| Category | Names |
|---|---|
| Vector DBs | Pinecone, Qdrant, Weaviate, Milvus, pgvector |
| Knowledge graphs | Neo4j, TigerGraph, ArangoDB, Memgraph, KuzuDB, Microsoft GraphRAG |
| RAG infra | Unstructured.io, Apache Tika, BGE / GTE-Qwen2 embeddings, hybrid + RRF, Cohere/Jina/BGE rerankers |
| Agent memory | Mem0, Letta (ex-MemGPT), Zep, Cognee, LangGraph state |
| Multi-agent | LangGraph, CrewAI, AutoGen, OpenAI Agents SDK |
| Orchestration | Airflow + Astronomer AI, Dagster, Prefect, Databricks Lakeflow / Genie Code, Snowflake Cortex Code |
| Lakehouse | Iceberg (engine-neutral), Delta Lake (Spark-heavy), Hudi, Paimon |
| Streaming | Kafka, Flink, Flink Agents, Confluent Streaming Agents, Materialize, RisingWave |
| Semantic / context | dbt Semantic Layer, Cube, AtScale, Atlan, Gable (contracts) |
| LLM observability | Langfuse (OSS), LangWatch, Arize, Maxim, Helicone, MLflow |
| Eval | RAGAS, TruLens, DeepEval, LangSmith, Confident AI |
| Protocol | MCP (Model Context Protocol) — Anthropic + OpenAI + Google |

---

## 3 spicy takes (deploy one or two)

1. The modern data stack got disrupted by the AI stack faster than it disrupted the warehouse.
2. Agentic pipelines replace the bottom 30% of DE tickets. The job moves up the stack.
3. Most RAG projects fail because nobody owns the data — ML doesn't know SCDs, DE doesn't know embeddings.
4. Vector databases as a separate category mostly disappear in 18 months. Postgres + warehouses + Elasticsearch eat them.

---

## If asked "where do you work?"

> "I'm at a large consumer tech company on a privacy-sensitive team, so I'll keep specifics to my earlier work at [prior company]. That vantage point on governance is actually why this AI conversation matters to me."

---

## Bridges when stuck

- "The way I think about it is…"
- "Let me give a concrete example from my time at [prior company]…"
- "I'd separate two things here — X and Y — because they're often confused."
- "Honest answer: the industry hasn't figured this out yet, but here's where I'd bet…"
- "There's a version that's overhyped and a version that's underhyped — let me unpack both."

---

## Do NOT

- Name your current employer.
- Praise Firebolt (vendor-neutral show).
- Say "AI" when you mean "LLM" or "agent."
- Hedge every sentence.
- Read from notes.
- Run long per answer — 3–5 sentences, then let them follow up.
