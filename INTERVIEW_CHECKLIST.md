# Interview Checklist — Per-Version Coverage

> A mental walkthrough you can do in real time. Read once before the interview. During the interview, run the checklist in your head version-by-version as you whiteboard.
>
> **The cardinal rule**: every component you add must be justified by either (a) a concrete failure mode of the previous version, or (b) a scale/safety constraint that's been triggered. If you can't name the justification, drop the component.

---

## Beat 1 — Pre-design (60-90 seconds) — NEVER SKIP

Before drawing anything, do all of these:

- [ ] **Restate the problem in your own words.** "So we're designing X for {who} who need to {what}."
- [ ] **Restate the constraints from the prompt** — volume, latency SLO, cost target. Use the *exact numbers given* (this is where you lost points last time).
- [ ] **Ask the three clarifying questions:**
  1. "Does the answer need to ground in **private or fresh data**?" → drives RAG decision
  2. "Does the system need to **take actions in the world** beyond producing text?" → drives Agent decision
  3. "What's the **latency budget and daily volume**?" → drives model + caching + infra
- [ ] **Ask 1-2 domain-specific clarifiers** if relevant: multi-tenant? compliance regime? languages? offline support?
- [ ] **Commit to a path** out loud: LLM-only / RAG / Agent / RAG+Agent — with one sentence of reasoning.
- [ ] **Frame the version path** out loud: "v1 simplest viable, v2 fixes top failures, v3 production-grade."

---

## v1 Checklist — Simplest Viable

You're allowed to skip the fancy stuff. You're NOT allowed to skip these.

### Component picks (name each one + one-line justification)

- [ ] **LLM choice** — model + tier + reason (e.g., "Sonnet 4.5 for quality on instruction-following; Haiku for high-volume simple paths")
- [ ] **Static-vs-live data split** ← *the gap you missed last round*. Knowledge that changes daily/weekly → RAG (chunk + embed). Knowledge that changes second-by-second (logs, metrics, current state) → **tool calls at query time**, never pre-indexed.
- [ ] **If RAG**:
  - [ ] Chunking strategy + knobs: default **400 tokens, 50 overlap, recursive splitter**
  - [ ] Embedding model: default **text-embedding-3-small** unless domain demands otherwise
  - [ ] Vector DB: default **pgvector** if Postgres in stack and <10M vectors
  - [ ] Retrieval: **top-5 dense** in v1 (mention you'll upgrade in v2)
- [ ] **If Agent**:
  - [ ] **2–3 narrow tools**, mostly read-only in v1
  - [ ] **Strict JSON schemas** on every tool
  - [ ] **Idempotency keys** on any write tool
  - [ ] **Stop conditions** stated: max 8 steps, $0.50 budget cap, 30s timeout
- [ ] **Prompt structure**: system + context + user query — stable content first for caching
- [ ] **Prompt caching** stated explicitly ← *commonly forgotten, huge cost lever*
- [ ] **Streaming** (SSE for chat, WS for voice) ← *especially if user-facing*
- [ ] **Output schema validation** — Pydantic or strict JSON mode
- [ ] **Basic input guardrails** — rate limit per user, length cap on input

### Validation & operations (yes, even at v1)

- [ ] **Eval set: 50–100 hand-labeled examples** ← *don't push this to v3*
- [ ] **Per-call logging** — request_id, user_id, tokens (in/out/cached), latency, cost, errors
- [ ] **Honest list of v1 known limitations** — name 2–3 things you'll fix in v2

### Numbers (compute out loud)

- [ ] **Cost per query / per item**: $X
- [ ] **Daily cost**: $X × volume
- [ ] **p50 latency**: breakdown (retrieve + LLM + post)
- [ ] **TTFT** if streaming

---

## v2 Checklist — Address Top v1 Failure Modes

Before adding anything, **state the v1 failure modes** you're addressing. Then add only what fixes them.

### State the failures (out loud)

- [ ] Failure 1: "v1 had low retrieval precision on exact-term queries (acronyms, IDs)"
- [ ] Failure 2: "v1 hallucinated when retrieval missed"
- [ ] Failure 3: "v1 couldn't take any action / couldn't pull live data"
- [ ] Failure 4: "v1 felt slow because no streaming" (if applicable)

### Retrieval upgrades (RAG path)

- [ ] **Hybrid search** — BM25 + dense, fused with **RRF (k=60)**
- [ ] **Cross-encoder rerank** — top-50 → top-5 (Cohere Rerank or self-hosted bge-reranker)
- [ ] Two-stage retrieval explained: "wide for recall, tight for precision"
- [ ] Optional: **query rewrite** (HyDE, multi-query) if compound queries are common

### Tool upgrades (Agent path)

- [ ] **Grow from 2–3 tools to ~10** with a **tool router / intent classifier** (don't pass 30 tools to every call)
- [ ] **MCP servers** for any system with 3+ integrations across multiple AI surfaces ← *say this for any "many systems" use case*
- [ ] **Write tools** introduced with **per-tool approval thresholds** for high-impact actions
- [ ] **Per-tenant / per-user ACLs** on tools

### Memory

- [ ] **Working memory** — current conversation/session state
- [ ] **Episodic memory** — past interactions per user, retrieved via small per-user vector index
- [ ] **Semantic memory** — extracted facts in KV store
- [ ] Memories **timestamped + per-user isolated** (never shared)

### Quality gates

- [ ] **Faithfulness gate** — LLM-as-judge after generation: "is every claim in the answer supported by retrieved context?"
- [ ] **Citation validator** — reject outputs that cite chunk_ids that don't exist in retrieved set
- [ ] **Confidence-based abstention** — "I don't have authoritative info; routing to human"

### Cost & latency upgrades

- [ ] **Semantic cache** — embed query, cosine-match to past queries (threshold ~0.95). Target ≥20% hit rate.
- [ ] **Model routing** — cheap classifier sends simple queries to small model
- [ ] **Streaming if not already on**

### Validation & operations

- [ ] **Eval set grows to 300–500**, labeled per-axis (accuracy, helpfulness, safety)
- [ ] **RAGAS in CI** — faithfulness, answer relevance, context precision/recall. Fail CI on >5% regression.
- [ ] **Shadow mode for 1 week** before promote — same inputs, new version logs but doesn't reply
- [ ] **Updated numbers** — new $/query, new p50, expected lift on primary metric

---

## v3 Checklist — Scale, Safety, Production-Grade

State the **v2 ceiling** that justifies v3 — usually one of: traffic explosion, compound queries, regulatory weight, multi-tenancy, advanced workflows.

### Orchestration upgrades

- [ ] **Router LLM** to split simple vs complex paths (so 80% of traffic stays on v2-style cheap path)
- [ ] **Planner-Executor** split if workflow is long-horizon and decomposable
- [ ] **Critic / reflection** step for self-correction
- [ ] **Hard stop conditions** restated and tightened: max steps, budget cap, repeat detection, wall-clock timeout

### Multi-tenant & scale

- [ ] **Tenant isolation** — namespace per tenant in vector DB; ACL filter enforced server-side
- [ ] **Per-tenant cost caps**: per-query, per-session, per-tenant-per-day (three levels)
- [ ] **Auto-degrade** when budget exceeded (route to cheaper model, smaller top-k, force cache)
- [ ] **DiskANN** or sharded indexes if any tenant exceeds 50M vectors

### Safety & compliance

- [ ] **PII detection + redaction** on input AND output (Presidio / AWS Comprehend / in-house)
- [ ] **Prompt injection defense** — classifier on input, separator discipline in system prompt
- [ ] **Output safety filter** — toxicity, PII leak, brand-safety
- [ ] **Tool ACL** — destructive tools off by default; require role/workflow membership
- [ ] **Per-tool approval thresholds** with **dry-run mode** for state-changing tools
- [ ] **Compliance regime** named (GDPR / HIPAA / SOC 2 / PCI) with the specific controls (BAA, encryption, audit retention)
- [ ] **Audit log** — every query, every retrieval, every tool call, every guardrail decision. Separate from operational logs. 7+ year retention if regulated.

### Validation & operations

- [ ] **Adversarial test suite** — prompt injection variants, jailbreaks, PII probes, off-topic. Targets: <1% injection success, 0% PII leak.
- [ ] **A/B test in prod** with primary metric + guardrail metrics (cost, latency, refusal rate, CSAT)
- [ ] **Drift detection** — weekly LLM-as-judge sample of 100 prod queries; alert on >5% drop
- [ ] **Feedback loop** — prod failures auto-add to eval set; *carefully-curated* resolved cases feed back into KB
- [ ] **Cost dashboards** per tenant, per route, per model
- [ ] **Latency dashboards** with alerts on p95 > SLO

### Maturity signal

- [ ] **State "Don't go to v3 if {condition}"** — names the reverse case. Massive maturity signal.

---

## Cross-Cutting — Say at Every Version (lightly)

Even at v1, even briefly. Reference these so the interviewer knows they're not afterthoughts:

- [ ] **Validation layer** — schema + faithfulness + safety (depth grows by version)
- [ ] **Eval set** — size and labeling depth grow by version
- [ ] **Monitoring** — per-call logs, dashboards, alerts
- [ ] **Concrete numbers** — running cost + latency budget at each version
- [ ] **Honest failure modes** — what's still broken; what you'd add next

---

## Closing — Last 2 Minutes (offer without being asked)

When you wrap, land on this trio:

- [ ] **Failure modes I'd watch for in prod** — 3 specific ones (e.g., stale data, retrieval blind spots, runaway agent loops, cost spikes from a single bad input)
- [ ] **Day-2 ops** — who owns the prompts, change management, knowledge ownership boundaries
- [ ] **Where I'd be cautious** — what you'd NEVER auto-execute (high-impact write tools, irreversible side effects)
- [ ] **"Don't go to v3 if X"** — the reverse condition

---

## The "Don't Forget" Hit List

Your top gaps from practice. Re-read this 30 seconds before the call.

1. **ASK clarifying questions** — three of them — before drawing anything
2. **READ the prompt's numbers carefully** — restate volume / SLO / cost in your own words before computing
3. **Static knowledge → RAG. Live state (logs, metrics, deploys) → tools at query time.** Never pre-chunk live data.
4. **Prompt caching** at v1 — biggest cost lever you keep skipping
5. **Streaming + TTFT** — even when total latency budget is loose, perceived latency matters
6. **Eval set at v1** — 50 examples is fine. Eval is NOT a v3 thing.
7. **MCP** whenever 3+ external integrations are needed across multiple AI surfaces
8. **Idempotency keys** on every write tool
9. **Stop conditions** every time you say "agent" — max steps + budget cap + timeout
10. **State your "Don't go to v3 if X"** — proves you're not buzzword-stacking

---

## Mental Speed-Run (run this in your head while the interviewer is talking)

Read the prompt → restate constraints → ask 3 clarifiers → commit to path →

**v1**: pick LLM • split static/live data • RAG components OR tools • prompt cache • streaming • schema validate • 50-example eval • log everything • compute numbers • name 2 known limits

**v2**: name 2-3 v1 failures • hybrid + rerank • more tools + MCP • memory • faithfulness gate • semantic cache • model routing • 300-eval + RAGAS + shadow • new numbers

**v3**: name v2 ceiling • router + planner-executor • multi-tenant + cost caps • PII + injection + ACL + audit + compliance • adversarial set + A/B + drift • feedback loops • "don't go to v3 if X"

**Close**: failure modes • day-2 ops • where you'd be cautious

That's the whole interview in 15 minutes.

---

*Companion to SYSTEM_DESIGN_PREP.md and USE_CASE_DESIGN.md.*
