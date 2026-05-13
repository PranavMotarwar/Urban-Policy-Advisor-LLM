# Model Selection Guide for Parsing, RAG, and Inference

This document is a practical chooser for deciding which model to use in which scenario.

It is organized around the three parts of a typical LLM app:

1. Parsing / ingestion
2. Retrieval for RAG
3. Final answer generation / inference

## How to read this

- **Small** means lowest cost / lowest latency and usually the easiest to scale.
- **Medium** means balanced quality, cost, and throughput.
- **Large** means best quality or longest-reasoning use cases, with higher cost and slower latency.
- For **closed models**, parameter counts are usually **not public**.
- For **open-weight models**, parameter counts are public, so those are listed in **B** (billions of parameters).
- **Context window** is the main token number you should care about for long documents and RAG.

## Quick recommendations

### If your main goal is document parsing

- **Small:** `gemini-2.5-flash-lite` for cheap multimodal parsing at high volume.
- **Medium:** `gpt-5.4-mini` if you want strong structured extraction with function calling and a 400k context.
- **Large:** `gpt-5.5`, `claude-sonnet-4.6`, or `gemini-2.5-pro` for hard documents, messy layouts, and reasoning-heavy extraction.
- **Open / local:** `Mistral-Small-3.1-24B-Instruct-2503` if you need local multimodal parsing.

### If your main goal is RAG retrieval

- **Small:** `text-embedding-3-small` or `embed-v4.0` when index cost matters most.
- **Medium:** `text-embedding-3-large` for better retrieval quality on mixed domains.
- **Reranker:** `rerank-v4.0-fast` for speed, `rerank-v4.0-pro` for best quality.

### If your main goal is answer generation

- **Small:** `gpt-5-nano`, `gemini-2.5-flash-lite`, or `Qwen3-4B` for classification, routing, summaries, and light chat.
- **Medium:** `gpt-5.4-mini`, `gemini-2.5-flash`, `Mistral Small 3.1 24B`, or `Qwen3-32B` for most production assistants.
- **Large:** `gpt-5.5`, `gemini-2.5-pro`, `claude-sonnet-4.6`, or `Llama-3.3-70B-Instruct` for highest-quality reasoning and difficult enterprise Q&A.

---

## 1. Parsing / Ingestion Models

These are the models you use to read PDFs, screenshots, scanned pages, tables, forms, or mixed text-image inputs and convert them into clean text or structured JSON.

| Size | Model | Params | Context | Modality | Best for | Notes |
|---|---|---:|---:|---|---|---|
| Small | `gemini-2.5-flash-lite` | Not public | 1,048,576 input | Text, image, video, audio, PDF | Cheap high-volume document parsing | Very strong for multimodal ingestion when cost and latency matter most |
| Medium | `gpt-5.4-mini` | Not public | 400,000 | Text + image input | Structured extraction from PDFs, invoices, forms | Good when you need JSON outputs, tool use, and predictable API behavior |
| Medium | `Mistral-Small-3.1-24B-Instruct-2503` | 24B | 128k | Text + vision | Local parsing with private documents | Strong open model for local multimodal workloads |
| Large | `gpt-5.5` | Not public | 1,050,000 | Text + image input | Hard extraction, nested reasoning, messy docs | Best when document understanding itself needs deep reasoning |
| Large | `gemini-2.5-pro` | Not public | 1,048,576 input | Text, image, audio, video | Large document packs, multimodal enterprise parsing | Very good when documents are long and mixed-format |
| Large | `claude-sonnet-4.6` | Not public | 1M | Text + image input | Long reports, legal/technical docs | Strong fit for long-context synthesis and reasoning over docs |

### Parsing decision rules

- Use **Flash-Lite / Nano-class** models when parsing is mostly layout-to-text or simple field extraction.
- Use **Mini / Flash / 24B-class** models when you need structured extraction with validation logic.
- Use **large frontier models** when parsing also requires judgment, comparison, or multi-document reasoning.
- Use **Mistral Small 3.1 24B** if documents are sensitive and you want a local multimodal option.

### Example scenarios

- **Medical intake forms:** `gpt-5.4-mini` or `gemini-2.5-flash-lite`
- **Scanned insurance PDFs with messy formatting:** `gemini-2.5-pro` or `gpt-5.5`
- **On-prem contract parsing:** `Mistral-Small-3.1-24B-Instruct-2503`

---

## 2. RAG Retrieval Models

RAG is usually not one model. It is normally:

1. an **embedding model** to build vectors
2. optionally a **reranker** to reorder top results
3. a **generation model** to answer using retrieved chunks

### 2A. Embedding Models

| Size | Model | Dimensions / Params | Context | Best for | Notes |
|---|---|---:|---:|---|---|
| Small | `text-embedding-3-small` | 1536 dims by default | Text embedding workload | Cheapest general-purpose text embeddings | Best default if you need a low-cost vector index |
| Medium | `text-embedding-3-large` | 3072 dims by default | Text embedding workload | Higher-quality retrieval | Better for harder semantic retrieval and multilingual search |
| Medium | `embed-v4.0` | 256 / 512 / 1024 / 1536 dims | 128k | Multimodal retrieval including PDFs and images | Strong choice when your corpus includes mixed text-image documents |

### 2B. Rerank Models

| Size | Model | Context | Best for | Notes |
|---|---|---:|---|---|
| Small | `rerank-v4.0-fast` | Not stated in the model list page | High-throughput reranking | Use when latency matters more than absolute ranking quality |
| Medium | `rerank-v4.0-pro` | Not stated in the model list page | Best reranking quality | Best for serious enterprise retrieval |
| Legacy / stable | `rerank-v3.5` | 4096 | JSON and multilingual reranking | Still useful if you want a well-documented fixed context limit |

### Retrieval decision rules

- Use **small embeddings** if you have a very large corpus and want cheap indexing.
- Use **large embeddings** if your users ask subtle semantic questions and precision matters.
- Add a **reranker** once top-k retrieval quality starts to limit answer quality.
- If your source files are PDFs with figures or mixed layouts, use a **multimodal embedding** path like `embed-v4.0`.

### Example RAG stacks

- **Budget RAG:** `text-embedding-3-small` + vector DB + `gpt-5-nano`
- **Balanced enterprise RAG:** `text-embedding-3-large` + `rerank-v4.0-pro` + `gpt-5.4-mini`
- **Multimodal document RAG:** `embed-v4.0` + `rerank-v4.0-pro` + `gemini-2.5-flash`

---

## 3. Generation / Inference Models

These are the models that actually answer the user after parsing and retrieval are done.

## Small Inference Models

| Model | Params | Context | Best scenario | Why use it |
|---|---:|---:|---|---|
| `gpt-5-nano` | Not public | 400,000 | Classification, routing, summarization, low-cost helpers | Very fast and very cheap |
| `gemini-2.5-flash-lite` | Not public | 1,048,576 input | Cheap multimodal assistants and high-throughput chat | Fastest and most budget-friendly in Gemini 2.5 family |
| `Qwen3-4B` | 4.0B | 32,768 native, 131,072 with YaRN | Local assistants, low-cost OSS inference | Good open small model with thinking / non-thinking modes |
| `Llama-3.1-8B-Instruct` | 8B | 128k | General local chat and basic enterprise copilots | Reliable open baseline with long context |

### Use small models when

- the task is mostly extraction, formatting, classification, or short-answer generation
- you need high QPS and tight latency
- you want a first-pass router before calling a larger model
- you want local inference on smaller hardware

## Medium Inference Models

| Model | Params | Context | Best scenario | Why use it |
|---|---:|---:|---|---|
| `gpt-5.4-mini` | Not public | 400,000 | Production assistants, tool use, structured outputs | Best OpenAI middle tier for quality vs cost |
| `gemini-2.5-flash` | Not public | 1,048,576 input | General enterprise assistants with multimodal inputs | Strong price-performance model |
| `Mistral-Small-3.1-24B-Instruct-2503` | 24B | 128k | Local/private assistants, long docs, multimodal chat | Strong open local model; quantized deployment is practical |
| `Qwen3-32B` | 32.8B | 32,768 native, 131,072 with YaRN | Strong OSS reasoning assistant | Better quality than 4B/8B class without jumping to 70B |

### Use medium models when

- you want one main production model for most requests
- your RAG answers need some reasoning, not just retrieval stitching
- cost matters, but small models are no longer accurate enough
- you want a serious local or private deployment without going full multi-GPU 70B

## Large Inference Models

| Model | Params | Context | Best scenario | Why use it |
|---|---:|---:|---|---|
| `gpt-5.5` | Not public | 1,050,000 | Complex enterprise workflows, coding, difficult reasoning | Best when answer quality matters more than cost |
| `gemini-2.5-pro` | Not public | 1,048,576 input | Very long-context multimodal reasoning | Excellent for large corpora and complex synthesis |
| `claude-sonnet-4.6` | Not public | 1M | Long-context document reasoning and agentic workflows | Good speed/intelligence balance in the Claude family |
| `Llama-3.3-70B-Instruct` | 70B | 128k | High-quality open deployment | Best open-weight option here for high-end OSS chat and RAG |

### Use large models when

- users ask hard, ambiguous, multi-step questions
- your retrieved context is large and the answer requires synthesis across many chunks
- the cost of a wrong answer is higher than the cost of extra tokens
- your parsing and RAG pipeline already works, and the remaining bottleneck is reasoning quality

---

## Scenario-based recommendations

## Scenario A: Simple FAQ bot over clean docs

- **Parsing:** basic text extraction or `gemini-2.5-flash-lite`
- **Embedding:** `text-embedding-3-small`
- **Rerank:** skip initially
- **Generation:** `gpt-5-nano` or `Qwen3-4B`

Use this when your docs are clean, the questions are direct, and cost is your top constraint.

## Scenario B: Internal company knowledge assistant

- **Parsing:** `gpt-5.4-mini` or `gemini-2.5-flash`
- **Embedding:** `text-embedding-3-large`
- **Rerank:** `rerank-v4.0-pro`
- **Generation:** `gpt-5.4-mini`, `gemini-2.5-flash`, or `Qwen3-32B`

Use this when you want a balanced production RAG system.

## Scenario C: Medical / legal / compliance assistant

- **Parsing:** `gpt-5.5`, `gemini-2.5-pro`, or `claude-sonnet-4.6`
- **Embedding:** `text-embedding-3-large` or `embed-v4.0`
- **Rerank:** `rerank-v4.0-pro`
- **Generation:** `gpt-5.5`, `claude-sonnet-4.6`, or `Llama-3.3-70B-Instruct` if self-hosted

Use this when answers need careful reasoning over long and domain-specific material.

## Scenario D: Private on-prem RAG

- **Parsing:** `Mistral-Small-3.1-24B-Instruct-2503`
- **Embedding:** local embedding model or hosted embedding only if policy allows
- **Rerank:** local reranker if needed
- **Generation:** `Qwen3-32B`, `Mistral-Small-3.1-24B`, or `Llama-3.3-70B-Instruct`

Use this when data residency or privacy is more important than using the absolute best frontier API.

---

## Simple rule of thumb

- Start with **small** if your workload is mostly extraction, routing, or summarization.
- Move to **medium** when you need one reliable production assistant.
- Move to **large** when your failure mode is no longer retrieval but reasoning quality.
- In RAG, improving **retrieval quality** often gives a bigger win than upgrading the answer model.
- For document-heavy systems, choose by **context window + multimodality** first, not just by benchmark hype.

---

## Practical defaults

If you want a simple starting stack without overthinking it:

### Cheapest workable stack

- Parsing: `gemini-2.5-flash-lite`
- Embeddings: `text-embedding-3-small`
- Generation: `gpt-5-nano`

### Best balanced stack

- Parsing: `gpt-5.4-mini`
- Embeddings: `text-embedding-3-large`
- Rerank: `rerank-v4.0-pro`
- Generation: `gpt-5.4-mini` or `gemini-2.5-flash`

### Best quality stack

- Parsing: `gpt-5.5` or `gemini-2.5-pro`
- Embeddings: `text-embedding-3-large` or `embed-v4.0`
- Rerank: `rerank-v4.0-pro`
- Generation: `gpt-5.5`, `gemini-2.5-pro`, or `claude-sonnet-4.6`

---

## Sources

- OpenAI GPT-5.5: [developers.openai.com/api/docs/models/gpt-5.5](https://developers.openai.com/api/docs/models/gpt-5.5)
- OpenAI GPT-5.4 mini: [developers.openai.com/api/docs/models/gpt-5.4-mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
- OpenAI GPT-5 nano: [developers.openai.com/api/docs/models/gpt-5-nano](https://developers.openai.com/api/docs/models/gpt-5-nano)
- OpenAI embeddings guide: [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)
- Cohere Embed: [docs.cohere.com/docs/cohere-embed](https://docs.cohere.com/docs/cohere-embed)
- Cohere Rerank: [docs.cohere.com/docs/rerank](https://docs.cohere.com/docs/rerank)
- Google Gemini 2.5 Flash: [cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash)
- Google Gemini 2.5 Flash-Lite: [cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-lite](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-lite)
- Google Gemini 2.5 Pro: [cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro)
- Anthropic models overview: [platform.claude.com/docs/en/about-claude/models/overview](https://platform.claude.com/docs/en/about-claude/models/overview)
- Mistral Small 3.1 24B: [huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
- Qwen3-4B: [huggingface.co/Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- Qwen3-32B: [huggingface.co/Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
- Llama 3.1 8B Instruct: [huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Llama 3.3 70B Instruct: [huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)

## Notes

- A few recommendations in this file are **engineering judgments** based on the official specs above, especially the "best for" and "when to use" columns.
- For closed models, parameter counts are not disclosed, so context window, modality support, latency, and cost tier matter more than "B parameters".
