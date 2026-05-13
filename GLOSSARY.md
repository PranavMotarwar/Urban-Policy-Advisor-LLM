# Quick Lookup Glossary

This glossary covers the 25 major technical topics from [1763530668638.pdf](/Users/pranavmotarwar/Downloads/System%20Design%20AI/1763530668638.pdf). Each term is summarized in 4 short points for fast revision.

## Alignment Tax in RLHF

- **Definition:** The drop in a model's general capability after reinforcement learning makes it better at instruction-following but narrower overall.
- **Where used:** In RLHF and post-training pipelines where a policy model is optimized against a reward model.
- **Why it matters:** Without a KL penalty to a reference model, the policy can over-optimize the reward and become less useful on broad tasks.
- **Example:** In a medical chatbot, RLHF may improve safety wording, but without enough regularization it can become weaker at open-ended diagnosis explanations.

## Asynchronous Execution in GPU Benchmarking

- **Definition:** GPU kernels launch asynchronously, so the CPU can keep running before the GPU work actually finishes.
- **Where used:** In CUDA, PyTorch, and custom GPU kernel benchmarking scripts.
- **Why it matters:** If you do not synchronize before stopping the timer, your benchmark measures launch overhead instead of real execution time.
- **Example:** If you benchmark a custom attention kernel for a chatbot and skip `torch.cuda.synchronize()`, you may wrongly claim a huge speedup.

## Backpropagation Computational Asymmetry

- **Definition:** The backward pass costs more than the forward pass because it computes gradients for both weights and activations.
- **Where used:** In training cost estimation for neural networks and large language models.
- **Why it matters:** FLOPs are often underestimated; training is closer to `6 * params * tokens` than a simple forward-only estimate.
- **Example:** When budgeting a legal-domain model training run, using only forward FLOPs can make the GPU budget look 3x cheaper than reality.

## Benchmark Contamination

- **Definition:** Benchmark contamination happens when training data overlaps with evaluation data, even through paraphrases or semantic duplicates.
- **Where used:** In model evaluation, leaderboard reporting, and dataset audits.
- **Why it matters:** Inflated scores can hide weak real-world generalization and make a model look better than it actually is.
- **Example:** If a finance assistant trains on scraped exam solutions similar to MMLU questions, its benchmark score may rise without real reasoning improvement.

## Byte Pair Encoding (BPE)

- **Definition:** BPE is a tokenizer that repeatedly merges common symbol pairs into larger subword units.
- **Where used:** In Transformer tokenization pipelines for training and inference.
- **Why it matters:** It compresses sequences, which reduces attention cost and makes training much more compute-efficient than byte-level tokenization.
- **Example:** In a medical chatbot, BPE can learn terms like `cardiomyopathy` as fewer pieces, reducing token count and inference cost.

## Compute-Optimal Scaling Laws

- **Definition:** Scaling laws describe how loss changes with model size, data size, and compute budget.
- **Where used:** In planning pretraining runs and deciding model-to-data ratios.
- **Why it matters:** They help teams avoid wasting compute by choosing a model that is too large for the data or too small for the budget.
- **Example:** If you have 32 H100s for two weeks, scaling laws help decide whether to train a 7B model on more tokens or a 13B model on fewer.

## Data Curation for Fine-Tuning

- **Definition:** Fine-tuning curation means selecting high-intent, high-quality examples instead of dumping raw user traffic into training.
- **Where used:** In instruction tuning, post-training, and domain adaptation datasets.
- **Why it matters:** Raw traffic is dominated by low-value prompts, so unfiltered tuning can make the model worse on important use cases.
- **Example:** For a customer-support bot, you would fine-tune on resolved support conversations, not on thousands of vague one-line user messages.

## Data Curation Pipelines for Frontier Models

- **Definition:** A frontier data pipeline parses, filters, ranks, and deduplicates web-scale raw data into usable training corpora.
- **Where used:** In pretraining systems that ingest Common Crawl, PDFs, HTML, and other noisy large-scale sources.
- **Why it matters:** Model quality depends heavily on data quality, so "just train on the internet" usually produces noisy and contaminated datasets.
- **Example:** A research lab building a general chatbot may extract text from HTML and PDFs, filter spam, remove toxicity, and deduplicate near-identical pages.

## FlashAttention

- **Definition:** FlashAttention is an IO-aware attention algorithm that keeps attention computation in fast on-chip memory using tiling and fusion.
- **Where used:** In long-context Transformer training and inference kernels.
- **Why it matters:** The real bottleneck in attention is often memory movement, not just FLOPs, so reducing HBM traffic gives large speedups.
- **Example:** In a document QA chatbot with very long reports, FlashAttention helps process long prompts faster without materializing huge attention matrices.

## FLOPs Fallacy

- **Definition:** The FLOPs fallacy is the mistaken belief that equal FLOPs means equal inference speed.
- **Where used:** In comparisons like MHA vs GQA or other inference architecture choices.
- **Why it matters:** Decoding is often memory-bound, so KV-cache size and bandwidth matter more than raw arithmetic counts.
- **Example:** Two customer-service models may have similar FLOPs, but the one with smaller KV-cache traffic will answer faster in production.

## GRPO Length Normalization Trap

- **Definition:** A flawed RL objective can reward longer responses, causing the model to generate huge chains of thought without real gains.
- **Where used:** In GRPO-based reasoning model training, especially math and coding RL setups.
- **Why it matters:** This is a form of reward hacking that explodes inference cost and hurts production efficiency.
- **Example:** A tutoring chatbot may start producing 10,000-token solutions for simple algebra questions because the reward setup accidentally favors longer outputs.

## Grouped Query Attention (GQA)

- **Definition:** GQA shares key and value heads across groups of query heads instead of giving every query head its own KV pair.
- **Where used:** In modern LLM inference architectures to reduce KV-cache size.
- **Why it matters:** It cuts memory bandwidth pressure during decoding, often improving speed without changing parameter count much.
- **Example:** In a high-traffic chatbot API, switching from MHA to GQA can improve tokens-per-second because less KV data is read each step.

## KV Cache

- **Definition:** The KV cache stores past key and value tensors so the model does not recompute them for every generated token.
- **Where used:** In autoregressive LLM inference and serving systems.
- **Why it matters:** It speeds generation, but it also creates a memory bottleneck that can limit throughput and concurrency.
- **Example:** In a legal assistant answering long contract questions, the KV cache allows fast next-token generation after the prompt has been processed.

## Leaderboard Illusion

- **Definition:** Leaderboards can reward overfitting to evaluation quirks rather than genuine user value or broad intelligence.
- **Where used:** In arena rankings, public benchmarks, and model release comparisons.
- **Why it matters:** Optimizing for leaderboard score alone can misdirect post-training and produce worse outcomes for real customers.
- **Example:** A team may tune a model to gain 50 ELO on Chatbot Arena while actual enterprise support users see no improvement.

## Mantissa Trap in Mixed Precision

- **Definition:** Bfloat16 keeps range but loses precision, so tiny updates can vanish if everything is stored in low precision.
- **Where used:** In mixed-precision training with formats like `bf16` and `fp32`.
- **Why it matters:** Master weights and optimizer states must stay in FP32 or training can silently stall or diverge.
- **Example:** In pretraining a domain chatbot, storing optimizer state only in `bf16` can round away small gradient updates and freeze learning.

## Mixture of Experts (MoE) Router Collapse

- **Definition:** Router collapse happens when an MoE router sends too many tokens to a small subset of experts.
- **Where used:** In sparse MoE training systems with learned routing.
- **Why it matters:** Most experts stop learning, the model wastes capacity, and loss can flatline despite a huge parameter count.
- **Example:** In a multilingual assistant, the router may overuse a few English-heavy experts while many other experts receive almost no tokens.

## MuP (Maximal Update Parametrization)

- **Definition:** MuP is a scaling-aware parametrization that keeps update magnitudes stable as model width grows.
- **Where used:** In transferring hyperparameters from small proxy models to much larger models.
- **Why it matters:** It reduces the need for expensive retuning when scaling from millions to billions of parameters.
- **Example:** If a 1B proxy model trains well, MuP helps reuse similar learning-rate choices when scaling to a 70B production model.

## PagedAttention

- **Definition:** PagedAttention manages KV-cache memory in fixed-size blocks instead of large contiguous chunks.
- **Where used:** In high-throughput LLM serving engines such as vLLM-style systems.
- **Why it matters:** It reduces fragmentation and wasted VRAM, allowing more concurrent requests on the same GPU.
- **Example:** In a multi-user chatbot server, PagedAttention lets many requests with different prompt lengths share GPU memory more efficiently.

## Prefill and Decode Phases

- **Definition:** LLM inference has a compute-heavy prefill stage for the prompt and a memory-heavy decode stage for token generation.
- **Where used:** In chatbot serving, batch inference, and latency/throughput tuning.
- **Why it matters:** Different workloads stress the GPU differently, so good serving design depends on which phase dominates.
- **Example:** A long document summary job spends a lot in prefill, while a live chatbot feels slower mainly when decode latency is high.

## Pre-Layer Normalization (Pre-Norm)

- **Definition:** Pre-norm applies layer normalization before the attention or MLP sub-layer instead of after the residual addition.
- **Where used:** In deep Transformer architectures for stable large-scale training.
- **Why it matters:** It preserves cleaner gradient flow through the residual path and is much more stable than post-norm in deep models.
- **Example:** For a 100B legal-language model, pre-norm helps prevent exploding gradients during long training runs.

## RoPE (Rotary Position Embeddings)

- **Definition:** RoPE encodes relative position by rotating query and key vectors inside each attention layer.
- **Where used:** In Transformer attention blocks for long-context positional encoding.
- **Why it matters:** It must be applied to Q and K after projection, or the model loses the intended positional geometry and training suffers.
- **Example:** In a chatbot reading long medical reports, RoPE helps the model preserve token-distance relationships across the context window.

## Speculative Decoding

- **Definition:** Speculative decoding uses a small draft model to propose tokens and a large model to verify them in parallel.
- **Where used:** In lossless LLM inference acceleration when teams want faster decoding without changing output quality.
- **Why it matters:** It exploits fast verification vs slow generation, often giving 2-3x speedups while preserving the target model distribution.
- **Example:** In a hospital support chatbot, a small draft model can guess likely next tokens while the larger medical model verifies them for faster replies.

## SwiGLU

- **Definition:** SwiGLU is a gated activation that combines a linear path with a Swish-based gating branch.
- **Where used:** In Transformer feed-forward blocks as an alternative to ReLU-based activations.
- **Why it matters:** It consistently improves model quality in practice, even if the full theoretical reason is still debated.
- **Example:** When building a coding assistant, using SwiGLU in the feed-forward layers can improve perplexity compared with standard ReLU blocks.

## Thinking Mode Fusion

- **Definition:** Thinking mode fusion trains one model to switch between short direct answers and longer reasoning modes using control tags.
- **Where used:** In reasoning models that need both cheap simple answers and deep chain-of-thought on hard tasks.
- **Why it matters:** It avoids hosting multiple models and cuts inference cost on easy queries without losing advanced reasoning ability.
- **Example:** A math chatbot can answer `2+2` with a short `no_think` mode but use a `think` mode for a hard proof-style question.

## Throughput-Latency Paradox

- **Definition:** Increasing batch size can improve throughput at first, but eventually it adds latency without meaningful throughput gains.
- **Where used:** In LLM serving systems where operators try to reduce cost by batching more requests together.
- **Why it matters:** The best batch size is usually the peak of the throughput curve, not the maximum that fits in memory.
- **Example:** A customer-support chatbot may serve more requests per GPU at batch size 8, but going to batch size 32 may make each user wait longer.

## Tokenizer Fertility

- **Definition:** Fertility is the average number of tokens a tokenizer uses per word or unit of text.
- **Where used:** In tokenizer evaluation, especially for domain-specific and multilingual corpora.
- **Why it matters:** High fertility bloats sequence length, which makes attention much more expensive and shrinks effective context capacity.
- **Example:** In a biomedical chatbot, a bad tokenizer may split `glioblastoma` into many pieces, raising sequence length and compute cost.

