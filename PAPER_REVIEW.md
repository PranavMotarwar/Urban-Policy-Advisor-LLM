# Review: *Survey on Large Language Models and GPU*

**Reviewer pass:** full read of `ieee_project_report.tex` (2,034 lines), citation-integrity check, image/reference cross-check, and a compile attempt.
**Verdict:** Strong, ambitious survey with a clear thesis and genuinely good sections — but **not submittable as-is**. There are hard build blockers, broken references, and several structural/flow problems that need a revision pass first.

---

## 1. Submission blockers (must fix before arXiv)

These either break the build or produce visibly broken output.

1. **Empty Conclusion.** Line 2029 is `\section{Conclusion}` followed immediately by the bibliography — no text at all.
2. **`IEEEtran.cls` is a 0-byte file.** The paper does not compile (`File 'IEEEtran.cls' not found` → Emergency stop). On arXiv, either delete the local empty file so arXiv's own IEEEtran is used, or include a valid copy.
3. **Three undefined citations** render as `[?]`: `peters2018deep` (ELMo), `howard2018universal` (ULMFiT), `raffel2020exploring` (T5). All three are missing from `references.bib`, yet are discussed prominently in the PLM section and Table II.
4. **Four dangling figure references** in the TPU section produce "Figure **??**": `fig:cloud_tpu_stack`, `fig:llm_vs_accelerators`, `fig:tpu_v5e_cluster`, `fig:tpu_v5e_scaling`. The text says "Figure X shows…" but the corresponding figures are commented out. Either re-enable the figures or remove the sentences that reference them.
5. **Placeholder text left in the body** (will appear in the PDF):
   - Line 79: "...NVLink 3.0, **which means that <....>**."
   - Line 110: red "**(CITATION NEEDED HERE...)**"
   - Lines 121, 128, 132, 434, 436: bare "**(CITATION)**" markers.
6. **Two broken image paths:**
   - Line 100: `LLM_Survey_Paper/Images/software images/llm_gpu_coevolution_timeline.jpg` — wrong path prefix; file not found.
   - Line 1031: `\includegraphics{figure 5}` — no folder, no extension.

---

## 2. Content quality

### Strengths
- **Clear, well-defended thesis** — the co-evolution of LLM architectures and HPC/GPU infrastructure — and it is restated and reinforced consistently across sections.
- **Inference & Serving (Sec. on serving systems) is excellent.** ORCA, PagedAttention/vLLM, Sarathi-Serve, DistServe, FlashAttention/-2, ChunkAttention, and speculative decoding are covered accurately and are current. The prefill/decode framing is correct and well-explained.
- **Datasets & Benchmarks is unusually careful and honest** about disclosed vs. undisclosed training mixtures (GPT-4, Gemini, Claude, Mistral). This intellectual honesty is a real asset and rare in survey writing.
- **GPU-era section is detailed and accurate** — specs, HBM, NVLink, Tensor Cores, MIG, Transformer Engine, FP8/FP4 progression are correct and well-cited to NVIDIA whitepapers.
- **Breadth is impressive:** history → scaling laws → GPUs → TPUs → architecture internals → multimodal/agentic → serving → datasets → evaluation → challenges. 97 unique citations, mostly well-integrated.

### Weaknesses
- **The PLM subsection (1.3) is conspicuously sloppy** versus the rest: "lingusitic," "biredectional," "**BERY**" (BERT), "**Next Sequence Prediction**" (should be Next *Sentence* Prediction), "SQuaD." One paragraph of low-quality prose in an otherwise polished paper.
- **The hardware section reads as a spec catalog, not analysis.** The Pascal→Blackwell subsections are largely descriptive NVIDIA-whitepaper summaries. For a *co-evolution* survey, more of this should explicitly tie each hardware feature back to a concrete LLM bottleneck or training/serving consequence. The "AI/LLM Perspective" paragraphs do this, but they are short relative to the spec dumps.
- **~50 reproduced NVIDIA whitepaper figures** raise a copyright/permissions concern for arXiv. Confirm you have rights or redraw them as original diagrams with citation.
- A few **uncited claims** beyond the explicit `(CITATION)` markers (e.g., the n-gram Markov formulation, the transformer "stacked encoder-decoder" description).

---

## 3. Structure & flow

### Macro-ordering problem
Current order: Intro → Background → **HPC/GPU (~800 lines)** → TPU → **LLM Architectures (tokenization/embeddings/attention)** → Training & Alignment → Multimodal → Serving → Datasets → Evaluation → Challenges → Conclusion.

The reader gets ~800 lines of GPU microarchitecture *before* the section that explains what tokenization, embeddings, and self-attention are. That is pedagogically backwards. **Model fundamentals (the "LLM Architectures" section) should come before or right after Background**, so the hardware discussion can refer to model mechanics the reader already understands. As written, the paper defines attention twice and after the hardware deep-dive.

### Proportionality / balance
The GPU+TPU hardware block is by far the largest part of the paper and visually dominates it (heavy figure density). For a survey framed as *co-evolution*, the hardware half outweighs the model/algorithm half. Consider tightening the per-generation spec prose and converting repeated spec tables into a single consolidated comparison table.

### Fragmented and redundant training/alignment coverage
Alignment and fine-tuning are split across **three** separated places, with overlap:
- **"Training and Alignment Techniques"** section — but this section mostly *re-explains embeddings, positional encodings, and attention* (already covered in "LLM Architectures"), then only briefly mentions instruction tuning/RLHF/DPO.
- **Multimodal section** — repeats the alignment-pipeline point.
- **"Fine-Tuning Paradigms"** — placed *as a subsection of "Performance and Evaluation,"* which is the wrong home for it (fine-tuning is a training technique, not evaluation).

Within the fine-tuning subsection itself there is **internal redundancy**: a prose overview (lines 1939–1950) introduces full fine-tuning, PEFT/LoRA, instruction tuning, RLHF, and DPO — and then the bolded subsections (1953–2014) re-introduce the exact same five topics again.

**Recommendation:** Merge all training/alignment/fine-tuning material into a single coherent "Training, Alignment, and Adaptation" section placed after "LLM Architectures." Remove the duplicated embeddings/attention recap from the current "Training and Alignment" section (keep it only in Architectures). Cut the duplicate fine-tuning overview.

### Transitions
Section-to-section bridging sentences are generally good (the paper repeatedly ties back to the co-evolution thesis). The main flow break is the ordering issue above, not the local transitions.

---

## 4. Mechanical / cosmetic

- **Title** "Survey on Large Language Models and GPU" reads awkwardly ("and GPU" singular). Suggest "…and GPU Architectures" or "…and HPC Infrastructure."
- **Abstract** uses odd jargon ("multimodal Agent2Agent Systems") and is fine on length (~150 words).
- **Duplicate package imports** in the preamble: `graphicx`, `booktabs`, `amssymb`, `tabularx` are each loaded 2–3 times. Harmless but untidy.
- **T5 uncited** in the architecture comparison table (line 1331: "T5, PaLM").
- Inconsistent dashes/spacing in a few headings; not blocking.

---

## 5. Prioritized action list

**P0 — before any submission**
1. Write the Conclusion.
2. Fix `IEEEtran.cls` (delete the empty file or replace with a valid copy).
3. Add the 3 missing bib entries (ELMo, ULMFiT, T5).
4. Resolve the 4 dangling TPU figure refs (re-enable figures or delete the sentences).
5. Remove all `(CITATION)` / `<....>` placeholders; supply real citations.
6. Fix the 2 broken image paths.

**P1 — quality, strongly recommended**
7. Reorder: move "LLM Architectures" fundamentals ahead of the hardware deep-dive.
8. Consolidate training/alignment/fine-tuning into one section; remove the duplicated fine-tuning overview and the embeddings/attention recap.
9. Copy-edit the PLM subsection (typos: BERY→BERT, "Next Sequence"→"Next Sentence," etc.).
10. Confirm figure permissions for the NVIDIA whitepaper images or redraw them.

**P2 — polish**
11. Retitle; de-jargon the abstract.
12. De-duplicate preamble packages; cite T5 in the architecture table.
13. Trim hardware spec prose; consolidate redundant spec tables.

---

*Notes: The paper does not currently compile (empty class file), so this review is based on source reading and static checks rather than a rendered PDF. Once the class file is fixed, a full compile should be run to surface any remaining LaTeX/overfull-box warnings.*
