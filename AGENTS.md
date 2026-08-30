# AGENTS.md

## Project Overview

This repository implements a **Multilingual (Hindi) Visual Question Answering system for visually impaired users**, built on the VizWiz dataset (images captured by blind users, containing real-world noise: blur, poor lighting, off-center framing).

**Motivation (from Phase 2 proposal):** Most state-of-the-art VQA systems are optimized for English. Hindi is spoken by over 43% of the Indian population but is a low-resource language for VQA. Existing multilingual assistive VQA systems bridge this gap with an external machine-translation module bolted onto a decoupled vision/text pipeline, which introduces inference latency unsuitable for real-time assistance, and typically rely on frozen vision encoders that generalize poorly to noisy real-world images.

**Problem being solved:** Build a VQA system that is multilingual (Hindi), robust to real-world visual noise, and capable of real-time inference on standard/constrained hardware — without a decoupled translation step.

## Current State — Phase 1 (Completed)

Phase 1 built a **decoupled, dual-encoder baseline** and Hindi translation pipeline. This is the baseline Phase 2 must improve on (see Phase 2 Objectives). Do not assume Phase 1 code is production infrastructure — it was run as Kaggle notebooks with hardcoded Kaggle paths (`/kaggle/working/`, `/kaggle/input/...`) and manual checkpoint-existence checks used to skip recomputation. These paths are not portable and will need to be replaced/parameterized for any new environment.

### Phase 1 Pipeline Stages (as implemented)

1. **Data merging & path verification** — Merges VizWiz `train.json`/`val.json` with image directories and an `eng_hi.json` word-level dictionary; injects `image_path`; performs a stratified split (by `answer_type`) of the validation set into `val`/`test` (50/50 split ratio used, seed 42); validates for empty questions and malformed answer objects.
2. **Hindi translation pipeline** — Uses `facebook/nllb-200-distilled-600M` (`eng_Latn` → `hin_Deva`) to translate questions and all answers into Hindi, with a post-translation word-level dictionary substitution pass (`eng_hi.json`). Produces `question_hi` and `answers_hi` fields.
3. **Model A (dual-encoder baseline)** — `openai/clip-vit-large-patch14` (vision) + `xlm-roberta-base` (text) + a custom `CrossAttentionFusion` module (text queries attend to image patch embeddings) + a 4-class classifier head (`yes/no`, `number`, `other`, `unanswerable`).
   - Phase 4.1: trains only fusion + classifier heads (vision/text backbones frozen). AdamW, cosine schedule with warmup, LR 1e-4, batch 16, grad-accum 2, up to 20 epochs, early stopping patience 5.
   - Phase 4.2: adds a GPT-2 (`gpt2`) generative head for free-form answers. Two-stage fine-tuning: (a) stabilization of classifier/generator heads with everything else frozen, (b) constrained full fine-tuning with differential learning rates (text encoder 5e-6, fusion 1e-5, classifier/generator 2e-5) and the **vision encoder kept frozen throughout** ("visual features are language agnostic" — stated rationale in code comments).
   - Result: ~0.68 answer-type classification accuracy; VQA-style answer score (exact match, `min(1, matches/3)`) only ~0.21–0.23. During stabilization, generative head's real answer score was recorded as 0.0000, indicating the GPT-2 generation head performed very poorly.
4. **Feature extraction** — Pre-extracts and caches CLIP CLS-token image features (`clip_features.pt`, dim 1024) for reuse in Model B.
5. **Model B (bridge model)** — `google/mt5-small` with an image-feature projection (`img_proj`) that prepends the projected CLIP CLS feature as an extra input embedding token to the text input. Trained with a **class-balanced loss** (per-class inverse-frequency weighting + a balance penalty term) to counter severe class imbalance in the training data (`other` 66.9%, `unanswerable` 27.0%, `yes/no` 4.7%, `number` 1.5%). Saves two checkpoints: best-accuracy and best-balanced (lowest std of per-class accuracy).
   - Result: Overall VQA answer score 0.3429 on test, more class-balanced than Model A but still low absolute accuracy (`number` class especially poor, ~0.13).

### Known Phase 1 Limitations (Phase 2 should address these)
- Decoupled dual-encoder architecture with separate vision/text backbones → inference latency.
- Frozen vision encoder does not adapt to VizWiz-style visual noise (blur, poor lighting, off-center framing).
- Reliance on Hindi translation as a distinct upstream step (dataset creation), not integrated end-to-end.
- Generative answer quality is weak (near-zero real answer score for GPT-2 head; ~0.34 overall for mT5 bridge model).
- Severe class imbalance in `answer_type` (`number` and `yes/no` are minority classes).

## Phase 2 Objectives

**Project title (per proposal):** *QLoRA-Assisted Optimization of Unified VLMs for Low-Latency Hindi Assistive VQA*

Explicit objectives (from the Phase 2 proposal — authoritative for scope):
1. Transition from the Phase 1 decoupled dual-encoder baseline to a **unified, end-to-end Vision-Language architecture**.
2. Implement a **4-bit quantized QLoRA** fine-tuning strategy for the unified model.
3. Evaluate and reduce **inference latency** versus the Phase 1 decoupled pipeline.
4. Assess the unified model's ability to handle **multilingual (Hindi) queries** and **noisy, unstructured real-world images**.

### Expected Deliverables (explicit)
- A fully fine-tuned multilingual VLM weight file for Hindi assistive VQA.
- A comparative analysis report: inference latency, VRAM utilization, and accuracy, benchmarked **Phase 1 (decoupled) vs. Phase 2 (unified)**.
- Documented methodology and a reproducible codebase for hardware-constrained (**24GB VRAM** target, per proposal) fine-tuning using QLoRA.

### Dataset
- **VizWiz Hin**: base English VizWiz dataset (VizWiz project, UT Austin, public) + Hindi question/answer translations produced by the team in Phase 1 (`main_dataset_hi.json` and related files from the Phase 1 pipeline).
- Link: https://vizwiz.org/tasks-and-datasets/vqa/

## Requirements

### Functional
- Replace the decoupled CLIP+XLM-R / CLIP+mT5 pipelines with a single unified Vision-Language Model handling image + Hindi text input end-to-end.
- Fine-tune the unified VLM using 4-bit QLoRA.
- Produce Hindi-language answers to visual questions from the VizWiz Hin dataset.

### Non-Functional
- Reduce inference latency relative to the Phase 1 decoupled baseline (exact target latency is **not specified** in the source documents).
- Fit within a **24GB VRAM** budget for fine-tuning (per proposal; note Phase 1 experiments were run on a 16GB T4, so infrastructure has changed).
- Maintain/assess robustness to real-world image noise (blur, poor lighting, off-center framing) — this is an evaluation goal, not a specified technique.

### Not specified in source documents (do not assume)
- Which base unified VLM to use (e.g., a specific open-source VLM family) — **not named** in the proposal.
- QLoRA hyperparameters (rank, alpha, target modules, quantization details beyond "4-bit").
- Specific latency targets/thresholds or benchmark methodology beyond "compare to Phase 1."
- Repository/directory structure for Phase 2 code.
- Specific coding conventions, linting rules, or style guides.
- Automated testing requirements or CI setup.
- Build/run commands or environment setup scripts for Phase 2 (Phase 1 was Kaggle-notebook-based; whether Phase 2 continues on Kaggle is unspecified).

## Technical Context

### Confirmed from Phase 1 code (existing stack)
- Python, PyTorch, Hugging Face `transformers`.
- Vision: `openai/clip-vit-large-patch14` (`CLIPVisionModel`, `CLIPImageProcessor`).
- Text encoders used: `xlm-roberta-base`, `google/mt5-small`.
- Translation: `facebook/nllb-200-distilled-600M`.
- Generative head: `gpt2` (`GPT2LMHeadModel`).
- Data/eval tooling: `pandas`, `numpy`, `scikit-learn` (classification_report, confusion matrix), `matplotlib`, `tqdm`.
- Training utilities: `torch.cuda.amp` (mixed precision), gradient accumulation, gradient clipping, cosine LR schedules with warmup.

### Confirmed from Phase 2 proposal (new stack element)
- **QLoRA** (4-bit quantized low-rank adaptation) as the fine-tuning method for Phase 2. The specific QLoRA library/implementation is not specified — commonly `bitsandbytes` + `peft` are used for this technique, but this is **not stated in the source documents**, so verify/decide before assuming.

## Architecture

- **Phase 1 (existing, do not modify unless explicitly reworking the baseline for comparison)**: Two separate models —
  - Model A: CLIP-ViT-L/14 + XLM-RoBERTa-base + custom cross-attention fusion + classifier head + GPT-2 generative head.
  - Model B: CLIP CLS-token features (pre-extracted, cached) + mT5-small with an image-feature-as-prefix-token bridge.
  - These represent the "decoupled dual-encoder baseline" that Phase 2's comparative analysis report must benchmark against.
- **Phase 2 (to be built)**: A single unified end-to-end Vision-Language architecture, fine-tuned with QLoRA. No further architectural detail is specified in the source documents — this is an open design decision for the implementation phase (course plan indicates a "Design and solution" deliverable/review exists for this purpose, see below).

## Development Guidelines

No explicit coding conventions, project structure, or style requirements are specified in any source document. Absent other instruction, use conventional Python project practices, but do not present any specific convention as a project requirement — none was found in the source material.

## Testing & Verification

No automated testing framework or requirements are specified. Verification in Phase 1 was done via:
- Classification reports (`sklearn.metrics.classification_report`) and confusion matrices for answer-type classification.
- A custom VizWiz-style "ANS score" for generated answers: `min(1, matches / 3)` where `matches` counts exact (case-insensitive) matches against up to 10 ground-truth answers per question.
- Per-class score/accuracy breakdowns and "balance" metrics (std/spread across `answer_type` classes) to track the class-imbalance problem.

For Phase 2, verification should include (per the proposal's deliverables): accuracy comparison, inference latency comparison, and VRAM utilization comparison against the Phase 1 baseline. No specific test harness or acceptance thresholds are given.

## Constraints

- **Do not silently drop the multilingual (Hindi) requirement** — it is a stated objective and a stated deliverable.
- **Do not silently drop the noisy-image robustness evaluation** — the proposal explicitly frames frozen/decoupled vision encoders' inability to generalize to noise as a problem to address.
- Target hardware: fine-tuning should fit a **24GB VRAM** budget (proposal's stated deliverable constraint).
- QLoRA (4-bit) is an explicit required method for the fine-tuning strategy — do not substitute a different PEFT method without flagging the deviation.
- The comparative analysis report must benchmark against the **Phase 1 decoupled baseline** specifically (Model A and/or Model B as implemented) — not an arbitrary external baseline.
- Academic/documentation constraint (from both the course plan and the proposal's faculty remarks): submitted written deliverables (e.g., reports) must have **similarity index < 15%** and **AI-generated content score < 20%** as verified by Turnitin. This applies to written report deliverables, not necessarily to code, but source documents do not draw that distinction explicitly.

## Important Notes

- **Course plan (Document 3) is administrative, not a technical requirements source.** It specifies grading rubrics, review dates, and CO/PO mappings for course credit — it contains no functional, architectural, or technical requirements for the software itself. Relevant operationally-useful facts pulled from it: evaluation review windows (Progress review/code: 07–11 Sep 2026; Mid Review: 22–28 Sep 2026; Final Review: 12–16 Oct 2026; Project Report submission: 13 Nov 2026, per the version in this repo — treat these as the milestone calendar, not implementation deadlines with technical implications).
- **The Phase 2 proposal (Document 4) is the authoritative source for technical/functional Phase 2 requirements.** The Phase 1 notebook (Document 2) is the authoritative source for what currently exists. Where the course plan and proposal overlap (e.g., both mention plagiarism/AI-content thresholds), they agree — no conflict was found between the three documents.
- Source documents live under `docs/` in this repository — consult them directly for anything beyond what's summarized here:
  - `docs/phase_1_implementation.pdf` — Phase 1 notebook (data pipeline, Model A, Model B, evaluation).
  - `docs/phase_2_course_plan.pdf` — course/grading administrative plan.
  - `docs/phase_2_proposal.pdf` — Phase 2 technical objectives, dataset, deliverables, literature survey.
