# AGENTS.md

## Project Overview

This repository implements a **Multilingual (Hindi) Visual Question Answering system for visually impaired users**, built on the VizWiz dataset (images captured by blind users, containing real-world noise: blur, poor lighting, off-center framing).

**Why this exists (Phase 2 proposal):** State-of-the-art VQA is largely English-only. Hindi is spoken by 43%+ of the Indian population and is under-served. Existing multilingual assistive VQA systems bolt an external machine-translation module onto a decoupled vision/text pipeline — this adds inference latency that makes them unsuitable for real-time assistance, and their (usually frozen) vision encoders generalize poorly to the visual noise typical of images taken by blind users.

**Problem being solved:** Build a VQA system that is multilingual (Hindi), robust to real-world visual noise, and capable of real-time inference on constrained hardware, without a decoupled translation step.

---

## How to use this file

This file has three tiers of information. Treat them differently:

1. **Explicit Phase 2 requirements** — stated directly in the proposal. Non-negotiable scope; do not silently drop or substitute.
2. **Existing Phase 1 behavior** — what was actually built and measured. Treat as ground truth about the current state of the repo, and as the baseline Phase 2 must be compared against.
3. **Open engineering decisions** — things the source documents do *not* specify (model choice, hyperparameters, latency targets, repo layout, etc.). These are for you (or a human) to decide during implementation. **Do not treat any placeholder, prior conversation, or your own inference as an official requirement just because it sounds plausible.** If you pick a concrete VLM, QLoRA config, or similar, that is an implementation choice you're making — flag it as such, don't present it as something the project specifies.

---

## Explicit Phase 2 Requirements (from the proposal — authoritative, do not weaken)

1. Transition from the Phase 1 **decoupled dual-encoder** architecture to a **unified, end-to-end Vision-Language Model**.
2. Fine-tune using **4-bit quantized QLoRA**.
3. Support **Hindi** VQA (not English-only).
4. Produce an **inference latency comparison** against the Phase 1 decoupled baseline.
5. Produce a **VRAM utilization comparison** against the Phase 1 baseline.
6. Produce an **accuracy comparison** against the Phase 1 baseline.
7. Assess and improve **robustness to noisy, real-world images** (blur, poor lighting, off-center framing) — this is a stated objective, not incidental.
8. Fit fine-tuning within a **24GB VRAM** hardware target.
9. Deliver a **reproducible codebase** and a **fine-tuned VLM weight file**.

Every one of the above must show up somewhere in the eventual implementation and in the comparative analysis report. None of them may be quietly dropped in favor of "just get a model running."

## Explicitly NOT specified — do not hardcode these as requirements

The source documents do **not** name:
- A specific base VLM (no "use model X" anywhere in the proposal).
- QLoRA hyperparameters (rank, alpha, target modules, dropout, quantization library).
- Learning rate, batch size, epochs, schedule, or any other training hyperparameter for Phase 2.
- A numeric latency target (only "reduce vs. Phase 1," no threshold).
- A concrete unified-architecture design (how vision/text/fusion should be structured).
- Repository layout, coding conventions, or a testing framework.

If an agent (human or AI) proposes any of the above during implementation, that's a **design decision being made now**, not a rediscovered requirement. Record such decisions as decisions (e.g. in a design doc or PR description) rather than folding them into this file as if the proposal specified them. This file should only be updated with a concrete choice once a human has actually decided it — see "Design and Solution" milestone below.

---

## Current State — Phase 1 (Completed)

Phase 1 built a **decoupled, dual-encoder baseline** plus the Hindi dataset it will be evaluated against. This is not scaffolding to build on top of — it is the **baseline to be replaced and outperformed**. It exists as Kaggle notebooks with hardcoded Kaggle paths (`/kaggle/working/...`, `/kaggle/input/...`) and manual "does a checkpoint already exist" gating logic. This is not portable infrastructure; expect to reimplement, not import, when building Phase 2.

### Pipeline stages as implemented

1. **Data merging & path verification** — merges VizWiz `train.json`/`val.json` with image dirs and a word-level `eng_hi.json` dictionary; injects `image_path`; stratified-splits `val` into `val`/`test` (ratio 0.5, seed 42) by `answer_type`; validates for empty questions / malformed answers.
2. **Hindi translation pipeline** — `facebook/nllb-200-distilled-600M` (`eng_Latn`→`hin_Deva`) translates questions and answers, followed by a word-level dictionary substitution pass. Produces `question_hi` / `answers_hi`.
3. **Model A — dual-encoder baseline**: `openai/clip-vit-large-patch14` (vision) + `xlm-roberta-base` (text) + custom `CrossAttentionFusion` (text queries attend to image patches) + 4-class answer-type classifier (`yes/no`, `number`, `other`, `unanswerable`) + a later-added GPT-2 generative head.
   - Two-stage fine-tune: heads-only, then constrained full fine-tune with differential LRs. **Vision encoder stays frozen throughout** (code comment: "visual features are language agnostic" — this assumption is exactly what the Phase 2 proposal identifies as a limitation to fix).
   - Result: ~0.68 answer-type accuracy, but VQA-style exact-match answer score only ~0.21–0.23; the GPT-2 generative head's real answer score was measured at 0.0000 during stabilization — generation quality is effectively broken, not just weak.
4. **Feature extraction** — caches CLIP CLS-token features (1024-dim) for reuse.
5. **Model B — bridge model**: `google/mt5-small` with a projected CLIP CLS-feature prepended as an extra input embedding token. Trained with a class-balanced loss (inverse-frequency weights + a balance penalty) because `answer_type` is heavily skewed (`other` 66.9%, `unanswerable` 27.0%, `yes/no` 4.7%, `number` 1.5%).
   - Result: overall VQA answer score 0.3429 on test — better balanced than Model A, still low in absolute terms (`number` ~0.13).

### Known Phase 1 limitations (these are exactly what Phase 2's objectives target — keep the mapping explicit)

| Phase 1 limitation | Phase 2 objective addressing it |
|---|---|
| Decoupled vision + text pipeline → sequential latency | Unified end-to-end VLM |
| Vision encoder frozen, doesn't adapt to VizWiz-style noise | Robustness-to-noise evaluation objective |
| Hindi handled via a separate translation step (dataset creation, not inference-time) | Native multilingual (Hindi) VQA in the unified model |
| GPT-2 generative head effectively non-functional; mT5 bridge only 0.34 answer score | Accuracy comparison is a required deliverable — Phase 2 must actually beat this, not just differ from it |
| No QLoRA / parameter-efficient fine-tuning used | 4-bit QLoRA required in Phase 2 |
| No VRAM/latency instrumentation in Phase 1 code | VRAM + latency comparison are required deliverables |

---

## Deliverables (explicit)

- A fully fine-tuned multilingual VLM weight file for Hindi assistive VQA.
- A comparative analysis report: latency, VRAM, and accuracy, **Phase 1 (decoupled) vs. Phase 2 (unified)**.
- A documented, reproducible codebase for QLoRA fine-tuning under a 24GB VRAM constraint.

## Dataset

- **VizWiz Hin**: base English VizWiz (VizWiz project, UT Austin, public) + Hindi Q/A translations produced by the team in Phase 1.
- https://vizwiz.org/tasks-and-datasets/vqa/

---

## Development Workflow: Design → Implement → Verify → Demonstrate

The course plan (governing this project's evaluation) structures the phase as **design, then implementation, then verification, then demonstration** — not "get a model training ASAP." This has direct implications for how an agent should work:

- There is a **Design and Solution** milestone before implementation is expected to be complete. Architecture and approach decisions (e.g., which unified VLM, how QLoRA is applied) are meant to be deliberately chosen and documented, not defaulted to by an agent mid-task.
- **Progress Review** evaluates code against the design, not just whether it runs — "Design to Code Mapping" is graded explicitly.
- **Mid Review** and **Final Review** include result analysis and viva components that expect the implementer to justify design/implementation choices with technical reasoning, not just present output metrics.
- **Report submission** requires low plagiarism (similarity index < 15%) and low AI-generated-content score (< 20%) per Turnitin, per faculty remarks — applies to written report deliverables.

Practical implication for an agent: when asked to "implement Phase 2," don't jump straight to picking a model and training. Surface the open decisions (model choice, QLoRA config, evaluation harness for latency/VRAM/accuracy) as decisions to be made/confirmed first, consistent with the design-before-implementation expectation.

---

## Technical Context

### Confirmed existing stack (Phase 1)
Python, PyTorch, `transformers`; `openai/clip-vit-large-patch14`, `xlm-roberta-base`, `google/mt5-small`, `facebook/nllb-200-distilled-600M`, `gpt2`; `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`; `torch.cuda.amp`, gradient accumulation/clipping, cosine LR warmup schedules.

### New for Phase 2 (per proposal, method only — not implementation details)
**QLoRA** (4-bit quantized low-rank adaptation) is the required fine-tuning method. The specific library (e.g. `bitsandbytes`/`peft`) is not named in the source documents — verify/decide, don't assume.

---

## Constraints

- Do not drop the Hindi/multilingual requirement.
- Do not drop the noisy-image robustness evaluation.
- Fine-tuning must target a 24GB VRAM budget.
- QLoRA (4-bit) is required for fine-tuning; substituting a different PEFT method is a deviation that must be flagged, not silently made.
- The comparative report must benchmark against **Phase 1 Model A and/or Model B as actually implemented**, not a generic external baseline.
- Written report deliverables: similarity index < 15%, AI-generated-content score < 20% (Turnitin).
- Do not treat this file's "open engineering decisions" list, or any prior conversation about candidate models/hyperparameters, as project requirements. Only the "Explicit Phase 2 Requirements" section above is authoritative scope.

## Testing & Verification

No automated test framework or thresholds are specified. Phase 1 verification approach (reusable for Phase 2):
- `sklearn.metrics.classification_report` + confusion matrices for answer-type classification.
- VizWiz-style ANS score: `min(1, matches/3)` against up to 10 ground-truth answers, case-insensitive exact match.
- Per-class score/accuracy and balance (std/spread) tracking, given the class imbalance in `answer_type`.

Phase 2 additionally requires latency and VRAM measurement — no existing instrumentation for this exists in the Phase 1 code; it must be built.

## Important Notes

- The proposal (Phase 2 technical scope) and the course plan (grading/process) do not conflict; the course plan contributes the design→implement→verify→demo workflow requirement and the plagiarism/AI-content thresholds, nothing else technical.
- Source documents live under `docs/`:
  - `docs/phase_1_implementation.pdf`
  - `docs/phase_2_course_plan.pdf`
  - `docs/phase_2_proposal.pdf`
