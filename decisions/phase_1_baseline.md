• # Phase 1 Data Artifact Extension

  The JSON files confirm that the supplied implementation artifacts are a minimal/raw dataset setup, not the complete data pipeline
  described in the PDF.

  ## Exact schemas and record counts

   File                       Records    Top-level schema                                     Annotations
  ━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   train.json                  20,523    image, question, answers, answer_type, answerable    Yes
  ─────────────────────────  ─────────  ───────────────────────────────────────────────────  ───────────────────────────────
   val.json                     4,319    image, question, answers, answer_type, answerable    Yes
  ─────────────────────────  ─────────  ───────────────────────────────────────────────────  ───────────────────────────────
   test.json                    8,000    image, question                                      No
  ─────────────────────────  ─────────  ───────────────────────────────────────────────────  ───────────────────────────────
   vizwiz_train_hindi.json     20,523    Same as train.json, plus question_hi                 Yes, but English answers only

  For annotated files:

  - image: string filename identifier; present and non-empty for every record.
  - question: non-empty English string.
  - answers: exactly 10 objects per record.
  - Each answer object has answer and answer_confidence.
  - answer_type: non-empty string.
  - answerable: integer (0 or 1).

  There is no id, image_path, split, answers_hi, or other absolute/relative image-directory field in any supplied JSON file.

  Image filenames are unique within each file and have zero overlap between train, val, and test.

  ## Data/split findings

  ### Actual supplied splits

  - train.json: 20,523 annotated records.
  - val.json: 4,319 annotated records.
  - test.json: 8,000 unannotated records; it cannot be used for answer-based evaluation because it has no answers or answer types.

  train.json answer-type distribution:

   Answer type      Count
  ━━━━━━━━━━━━━━  ━━━━━━━━
   other           13,733
  ──────────────  ────────
   unanswerable     5,532
  ──────────────  ────────
   yes/no             957
  ──────────────  ────────
   number             301

  val.json answer-type distribution:

   Answer type     Count
  ━━━━━━━━━━━━━━  ━━━━━━━
   other           2,691
  ──────────────  ───────
   unanswerable    1,385
  ──────────────  ───────
   yes/no            195
  ──────────────  ───────
   number             48

  For training records, answerable=0 occurs exactly 5,532 times, matching the unanswerable count; answerable=1 occurs 14,991 times.

  ### Relation to the PDF-described split

  The PDF reports:

  - Train: 20,523
  - Validation: 2,160
  - Test: 2,159

  The artifact evidence shows:

  - The documented training count exactly matches supplied train.json.
  - The PDF’s validation and internal-test counts sum exactly to the supplied raw val.json count: 2,160 + 2,159 = 4,319.
  - However, no generated validation/test partition is present in the artifacts. val.json is not labelled or stored as two splits.
  - The supplied test.json is a separate, unannotated 8,000-record file and cannot be the PDF’s reported internal test set.

  There is also an inconsistency in the PDF’s printed split code. Its int(len(items) * 0.5) logic applied to the supplied val.json class
  counts would produce 2,158 test and 2,161 validation records, not the reported 2,159 test / 2,160 validation records. The exact split
  implementation that produced the PDF counts is therefore not available.

  ## Hindi dataset findings

  vizwiz_train_hindi.json is an exact record-for-record augmentation of train.json.

  Verified across all 20,523 aligned records:

  - Same record order.
  - Same image.
  - Same English question.
  - Same answers, byte-for-byte equivalent after JSON normalization.
  - Same answer_type.
  - Same answerable.
  - No core-record mismatches.
  - Every record has non-empty question_hi.

  Hindi question evidence:

  - 20,506 question_hi values contain Devanagari characters.
  - 4 question_hi values are exactly unchanged from their English question.
  - The remaining records without Devanagari are still non-empty, likely numbers, symbols, names, or non-Devanagari translations.

  Hindi answer evidence:

  - No record has an answers_hi field.
  - The original answers objects are unchanged from train.json.
  - Of 205,230 training-answer strings, zero contain Devanagari characters.

  Therefore, the supplied Hindi dataset contains translated questions only—not translated answers.

  ## Evidence for the reported Model A/Model B experiments

  The JSON artifacts provide no direct evidence that they were used in the PDF-reported experiments.

  They do establish that:

  - The claimed Phase 1 training-set size and class distribution are grounded in the supplied train.json.
  - The raw validation-set size is consistent with the PDF’s claimed later 50/50 internal partition.
  - vizwiz_train_hindi.json could supply Hindi questions for training/inference.

  They do not establish that the reported experiments were run because the necessary artifacts remain absent:

  - no main_dataset_en.json / main_dataset_hi.json;
  - no split-labelled records;
  - no image_path;
  - no answers_hi;
  - no cached CLIP features;
  - no Model A/B code;
  - no checkpoints;
  - no predictions;
  - no metric CSVs or logs.

  In particular, Model B’s PDF implementation expects Hindi answer targets via answers_hi. The supplied vizwiz_train_hindi.json cannot
  support that code as written, because answers_hi does not exist.

  ## Updated audit conclusions

  The previous conclusion is strengthened and refined:

  - The supplied JSON files substantiate only a question-translated Hindi training dataset.
  - They do not substantiate the PDF claim that questions and answers were translated using NLLB plus dictionary substitution.
  - They do not contain the PDF’s path-injected, split-labelled, 24,842-record merged dataset.
  - They do not contain the internal validation/test split used for reported metrics.
  - The unannotated 8,000-record test.json is unsuitable for reproducing the reported ANS evaluations.
  - The supplied question-only Hindi file is incompatible with the PDF’s reported Model A GPT-2 and Model B mT5 Hindi-answer generation code
    unless an unprovided answers_hi dataset existed elsewhere.

  - The one-feature CLIP-cache concern remains unresolved: these JSON files neither confirm nor disprove it.
