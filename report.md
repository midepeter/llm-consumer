# Research Challenge Report

## Research Goal
- Objective: Build a compact, high-performing AI model that can run comfortably on a constrained laptop-class machine for the ADTC Challenge.
- Key questions:
  - How can the model fit within the hardware limits of an Intel Core i5 / Ryzen 5 class machine with 8 GB RAM and integrated graphics?
  - How can the model balance response quality, throughput, memory efficiency, and thermal stability?
  - How do the scoring components affect the design trade-offs?
- Expected outcome: A model and deployment approach that performs well under the ADTC scoring framework while remaining practical for Ubuntu 22.04 on a laptop.

## Challenge Constraints
- Hardware target:
  - CPU: Intel Core i5 10th–12th gen OR AMD Ryzen 5 3000–5000 (x86-64)
  - RAM: 8 GB DDR4
  - Graphics: Integrated only
  - Storage: 256 GB SSD
  - OS: Ubuntu 22.04 LTS
- Constraint focus: The model must be practical to run on a typical laptop without discrete GPU support.

## Scoring Model
- Sacc (50%): Weighted average of model response score from 0 to 100 by the judge.
- Sperf (30%): 100 × (TPSact ÷ TPSmax), with TPS_REFERENCE = 15.0 provisional.
- Seff (20%): 100 × ((7 GB − Peak RAM) ÷ 7 GB), rewarding lower memory usage.
- Pthermal: -10 points if throttled or temperature > 85°C, otherwise 0.

## Current Status
- Date started: 2026-07-28
- Current phase: Dataset generation pipeline — building hybrid CoT distillation dataset
- Summary of progress so far:
  - Selected Qwen2.5-Coder 3B as the baseline model because it runs comfortably and conveniently on laptop-class hardware.
  - Identified the core challenge: improving accuracy and making the model behave more like an expert model without losing its low-resource advantages.
  - Narrowed the improvement strategy to two main options: fine-tuning and distillation.
  - Chosen direction: distill Qwen3-Coder-Next into Qwen2.5-Coder 3B, as this appears to offer the strongest performance jump while preserving the smaller model's efficiency.
  - Built the full `dataset-gen/` pipeline: prompt loading/cleaning, teacher-model CoT generation, two-layer evaluation (rule-based + LLM-as-judge), open-source dataset loading (LeetCode, CodeContests, SWE-Bench, HumanEvalPack, MBPP), and 70/30 hybrid merging.

## Decisions Made
- Chose Qwen2.5-Coder 3B as the baseline because it fits the target hardware constraints better than larger models.
- Decided to focus on distillation rather than fine-tuning as the primary improvement method because it may transfer much of the teacher model's capability into the smaller student model.
- Prioritized maintaining low-resource deployability while aiming to approach the quality of a stronger teacher model.
- Planned to evaluate whether distillation can close the gap to the expert-level behavior without sacrificing speed or memory efficiency.

## Findings So Far
- The main trade-off is between capability and efficiency: larger expert models are stronger, but they are less suitable for the ADTC hardware target.
- Distillation is attractive because it can potentially transfer the teacher model's behavior into a smaller student model more efficiently than training from scratch.
- The baseline model is a good starting point because it is already practical for laptop deployment, which is essential for the challenge.

## Milestones
- [x] Create the research report file
- [x] Capture the challenge objective and constraints
- [x] Document the scoring model
- [x] Choose the baseline model
- [x] Choose the primary improvement strategy
- [x] Prepare the distillation pipeline
- [ ] Benchmark performance and memory usage
- [ ] Evaluate thermal behavior
- [ ] Refine the design for best overall score

## Daily Progress Log
- 2026-07-28
  - Set up the research report structure.
  - Selected Qwen2.5-Coder 3B as the baseline model.
  - Chose distillation from Qwen3-Coder-Next as the main improvement strategy.
  - Documented the ADTC challenge constraints and scoring model.

- 2026-07-30
  - Built the `dataset-gen/` pipeline folder for chain-of-thought knowledge distillation.
  - Created `prompt.py`: flexible prompt loader supporting `.txt`, `.json`, `.jsonl`, `.csv`, and inline Python lists; includes a cleaning stage that strips control characters, collapses whitespace, and deduplicates prompts before they reach the teacher model.
  - Created `generator.py`: passes cleaned prompts through the teacher model (Qwen3 Coder Next via HuggingFace Inference API) to generate CoT rationales in `<think>…</think>` format, then immediately evaluates each record before streaming to `synthetic_data.jsonl`; supports resume-on-restart.
  - Created `evaluator.py`: two-layer evaluation system for synthetic records — Layer 1 is a fast rule-based scorer (checks for think block, answer length, refusal patterns, code block presence on coding tasks) and Layer 2 is an LLM-as-judge call back to the teacher model using a structured rubric (reasoning depth, answer quality, code quality, no-hallucination), combined as a weighted score (40% rule / 60% LLM); open-source records are soft-tagged only and never hard-filtered.
  - Created `open_source.py`: loads and normalises five open-source coding datasets — `mppk/leetcode_problems`, `deepmind/code_contests`, `swe-bench/SWE-bench`, `princeton-nlp/SWE-bench_Lite`, `bigcode/humanevalpack` (all six languages), and `google-research-datasets/mbpp` — into a shared schema; uses a registry pattern so new datasets can be added without touching other files.
  - Created `merger.py`: enforces the 70% open-source / 30% synthetic ratio using stratified sampling by source to preserve dataset diversity; shuffles the final merged output and writes it to `hybrid_dataset.jsonl`; also supports loading from pre-written JSONL files for standalone merge runs.
  - Created `pipeline.py`: top-level orchestrator that runs all five steps in sequence — load/clean prompts → generate synthetic data → load/soft-evaluate open-source datasets → merge → write final hybrid dataset.
  - Updated `requirements.txt` with `huggingface_hub>=0.24.0` and `datasets>=2.19.0`.

## Notes
- Distillation will be the main experiment path going forward.
- Keep track of teacher/student model comparisons, training setup, memory footprint, inference speed, and evaluation results here.
- Add a new dated entry each day so progress history is preserved.
