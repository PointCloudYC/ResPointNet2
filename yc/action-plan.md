# Action Plan: Finalizing the Industrial3D Paper (3-Week Schedule, deadline: 2025-10-30)

This action plan breaks down the completion of Sections 4-7 of the Industrial3D paper into a 3-week schedule, assuming approximately 4 effective work hours per day. It integrates the core ideas from your `princinles.md` (Exploration/Exploitation, Incremental Steps) and `writing-gpt.mdc` (structured writing prompts).

**Core Principle Reminder:** Maintain your "story journal" throughout this process. At the end of each day, spend 5-10 minutes jotting down the key finding, challenge, or insight from the day's work. This will be invaluable for writing a compelling narrative.

---

## Core Actions

- Run fully-supervised baselines (KPConv, RandLA-Net, PTv3, ResPointNet++)
- Run weakly-supervised methods (SQN, CPCM)
- Run self-supervised methods (PointContrast)
- Run foundation models (Point-SAM)
- Run domain-specific challenges (Data Preparation, Robustness to Clutter and Occlusion, Geometrically Ambiguous Components, Performance Under Sparse Supervision)

## Week 1: Benchmark Experiments & Results Analysis (Sec 4)

**Goal:** Complete all experiments, generate result tables, and write the first draft of the analysis for Section 4.

*   **Day 1: Fully-Supervised Experiments (4 hours)**
    *   **Task 1 (Exploration - 3.5h):** Run the experiments for the fully-supervised methods (KPConv, RandLA-Net, PTv3, ResPointNet++). Collate all raw output metrics (OA, mIoU, per-class IoU) into a spreadsheet.
    *   **Task 2 (Documentation - 0.5h):** In your story journal, note any immediate surprising results or difficulties encountered during the experiments.

*   **Day 2: Fully-Supervised Analysis & Table Generation (4 hours)**
    *   **Task 1 (Exploitation - 2h):** Populate `\Cref{tab:benchmark_results_fs}` in `main.tex` with the final, validated numbers from your spreadsheet.
    *   **Task 2 (Writing - 2h):** Write the analysis for Section 4.4.1.
        *   **Principle:** Hypothesis-Driven Experimentation. Frame your paragraph around answering: *Which architectural paradigm (point-based vs. voxel-based) performs best and why? Where do they succeed and fail?*
        *   **Writing-GPT Prompt:** Use `Discuss these results: [PASTE YOUR KEY FINDINGS]` to generate an initial draft of the analysis.

*   **Day 3: Weakly/Self-Supervised Experiments & Analysis (4 hours)**
    *   **Task 1 (Exploration - 2.5h):** Run experiments for SQN, CPCM, and PointContrast at 1%, 5%, and 10% label settings. Collate results.
    *   **Task 2 (Exploitation - 1.5h):** Populate `\Cref{tab:benchmark_results_wsl}` and write the first draft of the analysis for Section 4.4.2, focusing on the practical implications for reducing annotation costs.

*   **Day 4: Foundation Model Experiments & Analysis (4 hours)**
    *   **Task 1 (Exploration - 2.5h):** Run the zero-shot and few-shot experiments for Point-SAM.
    *   **Task 2 (Exploitation - 1.5h):** Populate `\Cref{tab:benchmark_results_fm}` and write the analysis for Section 4.4.3. Frame the narrative around the central question: *How well do large models transfer to specialized domains?*

*   **Day 5: Section 4 Review & Buffer Day (4 hours)**
    *   **Task 1 (Review - 2h):** Read through all of Section 4. Check for consistency, clarity, and flow between the text and the tables.
    *   **Task 2 (Writing-GPT Prompt - 1h):** Use `Improve the clarity and coherence of my writing [PASTE SECTION 4 TEXT]` to refine the language.
    *   **Task 3 (Buffer - 1h):** Use this time to catch up on any tasks that have fallen behind.

---

## Week 2: Domain-Specific Challenges (Sec 5)

**Goal:** Conduct targeted analyses for each challenge in Section 5 and write the corresponding subsections.

*   **Day 6: Data Prep & Occlusion Challenges (4 hours)**
    *   **Task 1 (Exploration - 2h):** Run targeted tests for Section 5.1 (e.g., compare sampling strategies) and 5.2 (e.g., isolate occluded components and calculate their scores).
    *   **Task 2 (Writing - 2h):** Write the full text for subsections 5.1 and 5.2 based on your findings.

*   **Day 7: Geometric Ambiguity & Sparse Supervision Challenges (4 hours)**
    *   **Task 1 (Exploration - 2h):** Conduct experiments for Section 5.3 (e.g., train with/without RGB/Intensity) and 5.4 (e.g., plot performance degradation curves).
    *   **Task 2 (Writing - 2h):** Write the full text for subsections 5.3 and 5.4.

*   **Day 8: Class Imbalance & Generalization Challenges (4 hours)**
    *   **Task 1 (Exploration - 2h):** Run tests for Section 5.5 (e.g., apply a class-reweighting loss) and 5.6 (e.g., test S3DIS-trained model on Industrial3D).
    *   **Task 2 (Writing - 2h):** Write the full text for subsections 5.5 and 5.6.

*   **Day 9: Key Insights & Section 5 Review (4 hours)**
    *   **Task 1 (Writing - 2h):** Write the "Key Insights and Takeaways" subsection (5.7).
        *   **Writing-GPT Prompt:** Use `Identify the key findings and implications of this: [PASTE THE TEXT FROM SEC 5.1-5.6]` to help synthesize the main points.
    *   **Task 2 (Review - 2h):** Read through all of Section 5. Ensure the narrative is strong and the evidence from your experiments clearly supports the claims for each challenge.

*   **Day 10: Figure & Table Generation (4 hours)**
    *   **Task 1 (Visualization - 4h):** Create all figures mentioned in the paper (`overview.png`, `lidar_examples.png`, etc.). This is a critical but time-consuming task. Focus on creating high-quality, publication-ready visuals. Ensure all captions are descriptive.

---

## Week 3: Finalization & Polish (Sec 6-7 & Submission Prep)

**Goal:** Finalize the discussion and conclusion, and perform all necessary checks before considering the draft complete.

*   **Day 11: Discussion & Limitations (Sec 6) (4 hours)**
    *   **Task 1 (Writing - 4h):** Write a detailed draft of Section 6. Expand on the points already outlined.
        *   **Principle:** Be realistic and critical. Acknowledge the boundaries of your work clearly.
        *   **Writing-GPT Prompt:** For future work, use `Can you suggest 3 directions for future research on this topic: [YOUR ABSTRACT]?` to brainstorm ideas. For limitations, use `What are the limitations of using [YOUR METHODOLOGY] in [YOUR DOMAIN]?`

*   **Day 12: Conclusion (Sec 7) & Abstract/Title Review (4 hours)**
    *   **Task 1 (Writing - 2h):** Refine the Conclusion to ensure it powerfully summarizes the paper's contributions, especially the findings from Section 5.
    *   **Task 2 (Review - 2h):** Reread the Abstract and Title. Do they still accurately reflect the full scope of the paper? Use `Suggest 5 titles for the following abstract: [YOUR ABSTRACT]` if you want to explore alternatives.

*   **Day 13: Reference Management (4 hours)**
    *   **Task 1 (Verification - 4h):** Go through the entire paper and check every citation. Ensure all `\cite{misc}` placeholders are replaced with correct BibTeX keys. Verify that every entry in `references.bib` is complete and correctly formatted. This will resolve the linter errors.

*   **Day 14: Full Paper Proofread (4 hours)**
    *   **Task 1 (Grammar & Spelling - 2h):** Do a full read-through focusing only on technical errors.
        *   **Writing-GPT Prompt:** Use `Proofread the following text for spelling and grammatical errors and rewrite it with corrections. [PASTE ENTIRE PAPER TEXT CHUNK BY CHUNK]`.
    *   **Task 2 (Clarity & Flow - 2h):** Do a second read-through focusing on the narrative. Do the sections transition smoothly? Is the argument easy to follow? Read it aloud to catch awkward phrasing.

*   **Day 15: Final Review & GitHub Prep (4 hours)**
    *   **Task 1 (Final Polish - 2h):** Check all formatting, figure placements, and table styles one last time.
    *   **Task 2 (Repo Setup - 2h):** Prepare the paper's GitHub repository as mentioned in your `TODOs.md`. Write a clear `README.md`, clean up any unnecessary files, and ensure the dataset access instructions are clear.
