## 14-day SMART action plan

A comprehensive 14-day SMART action plan designed to get your paper submission-ready.

### Guiding Principles for the 2-Week Sprint

*   **SMART:** Every task is **S**pecific, **M**easurable, **A**chievable, **R**elevant, and **T**ime-bound.
*   **Parallel Tracks:** The plan is split into two concurrent tracks: **(E) Experiments & Data** and **(W) Writing & Analysis**. You will work on both every day.
*   **Prioritization:** The focus is on completing "Must-Have" items for a strong initial submission. "Should-Have" items are integrated, and "Nice-to-Haves" are deferred to post-submission.
*   **Daily `\cite{misc}` Goal:** Each day, aim to replace at least 3-5 `\cite{misc}` placeholders with real references. This prevents a massive task on the final day.

### Overview & Guiding Principles
This plan outlines a 14-day sprint to finalize the paper. It's designed for parallel experiment and writing tracks.

-   **Objective:** Produce a complete, polished paper draft in two weeks.
-   **Methodology:** Follow a daily schedule with parallel tracks for experiments and writing.
-   **Prioritization:** Focus on "Must-Have" components first, integrating "Should-Haves" where possible.

---

### Pre-flight Checklist (Day 0)

*   **[x] Environment Setup:** Ensure all experimental environments are stable.
*   **[x] Data Validation:** Double-check the PSNet12 dataset is ready for training.
*   **[x] Version Control:** Create a new git branch (e.g., `submission-sprint`).
*   **[x] Reference Manager:** Set up your `references.bib` file to easily add citations.

---

### Week 1: Core Results, Methodology & First Draft

**Day 1: Finalize Core Experiments & Methodology**
*   **(E) Experiments:**
    *   **Launch:** Start final training runs for **LD-CB Loss** and **GC-CB Loss**.
*   **(W) Writing & Analysis:**
    *   **Task:** Refine **Section 3 (Methodology)**. Add details on local density computation and complexity for both new losses.
    *   **`\cite` Goal:** Replace the 3 `\cite{misc}` in the introduction's first paragraph.

**Day 2: Main Results Table & Analysis**
*   **(E) Experiments:**
    *   **Monitor:** Check LD-CB/GC-CB runs.
    *   **Launch:** Start the final training run for the **Combined Loss**.
*   **(W) Writing & Analysis:**
    *   **Task:** Draft **Section 4.2 (Main Results)**.
    *   **Specifics:** Update **Table 1 (`main_results`)** with final numbers for LD-CB/GC-CB. Write the analysis paragraphs based on these results.
    *   **`\cite` Goal:** Add references for backbone architectures in Section 1.

**Day 3: Per-Class Analysis & Related Work**
*   **(E) Experiments:**
    *   **Collect:** Finalize all metrics for all 5 methods (Baseline, CB, LD-CB, GC-CB, Combined).
*   **(W) Writing & Analysis:**
    *   **Task:** Complete per-class analysis; refine **Section 2 (Related Works)**.
    *   **Specifics:** Update **Table 2 (`per_class_iou`)** with all final results. Write the "Per-Class Analysis" paragraph.
    *   **`\cite` Goal:** Replace `\cite{misc}` in the Related Works section.

**Day 4: Ablation Experiments & Computational Overhead**
*   **(E) Experiments:**
    *   **Launch:** Start ablation studies for **LD-CB radius `r`** and **GC-CB neighborhood size `k`**.
*   **(W) Writing & Analysis:**
    *   **Task:** Add **Computational Overhead Analysis**.
    *   **Specifics:** Create a small table or paragraph in Section 4.3 (Ablations) to present the training time and memory overhead.

**Day 5: Ablation Results & Introduction Polish**
*   **(E) Experiments:**
    *   **Collect:** Log results from `r` and `k` ablations.
*   **(W) Writing & Analysis:**
    *   **Task:** Draft **Section 4.3 (Ablation Studies)** and polish the **Introduction**.
    *   **Specifics:** Update/create tables for ablation results and write the analysis. Strengthen the "dual crisis" narrative in the Introduction.

**Day 6: Visualization Day 1 - Core Figures**
*   **(E) Experiments:**
    *   **Data Prep:** Generate prediction files for the baseline and your best model on representative scenes.
*   **(W) Writing & Analysis:**
    *   **Task:** Create core visualizations.
    *   **Specifics:** Create a plot for the "Dual Crisis" (local density distribution) and replace the placeholder **Figure 4 (`qualitative_results`)** with a real comparison.

**Day 7: Discussion, Conclusion & First Full Draft Review**
*   **(W) Writing & Analysis:**
    *   **Task:** Revise **Section 5 (Discussion)** and **Section 6 (Conclusion)** based on final results.
    *   **Review:** Read the paper start to finish. Check for flow and consistency.

---

### Week 2: Polishing, Visualizations & Submission Prep

**Day 8: Address Review Feedback & Refine Abstract**
*   **(W) Writing & Analysis:**
    *   **Task:** Systematically address issues found during the review. Polish the **Abstract** to lead with the problem, solution, and key quantitative results.

**Day 9: Visualization Day 2 - Graphical Abstract**
*   **(W) Writing & Analysis:**
    *   **Task:** Create the **Graphical Abstract**.
    *   **Specifics:** Update the concept in `main-CB.tex` comments to show your geometry-aware loss instead of the old decoupling idea.

**Day 10: Final `\cite{misc}` Hunt & Reference Check**
*   **(W) Writing & Analysis:**
    *   **Task:** Eliminate all remaining `\cite{misc}` placeholders and format the bibliography.

**Day 11: Write Highlights, Keywords & Front Matter**
*   **(W) Writing & Analysis:**
    *   **Task:** Finalize all front matter (Highlights, Keywords, author details).

**Day 12: Co-author Review**
*   **(W) Writing & Analysis:**
    *   **Task:** Send the complete draft to co-authors for a final 24-hour review.

**Day 13: Final Revisions**
*   **(W) Writing & Analysis:**
    *   **Task:** Incorporate feedback from co-authors.

**Day 14: Final Polish & Submission**
*   **(W) Writing & Analysis:**
    *   **Task:** Perform a final proofread and submit the paper.

This aggressive but structured plan provides a clear path to completing your paper. Good luck!
