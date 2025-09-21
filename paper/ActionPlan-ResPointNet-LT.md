# logs

- 250911, re-evaluate the current paper, add todos and roadmap, 2P
- 250914, review previous notes and re-start the paper, 3P
- 250915, idea(ResPointNet-LT), action plan, paper draft(main-FSL.tex), 5P
- 250915, coding (set up the environment and run the code w. success), 10P
- 250916, identify tail class problem, LTL techniques implementation and training, 2P
- 250917, critically review and update the action plan with Phase 0 validation and stronger narrative structure, 3h
- 250918, conduct experiments 1 and 2 and complete over 95%, 9P

- 250920, fix minor bugs, 2P
- 250921, add PSNet12 new experiment results and fix bugs, 2P

# ResPointNet-LT Action Plan (Version 3)

**Paper Title:** ResPointNet-LT: Decoupling Representation and Classifier Learning to Resolve the Dual Crisis in 3D Long-Tailed Segmentation

**Updated Overview:**
This project provides the first systematic investigation into the "dual crisis" of **statistical bias** and **geometric ambiguity** in long-tailed 3D semantic segmentation. We demonstrate that conventional end-to-end training on imbalanced 3D data leads to a distorted feature space, harming both head and tail classes (head?), which stadard deep learning approaches fail to address. We show that a simple yet powerful two-stage decoupled training framework, **ResPointNet-LT**, resolves this issue by preserving feature integrity, improving performance on critical but rare industrial components (like pumps) while maintaining strong overall segmentation quality. Our work establishes decoupling as a highly effective and principled approach for 3D LTL, validated on both a challenging industrial dataset and the public ScanNet200 benchmark.
---

## Phase 0: Go/No-Go Crucial Experiments (Weeks 1-2)
**Goal:** Rapidly validate the core hypothesis that the "dual crisis" exists and that LTL techniques can address it.

### Week 1: Baseline Validation & "Dual Crisis" Proof
- [x] **Experiment 1: Baseline Performance Analysis (2 days)**
  - **Action:** Train and evaluate baseline ResPointNet++ on the industrial dataset using standard cross-entropy loss.
  - **Critical Analysis:** Document the severe performance gap between head classes (pipes) and tail classes (pumps). Calculate "Tail-mIoU" metric.
  - **Go/No-Go Criteria:** **GO** if tail classes show IoU < 20% while head classes achieve > 60% IoU, confirming a significant performance imbalance problem exists. **NO-GO** if performance is relatively balanced.
  - **Decision (250917):** Experiment complete. The PSNet12 dataset demonstrates a critical long-tail problem (Head-mIoU: 87.6% vs. Tail-mIoU: 21.6%), with complete model failure on the two rarest classes. **Decision is GO.**

- [x] **Experiment 2: Quantifying the "Dual Crisis" (2 days)**
  - **Action:** Prove the "dual crisis" narrative with hard data.
    1.  **Statistical Bias:** Plot the point-level long-tail distribution curve.
    2.  **Geometric Ambiguity:** **Systematically quantify sparsity** by calculating the average number of points per object instance for **all classes**. Plot this against the per-class IoU from Experiment 1 to show a clear correlation between point sparsity and model failure.
  - **Deliverable:** Hard evidence for the narrative, including (1) the long-tail plot and (2) **a scatter plot correlating point count with accuracy**, demonstrating geometric ambiguity is a systemic problem.
  - **Conclusion (250918):** Experiment complete. The "dual crisis" is confirmed. **Statistical bias** is evident from the long-tailed distribution plot. **Geometric ambiguity** is demonstrated; the initial case study on SPH shows up to a 100x disparity, and **this will be expanded to a systematic analysis**.

  **TODO:**
  - [ ] Add the scatter plot correlating point count with accuracy.
  - [ ] Expand the systematic analysis to all classes.

### Week 2: Core LTL Method Validation
- [ ] **Experiment 3: Simple Re-weighting Baseline (2 days)**
  - **Action:** Implement and train ResPointNet++ with Class-Balanced Loss.
  - **Go/No-Go Criteria:** **GO** if tail-class IoU improves by > 5% while head-class IoU degradation remains acceptable (< 10% overall).

- [ ] **Experiment 4: Two-Stage Feasibility Test (3 days)**
  - **Action:** Implement a rapid prototype of ResPointNet-LT with a frozen encoder in Stage 2 and class-balanced sampling.
  - **Go/No-Go Criteria:** **GO** if ResPointNet-LT shows > 1.5x improvement in Tail-mIoU over the re-weighting baseline AND maintains head-class performance.

**Decision Point:** A "No-Go" on Experiments 3 or 4 indicates fundamental issues. The project must be re-evaluated.

---

## Phase 1: EXPLORATION & Foundational Benchmarking (Weeks 3-6)
**Goal:** Implement the full ResPointNet-LT architecture and benchmark it against strong LTL baselines.

### Week 3: Multi-Dataset Framework Implementation
- [ ] **Action:** Implement core LTL components: Class-Balanced Loss and a class-balanced data sampler.
- [ ] **Action:** **Front-load a strong baseline:** Implement **Balancing Logit Variation (BLV)**.
- [ ] **Action: Prepare Public Benchmark:** Set up the data loading and evaluation pipeline for **ScanNet200**.
- [ ] **Deliverable:** A robust experimental framework for both PSNet12 and ScanNet200.

### Week 4-5: Foundational Benchmarking on Both Datasets
- [ ] **Action:** Develop and refine the full two-stage training pipeline for ResPointNet-LT.
- [ ] **Action:** Systematically run baseline experiments **on both PSNet12 and ScanNet200**:
  - ResPointNet++ (vanilla)
  - ResPointNet++ with Class-Balanced Loss
  - ResPointNet++ with BLV
  - ResPointNet++ with simple re-sampling
- [ ] **Deliverable:** Complete ResPointNet-LT implementation and baseline results tables for both datasets.

### Week 6: Cross-Validation & Early Writing
- [ ] **Action:** Conduct initial experiments across multiple industrial scenes to validate generality.
- [ ] **Action (Continuous):** Begin drafting the Introduction, Related Work, and Methods sections.

---

## Phase 2: EXPLOITATION & SOTA Comparison (Weeks 7-10)
**Goal:** Rigorous evaluation of ResPointNet-LT and deep analysis to prove the core novelty.

### Week 7-8: Hypothesis-Driven Experimentation
- [ ] **Experiment 1 - Baseline Trade-offs Analysis (2 days)**
  - **Hypothesis:** "Re-weighting (CB-Loss, BLV) will provide solid gains but at the cost of feature quality(?), while re-sampling will show stronger tail gains but risk overfitting(?)."
  - [ ] Analyze performance trade-offs from Phase 1 experiments.

- [ ] **Experiment 2 - Full ResPointNet-LT Validation (4 days)**
  - **Hypothesis:** "Two-stage decoupling will achieve the best balance, significantly improving Tail-mIoU over all single-stage baselines while maintaining head-class performance."
  - [ ] Run comprehensive evaluation of ResPointNet-LT (n ≥ 3 runs) on both datasets.

- [ ] **NEW Experiment 3 - Feature Space Integrity Analysis (4 days) - CRITICAL FOR NOVELTY**
  - **Hypothesis:** "Single-stage LTL methods (re-weighting, re-sampling) distort the feature space to accommodate tail classes, hurting overall representation quality. Decoupling preserves a more robust and generalizable feature space from Stage 1."
  - **Action:** For each model (vanilla, CB-Loss, BLV, and ResPointNet-LT's Stage 1 encoder):
    1.  Extract point features from the validation set.
    2.  **Analyze Feature Norms:** Compute the mean L2-norm of features for each class. Test if single-stage methods lead to larger norms for tail classes, a sign of overfitting.
    3.  **Analyze Inter-Class Separation:** Compute the cosine similarity between class prototypes (mean class features). Show that the feature space from the decoupled encoder maintains better separation.
  - **Deliverable:** Quantitative results (tables of norms/similarities) and visualizations (t-SNE/UMAP plots) that provide clear evidence for the paper's central claim.

### Week 9-10: SOTA Comparison and Generalization
- [ ] **Action:** Implement and evaluate a SOTA 3D-native LTL method like **Subspace Prototype Guidance (SPG)** as a final benchmark.
- [ ] **Action:** Conduct thorough efficiency analysis of the top methods (ResPointNet-LT, BLV, SPG).
- [ ] **Action:** Finalize all experiments, ensuring results for all key methods are present for both datasets.

---

## Phase 3: PUBLICATION PREPARATION (Weeks 11-16)

### Week 11-12: Rigorous Analysis & Ablation Studies
- [ ] **Action:** Complete comprehensive ablation studies for ResPointNet-LT:
  - Impact of freezing vs. fine-tuning the encoder in Stage 2
  - Effect of different class-balanced sampling strategies
- [ ] **Action:** Conduct a failure case analysis for ResPointNet-LT.

### Week 13-14: Visualization & Scientific Interpretation
- [ ] **Action:** Create key figures for the paper that tell the complete story:
  - "Dual Crisis" evidence: long-tail plot + sparsity-vs-accuracy plot.
  - Main results tables for **both** PSNet12 and ScanNet200.
  - **Core Evidence Figure:** A multi-panel figure from the **Feature Space Integrity Analysis** showing superior feature norms and separation for the decoupled encoder. This will be a cornerstone of the paper.
  - Powerful qualitative results on both datasets showing improvements on tail classes.
- [ ] **Action:** Finalize analysis, connecting all results back to the core "feature space distortion" narrative.

### Week 15-16: Paper Writing & Submission
- [ ] **Action:** Finalize manuscript draft, ensuring a clear and compelling narrative.
- [ ] **Action:** Refine contributions, framing them around the "less is more" insight that principled decoupling is superior to complex end-to-end solutions for 3D LTL.
- [ ] **Action:** Prepare the reproducibility package: code, dataset, and pretrained models, etc.

---

## Timeline Summary & Risk Management

- **Weeks 1-2:** Go/No-Go Validation
- **Weeks 3-6:** Exploration & Foundational Benchmarking
- **Weeks 7-10:** Exploitation & SOTA Comparison
- **Weeks 11-16:** Publication Prep, Analysis & Buffer
- **Total Duration:** 4 months (16 weeks)

### High-Risk Items & Mitigation
1.  **Limited Baseline Improvement** → Fallback: Pivot the paper to be a systematic analysis of *why* certain LTL methods fail, using the feature space analysis to provide valuable insights.
2.  **Novelty Perceived as Incremental** → Mitigation: This is the highest risk. **Mitigate by executing the "Feature Space Integrity Analysis" with rigor.** The paper's success hinges on providing convincing evidence that decoupling is not just effective, but works because it solves a fundamental feature-level problem that other methods do not.
3.  **Dataset-Specific Results** → Mitigation: Already well-mitigated by using both PSNet12 and ScanNet200, demonstrating broad applicability of the findings.