### logs

- 250911, re-evaluate the current paper, add todos and roadmap, 2P
- 250914, review previous notes and re-start the paper, 3P
- 250915, idea(ResPointNet-LT), action plan, paper draft(main-FSL.tex), 5P
- 250915, coding (set up the environment and run the code w. success), 10P
- 250916, identify tail class problem, LTL techniques implementation and training? 

### Guiding Principles Checklist
- [x] **Exploration-Exploitation Cycle:** The plan is structured in distinct phases.
- [x] **Executable & Incremental:** Each phase is broken into small, time-boxed steps.
- [x] **Pre-flight Check:** Phase 1 is dedicated to validating the baseline.
- [x] **Hypothesis-Driven:** Phase 3 frames experiments with clear hypotheses.
- [x] **Pivot/Persevere Checkpoints:** Added at the end of key phases.
- [x] **Plan the Narrative:** A "Story Journal" is integrated, and Phase 4 focuses on writing.

---

### Phase 1: Foundation & Baseline Replication (2-3 Days) - [Exploration]

**Goal:** Ensure the existing `ResPointNet++` codebase is functional and replicate the baseline performance reported in the previous paper. This serves as the "Pre-flight Check."

*   **1.1: Environment Setup & Sanity Check (0.5 day)**
    *   [ ] Set up the environment with all necessary dependencies.
    *   [ ] Run the existing training and evaluation scripts on a small subset of the data (e.g., one tile) to ensure everything runs without crashing.
    *   [ ] **Overfit a single batch:** Verify that the model can achieve ~100% training accuracy on a single, fixed batch of data.

*   **1.2: Baseline Performance Replication (1-2 days)**
    *   [ ] Run the full training and evaluation for the baseline `ResPointNet++` on the original long-tailed dataset.
    *   [ ] **Crucial:** Confirm that the resulting mIoU and per-class IoU metrics are consistent with your previous paper's results. This validates your starting point.

*   **1.3: Story Journal - Initial Entry (0.5 day)**
    *   [ ] Create a "story journal" document.
    *   [ ] Document the initial hypothesis: "Standard deep learning models like ResPointNet++ struggle with real-world industrial point clouds due to severe class imbalance, leading to poor performance on infrequent but critical classes (e.g., pumps)."
    *   [ ] Document the baseline results, highlighting the specific IoU failures on tail classes.

---

### Phase 2: LTL Method Implementation (3-5 Days) - [Exploration]

**Goal:** Implement the core LTL techniques to be benchmarked. The focus here is on correct implementation, not exhaustive tuning.

*   **2.1: Class-Balanced Loss (1-2 days)**
    *   [ ] Find a reliable PyTorch implementation of Class-Balanced (CB) Loss for focal loss.
    *   [ ] Integrate it into your training framework as a new loss function option.
    *   [ ] Test it on a small run to ensure the loss values are stable and the model trains.

*   **2.2: Re-sampling Sampler (1 day)**
    *   [ ] Implement a class-balanced data sampler for your `torch.utils.data.Dataset`.
    *   [ ] This sampler should oversample points from tail classes to create more balanced batches.
    *   [ ] Visualize a few batches to confirm that tail classes appear more frequently.

*   **2.3: Two-Stage Training Script (1-2 days)**
    *   [ ] Create a new training script for the `ResPointNet-LT` two-stage approach.
    *   [ ] **Stage 1:** It should load a pre-trained model from your baseline run.
    *   [ ] **Stage 2:** It must have functionality to:
        *   Freeze the weights of the encoder layers.
        *   Connect the class-balanced sampler from step 2.2.
        *   Fine-tune only the decoder layers.

---

### **[Checkpoint: Pivot or Persevere]**

*   **Review:** Have all components been implemented correctly? Do they run without major bugs?
*   **Decision:** Is the proposed two-stage framework technically feasible? If not, pivot back to exploring simpler methods. If yes, proceed to exploitation.

---

### Phase 3: Hypothesis-Driven Experimentation (7-10 Days) - [Exploitation]

**Goal:** Systematically evaluate the implemented LTL methods and build a strong empirical argument for the proposed `ResPointNet-LT` framework.

*   **3.1: Experiment 1 - Re-weighting (2 days)**
    *   **Hypothesis:** "Applying a Class-Balanced Loss directly to `ResPointNet++` will improve tail-class IoU but may slightly degrade head-class performance, leading to a modest overall mIoU gain."
    *   [ ] Train `ResPointNet++` with CB-Loss.
    *   [ ] Evaluate and log results.

*   **3.2: Experiment 2 - Re-sampling (2 days)**
    *   **Hypothesis:** "Using a simple class-balanced sampler from the start of training will provide a stronger boost to tail classes than re-weighting but might lead to overfitting on tail-class features."
    *   [ ] Train `ResPointNet++` with the re-sampling dataloader.
    *   [ ] Evaluate and log results.

*   **3.3: Experiment 3 - The Full ResPointNet-LT (3-4 days)**
    *   **Hypothesis:** "The proposed two-stage decoupled training (ResPointNet-LT) will yield the best balance, significantly improving tail-class IoU while preserving head-class performance, thus achieving the highest overall mIoU."
    *   [ ] Run the full two-stage training:
        *   Use the baseline model from Phase 1 as the starting point for Stage 1.
        *   Run the Stage 2 fine-tuning with the frozen encoder and class-balanced sampler.
    *   [ ] Evaluate and log results.

*   **3.4: Ablation Study (Optional but recommended, 2-3 days)**
    *   **Hypothesis:** "Freezing the encoder in Stage 2 is critical. Fine-tuning the whole network on balanced data will corrupt the learned representations and harm performance."
    *   [ ] Run the two-stage training *without* freezing the encoder.
    *   [ ] Compare results to Experiment 3.3.

---

### **[Checkpoint: Pivot or Persevere]**

*   **Review:** Do the results support the main hypothesis? Are the gains significant?
*   **Decision:** If `ResPointNet-LT` is the clear winner, proceed to writing. If another method performs better, pivot the paper's narrative to focus on that surprising result. If the gains are marginal, return to exploration to identify the bottleneck.

---

### Phase 4: Analysis, Visualization, and Paper Writing (5-7 Days) - [Narrative]

**Goal:** Consolidate results and write a compelling paper.

*   **4.1: Quantitative Analysis (1 day)**
    *   [ ] Compile all results into clear tables: OA, mIoU, per-class IoU.
    *   [ ] Calculate the "Tail-mIoU" metric proposed in the idea document.

*   **4.2: Qualitative Analysis & Visualization (2 days)**
    *   [ ] **This is critical.** Generate high-quality, side-by-side visualizations of the segmentation results.
    *   [ ] Focus on scenes where the baseline fails on a tail class (e.g., misclassifies a pump) and `ResPointNet-LT` succeeds.
    *   [ ] Create visualizations for the introduction and results sections.

*   **4.3: Paper Writing (2-4 days)**
    *   [ ] Draft the Abstract, Introduction, and Method sections using your story journal as a guide.
    *   [ ] Write the Experiments and Results sections, integrating your tables and figures.
    *   [ ] Write the Conclusion, summarizing the findings and contributions.
    *   [ ] Review and refine the entire manuscript.


