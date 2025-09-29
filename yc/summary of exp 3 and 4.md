I've analyzed the performance of the Class-Balanced (CB) Loss and the Two-Stage training methods based on the logs you provided. Here is a comprehensive comparison of their performance, going beyond the overall mIoU to look at per-class results and the specialized long-tail metrics from `yc/metrics.md`.

### Head vs. Tail Class Definition

To analyze the long-tail performance, I've defined the "head" and "tail" classes based on the baseline model's performance (`ResPointNet++` from your stage 1 training). Following the methodology in `yc/metrics.md`, I've designated the 6 classes (50%) with the lowest IoU in the baseline as **tail classes**.

*   **Head Classes (Indices):** 0, 2, 3, 4, 6, 9
*   **Tail Classes (Indices):** 1, 5, 7, 8, 10, 11

### Per-Class IoU Comparison

The table below shows the per-class IoU for the baseline, CB Loss, and Two-Stage training methods. I've highlighted the improvements over the baseline in green.

| Class Index | Group | Baseline IoU | CB Loss IoU | 2-Stage IoU |
| :---------- | :---- | :----------- | :---------- | :---------- |
| 0           | Head  | 0.907        | **<font color="green">0.914 (+0.7%)</font>** | **<font color="green">0.914 (+0.7%)</font>** |
| 1           | Tail  | 0.408        | **<font color="green">0.419 (+2.7%)</font>** | **<font color="green">0.465 (+14.0%)</font>**|
| 2           | Head  | 0.452        | **<font color="green">0.507 (+12.2%)</font>**| **<font color="green">0.508 (+12.4%)</font>**|
| 3           | Head  | 0.987        | **<font color="green">0.991 (+0.4%)</font>** | **<font color="green">0.992 (+0.5%)</font>** |
| 4           | Head  | 0.763        | 0.757 (-0.8%)| **<font color="green">0.775 (+1.6%)</font>** |
| 5           | Tail  | 0.431        | **<font color="green">0.438 (+1.6%)</font>** | **<font color="green">0.436 (+1.2%)</font>** |
| 6           | Head  | 0.948        | **<font color="green">0.951 (+0.3%)</font>** | **<font color="green">0.950 (+0.2%)</font>** |
| 7           | Tail  | 0.000        | **<font color="green">0.149 (N/A)</font>**  | 0.000 (0.0%) |
| 8           | Tail  | 0.000        | 0.000 (0.0%) | 0.000 (0.0%) |
| 9           | Head  | 0.990        | **<font color="green">0.990 (+0.0%)</font>** | 0.990 (0.0%)|
| 10          | Tail  | 0.031        | **<font color="green">0.047 (+51.6%)</font>**| **<font color="green">0.090 (+190.3%)</font>**|
| 11          | Tail  | 0.381        | 0.328 (-13.9%)| **<font color="green">0.390 (+2.4%)</font>** |

### Long-Tail Performance Metrics Summary

This table summarizes the key performance metrics, including the Harmonic Mean IoU (H-IoU) and Tail-mIoU, which are crucial for evaluating long-tail performance.

| Metric     | Baseline | CB Loss                             | 2-Stage Training                    |
| :--------- | :------- | :---------------------------------- | :---------------------------------- |
| **mIoU**   | 52.48%   | **<font color="green">54.09% (+1.61pp)</font>** | **<font color="green">54.25% (+1.77pp)</font>** |
| **Head mIoU**| 84.12%   | **<font color="green">85.17% (+1.05pp)</font>** | **<font color="green">85.48% (+1.36pp)</font>** |
| **Tail mIoU**| 20.85%   | **<font color="green">23.01% (+10.4%)</font>** | **<font color="green">23.01% (+10.4%)</font>** |
| **H-IoU**  | 33.42%   | **<font color="green">36.23% (+2.81pp)</font>** | **<font color="green">36.28% (+2.86pp)</font>** |

---

### Analysis

1.  **Both Methods Improve Performance:** Both CB Loss and Two-Stage training successfully improve upon the baseline across all major metrics. The overall mIoU, head-class performance, and tail-class performance are all boosted.

2.  **Tail-mIoU Improvement is Significant:** Both methods achieve a **10.4%** relative improvement in Tail-mIoU. This is a strong signal that the long-tail problem is being addressed. According to your `ActionPlan-ResPointNet-LT.md`, this surpasses the "> 5%" Go criteria for the re-weighting experiment.

3.  **H-IoU Confirms Balanced Improvement:** The H-IoU metric increases from 33.42% to ~36.25% for both methods. This is important as it confirms that the improvement in tail classes did not come at a significant cost to head-class performance. In fact, head-class performance also improved, indicating a well-balanced model.

4.  **Two-Stage Training is Marginally Better Overall:** While both methods yield a very similar Tail-mIoU, the Two-Stage method shows slightly better performance on the overall mIoU, Head mIoU, and H-IoU. The key difference seems to come from stronger performance on tail classes 1 and 10, though it did fail to learn class 7 where CB Loss made some progress.

In conclusion, both strategies are effective at improving long-tail performance. The two-stage training approach appears to have a slight edge in overall balanced performance.