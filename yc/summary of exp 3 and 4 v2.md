### Class Definitions

*   **Head Classes**: `Duct` (0), `Pipe` (4), `RectangularBeam` (6)
*   **Common Classes**: `Ibeam` (3), `Tank` (9)
*   **Tail Classes**: `Flange` (2), `Elbow` (1), `Tee` (10), `Valve` (11), `Pump` (5), `Reducer` (7), `Strainer` (8)

### Per-Class IoU Comparison

The table below shows the per-class IoU for the baseline, CB Loss, and Two-Stage training methods. Improvements over the baseline are highlighted in green.

| Class (ID)        | Group  | Baseline IoU | CB Loss IoU                        | 2-Stage IoU                        |
| :---------------- | :----- | :----------- | :--------------------------------- | :--------------------------------- |
| `Duct` (0)        | Head   | 0.907        | **<font color="green">0.914 (+0.8%)</font>** | **<font color="green">0.914 (+0.8%)</font>** |
| `Pipe` (4)        | Head   | 0.763        | 0.757 (-0.8%)                      | **<font color="green">0.775 (+1.6%)</font>** |
| `RectangularBeam` (6)| Head | 0.948        | **<font color="green">0.951 (+0.3%)</font>** | **<font color="green">0.950 (+0.2%)</font>** |
| `Ibeam` (3)       | Common | 0.987        | **<font color="green">0.991 (+0.4%)</font>** | **<font color="green">0.992 (+0.5%)</font>** |
| `Tank` (9)        | Common | 0.990        | 0.990 (0.0%)                       | 0.990 (0.0%)                       |
| `Flange` (2)      | Tail   | 0.452        | **<font color="green">0.507 (+12.2%)</font>**| **<font color="green">0.508 (+12.4%)</font>**|
| `Elbow` (1)       | Tail   | 0.408        | **<font color="green">0.419 (+2.7%)</font>** | **<font color="green">0.465 (+14.0%)</font>**|
| `Tee` (10)        | Tail   | 0.031        | **<font color="green">0.047 (+51.6%)</font>**| **<font color="green">0.090 (+190.3%)</font>**|
| `Valve` (11)      | Tail   | 0.381        | 0.328 (-13.9%)                     | **<font color="green">0.390 (+2.4%)</font>** |
| `Pump` (5)        | Tail   | 0.431        | **<font color="green">0.438 (+1.6%)</font>** | **<font color="green">0.436 (+1.2%)</font>** |
| `Reducer` (7)     | Tail   | 0.000        | **<font color="green">0.149 (N/A)</font>**  | 0.000 (0.0%)                       |
| `Strainer` (8)    | Tail   | 0.000        | 0.000 (0.0%)                       | 0.000 (0.0%)                       |

### Long-Tail Performance Metrics Summary

This table summarizes the key performance metrics using the new class definitions.

| Metric        | Baseline | CB Loss                             | 2-Stage Training                    |
| :------------ | :------- | :---------------------------------- | :---------------------------------- |
| **mIoU**      | 52.48%   | **<font color="green">54.09% (+1.61pp)</font>** | **<font color="green">54.25% (+1.77pp)</font>** |
| **Head mIoU**   | 87.27%   | 87.40% (+0.13pp)                    | **<font color="green">87.97% (+0.70pp)</font>** |
| **Common mIoU** | 98.85%   | **<font color="green">99.05% (+0.20pp)</font>** | **<font color="green">99.10% (+0.25pp)</font>** |
| **Tail mIoU**   | 24.33%   | **<font color="green">26.97% (+10.8%)</font>** | 26.99% (+10.9%)                     |
| **H-IoU**     | 38.06%   | **<font color="green">41.13% (+3.07pp)</font>** | **<font color="green">41.22% (+3.16pp)</font>** |

---

### Analysis

1.  **Tail Classes See Major Gains:** The primary benefit of both CB Loss and Two-Stage training is a significant improvement in the performance of tail classes. The Tail-mIoU increases by nearly **11%** with both methods, which is a substantial gain and confirms the effectiveness of these strategies for long-tail recognition.

2.  **Two-Stage Training Shows Best Overall Balance:** While both methods are effective, the Two-Stage training method provides the most balanced performance lift. It slightly outperforms CB Loss across every category group (Head, Common, and Tail) and achieves the highest H-IoU, indicating a robust improvement that doesn't sacrifice performance on common classes to boost rare ones.

3.  **Specific Class Improvements are Notable:**
    *   The Two-Stage method provides a remarkable **+190%** IoU boost to the `Tee` (10) class and a **+14%** boost to the `Elbow` (1) class.
    *   CB Loss was uniquely able to learn the `Reducer` (7) class, which the other models failed to identify. However, this came at the cost of a significant performance drop on the `Valve` (11) class.

In summary, both methods successfully address the long-tail problem as intended. The Two-Stage training approach appears to be the superior method, delivering slightly better and more consistent performance gains across the board.