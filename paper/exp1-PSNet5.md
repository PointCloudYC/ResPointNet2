## Point Distribution by Category in PSNet5 Dataset (%)

| ID | Category | Train    | Test     | Total    |
|----------|----------|----------|----------|
| 0 | Ibeam    | 2.23%    | 9.33%    | 5.57%    |
| 1 | Pipe     | 47.43%   | 25.95%   | 49.40%   |
| 2 | Pump     | 3.90%    | 7.58%    | 6.24%    |
| 4 | RBeam    | 23.07%   | 27.94%   | 30.16%   |
| 5 | Tank     | 8.20%    | 4.70%    | 8.63%    |
| / | TOTAL    | 100.00%  | 100.00%  | 100.00%  |

## Baseline Performance Analysis

Based on the training set distribution, we can categorize the classes as follows:
- **Head Classes**: `Pipe` (47.43%) and `RBeam` (23.07%), which are the most frequent.
- **Common Class**: `Tank` (8.20%), representing a medium frequency.
- **Tail Classes**: `Pump` (3.90%) and `Ibeam` (2.23%), which are the least frequent.

Analysis of the baseline model performance at its peak (Epoch 90, mIoU: 84.9%) reveals a significant and nuanced performance gap between these categories. The per-class IoU values are:

| Category | Class  | IoU (%) |
|:---------|:-------|:--------|
| Head     | RBeam  | 98.99   |
| Common   | Tank   | 96.86   |
| Tail     | Ibeam  | 90.54   |
| Head     | Pipe   | 81.41   |
| Tail     | Pump   | 56.71   |

### Performance Gap and Tail-mIoU

The results highlight a severe performance disparity that doesn't strictly follow the head-tail distribution:

- The `Pump` class, a tail class, exhibits significantly poor performance (56.71% IoU), which is a classic symptom of data imbalance.
- The most frequent class, `Pipe`, does not have the highest IoU, suggesting that frequency alone does not guarantee performance.
- Surprisingly, the rarest class, `Ibeam`, achieves a very high IoU (90.54%), indicating that other factors like geometric simplicity may play a larger role in its identification.

To quantify the issue for the most under-represented classes, we calculate the **Tail-mIoU** for the bottom 50% least frequent classes (`Pump` and `Ibeam`):

\[ \text{Tail-mIoU} = \frac{\text{IoU}(\text{Pump}) + \text{IoU}(\text{Ibeam})}{2} = \frac{56.71\% + 90.54\%}{2} = 73.63\% \]

While the overall mIoU is high, the low performance on the `Pump` class confirms a critical performance imbalance problem, satisfying the "Go" criteria for investigating long-tail recognition solutions.

