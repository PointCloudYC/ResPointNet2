## Baseline Performance

The following table shows the per-class IoU and overall mIoU for baseline models on the PSNet12 test set.

| Network       | mIoU   | Duct  | Elbow | Flange | Ibeam | Pipe  | Pump  | RBeam | Reducer | Strainer | Tank  | Tee   | Valve |
| :------------ | :----- | :---- | :---- | :----- | :---- | :---- | :---- | :---- | :------ | :------- | :---- | :---- | :---- |
| ResPointNet++ | 53.31% | 91.36 | 46.05 | 50.92  | 99.41 | 76.23 | 35.95 | 95.19 | **0.00**| **0.00** | 97.06 | 12.03 | 35.56 |
| ResPointNet++ (2025-09-26) | 52.48% | 90.73 | 40.77 | 45.23  | 98.66 | 76.33 | 43.08 | 94.79 | **0.00**| **0.00** | 98.99 | 3.10  | 38.06 |
| ResPointNet++ (2025-09-20) | 49.52% | 88.21 | 34.72 | 43.75 | 96.47 | 73.77 | 47.29 | 93.15 | **0.00**| **0.00** | 98.44 | **0.00** | 18.39 |
| PseudoGrid    | 53.65% | 91.35 | 45.44 | 50.95  | 98.35 | 75.96 | 42.60 | 95.04 | **0.00**| **0.00** | 97.72 | 7.72  | 38.69 |
| PosPool       | 53.18% | 90.25 | 32.93 | 42.88  | 98.93 | 76.35 | 31.27 | 94.18 | 34.48   | **0.00** | 95.59 | 9.17  | 32.14 |

new experiments, 2025-09-26(early stopping with patience 50 and save the best model), details are in the log `log/psnet12/respointnet2_dp_fi_df_fc1_max_20250924205054`, **the best mIou is 52.479% @epoch 165, took ~28 hours with 215 epochs on 2 RTX 3090 GPUs**

```
[09/25 18:24:31] psnet12 INFO: E165 V1 * mIoU 52.479%
[09/25 18:24:31] psnet12 INFO: E165 V1  * msIoU [0.90727645 0.40771802 0.452336   0.98661473 0.76332283 0.43079931
 0.9479247  0.         0.         0.98986282 0.03102795 0.38061437]
[09/25 18:24:31] psnet12 INFO: New best mIoU: 52.479%
[09/25 18:24:31] psnet12 INFO: ==> Saving...
[09/25 18:24:33] psnet12 INFO: Saved best model in log/psnet12/respointnet2_dp_fi_df_fc1_max_20250924205054/best.pth
```

new experiments, 2025-09-20(early stopping with patience 20 and save the best model), details are in the log `log/psnet12/respointnet2_dp_fi_df_fc1_max_20250919163927`, the best mIou is 49.516% @epoch 125

```
[09/20 17:15:23] psnet12 INFO: E125 V1 * mIoU 49.516%
[09/20 17:15:23] psnet12 INFO: E125 V1  * msIoU [0.88213506 0.34722973 0.4375355  0.96466579 0.7377253  0.47287575
 0.93147885 0.         0.         0.98444496 0.         0.18386864]
[09/20 17:15:23] psnet12 INFO: New best mIoU: 49.516%
[09/20 17:15:23] psnet12 INFO: ==> Saving...
[09/20 17:15:24] psnet12 INFO: Saved best model in log/psnet12/respointnet2_dp_fi_df_fc1_max_20250919163927/psnet12/respointnet2_dp_fi_df_fc1_max_20250920162850/best.pth
```

## Performance Analysis on PSNet12

The PSNet12 dataset exhibits a significant long-tail distribution. We can group the classes into three categories based on their frequency in the training set:

- **Head Classes**: `RectangularBeam` (32.44%), `Pipe` (24.08%), and `Duct` (21.0%). These three classes constitute over 77% of the dataset.
- **Common Classes**: `Tank` (7.82%) and `Ibeam` (5.64%).
- **Tail Classes**: The remaining seven classes, including `Flange`, `Valve`, `Elbow`, `Tee`, `Pump`, `Reducer`, and `Strainer`, each accounting for less than 3% of the total points.

### Performance Gap and Tail-mIoU

Using the results from the baseline `ResPointNet++` model, we can observe a severe performance disparity between the head and tail classes.

| Category | Class (ID) | IoU (%) | Point Ratio |
|:---|:---|:---|:---|
| Head | RectangularBeam (6) | 95.19 | 32.44% |
| Head | Duct (0) | 91.36 | 21.00% |
| Head | Pipe (4) | 76.23 | 24.08% |
| Tail | Elbow (1) | 46.05 | 1.63% |
| Tail | Pump (5) | 35.95 | 0.99% |
| Tail | Valve (11) | 35.56 | 1.78% |
| Tail | Tee (10) | 12.03 | 1.29% |
| Tail | Reducer (7) | 0.00 | 0.18% |
| Tail | Strainer (8) | 0.00 | 0.15% |

The head classes achieve an average IoU of **87.59%**, while the performance on tail classes is drastically lower. To quantify this, we calculate the **Tail-mIoU** for the bottom 50% of classes (6 classes: `Strainer`, `Reducer`, `Pump`, `Tee`, `Elbow`, `Valve`):

\[ \text{Tail-mIoU} = \frac{0.00 + 0.00 + 35.95 + 12.03 + 46.05 + 35.56}{6} = 21.60\% \]

The stark contrast between the Head-mIoU (87.59%) and Tail-mIoU (21.60%) confirms a critical performance imbalance, highlighting the model's struggle with infrequent classes. Notably, the two least frequent classes, `Reducer` and `Strainer`, completely fail to be recognized (0% IoU), underscoring the severity of the long-tail problem.

## Point Distribution by Category in PSNet12 Dataset (%)

| Area | Name | Duct/0 | Elbow/1 | Flange/2 | Ibeam/3 | Pipe/4 | Pump/5 | RectangularBeam/6 | Reducer/7 | Strainer/8 | Tank/9 | Tee/10 | Valve/11 | TOTAL | Ratio |
|------|------|--------|---------|----------|---------|--------|--------|-------------------|-----------|------------|--------|--------|----------|-------|-------|
| 1 | Chiller | 5,311,209 | 2,770,052 | 5,149,969 | 359,610 | 22,799,721 | 1,478,449 | 5,821,433 | 29,453 | 595,153 | 0 | 1,252,257 | 1,389,285 | 46,956,591 | 0.08 |
| 2 | Service gallery | 29,442,798 | 1,432,104 | 142,104 | 184,875 | 18,739,827 | 878,080 | 25,016,033 | 2,581 | 0 | 0 | 2,010,975 | 1,743,055 | 79,592,432 | 0.13 |
| 3 | 93m Tank | 0 | 592,328 | 558,215 | 0 | 9,737,560 | 0 | 13,339,540 | 27,207 | 0 | 37,391,543 | 129,762 | 148,148 | 61,924,303 | 0.10 |
| 4 | 99.2 corridor | 24,006,510 | 145,436 | 73,538 | 367,581 | 5,668,017 | 0 | 22,232,498 | 0 | 0 | 240,150 | 47,384 | 0 | 52,781,114 | 0.09 |
| 5 | 99.2 Ozone Generation room | 17,920,499 | 540,905 | 773,649 | 0 | 11,609,415 | 0 | 17,204,263 | 175,521 | 0 | 0 | 285,862 | 300,015 | 48,810,129 | 0.08 |
| 6 | 93m Psu 99.2 VSPA-1 | 12,818,964 | 440,953 | 130,557 | 1,906,680 | 6,121,864 | 0 | 25,809,149 | 0 | 0 | 0 | 163,234 | 558 | 47,391,959 | 0.08 |
| 7 | 93m OxygenCompressorRm1-2 99.2 instrument | 6,775,378 | 372,372 | 385,482 | 2,324,899 | 10,764,037 | 0 | 9,835,747 | 0 | 53,492 | 0 | 97,079 | 217,343 | 30,825,829 | 0.05 |
| 8 | 99.2 VSPA-{2-6} | 16,823,642 | 1,313,416 | 688,602 | 8,442,834 | 10,957,479 | 159,266 | 26,961,414 | 5,904 | 0 | 3,550,356 | 273,792 | 259,829 | 69,436,534 | 0.11 |
| 9 | OSCG | 0 | 347,879 | 0 | 467,782 | 5,712,622 | 279,617 | 2,488,327 | 0 | 40,972 | 5,455,825 | 123,248 | 206,200 | 15,122,472 | 0.02 |
| 10 | PBF2_1 | 0 | 868,044 | 3,369,258 | 11,810,539 | 26,862,995 | 0 | 4,038,311 | 338,128 | 0 | 0 | 1,628,932 | 3,181,606 | 52,097,813 | 0.09 |
| 11 | PBF2_2 | 3,459,422 | 676,789 | 1,029,153 | 5,167,190 | 7,842,605 | 0 | 30,862,340 | 0 | 0 | 0 | 1,016,705 | 160,368 | 50,214,572 | 0.08 |
| 12 | SPH | 7,942,541 | 405,221 | 1,824,076 | 3,023,003 | 8,408,907 | 2,455,212 | 9,055,639 | 239,815 | 243,577 | 1,523,331 | 363,125 | 2,027,098 | 37,511,545 | 0.06 |
| 13 | WRT2 | 4,785,101 | 138,838 | 1,365,042 | 656,071 | 3,052,706 | 836,554 | 7,040,629 | 256,929 | 0 | 0 | 535,844 | 1,331,633 | 19,999,347 | 0.03 |
| | TOTAL | 129,286,064 | 10,044,337 | 15,489,645 | 34,711,064 | 148,277,755 | 6,087,178 | 199,705,323 | 1,075,538 | 933,194 | 48,161,205 | 7,928,199 | 10,965,138 | 612,664,640 | 1.00 |
| | Ratio | 20.999% | 1.631% | 2.516% | 5.638% | 24.083% | 0.989% | 32.436% | 0.175% | 0.152% | 7.822% | 1.288% | 1.781% | 100.000% | |
| | Rank | 3 | | | 5 | 2 | | 1 | Lowest | Lowest | 4 | | | | |
| | Training | 107,906,143 | 8,826,643 | 13,686,843 | 29,543,874 | 128,825,735 | 6,087,178 | 151,638,720 | 900,017 | 933,194 | 48,161,205 | 6,625,632 | 10,504,755 | 513,639,939 | |

## Conclusion

The severe performance gap between head and tail classes, with a Tail-mIoU of only 21.60% and complete failure on the rarest classes, makes the PSNet12 dataset an excellent and challenging benchmark for developing and evaluating long-tail 3D point cloud segmentation models. Consequently, we will conduct the rest of our experiments on PSNet12 to address this pronounced long-tail problem.

