## 2-stage training (S1 strong CB loss's best checkpoint + S2 CB loss w. lr 0.0001)
### TODOs

- How come the best mIoU from stage 2 is 54.253% with pretrained checkpoint w.o. strong CB loss is almost the same as the best mIoU from stage 2 with strong CB loss?

### stage 1

The best miou is 54.088% @epoch 212, detailed results are as follows:

```
09/27 05:30:54] psnet12 INFO: E212 V1 * mIoU 54.088%
[09/27 05:30:54] psnet12 INFO: E212 V1  * msIoU [0.91398145 0.41882375 0.50713881 0.99084241 0.7569293  0.43806582
 0.95076358 0.14915307 0.         0.99042162 0.04676464 0.32763662]
[09/27 05:30:54] psnet12 INFO: New best mIoU: 54.088%
[09/27 05:30:54] psnet12 INFO: ==> Saving...
[09/27 05:30:56] psnet12 INFO: Saved best model in log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250926015005/best.pth
```


### stage 2

The best miou is 54.253% @stage 2 E4, detailed results are as follows:

```
[09/27 13:36:26] psnet12_two_stage INFO: ES2-E4 V1 * mIoU 54.253%
[09/27 13:36:26] psnet12_two_stage INFO: ES2-E4 V1  * msIoU [0.91409964 0.46524689 0.508406   0.99205893 0.77504353 0.4357597
 0.95021416 0.         0.         0.98978268 0.08950846 0.39026746]
[09/27 13:36:26] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250927125709/stage2/stage2_best.pth

# another epoch w. strong CB loss
[09/28 13:09:37] psnet12_two_stage INFO: ES2-E4 V1 * mIoU 54.252%
[09/28 13:09:37] psnet12_two_stage INFO: ES2-E4 V1  * msIoU [0.91409969 0.46523941 0.50841067 0.9920567  0.77503195 0.43575699
 0.95021511 0.         0.         0.98978268 0.08950846 0.39017864]
[09/28 13:09:38] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250928123008/stage2/stage2_best.pth


# another epoch w. strong CB loss @ stage lr 0.0005
[09/29 07:14:20] psnet12_two_stage INFO: ES2-E1 V1 * mIoU 53.804%
[09/29 07:14:20] psnet12_two_stage INFO: ES2-E1 V1  * msIoU [0.91200441 0.47632465 0.48208824 0.99155347 0.76492493 0.40155291
 0.9507611  0.         0.         0.98975985 0.10819642 0.37928658]
[09/29 07:14:21] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250929070200_0.0005/stage2/stage2_best.pth

# another epoch w. strong CB loss @ stage lr 0.001
[09/29 13:52:59] psnet12_two_stage INFO: ES2-E20 V1 * mIoU 54.283%
[09/29 13:52:59] psnet12_two_stage INFO: ES2-E20 V1  * msIoU [0.91188613 0.45996833 0.49436878 0.99132645 0.77348399 0.45386596
 0.95045892 0.         0.         0.97325653 0.13410498 0.37126222]
[09/29 13:53:00] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250929104930_0.005/stage2/stage2_best.pth
[09/29 13:53:00] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250929104930_0.005/stage2/stage2_epoch_20.pth
[09/29 13:53:00] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250929104930_0.001/stage2/stage2_final.pth
[09/29 13:53:00] psnet12_two_stage INFO: Stage 2 completed. Best mIoU: 54.283%


# another epoch w. strong CB loss @ stage lr 0.005
09/29 17:40:55] psnet12_two_stage INFO: ES2-E20 V1 * mIoU 53.834%
[09/29 17:40:55] psnet12_two_stage INFO: ES2-E20 V1  * msIoU [0.90929211 0.46227165 0.49264116 0.98966262 0.77119163 0.4097618
 0.94944094 0.         0.         0.97799146 0.09989584 0.39787822]
[09/29 17:40:55] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250929143656_0.005/stage2/stage2_best.pth
[09/29 17:40:55] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250929143656_0.005/stage2/stage2_epoch_20.pth
[09/29 17:40:55] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250929143656_0.005/stage2/stage2_final.pth
[09/29 17:40:55] psnet12_two_stage INFO: Stage 2 completed. Best mIoU: 53.834%
```


## 2-stage training (S1 normal CE loss's best checkpoint + S2 CB loss w. lr 0.0001)
### stage 1

The best miou is 52.479% @epoch 165, detailed results are as follows:

```
[09/25 18:24:31] psnet12 INFO: E165 V1 * mIoU 52.479%
[09/25 18:24:31] psnet12 INFO: E165 V1  * msIoU [0.90727645 0.40771802 0.452336   0.98661473 0.76332283 0.43079931
 0.9479247  0.         0.         0.98986282 0.03102795 0.38061437]
[09/25 18:24:31] psnet12 INFO: New best mIoU: 52.479%
[09/25 18:24:31] psnet12 INFO: ==> Saving...
[09/25 18:24:33] psnet12 INFO: Saved best model in log/psnet12/respointnet2_dp_fi_df_fc1_max_20250924205054/best.pth
```
### stage 2

The best miou is 52.479% @epoch 165, detailed results are as follows:

```
# another epoch w. strong CB loss @ stage lr 0.0001
[09/28 18:27:09] psnet12_two_stage INFO: ES2-E1 V1 * mIoU 52.474%
[09/28 18:27:09] psnet12_two_stage INFO: ES2-E1 V1  * msIoU [0.90431588 0.41096559 0.46658512 0.97553431 0.75446722 0.40926494
 0.94609958 0.         0.         0.98865832 0.02725613 0.41373311]
[09/28 18:27:10] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250928181458/stage2/stage2_best.pth


# another epoch w. strong CB loss @ stage lr 0.0005
[09/29 19:49:21] psnet12_two_stage INFO: ES2-E9 V1 * mIoU 51.906%
[09/29 19:49:21] psnet12_two_stage INFO: ES2-E9 V1  * msIoU [0.89988726 0.42060293 0.46532558 0.9814353  0.76775728 0.37682854
 0.94180491 0.         0.         0.98768043 0.02060664 0.36680987]
[09/29 19:49:21] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250929182451_0.0005/stage2/stage2_best.pth

# another epoch w. strong CB loss @ stage lr 0.001
[09/30 00:03:55] psnet12_two_stage INFO: ES2-E12 V1 * mIoU 51.897%
[09/30 00:03:55] psnet12_two_stage INFO: ES2-E12 V1  * msIoU [0.91011574 0.43834395 0.46462298 0.98394628 0.75482586 0.37679755
 0.9476453  0.         0.         0.98920951 0.04219649 0.31998412]
[09/30 00:03:55] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250929221227_0.001/stage2/stage2_best.pth

# another epoch w. strong CB loss @ stage lr 0.005
[09/30 05:03:29] psnet12_two_stage INFO: ES2-E20 V1 * mIoU 51.577%
[09/30 05:03:29] psnet12_two_stage INFO: ES2-E20 V1  * msIoU [0.90915916 0.41785683 0.41039517 0.97631972 0.75209647 0.38430376
 0.94559131 0.         0.         0.98837085 0.04301874 0.36207897]
[09/30 05:03:29] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250930020008_0.005/stage2/stage2_best.pth
[09/30 05:03:30] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250930020008_0.005/stage2/stage2_epoch_20.pth
[09/30 05:03:30] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_20250930020008_0.005/stage2/stage2_final.pth
[09/30 05:03:30] psnet12_two_stage INFO: Stage 2 completed. Best mIoU: 51.577%
```

## 2-stage training (250924)

### stage 1

The best miou is 44.221% @epoch 66, detailed results are as follows:

```
[09/23 21:29:12] psnet12_two_stage INFO: ES1-E66 V1 * mIoU 44.221%
[09/23 21:29:12] psnet12_two_stage INFO: ES1-E66 V1  * msIoU [0.85296796 0.12082227 0.33369378 0.94933949 0.71839868 0.33281968
 0.9159994  0.         0.         0.96575291 0.         0.11668889]
[09/23 21:29:13] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_two_stage_20250923124932/stage1/stage1_best.pth
```

### stage 2

The best miou is 46.646% @epoch 4, detailed results are as follows:

```
[09/24 16:12:37] psnet12_two_stage INFO: ES2-E4 V1 * mIoU 46.646%
[09/24 16:12:37] psnet12_two_stage INFO: ES2-E4 V1  * msIoU [0.89345836 0.23063205 0.35393861 0.96975493 0.73533585 0.35207246
 0.93819791 0.         0.         0.98705385 0.         0.13702737]
[09/24 16:12:37] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_two_stage_20250923124932/stage2/stage2_best.pth

...
[09/24 20:00:26] psnet12_two_stage INFO: ES2-E4 V1 * mIoU 46.646%
[09/24 20:00:26] psnet12_two_stage INFO: ES2-E4 V1  * msIoU [0.89345836 0.23063205 0.35393055 0.96975493 0.73533559 0.35207436
 0.93819791 0.         0.         0.98705385 0.         0.13703678]
[09/24 20:00:27] psnet12_two_stage INFO: Checkpoint saved: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_two_stage_20250923124932/stage2/stage2_best.pth
[09/24 20:00:57] psnet12_two_stage INFO: STAGE2 Train: [5][0/333]	Loss 0.389 (0.389) [Enhanced Aug]
```

**Comparison and Conclusion:**

The two-stage training approach demonstrates clear performance improvement over single-stage training. Stage 2 achieves a 2.4% absolute improvement in mIoU compared to Stage 1, reaching 46.65% vs 44.22%. This represents a 5.5% relative improvement.

| Metric | Stage 1 | Stage 2 | Improvement |
|--------|---------|---------|-------------|
| Best mIoU | 44.221% | 46.646% | +2.4% (+5.5%) |
| Best Epoch | 66 | 4 | Much faster convergence |
| Class 1 IoU | 0.853 | 0.893 | +0.040 |
| Class 2 IoU | 0.121 | 0.231 | +0.110 |
| Class 3 IoU | 0.334 | 0.354 | +0.020 |
| Class 4 IoU | 0.949 | 0.970 | +0.021 |
| Class 5 IoU | 0.718 | 0.735 | +0.017 |
| Class 6 IoU | 0.333 | 0.352 | +0.019 |
| Class 7 IoU | 0.916 | 0.938 | +0.022 |
| Class 12 IoU | 0.117 | 0.137 | +0.020 |

**Conclusion:** The second stage training significantly improves performance across most classes, with particularly notable gains in Class 2 (+91% relative improvement). The approach also shows much faster convergence, achieving best results at epoch 4 compared to epoch 66 in stage 1. This suggests that the pre-trained features from stage 1 provide a strong foundation for rapid optimization in stage 2.
