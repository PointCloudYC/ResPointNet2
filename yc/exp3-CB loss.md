## info

Class distribution: {'duct': 1663319, 'elbow': 161915, 'flange': 265522, 'ibeam': 570126, 'pipe': 2154311, 'pump': 58965, 'rectangularbeam': 3527818, 'reducer': 11019, 'strainer': 9138, 'tank': 487956, 'tee': 99396, 'valve': 127480}

## TLDR

The **strong CB loss configuration (γ=3.0, α=2.0)** is the clear winner, providing the best overall performance while maintaining reasonable training stability. The plain CB loss actually hurt performance significantly, suggesting the hyperparameters were too aggressive for this imbalanced dataset.

## Key Findings

**Strong CB loss wins decisively** with 50.072% mIoU vs baseline 48.154% vs plain CB 40.047%. The progression shows:

- **Baseline**: 48.154% mIoU @ epoch 111
- **Plain CB loss**: 40.047% mIoU @ epoch 36 (regression)
- **Strong CB loss**: 54.088% mIoU @ epoch 212 (improvement)

## Comparison Table

| Metric | Baseline | Plain CB Loss | Strong CB Loss |
|--------|----------|---------------|----------------|
| **mIoU** | 52.479% | 40.047% | **54.088%** |
| **Epoch** | 165 | 36 | 212 |
| **Improvement vs Baseline** | - | -12.385% | **+1.6%** |
| **Convergence Speed** | Medium | Fast (early stop) | Slow |
| **Best Performing Classes** | duct(89%), ibeam(97%), pipe(73%) | duct(88%), pipe(67%), tank(94%) | duct(88%), ibeam(98%), pipe(76%) |
| **Worst Performing Classes** | reducer/strainer/tee (0%) | elbow/flange/reducer/strainer/tee/valve (0%) | reducer/strainer (0%) |


## baseline

- results(250926): new experiments, 2025-09-26(early stopping with patience 50 and save the best model), details are in the log `log/psnet12/respointnet2_dp_fi_df_fc1_max_20250924205054`, **the best mIou is 52.479% @epoch 165, took ~28 hours with 215 epochs on 2 RTX 3090 GPUs**

```
[09/25 18:24:31] psnet12 INFO: E165 V1 * mIoU 52.479%
[09/25 18:24:31] psnet12 INFO: E165 V1  * msIoU [0.90727645 0.40771802 0.452336   0.98661473 0.76332283 0.43079931
 0.9479247  0.         0.         0.98986282 0.03102795 0.38061437]
[09/25 18:24:31] psnet12 INFO: New best mIoU: 52.479%
[09/25 18:24:31] psnet12 INFO: ==> Saving...
[09/25 18:24:33] psnet12 INFO: Saved best model in log/psnet12/respointnet2_dp_fi_df_fc1_max_20250924205054/best.pth
```

- ~~results(250920): mIoU 48.154% @epoch 111 w. early stopping patience 20, detailed results are as follows, and the best model is saved in `log/psnet12/respointnet2_dp_fi_df_fc1_max_20250919163927/best.pth`~~


```
[09/20 07:15:05] psnet12 INFO: E111 V1 * mIoU 48.154%
[09/20 07:15:05] psnet12 INFO: E111 V1  * msIoU [0.89098106 0.28553554 0.40304174 0.96781021 0.72605487 0.38514389
 0.94526234 0.         0.         0.93258006 0.         0.24208114]
[09/20 07:15:05] psnet12 INFO: New best mIoU: 48.154%
[09/20 07:15:05] psnet12 INFO: ==> Saving...
[09/20 07:15:07] psnet12 INFO: Saved best model in log/psnet12/respointnet2_dp_fi_df_fc1_max_20250919163927/best.pth
```

## training plain CB loss(beta=0.9999, gamma=2.0, alpha=1.0)

- results: mIoU 40.047% @epoch 36 w. early stopping patience 20, detailed results are as follows, and the best model is saved in `log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_20250921214715/best.pth`

```
 0.923351   0.         0.         0.8808909  0.         0.        ]
[09/21 17:13:18] psnet12 INFO: E36 V1 * mIoU 40.047%
[09/21 17:13:18] psnet12 INFO: E36 V1  * msIoU [0.87584416 0.         0.25997546 0.84151705 0.66732977 0.28646128
 0.93827695 0.         0.         0.93629086 0.         0.        ]
[09/21 17:13:18] psnet12 INFO: New best mIoU: 40.047%
```


## training strong CB loss(beta=0.9999, gamma=3.0, alpha=2.0)

- results(250927): mIoU 54.088% @epoch 212 w. early stopping patience 50, detailed results are as follows, and the best model is saved in `log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250926015005/best.pth`

```
[09/27 05:30:54] psnet12 INFO: E212 V1 * mIoU 54.088%
[09/27 05:30:54] psnet12 INFO: E212 V1  * msIoU [0.91398145 0.41882375 0.50713881 0.99084241 0.7569293  0.43806582
 0.95076358 0.14915307 0.         0.99042162 0.04676464 0.32763662]
[09/27 05:30:54] psnet12 INFO: New best mIoU: 54.088%
[09/27 05:30:54] psnet12 INFO: ==> Saving...
[09/27 05:30:56] psnet12 INFO: Saved best model in log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250926015005/best.pth
```

- results(250922): mIoU 50.072% @epoch 119 w. early stopping patience 20, detailed results are as follows, and the best model is saved in `log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250921214715/best.pth`

```
[09/22 13:24:31] psnet12 INFO: E119 V1 * mIoU 50.072%
[09/22 13:24:31] psnet12 INFO: E119 V1  * msIoU [0.88017928 0.24303868 0.43321599 0.9810273  0.76277525 0.34421582
 0.92821206 0.         0.         0.98083735 0.13718397 0.31798426]
[09/22 13:24:31] psnet12 INFO: New best mIoU: 50.072%
[09/22 13:24:31] psnet12 INFO: ==> Saving...
[09/22 13:24:33] psnet12 INFO: Saved best model in log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250921214715/best.pth
```