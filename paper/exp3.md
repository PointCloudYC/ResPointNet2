## info

Class distribution: {'duct': 1663319, 'elbow': 161915, 'flange': 265522, 'ibeam': 570126, 'pipe': 2154311, 'pump': 58965, 'rectangularbeam': 3527818, 'reducer': 11019, 'strainer': 9138, 'tank': 487956, 'tee': 99396, 'valve': 127480}

## training plain CB loss

- results: mIoU 48.154% @epoch 111 w. early stopping patience 20, detailed results are as follows, and the best model is saved in `log/psnet12/respointnet2_dp_fi_df_fc1_max_20250919163927/best.pth`

```
[09/20 07:15:05] psnet12 INFO: E111 V1 * mIoU 48.154%
[09/20 07:15:05] psnet12 INFO: E111 V1  * msIoU [0.89098106 0.28553554 0.40304174 0.96781021 0.72605487 0.38514389
 0.94526234 0.         0.         0.93258006 0.         0.24208114]
[09/20 07:15:05] psnet12 INFO: New best mIoU: 48.154%
[09/20 07:15:05] psnet12 INFO: ==> Saving...
[09/20 07:15:07] psnet12 INFO: Saved best model in log/psnet12/respointnet2_dp_fi_df_fc1_max_20250919163927/best.pth
```

## training plain CB loss

- results: mIoU 40.047% @epoch 36 w. early stopping patience 20, detailed results are as follows, and the best model is saved in `log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_20250921214715/best.pth`

```
 0.923351   0.         0.         0.8808909  0.         0.        ]
[09/21 17:13:18] psnet12 INFO: E36 V1 * mIoU 40.047%
[09/21 17:13:18] psnet12 INFO: E36 V1  * msIoU [0.87584416 0.         0.25997546 0.84151705 0.66732977 0.28646128
 0.93827695 0.         0.         0.93629086 0.         0.        ]
[09/21 17:13:18] psnet12 INFO: New best mIoU: 40.047%
```


## training strong CB loss

- results: mIoU 50.072% @epoch 119 w. early stopping patience 20, detailed results are as follows, and the best model is saved in `log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250921214715/best.pth`

```
[09/22 13:24:31] psnet12 INFO: E119 V1 * mIoU 50.072%
[09/22 13:24:31] psnet12 INFO: E119 V1  * msIoU [0.88017928 0.24303868 0.43321599 0.9810273  0.76277525 0.34421582
 0.92821206 0.         0.         0.98083735 0.13718397 0.31798426]
[09/22 13:24:31] psnet12 INFO: New best mIoU: 50.072%
[09/22 13:24:31] psnet12 INFO: ==> Saving...
[09/22 13:24:33] psnet12 INFO: Saved best model in log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250921214715/best.pth
```