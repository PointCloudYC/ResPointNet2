## PSNet12(early stopping and saving best model)

The reproduced results surpassed the paper results and detailed results are as follows:

Evaluation Summary:
- Overall mIoU: TODO, with msIoU TODO
- Overall sub_mIoU: TODO
- Overall Accuracy (OA): TODO

Per-class F1-scores:
- TODO

details see: `log/psnet12/?`


## PSNet5(early stopping and saving best model)

The reproduced results surpassed the paper results and detailed results are as follows:

Evaluation Summary:
- Overall mIoU: 87.56%, with msIoU [0.93855127 0.87721277 0.7006057  0.98426166 0.87744268]
- Overall sub_mIoU: 85.21%
- Overall Accuracy (OA): 94.99%

Per-class F1-scores:
- ibeam: 0.9683
- pipe: 0.9346
- pump: 0.8239
- rectangularbeam: 0.9921
- tank: 0.9347

details see: `log/psnet5/respointnet2_dp_fi_df_fc1_max_20250918160047`

Note: the results with early stopping are in `log/psnet5/respointnet2_dp_fi_df_fc1_max_20250915201532`