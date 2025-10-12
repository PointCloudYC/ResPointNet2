# plan copy the whole project to another server

# compress the code
echo "start compressing the code"
cd /media/yinchao/code
tar -czf ResPointNet2_code.tar.gz \
    --exclude='ResPointNet2/log' \
    --exclude='ResPointNet2/log_eval' \
    --exclude='ResPointNet2/data' \
    --exclude='ResPointNet2/~' \
    --exclude='ResPointNet2/.venv' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' --exclude='*.pth' --exclude='*.pt' --exclude='*.so' --exclude='*/build' --exclude='*/dist' --exclude='*.egg-info' --exclude='*/.venv' --exclude='*/venv' --exclude='*/.uv' ResPointNet2/
echo "end compressing the code"

# compress the PSNet12 datasets
# echo "start compressing the PSNet12 datasets"
# tar -czf PSNet12_datasets.tar.gz \
#     --exclude='ResPointNet2/data/PSNet/PSNet5' \
#     --exclude='ResPointNet2/data/S3DIS' \
#     ResPointNet2/data/PSNet/PSNet12/Area_* \
#     ResPointNet2/data/PSNet/PSNet12/processed \
#     ResPointNet2/data/PSNet/PSNet12/meta \
#     ResPointNet2/data/PSNet/PSNet12/scripts \
#     ResPointNet2/data/PSNet/PSNet12/*.txt \
#     ResPointNet2/data/PSNet/PSNet12/*.csv
# echo "end compressing the PSNet12 datasets"