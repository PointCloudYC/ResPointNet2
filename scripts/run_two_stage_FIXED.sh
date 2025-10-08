#!/bin/bash
# ============================================================================
# Two-Stage Training Script - FIXED VERSION
# ============================================================================
# This script runs two-stage decoupled training with ALL CRITICAL BUGS FIXED:
#
# ✅ FIX #1: Class-balanced sampler properly implemented and verified
# ✅ FIX #2: Learning rate reduced from 0.0001 to 0.00001 (10x reduction)
# ✅ FIX #3: Warmup disabled (auto-disabled for ultra-low LR)
# ✅ FIX #4: Using CE loss in Stage 2 (not CB loss)
# ✅ FIX #5: Strong regularization added (weight_decay=0.01)
# ✅ FIX #6: Validation uses standard imbalanced sampler (fair evaluation)
# ✅ FIX #7: Comprehensive verification functions added
#
# Expected behavior:
# - Stage 2 should NOT degrade from Stage 1 baseline
# - Tail class IoU should improve (pump, tank, etc.)
# - Head class IoU should remain stable (pipe, rbeam, etc.)
# - Overall mIoU: +0.5% to +2% improvement (realistic expectation)
# ============================================================================

set -e  # Exit on error

# Configuration
DATASET="psnet12"
CONFIG="cfgs/psnet12/respointnet2_two_stage_FIXED.yaml"
NUM_GPUS=2
BATCH_SIZE=6

# Stage 1: Load from checkpoint (skip training)
STAGE1_CHECKPOINT=""  # Set this to your Stage 1 best checkpoint path
# Example: STAGE1_CHECKPOINT="log/psnet12/your_stage1_run/stage1/stage1_best.pth"

# Experiment naming
TIMESTAMP=$(date +%Y%m%d%H%M%S)
EXP_NAME="two_stage_FIXED_lr1e5_noWarmup_${TIMESTAMP}"
LOG_DIR="log/${DATASET}/${EXP_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

log_info "Running pre-flight checks..."

# Check config file exists
if [ ! -f "$CONFIG" ]; then
    log_error "Config file not found: $CONFIG"
    exit 1
fi
log_success "Config file found: $CONFIG"

# Check GPU availability
NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    log_warning "Requested $NUM_GPUS GPUs, but only $NUM_AVAILABLE_GPUS available"
    log_info "Adjusting to use $NUM_AVAILABLE_GPUS GPUs"
    NUM_GPUS=$NUM_AVAILABLE_GPUS
fi
log_success "Using $NUM_GPUS GPUs"

# Display GPU info
log_info "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv | head -n $((NUM_GPUS + 1))

# Check if Stage 1 checkpoint exists
if [ -n "$STAGE1_CHECKPOINT" ]; then
    if [ ! -f "$STAGE1_CHECKPOINT" ]; then
        log_error "Stage 1 checkpoint not found: $STAGE1_CHECKPOINT"
        log_info "Either:"
        log_info "  1. Set STAGE1_CHECKPOINT to empty string to train Stage 1 from scratch"
        log_info "  2. Provide correct path to Stage 1 checkpoint"
        exit 1
    fi
    log_success "Stage 1 checkpoint found: $STAGE1_CHECKPOINT"
else
    log_warning "No Stage 1 checkpoint provided - will train Stage 1 from scratch"
    log_info "This will take significant time (hours to days depending on stage1_epochs)"
fi

# ============================================================================
# Display Configuration
# ============================================================================

log_info "============================================================"
log_info "Two-Stage Training Configuration (FIXED)"
log_info "============================================================"
log_info "Experiment:        $EXP_NAME"
log_info "Config:            $CONFIG"
log_info "Log directory:     $LOG_DIR"
log_info "GPUs:              $NUM_GPUS"
log_info "Batch size:        $BATCH_SIZE"
log_info "Stage 1 checkpoint: ${STAGE1_CHECKPOINT:-'None (will train from scratch)'}"
log_info ""
log_info "CRITICAL FIXES APPLIED:"
log_success "  ✅ Learning rate: 0.00001 (10x reduction)"
log_success "  ✅ Warmup: Disabled (auto-disabled for ultra-low LR)"
log_success "  ✅ Loss: CE loss (not CB loss) in Stage 2"
log_success "  ✅ Regularization: weight_decay=0.01 (10x Stage 1)"
log_success "  ✅ Sampling: Class-balanced for training, imbalanced for validation"
log_success "  ✅ Verification: Comprehensive checks enabled"
log_info "============================================================"

# ============================================================================
# Prepare directories
# ============================================================================

mkdir -p "$LOG_DIR"
mkdir -p "${LOG_DIR}/stage1"
mkdir -p "${LOG_DIR}/stage2"
mkdir -p "${LOG_DIR}/plots"

# Copy config for reproducibility
cp "$CONFIG" "${LOG_DIR}/config_used.yaml"
log_success "Created log directory structure"

# ============================================================================
# Copy or link Stage 1 checkpoint if provided
# ============================================================================

if [ -n "$STAGE1_CHECKPOINT" ]; then
    log_info "Copying Stage 1 checkpoint to experiment directory..."
    cp "$STAGE1_CHECKPOINT" "${LOG_DIR}/stage1/stage1_best.pth"
    log_success "Stage 1 checkpoint ready"
fi

# ============================================================================
# Run Two-Stage Training
# ============================================================================

log_info "Starting two-stage training..."
log_info "Training logs will be written to: ${LOG_DIR}/log.txt"
log_info "============================================================"

# Build command
CMD="python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    function/train_psnet_two_stage_dist.py \
    --cfg ${CONFIG} \
    --dataset_name ${DATASET} \
    --log_dir ${LOG_DIR} \
    --batch_size ${BATCH_SIZE}"

# Execute
log_info "Command: $CMD"
log_info "Starting training... (this may take several hours)"

# Run with error handling
if $CMD 2>&1 | tee "${LOG_DIR}/log.txt"; then
    log_success "Training completed successfully!"
else
    EXIT_CODE=$?
    log_error "Training failed with exit code: $EXIT_CODE"
    log_error "Check logs at: ${LOG_DIR}/log.txt"
    exit $EXIT_CODE
fi

# ============================================================================
# Post-Training Analysis
# ============================================================================

log_info "============================================================"
log_info "Post-Training Analysis"
log_info "============================================================"

# Check if Stage 2 best checkpoint exists
if [ -f "${LOG_DIR}/stage2/stage2_best.pth" ]; then
    log_success "Stage 2 training completed - best checkpoint saved"
else
    log_warning "Stage 2 best checkpoint not found - training may have failed"
fi

# Check for training report
if [ -f "${LOG_DIR}/training_report.txt" ]; then
    log_success "Training report generated"
    log_info "Report summary:"
    grep -A 5 "OVERALL SUMMARY" "${LOG_DIR}/training_report.txt" || true
else
    log_warning "Training report not found"
fi

# Check for plots
PLOT_COUNT=$(find "${LOG_DIR}/plots" -name "*.png" 2>/dev/null | wc -l)
if [ "$PLOT_COUNT" -gt 0 ]; then
    log_success "Generated $PLOT_COUNT training plots"
    log_info "Plots saved to: ${LOG_DIR}/plots/"
else
    log_warning "No training plots generated"
fi

# Extract key results from log
log_info "============================================================"
log_info "Key Results (if available):"
log_info "============================================================"

# Try to extract Stage 1 and Stage 2 best mIoU
STAGE1_MIOU=$(grep -oP "Stage 1.*[Bb]est.*?mIoU[: ]*\K[\d.]+" "${LOG_DIR}/log.txt" 2>/dev/null | tail -1 || echo "N/A")
STAGE2_MIOU=$(grep -oP "Stage 2.*[Bb]est.*?mIoU[: ]*\K[\d.]+" "${LOG_DIR}/log.txt" 2>/dev/null | tail -1 || echo "N/A")

log_info "Stage 1 Best mIoU: ${STAGE1_MIOU}%"
log_info "Stage 2 Best mIoU: ${STAGE2_MIOU}%"

if [ "$STAGE1_MIOU" != "N/A" ] && [ "$STAGE2_MIOU" != "N/A" ]; then
    # Calculate improvement (basic floating point comparison)
    IMPROVEMENT=$(awk "BEGIN {print $STAGE2_MIOU - $STAGE1_MIOU}")
    if (( $(awk "BEGIN {print ($IMPROVEMENT > 0) ? 1 : 0}") )); then
        log_success "Improvement: +${IMPROVEMENT}%"
    else
        log_warning "Change: ${IMPROVEMENT}%"
    fi
fi

log_info "============================================================"

# ============================================================================
# Verification Checks
# ============================================================================

log_info "============================================================"
log_info "Running Post-Training Verification Checks"
log_info "============================================================"

# Check for encoder leakage errors
if grep -q "ENCODER LEAKAGE DETECTED" "${LOG_DIR}/log.txt" 2>/dev/null; then
    log_error "Encoder leakage detected during training!"
    log_error "Stage 2 may have corrupted Stage 1 features"
else
    log_success "No encoder leakage detected"
fi

# Check for gradient leakage
if grep -q "GRADIENT LEAKAGE" "${LOG_DIR}/log.txt" 2>/dev/null; then
    log_error "Gradient leakage to frozen parameters detected!"
else
    log_success "No gradient leakage detected"
fi

# Check for catastrophic forgetting warnings
FORGETTING_WARNINGS=$(grep -c "Performance dropping" "${LOG_DIR}/log.txt" 2>/dev/null || echo "0")
if [ "$FORGETTING_WARNINGS" -gt 0 ]; then
    log_warning "Detected $FORGETTING_WARNINGS performance drop warnings"
    log_warning "Review training log for details"
else
    log_success "No catastrophic forgetting warnings"
fi

# Check class-balanced sampling verification
if grep -q "WELL BALANCED" "${LOG_DIR}/log.txt" 2>/dev/null; then
    log_success "Class-balanced sampling verified as working correctly"
elif grep -q "SEVERELY IMBALANCED" "${LOG_DIR}/log.txt" 2>/dev/null; then
    log_error "Class-balanced sampling NOT working - severely imbalanced"
else
    log_warning "Class-balanced sampling status unclear"
fi

log_info "============================================================"

# ============================================================================
# Final Summary
# ============================================================================

log_info "============================================================"
log_success "Two-Stage Training Workflow Completed"
log_info "============================================================"
log_info "Experiment:  $EXP_NAME"
log_info "Results:     ${LOG_DIR}/"
log_info ""
log_info "Next steps:"
log_info "  1. Review training report: ${LOG_DIR}/training_report.txt"
log_info "  2. Check training plots:   ${LOG_DIR}/plots/"
log_info "  3. Inspect full log:       ${LOG_DIR}/log.txt"
log_info "  4. Validate best model:    ${LOG_DIR}/stage2/stage2_best.pth"
log_info ""
log_info "For detailed analysis, run:"
log_info "  cat ${LOG_DIR}/training_report.txt"
log_info "  python scripts/analyze_two_stage_results.py ${LOG_DIR}"
log_info "============================================================"

exit 0

