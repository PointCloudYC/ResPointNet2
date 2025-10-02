# GPU Resource Management Guide
## Efficient Ways to Run CB Loss Experiments with Limited GPU Resources

**Date**: 2025-10-01  
**Status**: Ready to use

---

## 🎯 Problem: Limited GPU Resources

You have limited GPU resources and want to run comprehensive CB loss experiments efficiently. Here are **4 different approaches** from simple to sophisticated:

---

## 📋 Available Solutions

### 1. 🚀 **GPU Daemon** (RECOMMENDED for Limited Resources)

**Best for**: Limited GPU access, want to run experiments automatically when GPU becomes available

```bash
# Start daemon (runs in background, checks every 3 hours)
bash scripts/gpu_daemon.sh start

# Check status
bash scripts/gpu_daemon.sh status

# View logs
bash scripts/gpu_daemon.sh logs

# Stop daemon
bash scripts/gpu_daemon.sh stop
```

**Features**:
- ✅ Runs in background
- ✅ Checks GPU every 3 hours
- ✅ Automatically starts experiments when >19GB available
- ✅ Logs all activity
- ✅ Can be started/stopped easily

---

### 2. ⚡ **Efficient GPU Runner** (ADAPTIVE)

**Best for**: Want to maximize GPU usage with adaptive settings

```bash
# Check current GPU status
bash scripts/efficient_gpu_runner.sh status

# Run with adaptive resource usage (RECOMMENDED)
bash scripts/efficient_gpu_runner.sh adaptive

# Run quick tests (10 epochs each)
bash scripts/efficient_gpu_runner.sh quick

# Monitor and run when resources available
bash scripts/efficient_gpu_runner.sh monitor
```

**Features**:
- ✅ Automatically adjusts batch size based on available memory
- ✅ Uses smaller models when memory is limited
- ✅ Can handle 8GB+ GPUs efficiently
- ✅ Runs experiments in sequence to maximize usage

**Memory Requirements**:
- **High (19GB+)**: Full experiments with batch_size=6
- **Medium (12-19GB)**: Adaptive experiments with batch_size=4  
- **Low (8-12GB)**: Reduced experiments with batch_size=2
- **Insufficient (<8GB)**: Cannot run experiments

---

### 3. 📊 **GPU Scheduler** (ADVANCED)

**Best for**: Want sophisticated queue management and priority-based execution

```bash
# Start scheduler (monitors and runs experiments automatically)
bash scripts/gpu_scheduler.sh start

# Check status
bash scripts/gpu_scheduler.sh status

# Reset if needed
bash scripts/gpu_scheduler.sh reset
```

**Features**:
- ✅ Sophisticated queue management
- ✅ Priority-based experiment execution
- ✅ Progress tracking and resumption
- ✅ Notifications on completion/failure
- ✅ Handles interruptions gracefully

---

### 4. 🎯 **Manual Comprehensive Script** (DIRECT)

**Best for**: Want full control, have dedicated GPU time

```bash
# Interactive menu
bash scripts/comprehensive_CB_loss_experiments.sh

# Run specific experiment
bash scripts/comprehensive_CB_loss_experiments.sh 3  # LD-CB
bash scripts/comprehensive_CB_loss_experiments.sh 4  # GC-CB
```

**Features**:
- ✅ Full control over experiment execution
- ✅ Interactive menu
- ✅ 15+ experiments available
- ✅ Can run specific experiments

---

## 🎯 **Recommended Approach for Your Situation**

### If you have **limited GPU access** (shared resources):

```bash
# Option 1: Use GPU Daemon (AUTOMATIC)
bash scripts/gpu_daemon.sh start
# This will run in background and automatically start experiments when GPU becomes available

# Option 2: Use Efficient Runner (ADAPTIVE)
bash scripts/efficient_gpu_runner.sh adaptive
# This will adapt to your available GPU memory and run experiments efficiently
```

### If you have **dedicated GPU time**:

```bash
# Use Manual Comprehensive Script
bash scripts/comprehensive_CB_loss_experiments.sh
# Choose experiments from the interactive menu
```

### If you want to **test quickly**:

```bash
# Run quick tests (10 epochs each)
bash scripts/efficient_gpu_runner.sh quick
# This will test all methods in ~2 hours
```

---

## 📊 **Resource Usage Comparison**

| Approach | GPU Memory Required | Control Level | Automation | Best For |
|----------|-------------------|---------------|------------|----------|
| **GPU Daemon** | 19GB+ | Low | High | Limited access |
| **Efficient Runner** | 8GB+ | Medium | Medium | Adaptive usage |
| **GPU Scheduler** | 19GB+ | High | High | Advanced management |
| **Manual Script** | 19GB+ | High | Low | Dedicated time |

---

## 🚀 **Quick Start Guide**

### Step 1: Check Your GPU Status

```bash
# Check available GPU memory
nvidia-smi

# Check which approach is best for you
bash scripts/efficient_gpu_runner.sh status
```

### Step 2: Choose Your Approach

**If you have 19GB+ memory and dedicated time**:
```bash
bash scripts/comprehensive_CB_loss_experiments.sh
```

**If you have 8GB+ memory and want adaptive usage**:
```bash
bash scripts/efficient_gpu_runner.sh adaptive
```

**If you have limited access and want automation**:
```bash
bash scripts/gpu_daemon.sh start
```

### Step 3: Monitor Progress

```bash
# Check experiment logs
tail -f log/psnet12/[experiment_dir]/log_train.txt

# Check daemon status (if using daemon)
bash scripts/gpu_daemon.sh status

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## 💡 **Pro Tips for Limited Resources**

### 1. **Use Adaptive Settings**
- The efficient runner automatically adjusts batch size based on available memory
- Smaller batch sizes = less memory usage
- More epochs may be needed for convergence

### 2. **Run Experiments in Sequence**
- Don't run multiple experiments simultaneously
- Let one complete before starting the next
- This maximizes GPU utilization

### 3. **Use Quick Tests First**
```bash
# Test everything quickly (2 hours total)
bash scripts/efficient_gpu_runner.sh quick
```

### 4. **Monitor GPU Usage**
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Check if training is running
pgrep -f train_psnet_dist.py
```

### 5. **Save Progress**
- All approaches save checkpoints automatically
- You can resume from interruptions
- Logs are saved for analysis

---

## 🛠️ **Troubleshooting**

### If you get "Out of Memory" errors:

```bash
# Check available memory
nvidia-smi

# Use adaptive mode with smaller batch sizes
bash scripts/efficient_gpu_runner.sh adaptive

# Or reduce batch size in config files
# Edit cfgs/psnet12/*.yaml and change batch_size: 6 to batch_size: 2
```

### If experiments are too slow:

```bash
# Use quick mode for testing
bash scripts/efficient_gpu_runner.sh quick

# Or reduce epochs in config files
# Edit cfgs/psnet12/*.yaml and change epochs: 600 to epochs: 100
```

### If daemon doesn't start:

```bash
# Check if daemon is already running
bash scripts/gpu_daemon.sh status

# Reset daemon state
bash scripts/gpu_daemon.sh reset

# Start fresh
bash scripts/gpu_daemon.sh start
```

---

## 📈 **Expected Timeline**

### With 19GB+ GPU (Full Resources):
- **Baseline**: 2-3 days
- **LD-CB**: 2-3 days  
- **GC-CB**: 2-3 days
- **Key Ablations**: 8-10 days
- **Total**: ~2-3 weeks

### With 8-12GB GPU (Adaptive):
- **Baseline**: 3-4 days
- **LD-CB**: 3-4 days
- **GC-CB**: 3-4 days  
- **Key Ablations**: 10-12 days
- **Total**: ~3-4 weeks

### With Limited Access (Daemon):
- **Depends on when GPU becomes available**
- **Can take 1-2 months if GPU is heavily shared**
- **But runs automatically when resources are free**

---

## 🎯 **Success Metrics**

Your experiments will be successful if:

✅ **LD-CB beats Standard CB by ≥2% overall mIoU**  
✅ **LD-CB beats Standard CB by ≥5-10% on tail classes** (MOST IMPORTANT!)  
✅ **Improvements are statistically significant**  
✅ **Head class performance remains stable** (±2%)  

---

## 📞 **Getting Help**

### Check Status:
```bash
# GPU status
nvidia-smi

# Daemon status
bash scripts/gpu_daemon.sh status

# Efficient runner status  
bash scripts/efficient_gpu_runner.sh status
```

### View Logs:
```bash
# Daemon logs
bash scripts/gpu_daemon.sh logs

# Experiment logs
tail -f log/psnet12/[experiment_dir]/log_train.txt
```

### Reset Everything:
```bash
# Reset daemon
bash scripts/gpu_daemon.sh reset

# Reset scheduler
bash scripts/gpu_scheduler.sh reset
```

---

## 🚀 **Ready to Start?**

**Choose your approach based on your situation:**

1. **Limited GPU access** → Use GPU Daemon
2. **8GB+ GPU, want efficiency** → Use Efficient Runner  
3. **19GB+ GPU, dedicated time** → Use Manual Script
4. **Want to test quickly** → Use Quick Mode

**All approaches will give you the same results - just different ways to get there!**

---

**Good luck with your experiments! 🎉**

