# Colab 2: LoRA Fine-tuning with SmolLM2-135M

## 📋 Overview
This directory contains a Google Colab notebook demonstrating **LoRA (Low-Rank Adaptation)** fine-tuning with SmolLM2-135M using Unsloth.ai.

## 🎯 Objectives
- Learn parameter-efficient fine-tuning with LoRA
- Compare LoRA (r=16) with full fine-tuning (r=256 from Colab 1)
- Understand the efficiency benefits of LoRA
- Use the same dataset as Colab 1 for fair comparison
- Demonstrate significant resource savings

## 📊 Model Details
- **Model**: SmolLM2-135M-Instruct
- **Parameters**: 135 million total
- **Method**: LoRA with r=16 (low rank)
- **Trainable Params**: ~4M (~2% of model)
- **Dataset**: yahma/alpaca-cleaned (100 samples)

## 🔑 Key Concepts

### What is LoRA?
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
- Freezes the original model weights
- Adds small trainable adapter layers
- Updates only ~2% of parameters
- Achieves comparable performance to full fine-tuning

### How LoRA Works
```
Original Weight Matrix: W (frozen)
LoRA Adaptation: W + ΔW
Where: ΔW = B × A (low-rank matrices)
Rank: r=16 (much smaller than full matrix dimensions)
```

## 📊 Comparison with Colab 1

| Metric | Colab 1 (Full, r=256) | Colab 2 (LoRA, r=16) | Improvement |
|--------|----------------------|---------------------|-------------|
| Trainable Params | ~78M (36.75%) | ~4M (~2%) | **95% fewer** |
| Training Speed | Baseline | Faster | **~20-30% faster** |
| Model Size | ~270MB | ~10-20MB | **90% smaller** |
| Memory Usage | Higher | Lower | **~40% less** |
| Performance | Excellent | Comparable | ~Similar |

## 🚀 Quick Start

### Prerequisites
- Google Colab account
- GPU runtime (T4)
- ~5-10 minutes of time

### Steps to Run
1. Open `lora_finetuning_smollm2_135m.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells: Runtime → Run all
4. Wait ~8-12 minutes (faster than Colab 1!)
5. Compare results with Colab 1

## 📝 Notebook Structure

### 1. **Setup** (Steps 1-2)
   - Install Unsloth and dependencies
   - Import libraries and disable wandb

### 2. **Model Loading** (Steps 3-4)
   - Configure parameters (same as Colab 1)
   - Load SmolLM2-135M model

### 3. **LoRA Configuration** (Step 5) ⭐
   - **KEY**: Set r=16 for LoRA
   - Configure adapter modules
   - Much fewer trainable parameters!

### 4. **Dataset Preparation** (Steps 6-7)
   - Load same dataset as Colab 1
   - Format with Alpaca template
   - 100 samples for fair comparison

### 5. **Training** (Steps 8-10)
   - Same training config as Colab 1
   - Watch faster training speed!
   - Lower loss over 60 steps

### 6. **Testing** (Steps 11-12)
   - Test with various prompts
   - Compare quality with Colab 1
   - Similar performance, faster training!

### 7. **Model Saving** (Step 13)
   - Save LoRA adapters
   - Much smaller files (~10-20MB)
   - Easy to share and deploy

### 8. **Comparison Summary** (Step 14)
   - Side-by-side comparison with Colab 1
   - Efficiency metrics
   - Trade-offs analysis

## 💡 Key Parameters

### LoRA Configuration
```python
r=16,              # Rank: Lower = more efficient
lora_alpha=16,     # Scaling factor (usually = r)
lora_dropout=0,    # Dropout for regularization
target_modules=[   # Which layers to adapt
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### Why r=16?
- **r=8**: Very efficient, but may sacrifice performance
- **r=16**: Sweet spot - efficient + good performance ⭐
- **r=32**: More parameters, diminishing returns
- **r=256**: Approaches full fine-tuning (Colab 1)

## 📊 Expected Results

### Training Metrics
```
Trainable parameters: ~4M of 135M (~2%)
Training time: ~8-12 minutes (faster than Colab 1)
Final loss: ~0.4-0.6 (similar to Colab 1)
```

### Model Quality
- ✅ Follows instructions accurately
- ✅ Generates coherent responses
- ✅ Comparable to full fine-tuning
- ✅ Much more efficient!

## 🎥 Video Walkthrough Topics

When recording your video, cover:
1. **LoRA Concept**: What it is and how it works
2. **Parameter Efficiency**: Show ~2% vs 36.75% trainable params
3. **Speed Comparison**: Compare training time with Colab 1
4. **Code Walkthrough**: Especially Step 5 (LoRA config)
5. **Test Results**: Show model performance
6. **File Sizes**: Compare adapter size vs full model
7. **Use Cases**: When to use LoRA vs full fine-tuning

## ✅ Advantages of LoRA

### 1. **Resource Efficiency**
- 95% fewer trainable parameters
- 40% less memory usage
- 20-30% faster training

### 2. **Storage Efficiency**
- Adapter files: ~10-20MB
- Full model: ~270MB
- 90% storage savings!

### 3. **Deployment Flexibility**
- Easy to switch between tasks
- Multiple adapters on one base model
- Quick to download and share

### 4. **Comparable Performance**
- Often matches full fine-tuning quality
- Especially good for instruction following
- Better generalization in some cases

## ⚠️ When to Use LoRA vs Full Fine-tuning

### Use LoRA When:
- ✅ Limited GPU resources
- ✅ Need quick iterations
- ✅ Training multiple task-specific models
- ✅ Need to share/deploy models
- ✅ Instruction following tasks

### Use Full Fine-tuning When:
- ✅ Maximum performance is critical
- ✅ Have ample resources
- ✅ Drastic domain shift
- ✅ Single, important task

## 🔬 Technical Details

### LoRA Mathematics
```
Forward pass:
h = W₀x + ΔWx = W₀x + BAx

Where:
- W₀: Frozen pre-trained weights
- B ∈ ℝ^(d×r): Down-projection matrix
- A ∈ ℝ^(r×k): Up-projection matrix
- r << min(d, k): Rank bottleneck
```

### Memory Savings
```
Full fine-tuning memory: O(d × k)
LoRA memory: O(r × (d + k))
With r=16, d=k=4096: 99.6% memory reduction!
```

## 🔗 Resources

- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Unsloth Docs**: https://docs.unsloth.ai/
- **LoRA Guide**: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide
- **PEFT Library**: https://huggingface.co/docs/peft
- **SmolLM2**: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct

## 🐛 Troubleshooting

### Issue: "Out of memory"
**Solution**: Already using r=16 (low rank), should not happen with T4 GPU

### Issue: "Training too slow"
**Solution**: Reduce dataset size from 100 to 50 samples

### Issue: "Poor model performance"
**Solution**: 
- Increase r from 16 to 32
- Increase max_steps from 60 to 100
- Check if you're comparing with properly trained Colab 1 model

## 📈 Performance Benchmarks

### Expected Training Time
- **T4 GPU**: 8-12 minutes
- **V100 GPU**: 5-8 minutes
- **A100 GPU**: 3-5 minutes

### Compared to Colab 1
- **Speedup**: ~1.2-1.3x faster
- **Memory**: ~40% less
- **Quality**: 95-100% of full fine-tuning performance

## 🎓 Learning Outcomes

After completing this notebook, you will:
- ✅ Understand LoRA and parameter-efficient fine-tuning
- ✅ Know how to configure LoRA rank and alpha
- ✅ Be able to compare LoRA vs full fine-tuning
- ✅ Understand the efficiency trade-offs
- ✅ Know when to choose LoRA over full fine-tuning
- ✅ Be able to save and deploy LoRA adapters

## 🔄 Next Steps

After completing Colab 2:
1. ✅ Compare training time and model size with Colab 1
2. ✅ Record video explaining LoRA benefits
3. ✅ Test model quality side-by-side with Colab 1
4. ➡️ Move to **Colab 3** for DPO Reinforcement Learning

---

**Status**: ✅ Notebook created and ready for execution
**Estimated Time**: 8-12 minutes
**Difficulty**: Intermediate
**Last Updated**: November 9, 2025
