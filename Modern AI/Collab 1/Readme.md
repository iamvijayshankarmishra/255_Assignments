# Colab 1: Full Fine-tuning with SmolLM2-135M

## 📋 Overview
This directory contains a comprehensive Google Colab notebook demonstrating **full fine-tuning** of the SmolLM2-135M model using Unsloth.ai.

## 🎯 Objectives
- Learn how to perform full fine-tuning (updating all model parameters)
- Understand the difference between full fine-tuning and LoRA
- Work with SmolLM2-135M, a compact 135 million parameter model
- Use Unsloth.ai for 2x faster training with 70% less memory
- Format datasets properly for instruction following tasks

## 📊 Model Details
- **Model**: SmolLM2-135M-Instruct
- **Parameters**: 135 million
- **Method**: Full fine-tuning (r=0, all parameters updated)
- **Task**: Instruction following / Chat completion
- **Dataset**: yahma/alpaca-cleaned (100 samples for quick training)

## 🔑 Key Concepts

### What is Full Fine-tuning?
Full fine-tuning updates **ALL** parameters of the model during training:
- More computationally expensive than LoRA
- Requires more GPU memory
- Can achieve better task-specific performance
- Updates all 135M parameters in this case

### Comparison with LoRA
| Aspect | Full Fine-tuning | LoRA |
|--------|-----------------|------|
| Parameters Updated | ALL (135M) | Few adapter weights (< 1M) |
| Memory Usage | Higher | Lower |
| Training Speed | Slower | Faster |
| Performance | Potentially better | Good with less resources |

## 📝 Notebook Structure

### 1. **Setup & Installation**
   - Install Unsloth and dependencies
   - Import required libraries

### 2. **Model Configuration**
   - Configure SmolLM2-135M parameters
   - Set max sequence length (512)
   - Enable 4-bit quantization for memory efficiency

### 3. **Load Pre-trained Model**
   - Load SmolLM2-135M-Instruct from Hugging Face
   - Initialize tokenizer

### 4. **Prepare for Full Fine-tuning**
   - Set `r=0` for full parameter updates (NOT LoRA)
   - Configure gradient checkpointing for memory optimization

### 5. **Dataset Preparation**
   - Load yahma/alpaca-cleaned dataset
   - Select 100 samples for quick training
   - Format using Alpaca prompt template

### 6. **Training Configuration**
   - Batch size: 2 per device
   - Max steps: 60 (quick training)
   - Learning rate: 2e-4
   - Optimizer: AdamW 8-bit

### 7. **Training Process**
   - Initialize SFTTrainer
   - Train model (updates all 135M parameters!)
   - Monitor training metrics

### 8. **Testing & Inference**
   - Test with sample prompts
   - Generate responses
   - Evaluate model performance

### 9. **Model Export**
   - Save fine-tuned model locally
   - Optional: Export to GGUF for Ollama
   - Optional: Push to Hugging Face Hub

## 🚀 Quick Start

### Prerequisites
- Google Colab account (free tier works!)
- GPU runtime (T4 recommended)
- Basic understanding of LLMs and fine-tuning

### Steps to Run
1. Open `full_finetuning_smollm2_135m.ipynb` in Google Colab
2. Enable GPU runtime: Runtime → Change runtime type → GPU (T4)
3. Run all cells sequentially
4. Wait for training to complete (~5-10 minutes)
5. Test the model with your own prompts!

## 📊 Dataset Format

The notebook uses the Alpaca instruction format:

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

### Example:
```json
{
  "instruction": "Explain what machine learning is in simple terms.",
  "input": "",
  "output": "Machine learning is a type of artificial intelligence that allows computers to learn from data and make predictions or decisions without being explicitly programmed..."
}
```

## 🎥 Video Walkthrough

**TODO**: Record a YouTube video covering:
1. Overview of full fine-tuning concept
2. Step-by-step code walkthrough
3. Explanation of each cell and its purpose
4. Input format and dataset structure
5. Training process and metrics
6. Testing and inference demonstrations
7. Output examples and analysis
8. Comparison with LoRA (teaser for Colab 2)

## 📈 Expected Results

After training, the model should be able to:
- Follow instructions accurately
- Generate coherent responses
- Handle various types of prompts (questions, tasks, summarization)
- Show improvement over the base model for instruction-following

## 💡 Key Takeaways

1. **Full Fine-tuning** updates all 135M parameters
2. **Unsloth** makes training 2x faster with less memory
3. **Small datasets** (100 samples) are enough for demonstration
4. **4-bit quantization** enables training on free Colab GPUs
5. **Proper formatting** is crucial for model performance

## 🔗 Resources

- **Unsloth Documentation**: https://docs.unsloth.ai/
- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **SmolLM2 Model**: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- **Alpaca Dataset**: https://huggingface.co/datasets/yahma/alpaca-cleaned
- **Fine-tuning Guide**: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide

## ⚠️ Important Notes

- This is **full fine-tuning**, not LoRA (see Colab 2 for LoRA)
- GPU runtime is required (free T4 on Colab works)
- Training time: ~5-10 minutes with 100 samples
- Model size after training: ~270MB (4-bit quantized)

## 🎓 Learning Outcomes

After completing this notebook, you will understand:
- ✅ How to set up Unsloth for fine-tuning
- ✅ The difference between full fine-tuning and LoRA
- ✅ How to prepare datasets for instruction following
- ✅ How to configure training parameters
- ✅ How to test and evaluate fine-tuned models
- ✅ How to save and export models

## 🔄 Next Steps

After completing this notebook, proceed to:
- **Colab 2**: LoRA parameter-efficient fine-tuning with the same model
- Compare performance and training time between full fine-tuning and LoRA
- Understand the trade-offs between different fine-tuning approaches

---

**Status**: ✅ Notebook created and ready for execution
**Last Updated**: November 9, 2025
