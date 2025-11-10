# Colab 5: Continued Pre-training for Hindi Language

## 🎯 Overview

This notebook demonstrates **Continued Pre-training** to teach SmolLM2-135M a new language (Hindi). Unlike fine-tuning which adapts models to tasks, continued pre-training expands the model's fundamental knowledge base, enabling it to understand and generate text in languages it wasn't originally trained on.

## 🌟 What is Continued Pre-training?

**Continued Pre-training** extends a model's knowledge by training on raw text using next-token prediction:
- **Adds new languages**: Teach models Hindi, Arabic, Chinese, etc.
- **Expands domains**: Medical, legal, technical knowledge
- **Updates knowledge**: Recent events, new research
- **Maintains existing capabilities**: Doesn't forget original knowledge (with proper technique)

## 🌍 Why This Matters

### The Language Gap
- Most AI models are English-centric
- Billions of people speak other languages
- Regional languages are underserved
- Cultural context is often lost

### Making AI Accessible
Continued pre-training enables:
- 🌏 **Regional language support**: Hindi, Tamil, Bengali, etc.
- 🏛️ **Government services**: Local language AI
- 📱 **Mobile apps**: Underserved markets
- 📚 **Education**: Learning in native languages
- 🎭 **Cultural preservation**: Endangered languages

## 🔬 Method Comparison

| Aspect | Fine-tuning (Colab 1-4) | Continued Pre-training (Colab 5) |
|--------|-------------------------|----------------------------------|
| **Purpose** | Task adaptation | Knowledge expansion |
| **Input** | Instruction-response pairs | Raw text |
| **Training** | Supervised learning | Next-token prediction |
| **Data Size** | 100s-1000s examples | Millions of tokens |
| **Goal** | Learn task | Learn language/domain |
| **Modules** | Query, key, value | Embeddings + all layers |
| **Use Case** | After pre-training | Extend base knowledge |

## 📚 Dataset Design

### Format
Raw text with bilingual content to prevent catastrophic forgetting:

```python
{
    "text": "Hindi sentence. (English translation.) More Hindi..."
}
```

### Example
```python
{
    "text": "नमस्ते, मैं एक छात्र हूं। (Hello, I am a student.) मुझे हिंदी सीखना पसंद है। (I like learning Hindi.)"
}
```

### Why Bilingual?
- ✅ **Prevents forgetting**: Maintains English capability
- ✅ **Translation bridges**: Learns language connections
- ✅ **Efficient learning**: Leverages existing knowledge
- ✅ **Better alignment**: Maps concepts across languages
- ✅ **Code-switching**: Handles mixed-language input

### Dataset Coverage
Our 120 examples include:
- Basic greetings and introductions
- Numbers and counting (1-100)
- Daily activities and routines
- Family and relationships
- Food and cultural terms
- Common phrases and questions
- Colors, days, months
- Simple conversations

## 🚀 Quick Start

### Prerequisites
- Google Colab account (free tier works!)
- GPU runtime (T4 recommended)
- ~12-15 minutes for training

### Steps to Run

1. **Upload to Google Colab**
   ```
   - Go to https://colab.research.google.com/
   - File → Upload notebook
   - Select `continued_pretraining_hindi_smollm2.ipynb`
   ```

2. **Enable GPU**
   ```
   - Runtime → Change runtime type
   - Hardware accelerator: T4 GPU
   - Save
   ```

3. **Run All Cells**
   ```
   - Runtime → Run all (Ctrl/Cmd + F9)
   - Wait ~12-15 minutes
   - Watch Hindi learning happen!
   ```

4. **Test Bilingual Capability**
   ```
   - Try Hindi prompts
   - Test English→Hindi
   - Experiment with code-switching
   ```

## ⚙️ Configuration

### Model Configuration
```python
model_name = "unsloth/SmolLM2-135M-Instruct"
max_seq_length = 512
load_in_4bit = True
```

### LoRA Configuration (Critical!)
```python
r = 16
lora_alpha = 16
lora_dropout = 0
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
modules_to_save = ["embed_tokens", "lm_head"]  # CRITICAL!
```

### Why `modules_to_save`?

**embed_tokens** (Input Embeddings):
- Converts tokens → vectors
- Must adapt to Hindi tokens
- If frozen, can't understand Hindi
- **Must be trainable!**

**lm_head** (Output Layer):
- Predicts next tokens
- Must generate Hindi tokens
- If frozen, can't produce Hindi
- **Must be trainable!**

Without these, language learning fails!

### Training Arguments
```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4           # Higher than fine-tuning
max_steps = 100                # More steps for language
warmup_steps = 10
```

**Why more steps?**
- Language learning needs more exposure
- Need to see patterns repeatedly
- Building fundamental knowledge, not task adaptation

## 📈 Expected Results

### Training Metrics
- **Initial Loss**: ~3.0-3.5 (model doesn't know Hindi)
- **Final Loss**: ~1.5-2.0 (learning patterns)
- **Training Time**: 12-15 minutes on T4 GPU
- **Trainable Params**: ~4M LoRA + embeddings + lm_head

### Language Capability

**Before Continued Pre-training**:
```
Input: "नमस्ते"
Output: [gibberish or English fallback]
```

**After Continued Pre-training**:
```
Input: "नमस्ते, मैं"
Output: "एक छात्र हूं" (I am a student)
```

### What to Expect
- ✅ Generate basic Hindi sentences
- ✅ Understand common Hindi phrases
- ✅ Translate simple English→Hindi
- ✅ Handle bilingual/code-switched input
- ⚠️ Not fluent (needs more data)
- ⚠️ Limited vocabulary (120 examples)

## 🧪 Testing Examples

### Test 1: Hindi Generation
```python
prompt = "नमस्ते, मैं"
```
**Expected**: Continues in Hindi (e.g., "एक छात्र हूं")

### Test 2: English to Hindi
```python
prompt = "Translate to Hindi: Hello, how are you?"
```
**Expected**: "नमस्ते, आप कैसे हैं?"

### Test 3: Code-Switching
```python
prompt = "Hello, मैं"
```
**Expected**: Handles mixed language appropriately

### Test 4: Hindi Question
```python
prompt = "आप कैसे हैं?"
```
**Expected**: Appropriate Hindi response

## 📊 Performance Assessment

| Capability | Before | After (100 steps) | Production (1M steps) |
|------------|--------|-------------------|----------------------|
| Hindi Generation | 0% | 40-60% | 90%+ |
| Hindi Understanding | 0% | 50-70% | 95%+ |
| Translation | 0% | 30-50% | 85%+ |
| Code-Switching | 0% | 40-60% | 90%+ |
| English Retention | 100% | 85-95% | 95%+ |

## 💾 Model Saving Options

### 1. Save LoRA Adapters + Special Modules
```python
model.save_pretrained("smollm2_hindi")
tokenizer.save_pretrained("smollm2_hindi")
```
- **Size**: ~20-30MB (includes embeddings)
- **Use**: Load with base model

### 2. Save Merged Bilingual Model
```python
model.save_pretrained_merged(
    "smollm2_hindi_merged",
    tokenizer,
    save_method="merged_16bit"
)
```
- **Size**: ~270MB
- **Use**: Standalone bilingual model

### 3. Export to GGUF
```python
model.save_pretrained_gguf(
    "smollm2_hindi",
    tokenizer,
    quantization_method="q4_k_m"
)
```
- **Size**: ~80-100MB
- **Use**: Ollama for local bilingual inference

## 🎯 When to Use Continued Pre-training

### ✅ Use Continued Pre-training When:
- Teaching new languages
- Adding domain knowledge (medical, legal)
- Updating with recent information
- Adapting to specialized vocabulary
- Creating multilingual models
- Supporting underserved languages

### ❌ Don't Use Continued Pre-training When:
- Just need task adaptation (use fine-tuning)
- Teaching instruction-following (use SFT)
- Aligning preferences (use DPO)
- Adding reasoning (use GRPO)
- Have limited data (<100K tokens)

## 🏗️ Production Scaling

### Data Requirements

**Proof of Concept** (this notebook):
- Size: 120 bilingual examples (~20K tokens)
- Time: 12-15 minutes
- Result: Basic capability demonstration

**Research Quality**:
- Size: 10-100 million tokens
- Time: 10-100 GPU hours
- Result: Functional bilingual model

**Production Quality**:
- Size: 1-10 billion tokens
- Time: 1000+ GPU hours
- Result: Fluent multilingual model

### Data Sources for Hindi
- **OSCAR**: Large-scale web crawl
- **Wikipedia**: High-quality Hindi content
- **News corpora**: Current, clean text
- **Books**: Literary Hindi
- **Subtitles**: Conversational Hindi
- **Government docs**: Formal Hindi

### Training at Scale
```python
# Production configuration
max_steps = 100000              # Much longer
per_device_train_batch_size = 8 # Larger batches
gradient_accumulation_steps = 8 # Effective batch = 64
learning_rate = 1e-4            # May need tuning
eval_steps = 1000               # Evaluate periodically
```

## 🔍 Preventing Catastrophic Forgetting

### The Problem
When learning Hindi, model might forget English!

### Our Solutions

1. **Bilingual Training Data**
   - Mix Hindi and English in each example
   - Maintains both languages

2. **Translation Pairs**
   - Shows language equivalence
   - Reinforces both languages

3. **Balanced Curriculum**
   - Don't overwhelm with only Hindi
   - Keep English exposure

4. **Regular Evaluation**
   - Test English capability
   - Stop if degradation detected

### Monitoring
```python
# Check English retention
english_prompts = ["What is AI?", "Hello, how are you?"]
# Check Hindi learning
hindi_prompts = ["नमस्ते", "आप कैसे हैं?"]
```

## 🌍 Real-World Applications

### Language Accessibility
- 🇮🇳 **India**: 22 official languages
- 🇮🇩 **Indonesia**: Bahasa Indonesia
- 🇧🇷 **Brazil**: Portuguese variants
- 🌍 **Africa**: 2000+ languages

### Domain Adaptation
- 🏥 **Medical**: Clinical terminology
- ⚖️ **Legal**: Jurisdiction-specific law
- 💼 **Finance**: Industry jargon
- 🔬 **Science**: Research terminology

### Cultural Context
- 🎭 **Idioms**: Culture-specific expressions
- 📚 **Literature**: Classic texts
- 🗣️ **Dialects**: Regional variations
- 🏛️ **History**: Cultural knowledge

## 🐛 Troubleshooting

### Issue: Model outputs gibberish
**Solutions**:
- Check if `modules_to_save` includes `embed_tokens` and `lm_head`
- Increase training steps to 200
- Check data quality (proper Hindi encoding)
- Verify tokenizer handles Hindi

### Issue: Forgetting English
**Solutions**:
- Increase English proportion in data
- Add more bilingual examples
- Reduce learning rate to 1e-4
- Monitor English performance

### Issue: Slow training
**Solutions**:
- Normal! Language learning takes time
- Training embeddings + lm_head is slower
- Use larger GPU (V100/A100)
- Reduce max_seq_length to 384

### Issue: Poor Hindi quality
**Solutions**:
- More training data (need 1000s examples)
- More training steps (try 500+)
- Better quality Hindi corpus
- Native speaker data validation

## 📚 References

### Papers
- [BLOOM](https://arxiv.org/abs/2211.05100) - 46-language model
- [mT5](https://arxiv.org/abs/2010.11934) - Multilingual T5
- [XLM-R](https://arxiv.org/abs/1911.02116) - Cross-lingual model

### Multilingual Datasets
- **OSCAR**: Web crawl (many languages)
- **mC4**: Multilingual C4
- **CC100**: 100+ language corpus
- **Wikipedia**: High-quality multilingual

### Tools & Resources
- [IndicNLP](https://indicnlp.ai4bharat.org/) - Indian language processing
- [AI4Bharat](https://ai4bharat.org/) - Indian language models
- [Samanantar](https://indicnlp.ai4bharat.org/samanantar/) - Hindi-English parallel corpus

## 🎓 Learning Outcomes

After completing this notebook, you will understand:

1. **Pre-training vs Fine-tuning**
   - Fundamental knowledge vs task adaptation
   - When to use each approach
   - How they complement each other

2. **Language Learning for AI**
   - Next-token prediction for languages
   - Importance of embeddings and output layers
   - Bilingual training strategies

3. **Practical Implementation**
   - Critical configuration (`modules_to_save`)
   - Preventing catastrophic forgetting
   - Dataset design for languages

4. **Real-World Impact**
   - Making AI accessible globally
   - Supporting underserved languages
   - Cultural preservation through AI

## 🌟 Key Takeaways

1. **modules_to_save is critical**: Must include `embed_tokens` and `lm_head`
2. **Bilingual prevents forgetting**: Mix languages in training data
3. **More data needed**: 120 examples prove concept, need millions for fluency
4. **Next-token prediction**: Different from instruction fine-tuning
5. **Global accessibility**: This makes AI work for everyone

## 🔗 Related Notebooks

- **[Colab 1](../Collab%201/)**: Full fine-tuning - task learning
- **[Colab 2](../Collab%202/)**: LoRA fine-tuning - efficient task learning
- **[Colab 3](../Collab%203/)**: DPO - preference alignment
- **[Colab 4](../Collab%204/)**: GRPO - reasoning models

## 📖 Complete Series Pipeline

```
1. Continued Pre-training (Colab 5) → Learn Hindi
2. SFT (Colab 1-2)                 → Learn tasks in Hindi
3. DPO (Colab 3)                   → Align Hindi preferences
4. GRPO (Colab 4)                  → Add reasoning in Hindi
5. Deploy                          → Bilingual AI system!
```

## 💬 Support

Questions about multilingual AI? Want to adapt for other languages? Open an issue!

---

**Status**: ✅ Ready to Run  
**Estimated Time**: 12-15 minutes  
**Difficulty**: Intermediate to Advanced  
**GPU Required**: Yes (T4 or better)  
**Last Updated**: November 9, 2025

**Mission**: Making AI accessible to the 1.3 billion Hindi speakers! 🇮🇳🌍
