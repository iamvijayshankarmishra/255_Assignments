# When Tables Finally Started Making Sense: A Deep Dive into LLMs and Data Querying

**Course**: CMPE 255: Data Mining  
**Student**: Vijayshankar Mishra  
**Semester**: Fall 2025

---

## 📍 Abstract

In the domain of Data Mining, the gap between unstructured human intent and structured data storage (SQL databases) has historically been a significant bottleneck. This project, formatted as a "Short Story" survey, explores the paradigm shift brought by Large Language Models (LLMs) in solving the Text-to-SQL problem. We examine how modern architectures utilize Retrieval Augmented Generation (RAG) and Schema Linking to function as semantic compilers, effectively democratizing data access. The study highlights the transition from fragile rule-based systems to reasoning-heavy LLM pipelines and analyzes the performance disparity between academic benchmarks (Spider 1.0) and real-world enterprise scenarios (Spider 2.0).

---

## 📖 Introduction

Data Mining begins with data retrieval. However, the vast majority of organizational knowledge is locked in structured databases that require specialized knowledge of SQL (Structured Query Language) and complex schema architectures. This creates a reliance on technical intermediates (data analysts) to translate business questions into executable queries.

This project investigates the application of Generative AI to bridge this disconnect. By treating SQL generation as a language translation task, LLMs can theoretically enable any user to "mine" data using natural language. We explore the architectural evolution of these systems, focusing on how they handle the unique challenges of structured data: ambiguity, scale, and syntactic correctness.

---

## 🏗️ Architecture & Methodology

The transition from standard NLP to Text-to-SQL involves a specialized pipeline. This project surveys the following core architectural components:

### 1. The Context-Aware Pipeline

Unlike standard chat-bots, a Data Mining agent must be grounded in the specific database structure.

- **Schema Injection**: The LLM prompt is augmented with the Data Definition Language (DDL), including table names, column names, and data types.
- **RAG (Retrieval Augmented Generation)**: For large-scale databases (thousands of tables), relevant schema elements are retrieved dynamically based on the user's query to fit within the model's context window.

### 2. Schema Linking

This is the critical reasoning step where the model maps natural language tokens (e.g., "High Value Customer") to specific database entities (e.g., `SELECT * FROM orders WHERE total > 1000`).

- **Challenge**: Handling synonyms and ambiguous business logic without explicit instructions.
- **Solution**: Utilizing Chain-of-Thought (CoT) prompting to force the model to explain its linking logic before generating code.

---

## 📊 Evaluation & Ablation Studies

We analyzed recent benchmarks to understand what factors actually drive performance in Text-to-SQL systems.

### Key Metrics

- **Execution Accuracy (EX)**: Does the generated SQL return the correct data?
- **Exact Set Match (EM)**: Does the predicted SQL structure match the ground truth query?

### The "Spider" Reality Check

The study compares performance across two primary benchmarks:

- **Spider 1.0 (Academic)**: LLMs achieve >90% accuracy on standard, clean databases.
- **Spider 2.0 (Enterprise)**: Accuracy drops to ~10-15% when facing messy, real-world schemas with thousands of columns and lack of clean metadata.

### Ablation Findings

Research indicates that model size is less important than context quality:

- **Foreign Keys**: Explicitly providing FK constraints improves complex JOIN accuracy by ~15%.
- **Reasoning Steps**: Models prompted to "think step-by-step" significantly outperform those that output raw SQL directly.

---

## 🔗 Project Deliverables

| Artifact | Description | Link |
|----------|-------------|------|
| **Medium Article** | A narrative deep-dive into the technology and its implications for Data Mining. | [Read on Medium](https://medium.com/@vijayshankar.mishra/when-tables-finally-started-making-sense-my-deep-dive-into-llms-and-data-querying-3e8cf439b623?postPublishedType=initial) |
| **Presentation Slides** | A visual deck summarizing the architecture, challenges, and benchmarks. | [View Slides]([presentation/slides.pdf](https://github.com/iamvijayshankarmishra/255_Assignments/blob/main/Short%20Story/When%20Tables%20Finally%20Started%20Making%20Sense.pptx)) |
| **Video Presentation** | A 10-minute walkthrough of the concepts and slides. | [Watch on YouTube](https://www.youtube.com/watch?v=KWIr309PEvg) |


---

## 📚 References

The insights in this project are derived from the following key survey papers and benchmarks:

1. **Spider 2.0**: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows (arXiv 2024).
2. **A Survey on Text-to-SQL**: The Impact of Large Language Models on Structured Data Querying (arXiv 2023).
3. **BIRD**: Can LLM Already Serve as A Database Interface? A BIG Bench for Large-Scale Database Grounded Text-to-SQLs (NeurIPS 2024).
4. **Zhou et al. (2025)**: Table Question Answering in the Era of Large Language Models: A Comprehensive Survey of Tasks, Methods, and Evaluation. *arXiv:2510.09671*

---


