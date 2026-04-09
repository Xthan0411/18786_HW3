# Generative AI & Transformer Optimization: From Scratch to LLM Prompt Engineering
# 生成式 AI 与 Transformer 优化：从底层构建到大模型提示词工程

🌍 **[English](#english-version)** | 🇨🇳 **[中文版本](#中文版本)**

---

## English Version

This repository contains a comprehensive exploration of Large Language Models (LLMs) and Generative AI, ranging from implementing the foundational components of the Transformer architecture from scratch to deploying and optimizing modern open-source LLMs (e.g., Qwen3-1.7B). 

The project demonstrates a full-stack AI engineering capability, covering model pretraining, fine-tuning, positional encoding algorithmic optimizations, and Prompt Engineering for complex instruction-tuned models.

### 🚀 Key Features & Engineering Highlights

* **Transformer Core from Scratch:** Built custom Multi-Head Self-Attention, Feed-Forward Networks, and Encoder-Decoder blocks using native PyTorch operations, demonstrating a deep understanding of mathematical underpinnings.
* **Rotary Position Embedding (RoPE):** Implemented complex-valued rotation transformations for positional encoding. Shifted the model's spatial perception from absolute to relative positioning, significantly boosting spatial relationship awareness.
* **Pretraining vs. Fine-Tuning Paradigm:** Engineered a span corruption function for self-supervised pretraining on Wikipedia text, followed by supervised fine-tuning on a specialized factual dataset (Name-Birthplace mapping).
* **Modern LLM Deployment & Prompt Optimization:** Deployed the instruction-tuned `Qwen3-1.7B` model locally via the Hugging Face ecosystem. Designed automated pipelines to extract reasoning paths and systematically optimized prompt templates (Prompt Engineering) to mitigate hallucinations and constrain generative outputs.

### 📊 Quantitative Results & Benchmarking

The efficacy of our implementation and optimizations was rigorously benchmarked on the factual retrieval validation set.

| Model / Methodology | Accuracy | Notes |
| :--- | :---: | :--- |
| **Random Initialization (No Pretraining)** | 0.8% | Baseline without learned representations. |
| **London Baseline (Heuristic)** | 5.0% | Predicts the majority class ("London"). |
| **Transformer (Absolute Pos Embed) + Pretrain** | 24.6% | Demonstrates the critical necessity of the pretraining paradigm. |
| **Transformer (RoPE) + Pretrain** | **38.0%** | Massive improvement by encoding relative positional distances. |
| **Qwen3-1.7B (Zero-Shot + Prompt Eng)** | ~11.2% | Achieved through systematic prompt rewriting and constraint parsing. |

### 🛠️ Tech Stack & Dependencies

* **Frameworks:** PyTorch, Hugging Face `transformers`
* **Languages:** Python, Bash Shell
* **Core Concepts:** Attention Mechanisms, RoPE, Span Corruption Pretraining, Instruction-Tuning, Prompt Sensitivity Analysis.

### 📂 Repository Structure

```text
├── data/               # Text corpus and factual datasets
├── src/                # Core neural network modules and logic
│   ├── attention.py    # Multi-head attention & RoPE implementation
│   ├── models.py       # Network architecture definition
│   ├── modern_llm.py   # Qwen3 API and Hugging Face integration
│   ├── sensitivity.py  # Prompt sensitivity analysis scripts
│   └── trainer.py      # Training loops and evaluation logic
├── scripts/            # Automated bash scripts for training pipelines
├── docs/               # Detailed technical report and mathematical proofs
└── README.md
```


*(Note: Complete experiment logs, TensorBoard events, and model weights `.params` are excluded from this repository due to storage constraints. Please refer to `docs/18786_hw3_report.pdf` for exhaustive analytical breakdowns and mathematical derivations.)*

---

## 中文版本

本项目涵盖了对大语言模型 (LLM) 与生成式 AI 的系统性探索，包括从零使用 PyTorch 纯算子构建 Transformer 底层架构，以及针对现代开源大模型（如 Qwen3-1.7B）的本地部署与推理优化。

项目展现了完整的 AI 工程化能力，覆盖了模型的预训练 (Pretraining)、微调 (Fine-tuning)、位置编码算法优化，以及针对指令微调大模型的系统性提示词工程 (Prompt Engineering)。

### 🚀 核心特性与工程亮点

* **从零构建 Transformer 底层算子：** 纯手工使用 PyTorch 实现多头自注意力机制 (Multi-Head Self-Attention)、前馈神经网络及编解码器模块，扎实掌握前沿大模型的底层数学逻辑与生成机制。
* **旋转位置编码 (RoPE) 算法实现：** 创新性地引入复数空间旋转变换作为位置编码。将模型的空间感知能力从“绝对位置”升级为“相对位置”，极大提升了模型对长文本中实体关系的捕捉精度。
* **预训练与微调工程范式：** 开发基于 Span Corruption（片段掩码）的自监督预训练 Pipeline（维基百科语料），随后在特定事实数据集（人名-出生地映射）上完成监督微调。
* **现代大模型部署与 Prompt 调优：** 基于 Hugging Face 生态完成指令微调模型 `Qwen3-1.7B` 的本地部署。设计自动化脚本提取模型推理路径（Reasoning paths），并系统性调优 Prompt 模板，有效抑制了模型幻觉并规范了结构化输出。

### 📊 性能评测与结果分析

所有模型及优化策略均在事实检索验证集上进行了严谨的量化评估：

| 模型与技术方案 | 准确率 (Accuracy) | 结果分析 |
| :--- | :---: | :--- |
| **随机初始化 (无预训练)** | 0.8% | 纯白盒状态下的基线性能。 |
| **London Baseline (统计启发式)** | 5.0% | 强制预测数据集中出现频率最高的类别（伦敦）。 |
| **Transformer (绝对位置编码) + 预训练** | 24.6% | 证明了大规模自监督预训练范式对事实记忆的决定性作用。 |
| **Transformer (RoPE 旋转位置编码) + 预训练** | **38.0%** | 通过编码相对位置距离，模型对实体关系的感知能力得到巨大飞跃。 |
| **Qwen3-1.7B (零样本 + 提示词工程)** | ~11.2% | 通过系统性重写 Prompt 并结合后处理约束解析获得。 |

### 🛠️ 技术栈

* **深度学习框架：** PyTorch, Hugging Face `transformers`
* **编程语言：** Python, Bash Shell
* **核心技术：** 注意力机制, RoPE, 自监督预训练, 指令微调, 提示词敏感度评估与工程化处理。


*(注：为保证工程架构的简洁性，海量实验日志、TensorBoard 记录及数十 GB 的模型权重文件已通过 `.gitignore` 屏蔽。详尽的算法推理过程与置换等变性数学推导，请参阅 `docs/18786_hw3_report.pdf` 技术报告。)*
