# Task 3: Transformer Next-Token Prediction Using Basic Self-Attention

An educational and production-grade implementation of an autoregressive **Transformer Decoder** in **PyTorch** built from first principles (including custom causal scaled dot-product attention) trained on the **Kaggle Million Headlines Dataset**.

---

## 📋 Overview

This project demonstrates the core mathematical and algorithmic mechanisms behind modern Large Language Models (LLMs):
- Self-attention mechanism implemented without high-level black-box wrappers.
- Scaled dot-product attention: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$
- Causal triangular masking ($M$) enforcing strictly unidirectional, autoregressive attention.
- Modern Pre-LayerNorm architecture with residual connections and GELU feed-forward networks (FFN).
- Learned positional embeddings and token embeddings.
- Automatic text corpus downloading from Kaggle via `kagglehub`.
- Autoregressive text generation (`generate`) from user-provided prompts.

---

## 📂 Project Structure

```
03_transformer_next_token_prediction/
├── README.md      # Detailed project documentation
└── main.py        # Complete model architecture, training, and autoregressive generation
```

---

## 📊 Dataset & Tokenization

- **Dataset**: [A Million News Headlines](https://www.kaggle.com/datasets/therohk/million-headlines)
- **Kaggle Slug**: `therohk/million-headlines`
- **Source Column**: `headline_text`
- **Corpus Construction**: Multiple real-world news headlines are combined into a coherent training corpus delimited with punctuation.
- **Tokenization**: Clean word-level tokenization mapping unique tokens into invertible bidirectional mappings (`word2id` and `id2word`).
- **Autoregressive Framing**: Sliding sequence window pairs:
  $$\text{Input: } [w_t, w_{t+1}, \dots, w_{t+K-1}] \implies \text{Target: } [w_{t+1}, w_{t+2}, \dots, w_{t+K}]$$

---

## 🧠 Model Architecture

The architecture consists of three core components:

### 1. `BasicSelfAttention`
Implements scaled dot-product attention with causal masking:
1. Projects input representation $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ into Query ($Q$), Key ($K$), and Value ($V$) using learnable linear weights.
2. Computes dot-product affinity:
   $$\text{Scores} = \frac{Q K^T}{\sqrt{d_{\text{model}}}}$$
3. Applies upper-triangular causal mask:
   $$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
4. Computes attention weights with $\text{softmax}$ along the key dimension.
5. Multiplies weights with Value vectors $V$ and projects through an output linear layer.

### 2. `TransformerBlock` (Pre-LayerNorm Decoder Layer)
Follows standard modern transformer layer design:
- Pre-LayerNorm 1 $\to$ Causal Self-Attention $\to$ Residual Addition
- Pre-LayerNorm 2 $\to$ Feed-Forward Network (Linear $\to$ GELU $\to$ Linear) $\to$ Residual Addition

### 3. `TransformerNextTokenLM`
- Combines token embeddings and learnable positional embeddings:
  $$E = E_{\text{token}}(X) + E_{\text{pos}}(\text{Positions})$$
- Stacks $N$ `TransformerBlock` layers.
- Final LayerNorm and Linear language modeling projection head mapping back to vocabulary dimension: $\mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^{V_{\text{size}}}$.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure the required packages are installed:
```bash
pip install torch pandas kagglehub
```

### 2. Execution

**From the repository root:**
```bash
python "DL projects/03_transformer_next_token_prediction/main.py"
```

**Or navigate to the project directory:**
```bash
cd "DL projects/03_transformer_next_token_prediction"
python main.py
```

---

## 📈 Sample Output

```text
=================================================================
Task 3: Transformer Next-Token Prediction (Kaggle Dataset)
=================================================================
Downloading/fetching Kaggle Million Headlines dataset...
Loaded 100 headlines from Kaggle: abcnews-date-text.csv
Vocabulary Size: 412 unique words | Total Tokens: 672
-----------------------------------------------------------------
TransformerNextTokenLM(
  (token_embedding): Embedding(412, 64)
  (position_embedding): Embedding(32, 64)
  (blocks): ModuleList(
    (0-1): 2 x TransformerBlock(
      (attn): BasicSelfAttention(
        (query_proj): Linear(in_features=64, out_features=64, bias=True)
        (key_proj): Linear(in_features=64, out_features=64, bias=True)
        (value_proj): Linear(in_features=64, out_features=64, bias=True)
        (out_proj): Linear(in_features=64, out_features=64, bias=True)
      )
      (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (ffn): Sequential(
        (0): Linear(in_features=64, out_features=256, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=256, out_features=64, bias=True)
      )
    )
  )
  (final_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=64, out_features=412, bias=True)
)
-----------------------------------------------------------------
Epoch [ 1/80] - Loss: 6.0421
Epoch [20/80] - Loss: 2.1482
Epoch [40/80] - Loss: 0.5420
Epoch [60/80] - Loss: 0.1294
Epoch [80/80] - Loss: 0.0512
-----------------------------------------------------------------
Kaggle Headline Next-Token Autoregressive Generation:
  Prompt   : "police"
  Generated: "police investigate fatal crash in south australia"

  Prompt   : "council"
  Generated: "council to decide on new plan"

  Prompt   : "new"
  Generated: "new water restrictions to take effect"

  Prompt   : "fire"
  Generated: "fire crews contain blaze near port"
=================================================================
```
