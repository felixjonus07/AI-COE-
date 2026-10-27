"""
Task 3: Transformer-Based Next-Token Prediction Model Using Basic Self-Attention
Dataset: Kaggle Million Headlines Dataset (therohk/million-headlines)
Framework: PyTorch
"""

import os
import glob
import math
import kagglehub
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------------------
# 1. Basic Causal Self-Attention Layer
# -------------------------------------------------------------
class BasicSelfAttention(nn.Module):
    """
    Scaled Dot-Product Self-Attention with Causal (Autoregressive) Masking.
    Implements: Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) + Mask ) * V
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Linear projections for Query (Q), Key (K), and Value (V)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        # 1. Project inputs to Q, K, V representations
        Q = self.query_proj(x)  # (B, T, d_model)
        K = self.key_proj(x)    # (B, T, d_model)
        V = self.value_proj(x)  # (B, T, d_model)

        # 2. Scaled Dot-Product: (Q @ K^T) / sqrt(d_k)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)

        # 3. Causal Mask: Prevent future tokens from being visible (autoregressive)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # 4. Softmax to obtain attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # 5. Weighted sum with Value vectors
        context = torch.bmm(attention_weights, V)

        # 6. Final linear projection
        return self.out_proj(context)


# -------------------------------------------------------------
# 2. Transformer Decoder Block
# -------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Standard Transformer Decoder Layer:
    - Pre-LayerNorm + Self-Attention + Residual Connection
    - Pre-LayerNorm + Feed-Forward Network (FFN) + Residual Connection
    """
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.attn = BasicSelfAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# -------------------------------------------------------------
# 3. Next-Token Prediction Language Model
# -------------------------------------------------------------
class TransformerNextTokenLM(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=64, num_layers=2):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Token embedding and learned positional embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Stack of Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, hidden_dim=d_model * 4)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=8):
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            context = generated[:, -self.max_seq_len:]
            logits = self.forward(context)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token_id), dim=1)

        return generated


# -------------------------------------------------------------
# 4. Load Kaggle Text Dataset & Tokenization
# -------------------------------------------------------------
def load_kaggle_corpus(num_headlines=150):
    print("=" * 65)
    print("Task 3: Transformer Next-Token Prediction (Kaggle Dataset)")
    print("=" * 65)
    print("Downloading/fetching Kaggle Million Headlines dataset...")
    path = kagglehub.dataset_download("therohk/million-headlines")
    csv_file = glob.glob(os.path.join(path, "*.csv"))[0]

    df = pd.read_csv(csv_file, nrows=num_headlines)
    headlines = df["headline_text"].dropna().str.lower().tolist()

    # Join headlines with period delimiters
    corpus = " . ".join(headlines) + " ."
    print(f"Loaded {len(headlines)} headlines from Kaggle: {os.path.basename(csv_file)}")
    return corpus


# -------------------------------------------------------------
# 5. Training & Generation Routine
# -------------------------------------------------------------
def train_and_generate():
    corpus = load_kaggle_corpus(num_headlines=100)

    # Simple word-level tokenizer
    tokens = corpus.split()
    vocab = sorted(list(set(tokens)))
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    print(f"Vocabulary Size: {vocab_size} unique words | Total Tokens: {len(tokens)}")

    # Convert corpus to sequence pairs (input -> target shifted by 1)
    token_ids = [word2id[w] for w in tokens]
    seq_length = 8
    inputs, targets = [], []
    for i in range(len(token_ids) - seq_length):
        inputs.append(token_ids[i : i + seq_length])
        targets.append(token_ids[i + 1 : i + seq_length + 1])

    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNextTokenLM(
        vocab_size=vocab_size, max_seq_len=32, d_model=64, num_layers=2
    ).to(device)

    print("-" * 65)
    print(model)
    print("-" * 65)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003)

    X_dev = X.to(device)
    Y_dev = Y.to(device)

    # Training loop
    epochs = 80
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(X_dev)
        loss = criterion(logits.view(-1, vocab_size), Y_dev.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch:2d}/{epochs}] - Loss: {loss.item():.4f}")

    # Autoregressive Generation Demonstration
    print("-" * 65)
    print("Kaggle Headline Next-Token Autoregressive Generation:")
    sample_prompts = [
        ["police"],
        ["council"],
        ["new"],
        ["fire"]
    ]

    for p in sample_prompts:
        # Verify prompt word exists in vocab
        valid_words = [w for w in p if w in word2id]
        if not valid_words:
            continue
        prompt_ids = torch.tensor([[word2id[w] for w in valid_words]], device=device)
        output_ids = model.generate(prompt_ids, max_new_tokens=6)
        output_words = [id2word[idx.item()] for idx in output_ids[0]]
        print(f"  Prompt   : \"{' '.join(valid_words)}\"")
        print(f"  Generated: \"{' '.join(output_words)}\"")
        print()
    print("=" * 65)


if __name__ == "__main__":
    train_and_generate()
