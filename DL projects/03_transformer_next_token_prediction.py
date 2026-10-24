"""
Task 3: Transformer-Based Next-Token Prediction Model Using Basic Self-Attention
Framework: PyTorch
"""

import math
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
        self.key_proj = nn.Linear(d_model, d_model)
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
        # Result shape: (batch_size, seq_len, seq_len)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)

        # 3. Causal Mask: Prevent future tokens from being visible (autoregressive property)
        # Upper triangular entries above diagonal become -infinity
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # 4. Softmax to obtain attention weights (probabilities)
        attention_weights = torch.softmax(scores, dim=-1)

        # 5. Weighted sum with Value vectors
        context = torch.bmm(attention_weights, V)  # (B, T, d_model)

        # 6. Final linear transformation
        return self.out_proj(context)


# -------------------------------------------------------------
# 2. Transformer Decoder Block
# -------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Standard Transformer Decoder Layer consisting of:
    - Pre-LayerNorm + Self-Attention + Residual Connection
    - Pre-LayerNorm + Feed-Forward Network + Residual Connection
    """
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.attn = BasicSelfAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        # Residual connection around self-attention
        x = x + self.attn(self.norm1(x))
        # Residual connection around feed-forward network
        x = x + self.ffn(self.norm2(x))
        return x


# -------------------------------------------------------------
# 3. Next-Token Prediction Language Model
# -------------------------------------------------------------
class TransformerNextTokenLM(nn.Module):
    def __init__(self, vocab_size, max_seq_len=64, d_model=64, num_layers=2):
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
        # Head to map back to vocabulary logits
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # input_ids shape: (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape

        # Positional indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

        # Sum token embeddings and positional embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # Shape: (batch_size, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=10):
        """
        Autoregressive next-token generation (greedy decoding).
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to maximum supported sequence length
            context = generated[:, -self.max_seq_len:]
            logits = self.forward(context)

            # Focus only on the prediction for the last position
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append predicted token to sequence
            generated = torch.cat((generated, next_token_id), dim=1)

        return generated


# -------------------------------------------------------------
# 4. Tokenizer & Dataset Preparation
# -------------------------------------------------------------
corpus = (
    "artificial intelligence and neural networks are transforming technology . "
    "deep learning models learn patterns from data . "
    "language models predict the next token in a sequence . "
    "transformers use self attention to model relationships between words ."
)

# Simple word-level tokenizer
tokens = corpus.split()
vocab = sorted(list(set(tokens)))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# Convert whole corpus to token IDs
token_ids = [word2id[w] for w in tokens]

# Create next-token prediction training pairs:
# Input sequence (x) and Target sequence shifted by 1 token (y)
seq_length = 8
inputs, targets = [], []
for i in range(len(token_ids) - seq_length):
    inputs.append(token_ids[i : i + seq_length])
    targets.append(token_ids[i + 1 : i + seq_length + 1])

X = torch.tensor(inputs, dtype=torch.long)
Y = torch.tensor(targets, dtype=torch.long)


# -------------------------------------------------------------
# 5. Model Training & Generation Demo
# -------------------------------------------------------------
def run_transformer():
    print("=" * 60)
    print("Task 3: Transformer-Based Next-Token Prediction Model")
    print("=" * 60)
    print(f"Vocabulary Size: {vocab_size} unique words")
    print(f"Corpus token count: {len(token_ids)} | Training samples: {len(X)}")
    print("-" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNextTokenLM(
        vocab_size=vocab_size, max_seq_len=32, d_model=64, num_layers=2
    ).to(device)

    print(model)
    print("-" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003)

    X_dev = X.to(device)
    Y_dev = Y.to(device)

    # Training loop
    epochs = 120
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass: shape (B, T, vocab_size)
        logits = model(X_dev)

        # Reshape to (B*T, vocab_size) and (B*T) for cross-entropy
        loss = criterion(logits.view(-1, vocab_size), Y_dev.view(-1))

        # Backward pass & update
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{epochs}] - Loss: {loss.item():.4f}")

    # ---------------------------------------------------------
    # 6. Autoregressive Text Generation Test
    # ---------------------------------------------------------
    print("-" * 60)
    test_prompts = [
        ["language", "models"],
        ["transformers", "use"],
        ["deep", "learning"]
    ]

    print("Next-Token Generation Demo:")
    for prompt_words in test_prompts:
        prompt_ids = torch.tensor([[word2id[w] for w in prompt_words]], device=device)
        output_ids = model.generate(prompt_ids, max_new_tokens=6)
        output_tokens = [id2word[idx.item()] for idx in output_ids[0]]
        prompt_text = " ".join(prompt_words)
        generated_text = " ".join(output_tokens)
        print(f"  Prompt   : \"{prompt_text}\"")
        print(f"  Generated: \"{generated_text}\"")
        print()
    print("=" * 60)


if __name__ == "__main__":
    run_transformer()
