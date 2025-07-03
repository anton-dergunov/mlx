import torch
import torch.nn as nn
import math
from transformers.modeling_outputs import CausalLMOutput


# -----------------------------
# Multi-Head Self-Attention
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention mechanism.
    """
    def __init__(self, embed_dim, num_heads, dot_product_norm=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Separate projection layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dot_product_norm = dot_product_norm

    def forward(self, x, mask=None):
        """
        x: Input tensor of shape (B, N, D)
        mask: Optional attention mask of shape (1, 1, N, N)
        """
        B, N, D = x.shape  # Batch, Num tokens, Embedding dim

        # Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) # (B, num_heads, N, N)
        if self.dot_product_norm:
            scores /= math.sqrt(self.head_dim)

        # Apply mask if provided (for causal masking in decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = scores.softmax(dim=-1)

        # Apply attention weights to values
        attended = attn @ v  # (B, num_heads, N, head_dim)

        # Concatenate heads and project back to original embedding dim
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(attended)


# -----------------------------
# Transformer Decoder Block
# -----------------------------
class TransformerDecoderBlock(nn.Module):
    """
    Decoder block with self-attention and MLP (no cross-attention).
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0, dot_product_norm=True):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dot_product_norm)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask=None):
        # Masked self-attention + residual
        x = x + self.dropout(self.self_attn(self.norm1(x), tgt_mask))
        # MLP + residual
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


# -----------------------------
# Text Transformer Decoder
# -----------------------------
class TransformerDecoder(nn.Module):
    """
    Transformer decoder that generates digit sequences from image representations.
    """
    def __init__(self,
                 vocab_size,
                 pretrained_embed,
                 embed_dim=512,
                 num_heads=8,
                 mlp_dim=2048,
                 num_layers=4,
                 dropout=0.0):
        super().__init__()

        # Embeds digits
        self.tokens_embed = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)

        # Positional encoding for target sequence
        self.register_buffer("pos_encoding", self._build_sinusoidal_pos_encoding(1000, embed_dim), persistent=False)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        # Predicts logits over digits
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def _build_sinusoidal_pos_encoding(self, max_len, embed_dim):
        """Create (max_len, embed_dim) sinusoidal positional encodings."""
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        return pe

    def embed_tokens(self, tokens):
        token_embed = self.tokens_embed(tokens)
        return self._add_pos_embed(token_embed)

    def _add_pos_embed(self, token_embed):
        T = token_embed.size(1)
        device = token_embed.device

        if T > self.pos_encoding.size(1):
            # Extend pos_encoding if needed
            new_pe = self._build_sinusoidal_pos_encoding(T * 2, self.embed_dim).to(device)
            self.pos_encoding = new_pe
        pos_embed = self.pos_encoding[:, :T, :].to(device)
        return token_embed + pos_embed
        
    def forward(self, inputs_embeds, attention_mask, labels):
        T = inputs_embeds.size(1)
        device = inputs_embeds.device

        # Causal mask: allow attention only to current and previous tokens
        causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)

        # Expand attention_mask to (B, 1, 1, T) for broadcasting
        if attention_mask is not None:
            # Ensure the attention_mask from your dataloader is also boolean.
            attn_mask = attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            # Combine causal mask and attention mask
            mask = causal_mask & attn_mask  # (B, 1, T, T)
        else:
            mask = causal_mask  # (1, 1, T, T)

        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, mask)

        output = self.output_proj(x)  # (B, T, vocab_size)

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(output.view(-1, output.size(-1)), labels.view(-1))

        return CausalLMOutput(logits=output, loss=loss)

    # Note that attention_mask is ignored on purpose: in the current use case the input image embeddings
    # always have the same lengths.
    def generate(self, inputs_embeds, attention_mask, max_new_tokens, eos_token_id, pad_token_id):
        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]

        # This will hold the generated token IDs, starting empty
        generated_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        
        # Use inputs_embeds as the initial state for the autoregressive loop
        x = inputs_embeds
        
        # Keep track of which sequences in the batch have finished
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # The generation loop continues as long as the total sequence length is less than max_length
        while x.shape[1] < max_new_tokens:
            # 1. Get current sequence length
            T = x.shape[1]
            
            # 2. Create a causal (triangular) mask for self-attention
            causal_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(1)
            
            # 3. Pass the current sequence through the decoder layers
            h = x
            for layer in self.layers:
                h = layer(h, causal_mask)

            # 4. Get logits for the very last token in the sequence
            logits = self.output_proj(h[:, -1, :])  # Shape: (B, vocab_size)
            
            # 5. Select the next token ID with the highest probability (greedy search)
            next_token_id = torch.argmax(logits, dim=-1)  # Shape: (B)
            
            # For sequences that are already finished, fill with pad_token_id
            next_token_id[finished_sequences] = pad_token_id

            # 6. Append the new token IDs to our results
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(1)], dim=1)
            
            # 7. Check for the end-of-sequence token
            newly_finished = (next_token_id == eos_token_id) & ~finished_sequences
            finished_sequences |= newly_finished
            
            # Stop generation if all sequences in the batch have finished
            if finished_sequences.all():
                break

            # 8. Embed the newly generated tokens
            next_token_embed = self.tokens_embed(next_token_id.unsqueeze(1))  # Shape: (B, 1, D)
            
            # 9. Add the correct positional encoding for the new time step
            pos_embedding = self.pos_encoding[:, T:T+1, :]
            
            # 10. Concatenate the new embeddings to the sequence for the next iteration
            x = torch.cat([x, next_token_embed + pos_embedding], dim=1)
            
        return generated_ids
