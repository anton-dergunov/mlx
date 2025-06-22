import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import re
from collections import Counter

# --- 1. Tokenizer ---
def regex_tokenizer(text):
    return re.findall(r"\b\w+\b|[^\w\s]", text.lower())

# --- 2. Sample Text ---
corpus = "The quick brown fox jumps over the lazy dog. The fox is quick and the dog is lazy."
tokens = regex_tokenizer(corpus)

# --- 3. Vocabulary ---
min_freq = 1
word_freq = Counter(tokens)
vocab = {'<pad>': 0, '<unk>': 1}
for word, freq in word_freq.items():
    if freq >= min_freq:
        vocab[word] = len(vocab)
inv_vocab = {idx: word for word, idx in vocab.items()}
indexed = [vocab.get(word, vocab['<unk>']) for word in tokens]

# --- 4. Subsampling ---
t = 1e-4
total_count = len(indexed)
word_probs = {w: c / total_count for w, c in Counter(indexed).items()}
subsample_probs = {w: 1 - math.sqrt(t / word_probs[w]) if word_probs[w] > t else 0 for w in word_probs}
indexed = [w for w in indexed if random.random() > subsample_probs.get(w, 0)]

# --- 5. Training Pairs ---
window = 2
pairs = []
for i, center in enumerate(indexed):
    for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
        if i != j:
            pairs.append((center, indexed[j]))

# --- 6. Model ---
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, dim)
        self.out_embed = nn.Embedding(vocab_size, dim)

    def forward(self, center, context, negatives):
        v_c = self.in_embed(center)                # (B, D)
        v_o = self.out_embed(context)              # (B, D)
        v_n = self.out_embed(negatives)            # (B, K, D)

        pos_score = torch.mul(v_c, v_o).sum(dim=1)
        neg_score = torch.bmm(v_n.neg(), v_c.unsqueeze(2)).squeeze()
        loss = -(F.logsigmoid(pos_score) + F.logsigmoid(neg_score).sum(1)).mean()
        return loss

# --- 7. Train Setup ---
dim = 20
vocab_size = len(vocab)
model = SkipGramNegSampling(vocab_size, dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
neg_samples = 5
epochs = 10
batch_size = 4

# --- 8. Training Loop ---
for epoch in range(epochs):
    random.shuffle(pairs)
    total_loss = 0
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        if not batch:
            continue
        centers = torch.tensor([p[0] for p in batch], dtype=torch.long)
        contexts = torch.tensor([p[1] for p in batch], dtype=torch.long)
        negatives = torch.randint(0, vocab_size, (len(batch), neg_samples))

        loss = model(centers, contexts, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
