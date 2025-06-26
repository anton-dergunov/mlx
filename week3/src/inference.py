import torch
import torch.nn as nn

from data import START_TOKEN, END_TOKEN


def decode_sequence_greedy(model, img_patches, max_len=10):
    """
    Autoregressively decode output sequences from image patches using greedy decoding.
    Supports batch input.
    
    Args:
        model: Vision-to-sequence model with `.encode()` and `.decode()` methods.
        img_patches: Tensor of shape (B, N_patches) representing input images.
        max_len: Maximum sequence length to decode.

    Returns:
        List of decoded sequences (as Python lists of ints), one per batch item.
    """
    model.eval()
    device = next(model.parameters()).device

    B = img_patches.size(0)

    with torch.no_grad():
        # Encode image patches once
        memory = model.encode(img_patches.to(device))  # (B, N, D)

        # Initialize decoder input with <START> token
        sequences = torch.full((B, 1), START_TOKEN, dtype=torch.long, device=device)  # (B, 1)
        finished = torch.zeros(B, dtype=torch.bool, device=device)  # track finished sequences

        for _ in range(max_len):
            # Decode current sequence
            logits = model.decode(memory, sequences)   # (B, T, vocab_size)
            next_token = logits[:, -1].argmax(dim=-1)  # (B,)

            # Append predicted token to the sequence
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)  # (B, T+1)

            # Check if any sequences have reached <END>
            finished |= next_token == END_TOKEN
            if finished.all():
                break

        # Remove <START> token (first column)
        result = sequences[:, 1:]  # (B, T)

        # Convert to list of Python lists and truncate at END_TOKEN
        decoded = []
        for seq in result.tolist():
            if END_TOKEN in seq:
                idx = seq.index(END_TOKEN)
                decoded.append(seq[:idx])
            else:
                decoded.append(seq)

    return decoded


class InferenceWrapper(nn.Module):
    def __init__(self, model, max_len=10):
        super().__init__()
        self.model = model
        self.max_len = max_len

    def forward(self, img_patches):
        # img_patches: [B, 36, 196]
        memory = self.model.encode(img_patches)  # [B, N, D]
        B = img_patches.shape[0]
        sequences = torch.full((B, 1), START_TOKEN, dtype=torch.long, device=img_patches.device)

        for _ in range(self.max_len):
            logits = self.model.decode(memory, sequences)  # (B, T, vocab_size)
            next_token = logits[:, -1].argmax(dim=-1)      # (B,)
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)

        return sequences[:, 1:]  # remove START token

# TODO Implement beam search decoding

