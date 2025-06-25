import torch

from data import START_TOKEN, END_TOKEN


def decode_sequence_greedy(model, img_patches, max_len=10):
    """
    Autoregressively decode output sequences from image patches using greedy decoding.
    Supports batch input.
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
        decoded = sequences[:, 1:]  # (B, T)

        # Trim at <END> and optionally pad to max_len
        trimmed = []
        for seq in decoded:
            # Find first END_TOKEN position
            end_pos = (seq == END_TOKEN).nonzero(as_tuple=True)
            cut_pos = end_pos[0].item() if len(end_pos[0]) > 0 else len(seq)

            # Trim and optionally pad
            trimmed_seq = seq[:cut_pos]
            if len(trimmed_seq) < max_len:
                pad_len = max_len - len(trimmed_seq)
                trimmed_seq = torch.cat([trimmed_seq, torch.full((pad_len,), END_TOKEN, dtype=seq.dtype, device=seq.device)])
            trimmed.append(trimmed_seq)

        return torch.stack(trimmed).cpu().tolist()

# TODO Implement beam search decoding

