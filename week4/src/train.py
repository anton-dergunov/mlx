import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import wandb


SMOOTHIE = SmoothingFunction().method4


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for images, input_ids, attention_mask in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(images, input_ids, attention_mask)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_one_epoch(model, dataloader, device, num_examples=5):
    model.eval()
    total_loss = 0
    bleu_scores = []
    all_outputs = []

    example_count = 0

    with torch.no_grad():
        for images, input_ids, attention_mask in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # === Compute the token-level loss ===
            outputs = model(images, input_ids, attention_mask)
            loss = outputs.loss
            total_loss += loss.item()

            # === Generate captions ===
            generated_seqs = model.generate(images)

            for input_id_seq, generated_seq in zip(input_ids.tolist(), generated_seqs.tolist()):
                actual_text = model.gpt2_tokenizer.decode(input_id_seq, skip_special_tokens=True)
                generated_text = model.gpt2_tokenizer.decode(generated_seq, skip_special_tokens=True)

                # === Compute BLEU ===
                ref_tokens = actual_text.split()
                gen_tokens = generated_text.split()
                bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=SMOOTHIE)
                bleu_scores.append(bleu)

                # === Collect examples to print ===
                if example_count < num_examples:
                    print("\n=== Example ===")
                    print(f"Reference: {actual_text}")
                    print(f"Generated: {generated_text}")
                    example_count += 1

                # === For possible later use ===
                all_outputs.append({
                    "reference": actual_text,
                    "generated": generated_text,
                    "bleu": bleu
                })

    avg_loss = total_loss / len(dataloader)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return avg_loss, avg_bleu, all_outputs


def train_loop(train_loader, val_loader, device, model,
               num_epochs=10, lr=3e-4, log_wandb=True):

    model.to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.mapping.parameters(), lr=lr)

    if log_wandb:
        wandb.watch(model, log_freq=10)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss, val_bleu, _ = eval_one_epoch(model, val_loader, device)
        print(f"\nValidation loss: {val_loss:.4f} | Average BLEU: {val_bleu:.4f}")

        if log_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss
            })
