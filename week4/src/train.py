import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb


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

    with torch.no_grad():
        for images, input_ids, attention_mask in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            generated = model.generate(images)
            for input_id_seq, generated_seq in zip(input_ids.tolist(), generated.tolist()):
                actual_text = model.tokenizer.decode(input_id_seq, skip_special_tokens=True)
                generated_text = model.tokenizer.decode(generated_seq, skip_special_tokens=True)
                print(f"Actual: {actual_text}")
                print(f"Generated: {generated_text}")
            break
            
            # FIXME Actually compute the metrics

    return total_loss  # FIXME


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

        val_loss = eval_one_epoch(model, val_loader, device)
        print(f"Val   Loss: {val_loss:.4f}")

        if log_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss
            })
