import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import wandb


def train_loop(model: nn.Module, dataloader, cfg, device):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # avoid frozen embeddings
        lr=cfg.train.lr
    )

    if cfg.model.type == "dual_encoder":
        # TODO Customize this
        loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    elif cfg.model.type == "cross_encoder":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            q = batch["query"].to(device)
            pos = batch["pos"].to(device)
            neg = batch["neg"].to(device)

            if cfg.model.type == "dual_encoder":
                q_emb, pos_emb, neg_emb = model(q, pos, neg)
                loss = loss_fn(q_emb, pos_emb, neg_emb)

            elif cfg.model.type == "cross_encoder":
                # Positive pairs
                pos_scores = model(q, pos)
                pos_labels = torch.ones_like(pos_scores)

                # Negative pairs
                neg_scores = model(q, neg)
                neg_labels = torch.zeros_like(neg_scores)

                scores = torch.cat([pos_scores, neg_scores], dim=0)
                labels = torch.cat([pos_labels, neg_labels], dim=0)

                loss = loss_fn(scores, labels)

            else:
                raise ValueError("Unsupported model type")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")
        if cfg.log.wandb:
            wandb.log({"epoch": epoch, "loss": avg_loss})

    


def get_unique_params(*models):
    seen = set()
    unique_params = []
    for model in models:
        for param in model.parameters():
            if id(param) not in seen:
                unique_params.append(param)
                seen.add(id(param))
    return unique_params


def train_loop(query_model, document_model, dataloader, cfg, device):
    query_model.to(device)
    document_model.to(device)

    optimizer = torch.optim.Adam(
        get_unique_params(query_model, document_model),
        lr=cfg.train.lr
    )
    # TODO Implement loss from scratch to try using cosine similarity
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    for epoch in range(cfg.train.epochs):
        query_model.train()
        document_model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            q = batch["query"].to(device)
            pos = batch["pos"].to(device)
            neg = batch["neg"].to(device)

            q_emb = query_model(q)
            pos_emb = document_model(pos)
            neg_emb = document_model(neg)

            loss = triplet_loss(q_emb, pos_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        if cfg.log.wandb:
            wandb.log({"epoch": epoch, "loss": avg_loss})
