import torch
import wandb
from omegaconf import OmegaConf
import random
import numpy as np
import argparse

from embeddings import get_pretrained_w2v_embeddings, PAD_TOKEN
from data import get_dataloader, get_evaluation_data
from model import build_model, create_shared_embedding, save_model, load_model
from train import train_loop
from evaluate import evaluate
from utils import get_device
from config import load_config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cfg, modes):
    seed_all(cfg.train.seed)

    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading pretrained word2vec model...")
    embedding_matrix, word2idx = get_pretrained_w2v_embeddings(cfg)

    pad_token_idx = word2idx[PAD_TOKEN]
    shared_embedding = create_shared_embedding(embedding_matrix, pad_token_idx, freeze=cfg.model.freeze_embedding)

    model = build_model(cfg.model, shared_embedding, pad_token_idx)

    # TODO Introduce an option to use the same model instance for doc and query (share the weights)

    if cfg.log.wandb:
        wandb.init(
            project=cfg.log.project,
            name=cfg.log.run_name,
            config=OmegaConf.to_container(cfg, resolve=True))

    if "train" in modes:
        print("Loading dataset...")
        train_loader = get_dataloader(cfg, word2idx)

        if not model.requires_training:
            print("Skip training")
        else:
            print("Starting training...")
            train_loop(model, train_loader, cfg, device)
        
        if model.requires_training and cfg.model.save_path:
            save_model(model, cfg.model.save_path)
            print(f"Model saved to {cfg.model.save_path}")

        # TODO Upload the models to W&B

    if "test" in modes:
        if model.requires_training and cfg.model.save_path:
            load_model(model, cfg.model.save_path, device)
            print(f"Model restored from {cfg.model.save_path}")

        # TODO Make the split configurable
        query_loader, queries_orig, doc_loader, docs_orig, qrels = get_evaluation_data(cfg, "validation", word2idx)

        results, top_results = evaluate(query_loader, queries_orig, doc_loader, docs_orig, qrels, model, cfg, device)

        for result in top_results:
            print(f"\nQuery: {result['query_text']}")
            print("Top Documents:")
            for idx, doc in enumerate(result["top_documents"], 1):
                print(f"  {idx:2d}. [{doc['score']:.4f}] <{doc['rel_label']}> {doc['text']}")

        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

        if cfg.log.wandb:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--modes",
        type=lambda s: set(map(str.strip, s.split(','))),
        default="train,test",
        help="Comma-separated set of modes to run: train, test, etc. Example: --modes train,test"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.modes)
