import os
import torch
import wandb
import random
import numpy as np
from omegaconf import OmegaConf

from embeddings import get_pretrained_w2v_embeddings, PAD_TOKEN
from data import get_dataloader, get_evaluation_data
from model import get_model
from train import train_loop
from evaluate import evaluate
from utils import get_device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cfg, modes):
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading pretrained word2vec model...")
    embedding_matrix, word2idx = get_pretrained_w2v_embeddings(cfg)

    model = get_model(cfg, embedding_matrix, word2idx[PAD_TOKEN])

    if "train" in modes:
        seed_all(cfg.train.seed)

        if cfg.log.wandb:
            wandb.init(
                project=cfg.log.project,
                name=cfg.log.run_name,
                config=OmegaConf.to_container(cfg, resolve=True))

        print("Loading dataset...")
        train_loader = get_dataloader(cfg, word2idx)

        if not model.requires_training:
            print("Skip training")
        else:
            print("Starting training...")
            train_loop(model, train_loader, cfg, device)
        
        torch.save(model.state_dict(), cfg.train.output)
        print(f"Model saved to {cfg.train.output}")

        # TODO Upload the model to W&B

        if cfg.log.wandb:
            wandb.finish()

    if "test" in modes:
        if not model:
            model.load_state_dict(torch.load(cfg.train.output, map_location="cpu"))
            model.eval()
            print(f"Model restored from {cfg.train.output}")

        model.to(device)

        # TODO Make the split configurable
        query_loader, queries_orig, doc_loader, docs_orig, qrels = get_evaluation_data(cfg, "validation", word2idx)

        results, top_results = evaluate(query_loader, queries_orig, doc_loader, docs_orig, qrels, model, device)

        # TODO Log this into W&B as well (and make the number of queries and top docs configurable)
        for result in top_results:
            print(f"\nQuery: {result['query_text']}")
            print("Top Documents:")
            for idx, doc in enumerate(result["top_documents"], 1):
                print(f"  {idx:2d}. [{doc['score']:.4f}] <{doc['rel_label']}> {doc['text']}")

        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

        # FIXME Log these results to wandb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--modes",
        type=lambda s: set(map(str.strip, s.split(','))),
        default="train,test",
        help="Comma-separated set of modes to run: train, test, etc. Example: --modes train,test"
    )
    args = parser.parse_args()
    # TODO Load custom and base (default) configs
    cfg = OmegaConf.load(args.config)
    main(cfg, args.modes)
