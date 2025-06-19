import torch
import wandb
import random
import numpy as np
from omegaconf import OmegaConf

from embeddings import get_pretrained_w2v_embeddings, PAD_TOKEN
from data import get_dataloader, get_evaluation_data
from model import get_model, create_shared_embedding
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
    seed_all(cfg.train.seed)

    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading pretrained word2vec model...")
    embedding_matrix, word2idx = get_pretrained_w2v_embeddings(cfg)

    pad_token_idx = word2idx[PAD_TOKEN]
    shared_embedding = create_shared_embedding(embedding_matrix, pad_token_idx, freeze=True)

    query_model = get_model(cfg.query_model, shared_embedding, pad_token_idx)
    document_model = get_model(cfg.document_model, shared_embedding, pad_token_idx)

    if "train" in modes:
        if cfg.log.wandb:
            wandb.init(
                project=cfg.log.project,
                name=cfg.log.run_name,
                config=OmegaConf.to_container(cfg, resolve=True))

        print("Loading dataset...")
        train_loader = get_dataloader(cfg, word2idx)

        if not query_model.requires_training and not document_model.requires_training:
            print("Skip training")
        else:
            print("Starting training...")
            train_loop(query_model, document_model, train_loader, cfg, device)
        
        if query_model.requires_training and cfg.query_model.output:
            torch.save(query_model.state_dict(), cfg.query_model.output)
            print(f"Query model saved to {cfg.query_model.output}")

        if document_model.requires_training and cfg.document_model.output:
            torch.save(document_model.state_dict(), cfg.document_model.output)
            print(f"Document model saved to {cfg.document_model.output}")

        # TODO Upload the models to W&B

        if cfg.log.wandb:
            wandb.finish()

    if "test" in modes:
        if query_model.requires_training and cfg.query_model.output:
            query_model.load_state_dict(torch.load(cfg.query_model.output, map_location=device))
            print(f"Query model restored from {cfg.query_model.output}")

        if document_model.requires_training and cfg.document_model.output:
            document_model.load_state_dict(torch.load(cfg.document_model.output, map_location=device))
            print(f"Document model restored from {cfg.document_model.output}")

        # TODO Make the split configurable
        query_loader, queries_orig, doc_loader, docs_orig, qrels = get_evaluation_data(cfg, "validation", word2idx)

        results, top_results = evaluate(query_loader, queries_orig, doc_loader, docs_orig, qrels, query_model, document_model, device)

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
