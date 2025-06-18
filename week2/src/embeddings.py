from gensim.models import KeyedVectors
import torch
import os


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def get_pretrained_w2v_embeddings(cfg):
    # TODO Download the data from https://code.google.com/archive/p/word2vec/ or other paths
    path = os.path.expanduser(cfg.embeddings.path)
    w2v = KeyedVectors.load_word2vec_format(path, binary=cfg.embeddings.is_binary)
    # FIXME Ability to load glove embeddings without convertion. Use binary for them? Then remove convert_globe_emb.py

    # Get embedding dimensions and vocab size
    vector_size = w2v.vector_size
    original_vocab_size = len(w2v)
    print(f"Embedding vector size: {vector_size}")
    print(f"Original vocab size: {original_vocab_size}")

    # Build word2idx with consistent indices
    word2idx = {}
    all_words = list(w2v.index_to_key)

    for idx, word in enumerate(all_words):
        word2idx[word] = idx

    # Add PAD and UNK tokens at the end
    pad_idx = len(word2idx)
    word2idx[PAD_TOKEN] = pad_idx
    unk_idx = len(word2idx)
    word2idx[UNK_TOKEN] = unk_idx

    vocab_size = len(word2idx)  # Update to include PAD and UNK tokens

    # Initialize embedding matrix
    embedding_matrix = torch.zeros(vocab_size, vector_size)

    for word, idx in word2idx.items():
        if word in w2v:
            embedding_matrix[idx] = torch.tensor(w2v[word])
        # else: PAD and UNK are already zeros by default

    return embedding_matrix, word2idx
