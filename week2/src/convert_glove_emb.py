from gensim.scripts.glove2word2vec import glove2word2vec
import os


def main():
    dir = os.path.expanduser("~/experiment_data/models/glove.6B")
    glove_input_file = "glove.6B.50d.txt"
    word2vec_output_file = "glove.6B.50d.w2v.txt"
    glove2word2vec(os.path.join(dir, glove_input_file), os.path.join(dir, word2vec_output_file))
    print(f"Converted {glove_input_file} to {word2vec_output_file} in {dir}")


if __name__ == "__main__":
    main()
