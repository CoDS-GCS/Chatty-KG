import string
import os
import sys
import numpy as np
import statistics
from transformers import BertTokenizer, BertModel
import torch



class WordEmbeddings:
    def __init__(self, model_path):
        self.model_path = model_path
        self.w = None
        self.vocab = None
        self.ivocab = None
        self.vector_feature_size = 0
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def load_model(self):
        self.w, self.vocab, self.ivocab = self.load_vocab()

    def load_vocab(self):
        with open(self.model_path, "r", encoding='utf8') as f:
            words = [x.rstrip().split(" ")[0] for x in f.readlines()]
        with open(self.model_path, "r", encoding='utf8') as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(" ")
                vectors[vals[0]] = [float(x) for x in vals[1:]]

        vocab_size = len(words)
        vocab = {w: idx for idx, w in enumerate(words)}
        ivocab = {idx: w for idx, w in enumerate(words)}

        vector_dim = len(vectors[ivocab[0]])
        global vector_feature_size
        vector_feature_size = vector_dim
        W = np.zeros((vocab_size, vector_dim))
        for word, v in vectors.items():
            if word == "<unk>":
                continue
            W[vocab[word], :] = v

        # normalize each word vector to unit variance
        W_norm = np.zeros(W.shape)
        d = np.sum(W**2, 1) ** (0.5)
        W_norm = (W.T / d).T
        return (W_norm, vocab, ivocab)

    def semantic_distance(self, v1, v2):
        if v1 is None or v2 is None:
            print("unknowns")
            return -99
        else:
            sim = np.dot(v1, v2.T)
        return sim

    def get_embedding_for_word(self, word):
        # print("vocab", )
        # print(len(vocab))
        if word in self.vocab:
            return self.w[self.vocab[word], :]
        else:
            return None

    def mwe_semantic_distance(self, vs1, vs2):
        cmp_count = 0
        sims = list()
        for v1 in vs1:
            for v2 in vs2:
                cmp_count += 1
                if v1 is None:
                    # TODO: use minimum edit distance
                    sims.append(0.0)
                    continue
                if v2 is None:
                    sims.append(0.0)
                    continue
                sim = np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                sims.append(sim)
        else:
            return statistics.mean(sims)

    def get_embedding_for_mwe(self, mwe):
        mwe = mwe.translate(str.maketrans("", "", string.punctuation))
        words = mwe.strip().split()
        mwe_vecs = list()

        for w in words:
            if w in self.vocab:
                mwe_vecs.append(self.w[self.vocab[w], :])
            else:
                vec = self.get_bert_embedding(w)[:300]
                vec.astype(float)
                mwe_vecs.append(vec)
        else:
            return mwe_vecs

    def get_bert_embedding(self, text):
        # Tokenize the input text and convert to input IDs
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)

        # Get the embeddings from BERT
        with torch.no_grad():
            outputs = self.model(**inputs)

        # The embeddings are the outputs from the last hidden state of the model
        # We take the mean of the embeddings of all tokens for simplicity
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = np.array(embeddings.numpy().squeeze(), dtype=np.float64)
        return embeddings




if __name__ == "__main__":
    # get_embedding_for_mwe: Gets embeddings of words
    # mwe_semantic_distance: Gets semantic similarity between two vectors
    wiki_word_embed_path = os.path.join("data", "wiki-news-300d-1M.txt")
    wiki_model = WordEmbeddings(
        wiki_word_embed_path
    )
    wiki_model.load_model()
    print("Done loading")
    print(
        "ss: "
        + str(
            wiki_model.mwe_semantic_distance(
                wiki_model.get_embedding_for_mwe('wife'),
                wiki_model.get_embedding_for_mwe('spouse'),
            )
            )
        )
