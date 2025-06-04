import string
import os
import sys
import numpy as np
import statistics
from transformers import BertTokenizer, BertModel
import torch



class WordEmbeddings:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.w = None
        self.vocab = None
        self.ivocab = None
        self.vector_feature_size = 300  # BERT embeddings are 768, but we'll use first 300 dimensions
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        if model_path and os.path.exists(model_path):
            self.load_model()

    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            self.w, self.vocab, self.ivocab = self.load_vocab()
        else:
            print("Using BERT embeddings as fallback")

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
        # Try to get BERT embedding first
        vec = self.get_bert_embedding(word)[:300]
        vec = vec.astype(float)
        return vec

    def mwe_semantic_distance(self, vs1, vs2):
        cmp_count = 0
        sims = list()
        for v1 in vs1:
            for v2 in vs2:
                cmp_count += 1
                if v1 is None or v2 is None:
                    sims.append(0.0)
                    continue
                sim = np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                sims.append(sim)
        return statistics.mean(sims)

    def get_embedding_for_mwe(self, mwe):
        mwe = mwe.translate(str.maketrans("", "", string.punctuation))
        words = mwe.strip().split()
        mwe_vecs = list()

        for w in words:
            vec = self.get_bert_embedding(w)[:300]
            vec = vec.astype(float)
            mwe_vecs.append(vec)
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
    # Test the embeddings
    model = WordEmbeddings()
    print("Testing BERT embeddings...")
    print(
        "Similarity between 'wife' and 'spouse': "
        + str(
            model.mwe_semantic_distance(
                model.get_embedding_for_mwe('wife'),
                model.get_embedding_for_mwe('spouse'),
            )
        )
    )
