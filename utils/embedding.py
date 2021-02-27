import torch
import torch.nn as nn
import numpy as np


class Embedding(nn.Module):
    def __init__(self, word_emb_path, max_length, word_emb_size=50, pos_emb_size=5):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.word_emb_size = word_emb_size
        self.pos_embedding_idm = pos_emb_size

        self.word_embedding_map = torch.from_numpy(np.load(word_emb_path))

        self.word_embedding = nn.Embedding(self.word_embedding_map.shape[0],
                                           word_emb_size,
                                           padding_idx=self.word_embedding_map.shape[0]-1)
        self.word_embedding.weight.data.copy_(self.word_embedding_map)

        self.h_pos_embedding = nn.Embedding(2*max_length, pos_emb_size, padding_idx=0)
        self.t_pos_embedding = nn.Embedding(2*max_length, pos_emb_size, padding_idx=0)

    def forward(self, inputs):
        word = inputs["token"]
        h_pos = inputs["h_pos"]
        t_pos = inputs["t_pos"]
        x = torch.cat([self.word_embedding(word),
                       self.h_pos_embedding(h_pos),
                       self.t_pos_embedding(t_pos)], 3)

        return x
