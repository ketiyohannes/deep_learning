import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbeddding(nn.Module):
    def __init__(self, max_length, embed_dim, dropout=0.1):
        #max_length: the maximum length of the sequence
        #embed_dim: the dimension of the embeddings
        #dropout: the dropout rate
        super().__init__() 
        self.pos_embeddings = nn.Parameter(torch.randn(max_length, embed_dim)) # generating random positional embeddings
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x: (batch_size, seq_length, embed_dim)
        # self.pos_embeddings: (max_length, embed_dim)
        # we want to add the positional embeddings to the input x
        # we want to add the positional embeddings to the input x for each position in the sequence
        return self.dropout(x + self.pos_embeddings[:x.size(1)]) 
