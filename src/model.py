import torch
import torch.nn as nn


class MFRecommender(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 128):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)

        dot = (u * v).sum(dim=1)
        bias = self.user_bias(user_idx).squeeze() + self.item_bias(item_idx).squeeze()

        return dot + bias
    
def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()