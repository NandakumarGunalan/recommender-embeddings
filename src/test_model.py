from data import prepare_data
from model import MFRecommender, bpr_loss
from train_dataset import BPRDataset
from torch.utils.data import DataLoader
import torch

train_df, test_df, interactions, num_users, num_items = prepare_data()

ds = BPRDataset(train_df, interactions, num_items)
dl = DataLoader(ds, batch_size=256, shuffle=True)

model = MFRecommender(num_users, num_items, embedding_dim=32)

u, pos, neg = next(iter(dl))
pos_scores = model(u, pos)
neg_scores = model(u, neg)

loss = bpr_loss(pos_scores, neg_scores)
print("Batch loss:", float(loss))