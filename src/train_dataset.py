import random
import torch
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    """
    Returns (user, pos_item, neg_item) triples for BPR training.
    """
    def __init__(self, train_df, user_pos_dict, num_items, num_negatives=1, seed=42):
        self.train_df = train_df.reset_index(drop=True)
        self.user_pos = user_pos_dict
        self.num_items = num_items
        self.num_neg = num_negatives
        random.seed(seed)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]
        u = int(row["user_idx"])
        pos = int(row["item_idx"])

        # sample one negative not in user's positives
        while True:
            neg = random.randint(0, self.num_items - 1)
            if neg not in self.user_pos[u]:
                break

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )