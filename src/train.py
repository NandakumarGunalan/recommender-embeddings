import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os

from data import prepare_data
from model import MFRecommender, bpr_loss
from train_dataset import BPRDataset


def train_model(
    embedding_dim=128,
    batch_size=1024,
    lr=5e-4,
    weight_decay=1e-5,
    epochs=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print("Using device:", device)

    train_df, test_df, interactions, num_users, num_items = prepare_data()

    dataset = BPRDataset(train_df, interactions, num_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MFRecommender(num_users, num_items, embedding_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for user, pos, neg in tqdm(loader, desc=f"Epoch {epoch+1}"):
            user = user.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_scores = model(user, pos)
            neg_scores = model(user, neg)

            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": embedding_dim,
        },
        "artifacts/mf_model.pt",
    )
    print("Model saved to artifacts/mf_model.pt")

    return model


if __name__ == "__main__":
    train_model()
