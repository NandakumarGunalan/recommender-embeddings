import torch
import numpy as np
from tqdm import tqdm

from data import prepare_data
from model import MFRecommender


def recall_at_k(ranked_items, ground_truth, k):
    return int(ground_truth in ranked_items[:k])


def ndcg_at_k(ranked_items, ground_truth, k):
    if ground_truth in ranked_items[:k]:
        rank = ranked_items.index(ground_truth)
        return 1 / np.log2(rank + 2)
    return 0.0


def evaluate_model(model, train_interactions, test_df, num_items, k=10, device="cpu"):
    model.eval()

    recall_scores = []
    ndcg_scores = []

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            user = int(row["user_idx"])
            true_item = int(row["item_idx"])

            user_tensor = torch.tensor([user] * num_items).to(device)
            items_tensor = torch.arange(num_items).to(device)

            scores = model(user_tensor, items_tensor).cpu().numpy()

            # remove training items from ranking
            seen_items = train_interactions[user]
            scores[list(seen_items)] = -np.inf

            ranked_items = np.argsort(-scores).tolist()

            recall_scores.append(recall_at_k(ranked_items, true_item, k))
            ndcg_scores.append(ndcg_at_k(ranked_items, true_item, k))

    return np.mean(recall_scores), np.mean(ndcg_scores)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df, test_df, interactions, num_users, num_items = prepare_data()

    checkpoint = torch.load("artifacts/mf_model.pt")
    model = MFRecommender(
    checkpoint["num_users"],
    checkpoint["num_items"],
    checkpoint["embedding_dim"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(device)

    recall, ndcg = evaluate_model(
        model,
        interactions,
        test_df,
        num_items,
        k=10,
        device=device,
    )

    print(f"\nRecall@10: {recall:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")