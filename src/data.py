import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


DATA_PATH = "data/ml-100k/u.data"


def load_movielens():
    """
    Load MovieLens 100K interactions.
    Returns:
        df: pandas DataFrame with columns [user_id, item_id, rating, timestamp]
    """
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return df


def encode_ids(df):
    """
    Map raw user/item IDs to 0-based indices.
    """
    user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_map = {i: j for j, i in enumerate(df["item_id"].unique())}

    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)

    return df, user_map, item_map


def leave_one_out_split(df):
    """
    Leave-one-out split:
    For each user, last interaction -> test.
    """
    df = df.sort_values(["user_idx", "timestamp"])

    test_indices = df.groupby("user_idx").tail(1).index
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def build_interaction_dict(df):
    """
    Build user -> set(items) dictionary.
    Used for negative sampling.
    """
    interaction_dict = df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    return interaction_dict


def prepare_data():
    df = load_movielens()
    df, user_map, item_map = encode_ids(df)

    train_df, test_df = leave_one_out_split(df)

    train_interactions = build_interaction_dict(train_df)

    num_users = len(user_map)
    num_items = len(item_map)

    return (
        train_df,
        test_df,
        train_interactions,
        num_users,
        num_items,
    )