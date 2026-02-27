from data import prepare_data

train_df, test_df, interactions, num_users, num_items = prepare_data()

print("Train size:", len(train_df))
print("Test size:", len(test_df))
print("Users:", num_users)
print("Items:", num_items)
print("Sample interactions:", list(interactions.items())[:3])