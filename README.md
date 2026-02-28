# Embedding-Based Recommender System (Matrix Factorization with BPR)

## Overview

This repository implements an embedding-based collaborative filtering system optimized for implicit-feedback ranking tasks.

The model includes:

- Matrix Factorization with learned user and item embeddings  
- Bayesian Personalized Ranking (BPR) loss  
- Negative sampling training strategy  
- Offline ranking evaluation via Recall@K and NDCG@K  

The codebase is modular and structured for extensibility toward larger-scale recommender systems.

---

## Problem Definition

Given historical user–item interaction data, the objective is to learn latent representations that produce personalized top-K ranked recommendations.

Unlike regression-based rating prediction, this system optimizes directly for **pairwise ranking performance** using BPR loss.

For each user `u`, positive item `i`, and negative item `j`:

Loss = - log( sigmoid( r_hat(u,i) - r_hat(u,j) ) )

where:

- `r_hat(u,i)` is the predicted score for user `u` and item `i`
- `sigmoid(x)` = 1 / (1 + exp(-x))

This formulation directly optimizes ranking quality rather than minimizing rating error.

---

## Architecture

### Model Components

The recommender consists of:

- User embedding matrix `E_u ∈ R^{|U| × d}`
- Item embedding matrix `E_i ∈ R^{|I| × d}`
- Optional bias terms

Prediction function:

r_hat(u,i) = dot( E_u[u], E_i[i] ) + b_u + b_i

where `dot(·)` denotes the inner product.

---

## System Flow

1. Data loading  
2. Train/test split  
3. Negative sampling dataset construction  
4. Embedding model training (PyTorch)  
5. BPR optimization  
6. Offline evaluation (Recall@K, NDCG@K)

---

## Repository Structure

```text
recommender-embeddings/
├── src/
│   ├── data.py
│   ├── train_dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── test_data.py
│   └── test_model.py
├── data/                 # MovieLens dataset (excluded via .gitignore)
├── artifacts/            # Saved models (excluded via .gitignore)
├── requirements.txt
└── README.md
```

---

## Dataset

- MovieLens 100K  
- 100,000 explicit interactions  
- Converted to implicit feedback for ranking  

Data files are excluded from version control.

---

## Training

```bash
python src/train.py
```

Model checkpoints are written to:

```
artifacts/mf_model.pt
```

---

## Evaluation

```bash
python src/evaluate.py
```

Metrics reported:

- Recall@10  
- NDCG@10  

---

## Results (MovieLens 100K)

| Metric     | Value  |
|------------|--------|
| Recall@10  | 0.1007 |
| NDCG@10    | 0.0508 |

---

## Design Decisions

### Why BPR?
- Optimizes ranking directly  
- Suitable for implicit feedback  
- Scales efficiently with negative sampling  

### Why Embeddings?
- Compact latent representation  
- Efficient inference via dot product  
- Easily extendable to deeper architectures  

---

## Production Considerations

The project is structured to support:

- Hyperparameter tuning  
- Larger dataset integration  
- Model versioning  
- Experiment tracking integration  

Potential extensions include:

- Adaptive negative sampling  
- Temporal modeling  
- Regularization tuning  
- Batch inference pipeline  
- Approximate nearest neighbor retrieval (e.g., FAISS, ScaNN)

---

## Dependencies

See `requirements.txt`.

Core stack:

- PyTorch  
- NumPy  
- Pandas  
- scikit-learn  

---

## License

MIT License.
