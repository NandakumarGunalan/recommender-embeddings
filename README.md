# Embedding-Based Recommender System (Matrix Factorization with BPR)

## Overview

This repository implements a neural embedding–based collaborative filtering system optimized for implicit feedback ranking tasks.

The model uses:

- Matrix Factorization with learned user and item embeddings
- Bayesian Personalized Ranking (BPR) loss
- Negative sampling training strategy
- Offline ranking evaluation via Recall@K and NDCG@K

The implementation is modular, reproducible, and structured for extensibility toward large-scale recommender systems.

---

## Problem Definition

Given historical user–item interaction data, the objective is to learn latent representations that produce personalized top-K ranked recommendations.

Unlike regression-based rating prediction, this system optimizes directly for pairwise ranking performance using BPR loss:

For each user \( u \), positive item \( i \), and negative item \( j \):

\[
\mathcal{L}_{BPR} = - \log \sigma (\hat{r}_{ui} - \hat{r}_{uj})
\]

where:

- \( \hat{r}_{ui} \) is the predicted score for user \( u \) and item \( i \)
- \( \sigma \) is the sigmoid function

This directly optimizes ranking quality rather than rating error.

---

## Architecture

### Model Components

The recommender consists of:

- User embedding matrix \( E_u \in \mathbb{R}^{|U| \times d} \)
- Item embedding matrix \( E_i \in \mathbb{R}^{|I| \times d} \)
- Optional bias terms

Prediction function:

\[
\hat{r}_{ui} = \langle E_u(u), E_i(i) \rangle + b_u + b_i
\]

where \( \langle \cdot \rangle \) denotes dot product.

---

## System Flow

Data Loading  
↓  
Train/Test Split  
↓  
Negative Sampling Dataset  
↓  
Embedding Model (PyTorch)  
↓  
BPR Optimization  
↓  
Offline Evaluation (Recall@K, NDCG@K)

---

## Repository Structure

recommender-embeddings/
│
├── src/
│   ├── data.py
│   ├── train_dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── test_data.py
│   └── test_model.py
│
├── data/                # MovieLens dataset (excluded in .gitignore)
├── artifacts/           # Saved models (excluded in .gitignore)
├── requirements.txt
└── README.md

---

## Dataset

- MovieLens 100K
- 100,000 explicit interactions
- Converted to implicit feedback for ranking

Data is excluded from version control.

---

## Training

```bash
python src/train.py
```

Model checkpoints are stored in:

```
artifacts/mf_model.pt
```

---

## Evaluation

```bash
python src/evaluate.py
```

Metrics:

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

This implementation is intentionally modular to support:

- Hyperparameter tuning
- Large-scale dataset integration
- Model versioning
- Experiment tracking integration

Potential extensions:

- Adaptive negative sampling
- Temporal dynamics
- Regularization tuning
- Batch inference pipeline
- ANN-based retrieval (FAISS / ScaNN)

---

## Dependencies

See `requirements.txt`

Core stack:

- PyTorch
- NumPy
- Pandas
- scikit-learn

---

## License

MIT License
