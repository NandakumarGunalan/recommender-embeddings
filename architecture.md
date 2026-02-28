# System Architecture

## 1. Overview

This document describes the architectural design of the embedding-based recommender system implemented in this repository.

The system is structured to separate concerns across data processing, model definition, training logic, and evaluation to maintain modularity and extensibility.

The architecture follows a traditional offline recommendation pipeline:

Data → Feature Preparation → Model Training → Evaluation → Artifact Storage

---

## 2. Layered Design

### 2.1 Data Layer

Responsibilities:
- Load MovieLens interactions
- Normalize user and item identifiers
- Train/test split
- Return structured tabular representation

This layer is intentionally isolated from model logic to maintain clear data-model separation.

File:
- `src/data.py`

---

### 2.2 Dataset & Sampling Layer

Responsibilities:
- Construct triplets (user, positive_item, negative_item)
- Perform negative sampling
- Provide PyTorch-compatible dataset interface

Negative sampling strategy:
- For each observed interaction (u, i)
- Randomly sample item j such that j ∉ user(u)’s interaction history

This layer abstracts sampling logic away from the model and trainer.

File:
- `src/train_dataset.py`

---

### 2.3 Model Layer

Responsibilities:
- Define embedding tables
- Compute interaction scores
- Support BPR loss optimization

Core components:
- User embedding matrix: |U| × d
- Item embedding matrix: |I| × d
- Optional bias terms

Prediction function:

score(u, i) = dot(E_u[u], E_i[i]) + b_u + b_i

File:
- `src/model.py`

---

### 2.4 Training Engine

Responsibilities:
- Batch iteration
- Forward pass
- BPR loss computation
- Backpropagation
- Checkpoint persistence

The trainer does not contain data-loading logic or evaluation logic to preserve single-responsibility design.

File:
- `src/train.py`

Artifacts:
- Saved to `artifacts/mf_model.pt`

---

### 2.5 Evaluation Engine

Responsibilities:
- Score candidate items
- Rank items per user
- Compute Recall@K
- Compute NDCG@K

Evaluation is performed offline using a held-out test set.

File:
- `src/evaluate.py`

---

## 3. Data Flow

1. Load dataset
2. Encode user/item IDs
3. Split into train/test
4. Construct negative-sampled dataset
5. Train embedding model with BPR
6. Save trained model
7. Evaluate ranking metrics

---

## 4. Design Principles

### Separation of Concerns
Each module handles a single responsibility:
- Data processing
- Sampling
- Model definition
- Training loop
- Evaluation

### Modularity
Embedding model can be swapped with:
- Neural collaborative filtering
- Factorization machines
- Deep ranking architectures

### Reproducibility
- Deterministic train/test split
- Explicit artifact storage
- Clear execution commands

---

## 5. Scalability Considerations

While the current implementation targets MovieLens 100K, architectural decisions support scaling:

- Embedding layers scale linearly with |U| and |I|
- Negative sampling keeps training complexity manageable
- Model scoring remains O(d) per user-item pair

For large-scale production systems, potential extensions include:

- Distributed training
- Approximate nearest neighbor retrieval (e.g., FAISS)
- Candidate generation + ranking separation
- Feature store integration
- Online inference service

---

## 6. Future Extensions

- Hyperparameter configuration via CLI
- Logging and experiment tracking
- Regularization tuning
- Adaptive negative sampling
- Bias removal experiments
- Serving layer for real-time inference

---

## 7. Production Pathway (Conceptual)

Offline Training:
Data → Training Pipeline → Model Artifact

Online Serving:
User Request → Candidate Retrieval → Dot Product Scoring → Top-K Results

This repository currently implements the offline training and evaluation portion of the pipeline.
