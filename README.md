# DSCI 560 Lab 8

Group 8 Trojan Trio

Members: Jinyao Yang (4266900395), Sen Pang (8598139533), Qianshu Peng (5063709968)


# DSCI 560 Lab 8  
## Representing Document Concepts with Embeddings

**Team:** Trojan Trio  
**Members:**  
- Jinyao Yang  
- Qianshu Peng  
- Sen Pang  


# Project Overview

In this lab, we explore how document embeddings can be used to represent and cluster 
technology-related Reddit posts. The main goal is to compare two embedding-based approaches 
for representing document semantics:

1. **Doc2Vec document embeddings**
2. **Word2Vec-based bin-frequency document representations**

Both approaches are used to generate document vectors, which are then clustered using 
K-Means clustering. The clustering results are evaluated using several internal metrics 
and qualitative inspection of clusters.


# Dataset

The dataset contains **5,000 Reddit posts** related to technology topics.  
Each record includes fields such as:

- `title`
- `body`
- `subreddit`
- `permalink`
- `final_text` (cleaned text used for analysis)

The dataset used in this experiment:
posts_lab5_5000.json


---

# Preprocessing

Before generating embeddings, the text data undergoes several preprocessing steps:

- HTML unescaping
- Removal of URLs and HTML tags
- Lowercasing
- Removal of non-alphanumeric characters
- Whitespace normalization

Documents that are too short are removed, and extremely long documents are truncated.

Tokenized documents are then used as input for the embedding models.



# Task 1: Doc2Vec Document Embeddings

## Method

Doc2Vec learns document-level embeddings directly during training. Each Reddit post is 
represented as a dense vector that captures its semantic meaning.

Three initial configurations were tested:

| Config | Vector Size | Min Count | Epochs |
|------|------|------|------|
| D1 | 50 | 2 | 20 |
| D2 | 100 | 3 | 30 |
| D3 | 200 | 5 | 40 |

After the initial comparison, a broader parameter search was conducted using:

- different `min_count`
- different `epochs`
- both **DM** and **DBOW** architectures

The final semantic-best configuration selected was:
vector_size = 200
min_count = 3
epochs = 20
dm = 1

## Clustering

The document embeddings were clustered using **K-Means** with:
K = 8 clusters


Cluster quality was evaluated using:

- Silhouette Score (cosine distance)
- Calinski–Harabasz Index
- Davies–Bouldin Index
- Cluster balance statistics

---

# Task 2: Word2Vec-Based Document Representation

## Method

Instead of learning document vectors directly, this approach uses **Word2Vec word embeddings**.

Steps:

1. Train Word2Vec models with different embedding sizes:
   - 50
   - 100
   - 200

2. Cluster word embeddings into bins using **K-Means**

3. Represent each document using the **normalized frequency of word bins**

This produces a document representation known as a **bin-frequency representation**.

---

## Configurations

Three configurations were tested:

| Model | Vector Size | Number of Bins |
|------|------|------|
| A | 50 | 50 |
| B | 100 | 100 |
| C | 200 | 200 |

Among these, **Model A (50 bins)** achieved the best clustering performance according to 
internal evaluation metrics.

---

# Evaluation Metrics

Cluster quality was evaluated using three internal metrics:

### Silhouette Score
Measures how similar a document is to its own cluster compared to other clusters.

### Calinski–Harabasz Index
Measures cluster separation and compactness.

### Davies–Bouldin Index
Measures the average similarity between clusters (lower is better).

Additional diagnostics:

- cluster size distribution
- number of very small clusters
- cluster balance

---

# Visualization

Cluster structures were visualized using **PCA (Principal Component Analysis)** to project 
high-dimensional document embeddings into two dimensions.

Generated figures include:

- `doc2vec_best_pca.png`
- `doc2vec_semantic_best.png`
- `task2_best_w2v_pca.png`
- `w2v_semantic_best.png`

These plots help illustrate the spatial distribution of clusters.

---

# Comparative Analysis

Two embedding approaches were compared:

### Doc2Vec

- Learns document embeddings directly
- Captures document-level semantics
- Produces more balanced clusters
- Shows clearer topic separation

### Word2Vec + Bin Frequency

- Uses word embeddings to construct document representations
- Captures general topic signals
- Simpler representation
- Clusters can be broader and less semantically precise

Overall, **Doc2Vec produced more coherent document clusters**, especially for short 
technology-related Reddit posts.







