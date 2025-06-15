# Information Retrieval Project

## Overview
University assignment implementing core Information Retrieval algorithms and analysis.

## Parts

### 1. PageRank & HITS Analysis
- PageRank with personalization vector (50% weight to node 14)
- Damping factor analysis (α = 0.55 to 0.95)
- HITS algorithm (authorities & hubs)
- Eigenvalue analysis before/after adding edges

### 2. Zipf's Law Analysis  
- Analysis of "Pride and Prejudice" by Jane Austen
- Zipf constant calculation: 19,374.66
- R² = 0.0878 (moderate fit to Zipf's law)

### 3. Bayesian Personalized Ranking (BPR)
- Recommendation system experiments
- Datasets: Last.FM and MovieLens 1M
- Analysis of latent factors (10-100) and top-k recommendations (2-20)

## Key Results
- **PageRank**: Perfect ranking stability (Kendall τ = 1.0) across damping factors
- **Eigenvalues**: +67% improvement in primary eigenvalue after strategic edge addition
- **BPR**: MovieLens outperforms Last.FM (~2x higher precision/recall)
