
lastfm_dir = None
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from implicit.bpr import BayesianPersonalizedRanking
import warnings
warnings.filterwarnings('ignore')

# ===============================
# PART 3: BPR Analysis
# ===============================

def load_lastfm_data(data_dir):
    """Load and preprocess Last.FM dataset"""
    
    print("Loading Last.FM dataset...")
    
    # Load user-artist listening data
    user_artists_file = os.path.join(data_dir, "user_artists.dat")
    
    # Read the file (tab-separated)
    df = pd.read_csv(user_artists_file, sep='\t', encoding='utf-8')
    
    print(f"Loaded dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Rename columns for consistency
    df = df.rename(columns={'userID': 'user_id', 'artistID': 'item_id'})
    
    print(f"Last.FM: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} artists")
    
    # Create binary implicit feedback (any listening = positive)
    df['rating'] = 1  # Implicit feedback: any interaction is positive
    
    return df[['user_id', 'item_id', 'rating']]

def load_movielens_data(data_dir):
    """Load and preprocess MovieLens 1M dataset"""
    
    print("Loading MovieLens 1M dataset...")
    
    # Load ratings
    ratings_file = os.path.join(data_dir, "ratings.dat")
    
    # MovieLens format: UserID::MovieID::Rating::Timestamp
    df = pd.read_csv(ratings_file, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], 
                     engine='python', encoding='utf-8')
    
    print(f"MovieLens: {len(df)} ratings, {df['user_id'].nunique()} users, {df['item_id'].nunique()} movies")
    
    # Convert to implicit feedback (any rating = positive interaction)
    df['rating'] = 1  # Implicit feedback: any rating is positive
    
    return df[['user_id', 'item_id', 'rating']]

def create_user_item_matrix(df):
    """Create sparse user-item interaction matrix"""
    
    # Map user and item IDs to continuous indices
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()
    
    user_id_map = {uid: i for i, uid in enumerate(user_ids)}
    item_id_map = {iid: i for i, iid in enumerate(item_ids)}
    
    # Map to matrix indices
    df_mapped = df.copy()
    df_mapped['user_idx'] = df_mapped['user_id'].map(user_id_map)
    df_mapped['item_idx'] = df_mapped['item_id'].map(item_id_map)
    
    # Create sparse matrix
    matrix = csr_matrix(
        (df_mapped['rating'], (df_mapped['user_idx'], df_mapped['item_idx'])),
        shape=(len(user_ids), len(item_ids))
    )
    
    return matrix, user_id_map, item_id_map

def split_data(matrix, test_size=0.2, random_state=42):
    """Split user-item matrix into train/test sets"""
    
    # Get all user-item pairs
    users, items = matrix.nonzero()
    ratings = matrix.data
    
    # Split indices
    train_idx, test_idx = train_test_split(
        range(len(users)), test_size=test_size, random_state=random_state
    )
    
    # Create train matrix
    train_users = users[train_idx]
    train_items = items[train_idx]
    train_ratings = ratings[train_idx]
    
    train_matrix = csr_matrix(
        (train_ratings, (train_users, train_items)), 
        shape=matrix.shape
    )
    
    # Create test set (user, item pairs)
    test_users = users[test_idx]
    test_items = items[test_idx]
    test_set = list(zip(test_users, test_items))
    
    return train_matrix, test_set

def evaluate_recommendations(model, train_matrix, test_set, k=10):
    """Calculate Precision@K and Recall@K"""
    
    n_users, n_items = train_matrix.shape
    
    precisions = []
    recalls = []
    
    # Group test items by user
    test_by_user = {}
    for user, item in test_set:
        if user not in test_by_user:
            test_by_user[user] = []
        test_by_user[user].append(item)
    
    for user in test_by_user:
        # Get recommendations for this user
        user_items = train_matrix[user].toarray().flatten()
        
        # Get items user hasn't interacted with in training
        unrated_items = np.where(user_items == 0)[0]
        
        if len(unrated_items) == 0:
            continue
            
        # Get scores for unrated items
        scores = model.user_factors[user].dot(model.item_factors[unrated_items].T)
        
        # Get top-k recommendations
        top_k_idx = np.argsort(scores)[-k:][::-1]
        recommended_items = unrated_items[top_k_idx]
        
        # Calculate precision and recall
        relevant_items = set(test_by_user[user])
        recommended_set = set(recommended_items)
        
        intersection = len(relevant_items.intersection(recommended_set))
        
        if len(recommended_set) > 0:
            precision = intersection / len(recommended_set)
        else:
            precision = 0
            
        if len(relevant_items) > 0:
            recall = intersection / len(relevant_items)
        else:
            recall = 0
            
        precisions.append(precision)
        recalls.append(recall)
    
    return np.mean(precisions), np.mean(recalls)

def experiment_A(dataset_name, matrix, n_repetitions=10):
    """Experiment A: Vary latent factors (10-100), fix k=10"""
    
    print(f"\nRunning Experiment A for {dataset_name}...")
    print("Varying latent factors (10-100), top-10 recommendations")
    
    factors_range = range(10, 101, 10)  # [10, 20, 30, ..., 100]
    
    results = {
        'factors': [],
        'precision_mean': [],
        'precision_std': [],
        'recall_mean': [],
        'recall_std': []
    }
    
    for factors in factors_range:
        print(f"  Testing {factors} factors...")
        
        precisions = []
        recalls = []
        
        for rep in range(n_repetitions):
            # Split data with different random seed each time
            train_matrix, test_set = split_data(matrix, random_state=42+rep)
            
            # Train BPR model
            model = BayesianPersonalizedRanking(factors=factors, iterations=100, random_state=42+rep)
            model.fit(train_matrix)
            
            # Evaluate
            precision, recall = evaluate_recommendations(model, train_matrix, test_set, k=10)
            precisions.append(precision)
            recalls.append(recall)
        
        # Store results
        results['factors'].append(factors)
        results['precision_mean'].append(np.mean(precisions))
        results['precision_std'].append(np.std(precisions))
        results['recall_mean'].append(np.mean(recalls))
        results['recall_std'].append(np.std(recalls))
        
        print(f"    Precision@10: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"    Recall@10: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    
    return results

def experiment_B(dataset_name, matrix, n_repetitions=10):
    """Experiment B: Vary k (2-20), fix factors=50"""
    
    print(f"\nRunning Experiment B for {dataset_name}...")
    print("Varying top-k recommendations (2-20), 50 latent factors")
    
    k_range = range(2, 21, 2)  # [2, 4, 6, ..., 20]
    
    results = {
        'k': [],
        'precision_mean': [],
        'precision_std': [],
        'recall_mean': [],
        'recall_std': []
    }
    
    for k in k_range:
        print(f"  Testing top-{k} recommendations...")
        
        precisions = []
        recalls = []
        
        for rep in range(n_repetitions):
            # Split data with different random seed each time
            train_matrix, test_set = split_data(matrix, random_state=42+rep)
            
            # Train BPR model with fixed 50 factors
            model = BayesianPersonalizedRanking(factors=50, iterations=100, random_state=42+rep)
            model.fit(train_matrix)
            
            # Evaluate with varying k
            precision, recall = evaluate_recommendations(model, train_matrix, test_set, k=k)
            precisions.append(precision)
            recalls.append(recall)
        
        # Store results
        results['k'].append(k)
        results['precision_mean'].append(np.mean(precisions))
        results['precision_std'].append(np.std(precisions))
        results['recall_mean'].append(np.mean(recalls))
        results['recall_std'].append(np.std(recalls))
        
        print(f"    Precision@{k}: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"    Recall@{k}: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    
    return results

def plot_single_dataset_results(exp_a, exp_b, dataset_name):
    """Create plots for a single dataset"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Experiment A
    ax = axes[0]
    x = exp_a['factors']
    
    ax.errorbar(x, exp_a['precision_mean'], yerr=exp_a['precision_std'], 
                label='Precision@10', marker='o', capsize=5)
    ax.errorbar(x, exp_a['recall_mean'], yerr=exp_a['recall_std'], 
                label='Recall@10', marker='s', capsize=5)
    
    ax.set_xlabel('Number of Latent Factors')
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name}: Precision & Recall vs Latent Factors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Experiment B
    ax = axes[1]
    x = exp_b['k']
    
    ax.errorbar(x, exp_b['precision_mean'], yerr=exp_b['precision_std'], 
                label='Precision@K', marker='o', capsize=5)
    ax.errorbar(x, exp_b['recall_mean'], yerr=exp_b['recall_std'], 
                label='Recall@K', marker='s', capsize=5)
    
    ax.set_xlabel('K (Number of Recommendations)')
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name}: Precision & Recall vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'bpr_experiments_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
    print(f"Plots saved as 'bpr_experiments_{dataset_name.lower()}.png'")

def plot_results(lastfm_exp_a, lastfm_exp_b, movielens_exp_a, movielens_exp_b):
    """Create the 4 required plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Last.FM Experiment A
    ax = axes[0, 0]
    x = lastfm_exp_a['factors']
    
    ax.errorbar(x, lastfm_exp_a['precision_mean'], yerr=lastfm_exp_a['precision_std'], 
                label='Precision@10', marker='o', capsize=5)
    ax.errorbar(x, lastfm_exp_a['recall_mean'], yerr=lastfm_exp_a['recall_std'], 
                label='Recall@10', marker='s', capsize=5)
    
    ax.set_xlabel('Number of Latent Factors')
    ax.set_ylabel('Score')
    ax.set_title('Last.FM: Precision & Recall vs Latent Factors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Last.FM Experiment B
    ax = axes[0, 1]
    x = lastfm_exp_b['k']
    
    ax.errorbar(x, lastfm_exp_b['precision_mean'], yerr=lastfm_exp_b['precision_std'], 
                label='Precision@K', marker='o', capsize=5)
    ax.errorbar(x, lastfm_exp_b['recall_mean'], yerr=lastfm_exp_b['recall_std'], 
                label='Recall@K', marker='s', capsize=5)
    
    ax.set_xlabel('K (Number of Recommendations)')
    ax.set_ylabel('Score')
    ax.set_title('Last.FM: Precision & Recall vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: MovieLens Experiment A
    ax = axes[1, 0]
    x = movielens_exp_a['factors']
    
    ax.errorbar(x, movielens_exp_a['precision_mean'], yerr=movielens_exp_a['precision_std'], 
                label='Precision@10', marker='o', capsize=5)
    ax.errorbar(x, movielens_exp_a['recall_mean'], yerr=movielens_exp_a['recall_std'], 
                label='Recall@10', marker='s', capsize=5)
    
    ax.set_xlabel('Number of Latent Factors')
    ax.set_ylabel('Score')
    ax.set_title('MovieLens: Precision & Recall vs Latent Factors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: MovieLens Experiment B
    ax = axes[1, 1]
    x = movielens_exp_b['k']
    
    ax.errorbar(x, movielens_exp_b['precision_mean'], yerr=movielens_exp_b['precision_std'], 
                label='Precision@K', marker='o', capsize=5)
    ax.errorbar(x, movielens_exp_b['recall_mean'], yerr=movielens_exp_b['recall_std'], 
                label='Recall@K', marker='s', capsize=5)
    
    ax.set_xlabel('K (Number of Recommendations)')
    ax.set_ylabel('Score')
    ax.set_title('MovieLens: Precision & Recall vs K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bpr_experiments.png', dpi=300, bbox_inches='tight')
    print("All plots saved as 'bpr_experiments.png'")

def run_bpr_analysis():
    """Main function to run complete BPR analysis"""
    
    print("=" * 70)
    print("INFORMATION RETRIEVAL PROJECT - PART 3")
    print("Bayesian Personalized Ranking (BPR) Analysis")
    print("=" * 70)
    
    # Step 1: Check for existing datasets
    lastfm_dir = "data/hetrec2011-lastfm-2k"  # Example path for Last.FM dataset
    movielens_dir = "data/ml-1m"
    
    if lastfm_dir is None and movielens_dir is None:
        print("No datasets found. Please ensure datasets are in the data directory.")
        return None
    
    # Step 2: Load and preprocess data
    lastfm_df = None
    movielens_df = None
    
    if lastfm_dir is not None:
        lastfm_df = load_lastfm_data(lastfm_dir)
    
    if movielens_dir is not None:
        movielens_df = load_movielens_data(movielens_dir)
    
    if lastfm_df is None and movielens_df is None:
        print("Failed to load any datasets. Exiting.")
        return None
    
    # Step 3: Create user-item matrices
    print("\nCreating user-item matrices...")
    
    lastfm_matrix = None
    movielens_matrix = None
    
    if lastfm_df is not None:
        lastfm_matrix, _, _ = create_user_item_matrix(lastfm_df)
        print(f"Last.FM matrix shape: {lastfm_matrix.shape}")
    
    if movielens_df is not None:
        movielens_matrix, _, _ = create_user_item_matrix(movielens_df)
        print(f"MovieLens matrix shape: {movielens_matrix.shape}")
    
    # Step 4: Run experiments
    print("\n" + "="*50)
    print("RUNNING EXPERIMENTS (This may take 10-20 minutes)")
    print("="*50)
    
    # Initialize results
    lastfm_exp_a = lastfm_exp_b = None
    movielens_exp_a = movielens_exp_b = None
    
    # Run experiments for available datasets
    if lastfm_matrix is not None:
        print("\nRunning Last.FM experiments...")
        lastfm_exp_a = experiment_A("Last.FM", lastfm_matrix, n_repetitions=10)
        lastfm_exp_b = experiment_B("Last.FM", lastfm_matrix, n_repetitions=10)
    else:
        print("Skipping Last.FM experiments (data not available)")
    
    if movielens_matrix is not None:
        print("\nRunning MovieLens experiments...")
        movielens_exp_a = experiment_A("MovieLens", movielens_matrix, n_repetitions=10)
        movielens_exp_b = experiment_B("MovieLens", movielens_matrix, n_repetitions=10)
    else:
        print("Skipping MovieLens experiments (data not available)")
    
    # Step 5: Create plots (only for available data)
    if lastfm_exp_a and movielens_exp_a:
        plot_results(lastfm_exp_a, lastfm_exp_b, movielens_exp_a, movielens_exp_b)
    elif lastfm_exp_a:
        print("Creating plots for Last.FM only...")
        plot_single_dataset_results(lastfm_exp_a, lastfm_exp_b, "Last.FM")
    elif movielens_exp_a:
        print("Creating plots for MovieLens only...")
        plot_single_dataset_results(movielens_exp_a, movielens_exp_b, "MovieLens")
    
    # Step 6: Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY:")
    print("="*70)
    if lastfm_exp_a:
        print("✓ Last.FM: Experiment A & B completed")
    if movielens_exp_a:
        print("✓ MovieLens: Experiment A & B completed") 
    print("✓ All experiments repeated 10 times")
    print("✓ Mean and standard deviation calculated")
    print("✓ Plots generated and saved")
    print("="*70)
    
    return lastfm_exp_a, lastfm_exp_b, movielens_exp_a, movielens_exp_b

if __name__ == "__main__":
    # Run the complete BPR analysis
    results = run_bpr_analysis()