import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.linalg import eig

# ===============================
# PART 1A: Basic PageRank
# ===============================

def calculate_pagerank_1A():
    """Calculate PageRank for Part 1A with personalization"""
    
    # Create undirected graph (bidirectional links)
    G = nx.Graph()
    
    
    edges = [
        (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (2, 3), (3, 4), (3, 6), (4, 5), (4, 7),
        (5, 10), (6, 7), (6, 8), (6, 9), (7, 9), (8, 9), (10, 11), (10, 12), (10, 14), 
        (11, 13), (11, 14), (12, 13), (12, 14), (13, 14)
    ]
    G.add_edges_from(edges)
    
    # Create personalization vector: 50% to node 14, rest distributed equally
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    personalization = {}
    
    for node in nodes:
        if node == 14:
            personalization[node] = 0.5
        else:
            personalization[node] = 0.5 / (n_nodes - 1)
    
    # Calculate PageRank
    pagerank_values = nx.pagerank(G, alpha=0.65, personalization=personalization)
    
    # Sort by PageRank value
    sorted_pagerank = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)
    
    print("Part 1A - PageRank Results:")
    print("Node\tPageRank Value")
    print("-" * 25)
    for node, value in sorted_pagerank:
        print(f"{node}\t{value:.6f}")
    
    return pagerank_values

# ===============================
# PART 1B: Damping Factor Analysis
# ===============================

def calculate_pagerank_1B():
    """Analyze damping factor effects"""
    
    # Create directed graph
    G = nx.DiGraph()
    
    edges = [
        (0, 6), (0, 7), (0, 1), (1, 2), (1, 7), (2, 1), (2, 7),
        (3, 5), (3, 7), (4, 5), (5, 6), (6, 5), (7, 6)
    ]
    G.add_edges_from(edges)
    
    # Damping factors to test
    alphas = [0.55, 0.65, 0.75, 0.85, 0.95]
    
    # Store results
    results = {}
    rankings = {}
    
    for alpha in alphas:
        pagerank_values = nx.pagerank(G, alpha=alpha, max_iter=1000, tol=1e-06)
        results[alpha] = pagerank_values
        
        # Create ranking (sorted by PageRank value)
        sorted_nodes = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)
        rankings[alpha] = [node for node, _ in sorted_nodes]

    print("\n=== DEBUG: Checking Rankings ===")
    for alpha in alphas:
        print(f"α={alpha}: {rankings[alpha]}")
    
    # Plot PageRank vs Alpha for each node
    plot_pagerank_vs_alpha(results, alphas)
    
    # Calculate Kendall Tau correlations
    calculate_kendall_correlations(rankings, alphas)
    
    return results, rankings

def plot_pagerank_vs_alpha(results, alphas):
    """Create 8 separate plots showing PageRank vs Alpha for each node"""
    
    nodes = list(results[alphas[0]].keys())
    
    # Create subplots (2x4 grid for 8 nodes)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, node in enumerate(nodes):
        pagerank_values = [results[alpha][node] for alpha in alphas]
        
        axes[i].plot(alphas, pagerank_values, 'bo-', linewidth=2, markersize=6)
        axes[i].set_title(f'Node {node}')
        axes[i].set_xlabel('Damping Factor (α)')
        axes[i].set_ylabel('PageRank Value')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pagerank_vs_alpha.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_kendall_correlations(rankings, alphas):
    """Calculate Kendall Tau correlations between rankings"""
    
    n_alphas = len(alphas)
    correlation_matrix = np.zeros((n_alphas, n_alphas))
    
    print("\nKendall Tau Correlations between Rankings:")
    print("Alpha values:", alphas)
    print("-" * 50)
    
    for i, alpha1 in enumerate(alphas):
        for j, alpha2 in enumerate(alphas):
            if i <= j:
                tau, p_value = kendalltau(rankings[alpha1], rankings[alpha2])
                correlation_matrix[i][j] = tau
                correlation_matrix[j][i] = tau
                
                if i != j:
                    print(f"α={alpha1} vs α={alpha2}: τ = {tau:.4f} (p = {p_value:.4f})")
    
    return correlation_matrix

# ===============================
# PART 1C: HITS Algorithm
# ===============================

def calculate_hits():
    """Calculate HITS authorities and hubs"""
    
    # Use same graph as Part 1B
    G = nx.DiGraph()
    
    edges = [
        (5, 11), (11, 2), (11, 10), (11, 9), (7, 11), (7, 8), (8, 9), (3, 10), (3, 8)
    ]
    G.add_edges_from(edges)
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(G).toarray()

    print("Nodes:", list(G.nodes()))
    print("Edges:", list(G.edges()))
    print("Adjacency Matrix:\n", A)
    print(A)
     
    # Calculate authorities matrix: A^T * A
    authorities_matrix = A.T @ A
    
    # Calculate hubs matrix: A * A^T  
    hubs_matrix = A @ A.T
    
    # Calculate principal eigenvectors
    auth_eigenvals, auth_eigenvecs = eig(authorities_matrix)
    hub_eigenvals, hub_eigenvecs = eig(hubs_matrix)
    
    # Get principal eigenvector (largest eigenvalue)
    auth_principal = auth_eigenvecs[:, np.argmax(auth_eigenvals.real)]
    hub_principal = hub_eigenvecs[:, np.argmax(hub_eigenvals.real)]
    
    # Make sure eigenvectors are real and positive
    auth_principal = np.abs(auth_principal.real)
    hub_principal = np.abs(hub_principal.real)
    
    # Normalize
    auth_principal = auth_principal / np.sum(auth_principal)
    hub_principal = hub_principal / np.sum(hub_principal)
    
    print("Part 1C - HITS Results:")
    print("\nAuthorities Matrix:")
    print(authorities_matrix)
    print("\nHubs Matrix:")
    print(hubs_matrix)
    print("\nAuthorities Principal Eigenvector:")
    print(auth_principal)
    print("\nHubs Principal Eigenvector:")
    print(hub_principal)
    
    return authorities_matrix, hubs_matrix, auth_principal, hub_principal

# ===============================
# PART 1D: Eigenvalue Analysis
# ===============================

def calculate_eigenvalues_1D():
    """Compare eigenvalues before and after adding edges"""
    
    # Original graph from Part 1A
    G_original = nx.Graph()
    
    # TODO: Add edges from Part 1A
    original_edges = [
        (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (2, 3), (3, 4), (3, 6), (4, 5), (4, 7),
        (5, 10), (6, 7), (6, 8), (6, 9), (7, 9), (8, 9), (10, 11), (10, 12), (10, 14), 
        (11, 13), (11, 14), (12, 13), (12, 14), (13, 14)
    ]
    G_original.add_edges_from(original_edges)
    
    # Calculate PageRank matrix for original graph
    A_original = nx.adjacency_matrix(G_original).toarray()
    
    # Create modified graph with additional edges
    G_modified = G_original.copy()
    new_edges = [(5,11), (4,11), (7,13), (8,12), (9,13)]
    G_modified.add_edges_from(new_edges)
    
    A_modified = nx.adjacency_matrix(G_modified).toarray()
    
    # Calculate eigenvalues
    eigenvals_original = np.linalg.eigvals(A_original)
    eigenvals_modified = np.linalg.eigvals(A_modified)
    
    # Sort eigenvalues in descending order
    eigenvals_original = np.sort(eigenvals_original.real)[::-1]
    eigenvals_modified = np.sort(eigenvals_modified.real)[::-1]
    
    print("Part 1D - Eigenvalue Analysis:")
    print("\nOriginal Graph Eigenvalues:")
    for i, val in enumerate(eigenvals_original):
        print(f"λ{i+1} = {val:.6f}")
    
    print("\nModified Graph Eigenvalues:")
    for i, val in enumerate(eigenvals_modified):
        print(f"λ{i+1} = {val:.6f}")
    
    print("\nChanges in Eigenvalues:")
    for i in range(min(len(eigenvals_original), len(eigenvals_modified))):
        change = eigenvals_modified[i] - eigenvals_original[i]
        print(f"Δλ{i+1} = {change:.6f}")
    
    return eigenvals_original, eigenvals_modified

# ===============================
# MAIN EXECUTION
# ===============================

def run_all_parts():
    """Run all parts of the PageRank analysis"""
    
    print("=" * 60)
    print("INFORMATION RETRIEVAL PROJECT - PART 1")
    print("=" * 60)
    
    # Part 1A
    pagerank_1A = calculate_pagerank_1A()
    
    print("\n" + "=" * 60)
    
    # Part 1B  
    results_1B, rankings_1B = calculate_pagerank_1B()
    
    print("\n" + "=" * 60)
    
    # Part 1C
    auth_matrix, hub_matrix, auth_eigenvec, hub_eigenvec = calculate_hits()
    
    print("\n" + "=" * 60)
    
    # Part 1D
    eigenvals_orig, eigenvals_mod = calculate_eigenvalues_1D()

if __name__ == "__main__":
    run_all_parts()