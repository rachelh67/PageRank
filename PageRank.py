import numpy as np
from web_matrix import harvard_st

def build_web_graph(edges):
    """
    Constructs a web graph from a list of edges (directed links)

    Paramters:
    edges -- list of tuples (u, v) where u links to v

    Returns nodes and adj -- Nodes is a list of pages, adj represents the adjacency list
    """
    nodes = set()
    adj = {}

    for u, v in edges:
        #adding each page to the set of nodes
        nodes.add(u)
        nodes.add(v)
        #building adjacency list
        if u not in adj:
            adj[u] = []
        adj[u].append(v)
    #returning sorted list of nodes and adjacency list
    return sorted(nodes), adj

def build_transition_matrix(nodes, adj):
    """
    Constructs the transition matrix P from the web graph

    Parameters:
    nodes -- list of pages
    adj -- adjacency list representing the web graph

    Returns the transition matrix P
    """
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)} #creates a dict for node to index mapping
    P = np.zeros((n, n)) #n x n zero matrix

    #loop through the set of pages
    for u in nodes:
        if u in adj:
            #term 1/L(v_j)
            out_links = adj[u]
            prob = 1 / len(out_links)
            #assigning to the matrix
            for v in out_links:
                P[index[v], index[u]] = prob
        #dangling node case
        else:
            #assign probability 1/n
            for v in nodes:
                P[index[v], index[u]] = 1 / n
    return P

def build_google_matrix(P, alpha=0.85):
    """
    Constructs the Google matrix G from the transition matrix P

    Parameters:
    P -- transition matrix
    alpha -- damping factor (default 0.85)

    Returns the Google matrix G
    """
    n = P.shape[0]
    E = np.ones((n, n)) / n #term (1/n)11^T
    G = alpha * P + (1 - alpha) * E
    return G


def compute_pagerank(G, tol=1e-6, max_iter=100):
    """
    Computes the PageRank vector using the power iteration method

    Parameters:
    G -- Google matrix
    tol -- tolerance for convergence (default 1e-6)
    max_iter -- maximum number of iterations (default 100)

    Returns the PageRank vector
    """
    n = G.shape[0]
    r = np.ones(n) / n #initial PageRank, 1/n probability for all

    #repeatedly applying the Google matrix until convergence
    for _ in range(max_iter):
        r_new = G @ r
        #check for convergence
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new

    return r

def check_convergence(G, r):
    """
    Checks the convergence of the PageRank vector

    Parameters:
    G -- Google matrix
    r -- PageRank vector

    Checks if the column sums of G are close to 1, if the sum of r is close to 1, and if G @ r is close to r
    """
    col_dev = np.max(np.abs(np.sum(G, axis=0) - 1))
    eigen_error = np.linalg.norm(G @ r - r, 1)
    sum_r = np.sum(r)

    print(f"Column sums max deviation: {col_dev}")
    print(f"Sum of r: {sum_r}")
    print(f"Eigenvector error: {eigen_error}")

def main():
    i, j, v = harvard_st()
    edges = list(zip(i.tolist(), j.tolist()))
    print(f"Nodes: {max(max(i), max(j)) + 1}")
    print(f"Edges: {len(edges)}")

    nodes, adj = build_web_graph(edges)
    P = build_transition_matrix(nodes, adj)
    G = build_google_matrix(P)
    r = compute_pagerank(G)

    top_indices = np.argsort(-r)[:20]
    print("\nTop 20 pagerank nodes:")
    for idx in top_indices:
        print(f"Node {idx}: {r[idx]:.6f}")
    check_convergence(G, r)

if __name__ == "__main__":
    main()
