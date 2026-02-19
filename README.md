# PageRank
This project implements the PageRank algorithm in Python and applies to the harvard_st dataset from the web_matrix file. 

Mathematical formulation:
Let P be the transition matrix, alpha in (0,1) be the damping factor (default 0.85), and n be the number of pages. The google matrix is G = alpha(P) + (1-alpha)(1/n)11^T where 11^T is a matrix of ones. The PageRank vector r satisfies Gr = r. r is computed using power iteration r_{k+1}=Gr_k until convergence.

What this program does:
Loads the harvard_st dataset
Builds an adjacency list
Constructs a transition matrix (pure link-following probabilities)
Builds the Google matrix using the transition matrix and damping factor
Runs power iteration
Prints the top 20 ranked nodes
Verifies convergence numerically
