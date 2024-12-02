import numpy as np

def dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw = np.zeros((n+1, m+1))
    dtw[0, 1:] = np.inf
    dtw[1:, 0] = np.inf

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

    return dtw[n, m]

def compute_dtw(reference, target):
    n, m = len(reference), len(target)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(reference[i - 1] - target[j - 1])  # Euclidean distance
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # Insertion
                dtw_matrix[i, j - 1],    # Deletion
                dtw_matrix[i - 1, j - 1] # Match
            )

    # Backtracking, optimal path
    i, j = n, m
    path = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        choices = [
            (dtw_matrix[i - 1, j], i - 1, j),    # Up
            (dtw_matrix[i, j - 1], i, j - 1),    # Left
            (dtw_matrix[i - 1, j - 1], i - 1, j - 1) # Diagonal
        ]
        _, i, j = min(choices)

    path.reverse()
    return dtw_matrix[n, m], path
