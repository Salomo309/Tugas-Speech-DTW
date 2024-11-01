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
