import numpy as np

def DTW(P, Q, d = lambda x, y : np.linalg.norm(x - y)) :
    """Dynamic Time Warping
    input
    -------
    P, Q : array-like, data to be compared
    d : callable, distance to be used on data contained by P and Q
    """
    p, q = len(P), len(Q)
    dtw = np.zeros((2,q))
    dtw[0, 0] = d(P[0], Q[0])
    dtw[1, 0] = dtw[0, 0] + d(P[1], Q[0])
    
    for j in range(1, q) :
        dtw[0, j] = dtw[0, j-1] + d(P[0], Q[j])
        dtw[1, j] = min(dtw[0, j], dtw[1, j-1], dtw[0, j-1]) + d(P[1], Q[j])
    
    for i in range(1, p) :
        dtw[0, 0] = dtw[1, 0]
        dtw[1, 0] = dtw[0, 0] + d(P[i], Q[0])
        for j in range(1, q) :
            dtw[0, j] = dtw[1, j]
            dtw[1, j] = d(P[i], Q[j]) + min(dtw[0, j], dtw[1, j-1], dtw[0, j-1])
    
    return dtw[-1,-1]