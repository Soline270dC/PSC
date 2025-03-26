import numpy as np

def DTW(P, Q, d) :
    """Dynamic Time Warping
    input
    -------
    P, Q : array-like, data to be compared
    d : callable, distance to be used on data contained by P and Q
    """
    p, q = len(P), len(Q)
    dtw = np.zeros((p,q))
    dtw[0, 0] = d(P[0], Q[0])

    for i in range(1, p) :
        dtw[i, 0] = dtw[i-1, 0] + d(P[i], Q[0])
    
    for j in range(1, q) :
        dtw[0, j] = dtw[0, j-1] + d(P[0], Q[j])
    
    for i in range(1, p) :
        for j in range(1, q) :
            dtw[i, j] = d(P[i], Q[j]) + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[-1,-1]