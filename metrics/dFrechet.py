import numpy as np

def frechet_distance(P, Q, d) :
    """Fréchet distance
    input
    -------
    P, Q : array-like, data to be compared
    d : callable, distance to be used on data contained by P and Q
    """
    p, q = len(P), len(Q)
    c = np.zeros((p,q))
    c[0, 0] = d(P[0], Q[0])

    for i in range(1, p) :
        c[i, 0] = max(c[i-1, 0], d(P[i], Q[0]))
    
    for j in range(1, q) :
        c[0, j] = max(c[0, j-1], d(P[0], Q[j]))
    
    for i in range(1, p) :
        for j in range(1, q) :
            c[i, j] = max(d(P[i], Q[j]), min(c[i-1, j], c[i, j-1], c[i-1, j-1]))
            
    return c[-1,-1]
