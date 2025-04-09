import numpy as np

def frechet_distance(P, Q, d = lambda x, y : np.linalg.norm(x - y)) :
    """Fréchet distance
    input
    -------
    P, Q : array-like, data to be compared
    d : callable, distance to be used on data contained by P and Q
    """
    p, q = len(P), len(Q)
    c = np.zeros((2, q))
    c[0, 0] = d(P[0], Q[0])
    c[1, 0] = max(c[0, 0], d(P[1], Q[0]))
    
    for j in range(1, q) :
        c[0, j] = max(c[0, j-1], d(P[0], Q[j]))
        c[1, j] = max(d(P[1], Q[j]), min(c[0, j], c[1, j-1], c[0, j-1]))
    
    for i in range(1, p) :
        c[0, 0] = c[1, 0]
        c[1, 0] = max(c[0, 0], d(P[i], Q[0]))
        for j in range(1, q) :
            c[0, j] = c[1, j]
            c[1, j] = max(d(P[i], Q[j]), min(c[0, j], c[1, j-1], c[0, j-1]))

    return c[-1,-1]