import numpy as np

def frechet_distance(P, Q, d) :
    """Fréchet distance
    input
    -------
    P, Q : array-like, data to be compared
    d : callable, distance to be used on data contained by P and Q
    """
    p, q = len(P), len(Q)
    ca = np.full((p, q), -1.)

    def c(i, j, ca = ca) :
        assert 0 <= i < p and 0 <= j < q
        if ca[i, j] > -1 :
            return ca[i, j]
        elif i == 0 and j == 0 :
            ca[i, j] = d(P[i], Q[j])
        elif i > 0 and j == 0 :
            ca[i, j] = max(c[i-1, j], d(P[i], Q[j]))
        elif i == 0 and j > 0 :
            ca[i, j] = max(c[i, j-1], d(P[i], Q[j]))
        elif i > 0 and j > 0 :
            ca[i, j] = max(min(c[i-1, j], c[i-1, j-1], c[i, j-1]), d(P[i], P[j]))
        else :
            ca[i, j] = np.inf
        return ca[i, j]
    
    return c(p-1, q-1)
