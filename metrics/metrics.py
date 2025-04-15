import numpy as np
import scipy.stats as sps

def dFrechet(P, Q, d = lambda x, y : np.linalg.norm(x - y)) :
    """Fréchet distance
    input
    ------
    P : array-like, real data
    Q : array-like, generated data
    d : callable, distance on the P[i] and Q[j]s
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

def dWasserstein(P, Q, d = None, niter = 1000) :
    '''
    input
    ------
    P : array-like of array-like, real data
    Q : array-like of array-like, generated data

    output
    ------
    d - Wasserstein distance between the two distributions computed as such :
    draw a random vector of norm one from the ND-sphere, project x and y on the vector line borne by this vector and compute 1D Wasserstein distance between the two projected vectors. Repeat and compute the mean
    '''
    
    def project(x, u) :
        '''
        input
        ------
        x - (n,N) array
        u - (N,1) array of norm 1

        output
        ------
        x_u - (1,n) array such that x_u[i] is the orthogonal projection of x[i,:] over uR (i.e. x_u[i] = <x[i,:], u>)
        '''
        x_u = np.matmul(x, u)
        return x_u.T[0]

    N = len(P[0])
    x = P if isinstance(P, np.ndarray) else np.array(P)
    y = Q if isinstance(Q, np.ndarray) else np.array(Q)

    dist = 0
    for _ in range(niter) :
        # draw a ND vector at random following a N(0, I_N)
        u = sps.norm.rvs(loc = 0, scale = 1, size = N)
        # divide the vector by its norm to have a random vector of the N-sphere
        u = (u/np.linalg.norm(u))[:,np.newaxis]
        # project x and y lines on u
        x_u = project(x, u)
        y_u = project(y, u)
        # compute Wasserstein 1D distance between both projected vectors by using wasserstein_distance function of SciPy Stats
        dist += sps.wasserstein_distance(x_u, y_u)
    # return the mean of alld distances computed
    return dist/niter

def INND(P, Q, d = lambda x, y : np.linalg.norm(x - y)) :
    """
    measures estimation of the distribution function
    input
    ------
    P : array-like, real data
    Q : array-like, generated data
    d : callable, distance on the P[i] and Q[j]s
    """
    p, q = len(P), len(Q)
    innd = 0
    for i in range(q) :
        innd += min([d(Q[i], P[j]) for j in range(p)])
    return innd/q

def ONND(P, Q, d = lambda x, y : np.linalg.norm(x - y)) :
    """
    measures flexibility of the estimation
    input
    ------
    P : array-like, real data
    Q : array-like, generated data
    d : callable, distance on the P[i] and Q[j]s
    """
    p, q = len(P), len(Q)
    onnd = 0
    for j in range(p) :
        onnd += min([d(Q[i], P[j]) for i in range(q)])
    return onnd/p

def ICD(P, Q, d = lambda x, y : np.linalg.norm(x - y)) :
    """
    accounts for mode collapse issues
    input
    ------
    P : array-like, real data
    Q : array-like, generated data
    d : callable, distance on the P[i] and Q[j]s
    """
    p, q = len(P), len(Q)
    icdP = 0
    for i in range(p) :
        for j in range(p) :
            icdP += d(P[i], P[j])
    icdQ = 0
    for i in range(q) :
        for j in range(q) :
            icdQ += d(Q[i], Q[j])
    return (icdP/(p*p)-icdQ/(q*q))**2

def score(P, Q, d = lambda x, y : np.linalg.norm(x - y)) :
    pass