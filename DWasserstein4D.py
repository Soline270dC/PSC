import numpy as np
import scipy.stats as sps

def project(x, u) :
    '''
    ==input==
    x - n*4 array
    u - 4*1 array of norm 1

    ==output==
    x_u - 1*n array such that x_u[i] is the orthogonal projection of x[i,:] over uR (i.e. x_u[i] = x[i,:].u)
    '''
    x_u = np.matmul(x, u)
    return x_u.T[0]

def dWasserstein4D(x, y, niter = 100) :
    '''
    ==input==
    x, y - n*4 arrays containing n observation of 4D random vectors each

    ==output==
    d - Wasserstein distance between the two distributions computed as such : draw a random vector of norm one from the 4D-sphere, project x and y on the vector line borne by this vector and compute 1D Wasserstein distance between the two projected vectors. Repeat and compute the mean
    '''
    dist = 0
    for i in range(niter) :
        # draw a 4D vector at random following a N(0, I_4)
        u = sps.norm.rvs(loc = 0, scale = 1, size = 4)
        u = (u/np.linalg.norm(u))[:,np.newaxis]
        x_u = project(x, u)
        y_u = project(y, u)
        dist += sps.wasserstein_distance(x_u, y_u)
    return dist/niter