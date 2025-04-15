import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

def dWassersteinND(x, y, niter = 10000) :
    '''
    input
    ------
    x, y - (n,N) arrays containing n observation of N-D random vectors each (each line has the form [Y1..YN] where Yi is the yield of the i-th station)

    output
    ------
    d - Wasserstein distance between the two distributions computed as such :
    draw a random vector of norm one from the 4D-sphere, project x and y on the vector line borne by this vector and compute 1D Wasserstein distance between the two projected vectors. Repeat and compute the mean
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
    
    nn = np.array([10, 21, 46,   100,   215,   464,  1000,  2154,  4641, 10000])
    N = len(x[0])
    dist = 0
    dists = []
    for i in range(niter) :
        # draw a 4D vector at random following a N(0, I_N)
        u = sps.norm.rvs(loc = 0, scale = 1, size = N)
        # divide the vector by its norm to have a random vector of the N-sphere
        u = (u/np.linalg.norm(u))[:,np.newaxis]
        # project x and y lines on u
        x_u = project(x, u)
        y_u = project(y, u)
        # compute Wasserstein 1D distance between both projected vectors by using wasserstein_distance function of SciPy Stats
        dist += sps.wasserstein_distance(x_u, y_u)
        if i + 1 in nn :
            dists.append(dist/i)
    # return the mean of alld distances computed
    return dists

x = np.array([10, 21, 46,   100,   215,   464,  1000,  2154,  4641, 10000])
r = np.random.uniform(size = (1000, 4))
g = np.random.uniform(size = (1000, 4))
# y1 = dWassersteinND(r, g)
# y2 = dWassersteinND(r, g)
# y3 = dWassersteinND(r, g)
# y4 = dWassersteinND(r, g)

# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
# plt.plot(x, y4)
# plt.semilogx()
# plt.title("Evaluation de la distance de Wasserstein en fonction du nombre d'itérations")
# plt.xlabel("nombre d'itérations")
# plt.ylabel("distance de Wasserstein")
# plt.show()

# from time import time
# t = time()
# for _ in range(500) :
#     dWassersteinND(r, g, niter=1000)
# print((time()-t)/500)

from metrics.metrics import dFrechet, dWasserstein, ONND
# metrics = [dFrechet, dWasserstein, ONND]
# values = [0, 0, 0]

# N = 100
# for _ in range(N) :
#     r = np.random.uniform(size = (200, 4))
#     g = np.random.uniform(size = (200, 4))
#     for i in range(3) :
#         values[i]+=metrics[i](r, g)/N
# print("r (200, 4), g (200, 4) -", values)

# values= [0, 0, 0]
# for _ in range(N) :
#     r = np.random.uniform(size = (10, 4))
#     g = np.random.uniform(size = (100, 4))
#     # for i in range(3) :
#     values[1]+=metrics[1](r, g)/N
# print("r (100, 4), g (10, 4) -", values)

ndata = np.array([int(p) for p in np.logspace(1, 3, 10)])
valF = [0 for _ in range(10)]
valO = [0 for _ in range(10)]

N = 100

for _ in range(N) :
    r = np.random.uniform(size = (ndata[-1], 4))
    g = np.random.uniform(size = (ndata[-1], 4))
    for i in range(10) :
        valF[i] += dFrechet(r[:ndata[i]], g[:ndata[i]])/N
        valO[i] += ONND(r[:ndata[i]], g[:ndata[i]])/N
print("dFrechet -", valF)
print("ONND -", valO)