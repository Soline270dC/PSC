import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from grid_search.usefulFunctions2 import getGrid2

name = "result2nd-2"
savename = "lr-beta1_2"
testing = [np.linspace(6e-5, 1e-4, 5), np.linspace(0.2, 0.8, 5), list(range(40, 71, 5)), list(range(20, 51, 5))]
parameters = ['lr', 'beta1', 'num_epoch', 'nz']
# ticks = [(t, [str(x) for x in t]) for t in testing]
# ntest = len(testing)
# meanvalues = [len(t)//2 for t in testing]

scores = np.load("grid_search/"+name+".npy")
grid = getGrid2(*testing)
logScores = np.log(scores)
arg_min = np.unravel_index(scores.argmin(), scores.shape)
print(grid[arg_min[0]][arg_min[1]][arg_min[2]][arg_min[3]])
print([testing[i][arg_min[3-i]] for i in range(4)])
print(scores.shape)

# couples = [(s, t) for s in range(ntest) for t in range(ntest) if t > s]

'''
fig = plt.figure(figsize = (10, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(testing[3], scores[:, arg_min[1], arg_min[2], arg_min[3]])
ax1.set_title(parameters[3])

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(testing[2], scores[arg_min[0], :, arg_min[2], arg_min[3]])
ax2.set_title(parameters[2])

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(testing[1], scores[arg_min[0], arg_min[1], :, arg_min[3]])
ax3.set_title(parameters[1])

ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(testing[0], scores[arg_min[0], arg_min[1], arg_min[2], :])
ax3.set_title(parameters[0])
'''

fig = plt.figure()
ax = fig.add_subplot()
cont = ax.contourf(scores[arg_min[3],arg_min[2],:,:], locator=ticker.LogLocator(subs = 'all'))
ax.set_xlabel("lr")
ax.set_ylabel("beta1")
ax.set_xticks(range(len(testing[0])), testing[0].round(5))
ax.set_yticks(range(len(testing[1])), testing[1].round(3))
plt.colorbar(cont)
ax.scatter(arg_min[1], arg_min[3], c = 'red', marker= "x", label = "minimal $d_W$")
ax.legend()

# scores[arg_min[0], arg_min[1], arg_min[2], arg_min[3]]

# for s, t in couples :
#     ax = fig.add_subplot()

#     toPlot = [()]
#     cont = ax.contourf(toPlot, locator=ticker.LogLocator(subs = 'all'))
#     ax.set_xlabel(parameters[s])
#     ax.set_ylabel(parameters[t])
#     ax.set_xticks(*ticks[s])
#     ax.set_yticks(*ticks[t])
#     plt.colorbar(cont)
#     ax.scatter(amin[s], amin[t], c = 'red', marker= "x", label = "minimal $d_W$")
#     ax.legend()


# fig.savefig("grid_search/"+savename)
plt.show()