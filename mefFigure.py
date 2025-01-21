import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from grid_search.usefulFunctions import getGrid

scores = np.load("grid_search/result.npy")
testing = [[1, 3, 5], [30, 50]]
grid = getGrid(*testing)
logScores = np.log(scores)
amin = (scores.argmin()%len(grid), scores.argmin()//len(grid))
labelsx = [x[0] for x in grid[0]]
labelsN = [len(x[0]) for x in grid[0]]

fig = plt.figure()
ax = fig.add_subplot()
cont = ax.contourf(logScores, locator=ticker.LogLocator(subs = 'all'))
ax.set_xlabel("Complexity of the Generator (number of layers)")
ax.set_ylabel("Complexity of the Discriminator (number of layers)")
ax.set_xticks([0, 2, 10], ['1', '3', '5'])
ax.set_yticks([0, 2, 10], ['1', '3', '5'])
plt.colorbar(cont)
ax.scatter(*amin, c = 'red', marker= "x", label = "minimal $d_W$")
ax.legend()
fig.savefig("grid_search/result")
# plt.show()