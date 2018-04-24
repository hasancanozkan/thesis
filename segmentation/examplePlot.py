'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np



fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(1, 2, 0.25)
Y = np.arange(0, 0.75, 0.25)
X, Y = np.meshgrid(X, Y)
emArray = np.empty(shape=(3,4))

emArray[0] = [1,2,3,4]
emArray[1] = [1.5,2.5,3.5,4.5]
emArray[[2]] =(5,6,7,8)
Z = emArray
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# Make data.
X = np.arange(1, 2, 0.25)
#print str(X) + '\n'
#print X
#print X.ndim
Y = np.arange(0, 0.75, 0.25)
#print str(Y) + '\n'

X, Y = np.meshgrid(X, Y)

emArray = np.empty(shape=(3,4))

emArray[0] = [1,2,3,4]
emArray[1] = [1.5,2.5,3.5,4.5]
emArray[[2]] =(5,6,7,8)

print emArray.ndim
print len(emArray)
print 'here'
print emArray
