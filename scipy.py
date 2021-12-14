import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()
from scipy import signal

K = np.zeros((10,10))

K[:,:] = 1

K[:,0:5] = -1.

print(K)

img = np.zeros((100,100))

img[0:50, 0:40] = 1.
img[50:100, 0:60] = 1.

plt.figure(1)
plt.imshow(img)
plt.colorbar()

f1 = signal.convolve2d(img, K, mode='same')
#result = np.loadtxt("result.txt")

plt.figure(2)
plt.imshow(f1)
plt.colorbar()

"""
plt.figure(3)
plt.imshow(result)
plt.colorbar()
"""

plt.show()

