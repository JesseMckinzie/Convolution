import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.image import imread
import seaborn as sns; sns.set()
from scipy import signal
from PIL import Image


im = imread("checker.jpg")

np.savetxt("checkers", im[0])


K = np.zeros((10,10))

K[:,:] = 1

K[:,0:5] = -1.

np.savetxt('K.txt', K)

img = np.zeros((100,100))

img[0:50, 0:40] = 1.
img[50:100, 0:60] = 1.

np.savetxt('img', img)

f1 = signal.convolve2d(img, K,  fillvalue=0)

result = np.loadtxt('result.txt')

difference = f1 - result

for row in difference:
    for i in row:
        assert i == 0

plt.figure(1)
plt.imshow(result)
plt.colorbar()

plt.figure(2)
plt.imshow(f1)
plt.colorbar()

plt.figure(3)
plt.imshow(difference)
plt.colorbar()
plt.show()
