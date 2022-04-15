import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.image import imread
import seaborn as sns; sns.set()
from scipy import signal
from PIL import Image

im = imread("/home/ec2-user/Dev/Convolution/Convolution/img/checker.jpg")

np.savetxt("checkers.txt", im[0])

K = np.zeros((10,10))

K[:,:] = 1

K[:,0:5] = -1.

np.savetxt('K.txt', K)

img = np.zeros((100,100))

img[0:50, 0:40] = 1.
img[50:100, 0:60] = 1.

np.savetxt('img.txt', img)

f1 = signal.convolve2d(img, K,  fillvalue=0)

result = np.loadtxt('result_opt.txt')

difference = f1 - result

#for row in difference:
#    for i in row:
#        assert i == 0

plt.figure(1)
plt.imshow(result)
plt.colorbar()
plt.savefig('1.png')

plt.figure(2)
plt.imshow(f1)
plt.colorbar()
plt.savefig('2.png')

plt.figure(3)
plt.imshow(difference)
plt.colorbar()
plt.savefig('3.png')
plt.show()
