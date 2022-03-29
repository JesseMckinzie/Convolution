import numpy as np
from random import randrange

kernel_start_size = 10

image_start_size = 100

number_of_points = 12

max_pixel_value = 255

increment = 2

kernel_size = kernel_start_size
image_size = image_start_size
for i in range(1, number_of_points+1):
    print("Generating data for image size = " + str(image_size))
    # create kernel
    K = np.zeros((kernel_size, kernel_size))

    K[:,:] = 1
    K[:,0:kernel_size//2] = -1.

    name ="K{}".format(kernel_size)
    np.savetxt(name, K)

    kernel_size *= increment

    # create image
    image = np.zeros((image_size, image_size))

    for i in range(image_size):
        for j in range(image_size):
            image[i][j] = randrange(max_pixel_value+1)

    name ="img{}".format(image_size)
    np.savetxt(name, image)

    image_size *= increment

