import numpy as np
from random import randrange

def get_random_matrix(image_size, max_pixel_value):
    image = np.zeros((image_size, image_size))

    for i in range(image_size):
        for j in range(image_size):
            image[i][j] = randrange(max_pixel_value+1)

    return image
    

def get_four_kernels(start_size, image_size):
    end_size = image_size//4
    step_size = (end_size - start_size) // 3
    sizes = np.arange(start_size, end_size+1, step_size).tolist()
    sizes[-1] = end_size

    if(len(sizes) != 4):
        print("Size not length 4")
        exit(1)

    for size in sizes:
        kernel = get_random_matrix(size, 255)
        name = 'K' + str(size) + '_' + str(image_size) + '.txt'
        np.savetxt(name, kernel)


smallest = 256
largest = 5000

image_sizes = []


size = smallest
image_sizes.append(size)
while(size <= largest):
    size += 1000
    image_sizes.append(size)
    

for image_size in image_sizes:
    image = get_random_matrix(image_size, 255)
    image_name = 'img' + str(image_size) + '.txt'
    np.savetxt(image_name, image)
    get_four_kernels(32, image_size)



"""
kernel_start_size = 32

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
    np.savetxt(path + name, K)

    kernel_size *= increment

    # create image
    image = np.zeros((image_size, image_size))

    for i in range(image_size):
        for j in range(image_size):
            image[i][j] = randrange(max_pixel_value+1)

    name ="img{}".format(image_size)
    np.savetxt(path + name, image)

    image_size *= increment
"""
