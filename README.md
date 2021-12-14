# Convolution
The kernel is: <br />
[[-1. -1. -1. -1. -1.  1.  1.  1.  1.  1.] <br />
 [-1. -1. -1. -1. -1.  1.  1.  1.  1.  1.] <br />
 [-1. -1. -1. -1. -1.  1.  1.  1.  1.  1.] <br />
 [-1. -1. -1. -1. -1.  1.  1.  1.  1.  1.] <br />
 [-1. -1. -1. -1. -1.  1.  1.  1.  1.  1.] <br />
 [-1. -1. -1. -1. -1.  1.  1.  1.  1.  1.] <br />
 [-1. -1. -1. -1. -1.  1.  1.  1.  1.  1.] <br />
 [-1. -1. -1. -1. -1.  1.  1.  1.  1.  1.] <br />
 [-1. -1. -1. -1. -1.  1.  1.  1.  1.  1.] <br />
 [-1. -1. -1. -1. -1.  1.  1.  1.  1.  1.]] <br />
 
 The results from scipy
 
 ![scipy](https://user-images.githubusercontent.com/72471813/146014534-ebe947c4-e06d-4bf9-a250-f45ec932cffa.png)

The results from the C++ algorithm:

![cpp_algorithm](https://user-images.githubusercontent.com/72471813/146014589-707f172f-729b-4743-bf68-db4b519578f5.png)

There are still some differences between the two results: the lines at x=40 and x=60 are slightly offset from the lines on the scipy image. Also, the value of the lines are slightly different, in the middle of the line the C++ algorithm is 45 while it is 50 from scipy. The negative values on the left side of the image are also different, the C++ is -36 all the way to the left on the image while scipy is -50. 
