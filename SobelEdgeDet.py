# edge detection is used in games a lot to make
# them more stylized.

# also derivatives are commonly used in physics( first and second derivatives)

# and optimization(finding the min value of the function)

# this is both edge detection and FD first order

# Importing Required Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

sobelX_kernel = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=np.float32)
sobelY_kernel = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=np.float32)

# Loading image
img = cv2.imread('sun.jpg', cv2.IMREAD_GRAYSCALE)

# Applying Sobel Kernel for Edge Detection
sobel_x = cv2.filter2D(img, cv2.CV_64F, sobelX_kernel)
sobel_y = cv2.filter2D(img, cv2.CV_64F, sobelY_kernel)

# Combining the x and y edges
sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Displaying the Result
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(sobel, cmap='gray')
axs[1].set_title('Sobel Edge Detection Result')

plt.show()