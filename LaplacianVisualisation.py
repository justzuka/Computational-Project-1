# visualising laplacian

# Importing Required Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32)
# Loading image
img = cv2.imread('sun.jpg', cv2.IMREAD_GRAYSCALE)

# Applying Laplacian Kernel for Edge Detection
laplacian = cv2.filter2D(img, cv2.CV_64F, laplacian_kernel)

# Displaying the Result
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(laplacian, cmap='gray')
axs[1].set_title('Laplacian')

plt.show()