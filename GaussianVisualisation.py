# Visualizing Gaussian smoothing filter

# Importing Required Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel = np.array([[1, 4, 6, 4, 1],
          [4, 16, 24, 16, 4],
          [6, 24, 36, 24, 6],
          [4, 16, 24, 16, 4],
          [1, 4, 6, 4, 1]], dtype=np.float32)

# Loading image
img = cv2.imread('sun.jpg', cv2.IMREAD_GRAYSCALE)

Gaussian = cv2.filter2D(img, cv2.CV_64F, kernel)

# Displaying the Result
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(Gaussian, cmap='gray')
axs[1].set_title('Gaussian')

plt.show()