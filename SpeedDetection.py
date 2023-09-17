# Import necessary libraries
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from PIL import Image

# do edge detection on frame, than calculate difference between last frame and current frame,
# then blur result so we get rid of white pixels that are alone, then we take average position of
# other white pixels and we get moving target, we note every position of this target as well as when it
# happened, and at last we calculate average velocity based on this, accuracy is ofc based on camera angle, and also
# it has limit to how far it can detect moving target
# also answer will be printed after the video ends

# Read video file
cap = cv2.VideoCapture('2.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

lastFrame = None
frame_index = -1
positions = []
norm_threshold = 1200
# Loop through video frames
while cap.isOpened():
    frame_index += 1

    if frame_index % 5 > 0:
        continue


    # Read a frame
    ret, frame = cap.read()

    # Check if frame was read successfully
    if not ret:
        break

    # edge detection on current frame

    # Apply Gaussian blur to frame
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)

    # Convert blurred frame to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 150, 160)

    if lastFrame is not None:
        diff = cv2.absdiff(edges, lastFrame)
        norm = np.linalg.norm(diff, 1)


        # testing
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        diff = gaussian_filter(diff, sigma=7)

        # Find the indices of the white pixels
        white_pixel_indices = np.where(diff >= 25)

        if len(white_pixel_indices[0]) != 0:

            # Compute the number of white pixels
            num_of_white_pixels = len(white_pixel_indices[0])

            # Compute the sum of the row and column indices of the white pixels
            sum_of_row_indices = np.sum(white_pixel_indices[0])
            sum_of_col_indices = np.sum(white_pixel_indices[1])

            # Compute the average row and column indices of the white pixels
            avg_row_index = sum_of_row_indices / num_of_white_pixels
            avg_col_index = sum_of_col_indices / num_of_white_pixels

            img_with_circle = cv2.circle(frame, (int(avg_col_index), int(avg_row_index)), 10, (255, 0, 0), -1)

            if norm > norm_threshold:
                positions.append((int(avg_col_index), int(avg_row_index), frame_index))

            # Display the frame
            cv2.imshow('Video Frame', img_with_circle)

        else:
            # Display the frame
            cv2.imshow('Video Frame', frame)

    else:
        # Display the frame
        cv2.imshow('Video Frame', frame)

    lastFrame = edges

    # Wait for 25ms and check for key press
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
avg = 0
count = 0
for i in range(1, len(positions)):

    avg += math.dist((positions[i][0],positions[i][1]),(positions[i-1][0],positions[i-1][1]))
    count += 1

multiplayerBasedOnAngle = 3
deltaTime = 1 / 25
time = count * deltaTime

print(avg * multiplayerBasedOnAngle / count)

# Release video capture and destroy any open windows
cap.release()
cv2.destroyAllWindows()
