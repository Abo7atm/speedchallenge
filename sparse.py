# Standard library imports
import sys

# Third party imports
import cv2
import numpy as np

# Local imports
import helper

# TODO: to use argparse instead
if len(sys.argv) != 2:
    print('please pass video path as argument')
    exit()

# Parameters for Shi-Tomasi
feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2,
        blockSize=7)

# Parameters for Lucas-Kanade
lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Video feed to read
train_video = sys.argv[1]
cap = cv2.VideoCapture(train_video)

# Color for optical flow track
color = (0, 255, 0)

# read first frame
ret, first_frame = cap.read()

# resize first frame
first_frame = helper.crop_image(first_frame)

# # show image before cropping
# cv2.imshow('first frame', first_frame)
# cv2.waitKey(0)
 
# # show cropped image
# cropped = crop_image(first_frame)
# cv2.imshow('cropped image', cropped)
# cv2.waitKey(0)
 
# Convert first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Applying Shi Tomasi
prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create image with zero intensities
mask = np.zeros_like(first_frame)

# dbg -- counter to only work on 100 frames
counter=0

while(cap.isOpened()):
    # read frame from video
    ret, frame = cap.read()

    # resize images
    frame = helper.crop_image(frame)

    # Convert each frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate sparse optical flow using LK method
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

    # Handle tracked object going out of frame
    if nextPts is None:
        prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

    # Select good features for prev
    good_old = prev[status==1]

    print('good old: {}'.format(good_old))

    # Select good features for next
    good_new = nextPts[status==1]

    # Draw optical flow
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Contiguous flattend array for new points
        a, b = new.ravel()

        # Contiguous flattend array for old points
        c, d = old.ravel()

        # Draws line between old and new positions with green color and 2 thickness
        mask = cv2.line(mask, (a, b), (c, d), color, 2)

        # Draws filled circle of thickness -1 at new position with green color
        frame = cv2.circle(frame, (a, b), 3, color, -1)

    # Overlay optical flow on original frame
    output = cv2.add(frame, mask)

    # show frame in window
    cv2.imshow("Sparse optical flow", output)

    # Update previous frame
    prev_gray = gray.copy()

    # Update previous good features
    prev = good_new.reshape(-1, 1, 2)

    # set 'quit' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # dbg
    counter += 1
    print('counter: {}'.format(counter))

cap.release()
cv2.destroyAllWindows()
