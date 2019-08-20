# Standard library imports
import sys

# Third party imports
import cv2
import numpy as np

# Local imports
import helper

# TODO: 
# 1) to use argparse instead
# 2) create object for dense and sparse opt flow
if len(sys.argv) != 3:
    print('please pass video and vehicle speed path as argument')
    exit()
# vehicle speed file
data = open(sys.argv[2], 'r')

# Video feed to read
train_video = sys.argv[1]
cap = cv2.VideoCapture(train_video)

# Color for optical flow track
color = (0, 255, 0)

# read first frame
ret, first_frame = cap.read()

# resize first frame
first_frame = helper.crop_image(first_frame)

# Convert first frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# read vehicle speed at current frame
speed = data.readline()

# print speed on frame
font_face = cv2.FONT_HERSHEY_DUPLEX
font_scale = 1
color = (255, 255, 255)
cv2.putText(prev_gray, speed, (15, prev_gray.shape[1]//2), font_face, font_scale, (255, 255))

# mask for visualizing flow
mask = np.zeros_like(first_frame)

# set saturation to max
mask[..., 1] = 255

# dbg
counter = 0
print('{}: {}'.format(counter, speed))

while(cap.isOpened()):
    # read frame from video
    ret, frame = cap.read()

    # resize images
    frame = helper.crop_image(frame)

    # Convert each frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate dense flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # compute magnitutde and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # set image hue according to optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # set image hue accodring to optical flow magnitude
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # convert hsv to rgb
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    # read frame speed from file
    speed = '{:0.2f}'.format(float(data.readline()))

    # print speed on frame
    cv2.putText(rgb, speed, (110, 25), font_face, font_scale, (255, 255))

    # show frame in window
    cv2.imshow("Magic", rgb)

    # Update previous frame
    prev_gray = gray.copy()

    # set 'quit' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # dbg
    counter += 1
    print('counter: {}'.format(counter))

cap.release()
cv2.destroyAllWindows()
