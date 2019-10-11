import numpy as np
import cv2

#### --- helper functions
# Crop image and resize
def crop_image(image):
    cropped = image[100:360] # result image shape: (300, 640, 3)
    new_shape = (cropped.shape[1]//2, cropped.shape[0]//2)
    return cv2.resize(cropped, new_shape) # , interpolation=cv2.INTER_AREA)

# Draw points on frame and visualize flow
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


