import cv2

img = cv2.imread('time,jpg', 1)
dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #
cv2.imshow('image', dst)
cv2.waitKey(0)