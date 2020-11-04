import cv2
import numpy as np
import copy
import math


def camera():
    cap_region_x_begin = 0.5  # start point/total width
    cap_region_y_end = 0.8  # start point/total width
    threshold = 60  # BINARY threshold
    blurValue = 41  # GaussianBlur parameter
    bgSubThreshold = 50
    learningRate = 0

    camera = cv2.VideoCapture(0)
    camera.set(10, 200)
    cv2.namedWindow('trackbar')

    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)
        final = frame[0:int(cap_region_y_end * frame.shape[0]), int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        cv2.imshow('extracted area', final)
        grayFrame = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('grayscale', grayFrame)
        dim = (100, 100)
        resized = cv2.resize(grayFrame, dim, interpolation=cv2.INTER_AREA)
        print(type(resized))
        cv2.imshow('final resized', resized)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

# camera()    #if you need to run the code, unhash this


# # Keyboard OP
# k = cv2.waitKey(10)
# if k == 27:  # press ESC to exit
#     camera.release()
#     cv2.destroyAllWindows()
#     break
# elif k == ord('b'):  # press 'b' to capture the background
#     bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
#     isBgCaptured = 1
#     print('!!!Background Captured!!!')
# elif k == ord('r'):  # press 'r' to reset the background
#     bgModel = None
#     triggerSwitch = False
#     isBgCaptured = 0
#     print('!!!Reset BackGround!!!')
# elif k == ord('n'):
#     triggerSwitch = True
#     print('!!!Trigger On!!!')