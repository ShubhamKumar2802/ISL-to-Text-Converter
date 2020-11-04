import keras
import cv2
import numpy as np
from HandDetection.hd import camera

# model = keras.models.load_model("/Volumes/Samsung_T5/Data_Sets/isl_model_cnn/isl")
model_cnn = keras.models.load_model("C:\\Users\\Aniket\\Desktop\\MINI PROJECT\\newmodel3")
# model_cnn = keras.models.load_model("C:\\Users\\Aniket\\Desktop\\isl")
# model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# cam = cv2.VideoCapture(0)
# while True:
# received_img = camera()
# prediction = model_cnn.predict(received_img)
# print(prediction)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#    break
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
    # print(resized.shape)
    resized = resized.reshape(1, 100, 100, 1)
    prediction = np.argmax(model_cnn.predict(resized))
    print(prediction)
    # cv2.imshow('final resized', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
