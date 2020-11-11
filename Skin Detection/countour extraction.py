import cv2
import imutils
import keras
import numpy as np

# bg = None

model_cnn = keras.models.load_model("C:\\Users\\Aniket\\Desktop\\MINI PROJECT\\number_model")

# def run_avg(image, aWeight):
#     global bg
#     if bg is None:
#         bg = image.copy().astype("float")
#         return
#
#     cv2.accumulateWeighted(image, bg, aWeight)
#
#
# def segment(image, threshold=25):
#     global bg
#     diff = cv2.absdiff(bg.astype("uint8"), image)
#     thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
#     cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(cnts) == 0:
#         return
#     else:
#         segmented = max(cnts, key=cv2.contourArea)
#         return (thresholded, segmented)

background = None
accumulated_weight = 0.5

top = 10
bottom = 300
right = 450
left = 680


def run_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    contours, image = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand
        hand_segment_max_cont = max(contours, key=cv2.contourArea)

        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)


if __name__ == "__main__":
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    # top, right, bottom, left = 10, 450, 300, 700
    num_frames = 0
    while (True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        if num_frames < 70:
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (255, 0, 0), 1)
                cv2.imshow("Thresholded", thresholded)
                # thresholded_resized = thresholded.reshape(100, 100)
                thresholded_resized = cv2.resize(thresholded, (100, 100))
                thresholded_resized = thresholded_resized.reshape(1, 100, 100, 1)
                # thresholded_final = cv2.cvtColor(thresholded_resized, cv2.COLOR_BGR2GRAY)
                prediction = np.argmax(model_cnn.predict(thresholded_resized))
                print(prediction)
                # print(type(thresholded))
        cv2.rectangle(clone, (left, top), (right, bottom), (255, 128, 0), 3)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
