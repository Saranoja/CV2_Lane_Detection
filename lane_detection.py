import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')


def start_detection():
    while True:
        ret, frame = cam.read()
        if ret is False:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('Original', frame)


if __name__ == "__main__":
    start_detection()
    cam.release()
    cv2.destroyAllWindows()
