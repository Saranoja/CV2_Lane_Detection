import cv2
import numpy as np

# comments only for check up

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')
ret, frame = cam.read()


def get_resized_frame_dimensions(scale_ratio):
    width = int(frame.shape[1] * scale_ratio)
    height = int(frame.shape[0] * scale_ratio)
    return width, height


# dimensions = (width, height) tuple
def resize_frame(dimensions):
    resized_frame = cv2.resize(frame, dimensions)
    return resized_frame


def get_grayscale_frame(fr):
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)


def get_trapezoid_corners(window_width, window_height):
    upper_left = (0.45 * window_width, 0.75 * window_height)
    upper_right = (0.55 * window_width, 0.75 * window_height)
    lower_left = (0, window_height)
    lower_right = (window_width, window_height)
    return [upper_right, upper_left, lower_left, lower_right]


# dimensions = (height, width) tuple
def get_trapezoid_frame(trapezoid_bounds, frame_dimensions):
    trapezoid_bounds = np.array(trapezoid_bounds, dtype=np.int32)
    trapezoid_frame = np.zeros(frame_dimensions, dtype=np.uint8)
    cv2.fillConvexPoly(trapezoid_frame, trapezoid_bounds, 1)
    return trapezoid_frame


def get_greyscale_road(grayscale_frame, trapezoid_frame):
    return grayscale_frame * trapezoid_frame


def start_detection():
    while True:
        global ret, frame
        ret, frame = cam.read()

        if ret is False:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # cv2.imshow('Original', frame)

        width, height = get_resized_frame_dimensions(0.3)
        resized_frame = resize_frame((width, height))
        # cv2.imshow('Resized', resized_frame)

        grayscale_frame = get_grayscale_frame(resized_frame)
        # cv2.imshow('Grayscale', grayscale_frame)

        trapezoid_frame = get_trapezoid_frame(get_trapezoid_corners(width, height), (height, width))
        # cv2.imshow('Trapezoid', trapezoid_frame * 255)

        grayscale_road_frame = get_greyscale_road(grayscale_frame, trapezoid_frame)
        cv2.imshow('Road', grayscale_road_frame)


if __name__ == "__main__":
    start_detection()
    cam.release()
    cv2.destroyAllWindows()
