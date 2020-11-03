import cv2
import numpy as np

# comments only for check up

VERTICAL_SOBEL = np.float32([[-1, -2, -1],
                             [0, 0, 0],
                             [+1, +2, +1]])
HORIZONTAL_SOBEL = np.transpose(VERTICAL_SOBEL)

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')
ret, frame = cam.read()


def get_screen_corners_from_dimensions(width, height):
    upper_left_corner = (0, 0)
    upper_right_corner = (width, 0)
    lower_left_corner = (0, height)
    lower_right_corner = (width, height)
    return [upper_right_corner, upper_left_corner, lower_left_corner, lower_right_corner]


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


def get_birds_eye_view(road_frame, width, height):
    screen_corners = np.array(get_screen_corners_from_dimensions(width, height), dtype=np.float32)
    trapezoid_corners = np.float32(get_trapezoid_corners(width, height))
    magic_matrix = cv2.getPerspectiveTransform(trapezoid_corners, screen_corners)
    return cv2.warpPerspective(road_frame, magic_matrix, (width, height))


def get_burred_frame(fr):
    return cv2.blur(fr, ksize=(3, 3))


def filter_frame(fr_32, filter_matrix):
    return cv2.filter2D(fr_32, -1, filter_matrix)


def combine_filtered_versions_of_frame(fr):
    fr_32 = np.float32(fr)
    sobel_vertical_frame = cv2.filter2D(fr_32, -1, VERTICAL_SOBEL)
    sobel_horizontal_frame = cv2.filter2D(fr_32, -1, HORIZONTAL_SOBEL)
    sobel_result_frame = np.sqrt(sobel_vertical_frame ** 2 + sobel_horizontal_frame ** 2)
    return cv2.convertScaleAbs(sobel_result_frame)
    # separate filtered frames - add only for check up
    # sobel_vertical_frame = cv2.convertScaleAbs(sobel_vertical_frame)
    # cv2.imshow("Vertical", sobel_vertical_frame)
    # sobel_horizontal_frame = cv2.convertScaleAbs(sobel_horizontal_frame)
    # cv2.imshow("Horizontal", sobel_horizontal_frame)


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
        # cv2.imshow('Road', grayscale_road_frame)

        top_down_view = get_birds_eye_view(grayscale_road_frame, width, height)
        # cv2.imshow('Birds eye', top_down_view)

        blurred_top_down_view = get_burred_frame(top_down_view)
        # cv2.imshow('Blurred', blurred_top_down_view)

        sobel_filtered_top_down_view = combine_filtered_versions_of_frame(blurred_top_down_view)
        cv2.imshow('Sobel-filtered', sobel_filtered_top_down_view)


if __name__ == "__main__":
    start_detection()
    cam.release()
    cv2.destroyAllWindows()
