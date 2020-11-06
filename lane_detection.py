import cv2
import numpy as np

VERTICAL_SOBEL = np.float32([[-1, -2, -1],
                             [0, 0, 0],
                             [+1, +2, +1]])
HORIZONTAL_SOBEL = np.transpose(VERTICAL_SOBEL)

cam = cv2.VideoCapture('video.mp4')


def get_screen_corners_from_dimensions(width, height):
    upper_left_corner = (0, 0)
    upper_right_corner = (width, 0)
    lower_left_corner = (0, height)
    lower_right_corner = (width, height)
    return [upper_right_corner, upper_left_corner, lower_left_corner, lower_right_corner]


def get_resized_frame_dimensions(frame, scale_ratio):
    width = int(frame.shape[1] * scale_ratio)
    height = int(frame.shape[0] * scale_ratio)
    return width, height


def get_frame_grayscale(frame, width, height):
    newFrame = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            newFrame[i, j] = frame[i, j, 0] * .11 + frame[i, j, 1] * 0.6 + frame[i, j, 2] * 0.3
    return newFrame


def get_trapezoid_corners(width, height):
    upper_right = (0.55 * width, 0.75 * height)
    upper_left = (0.45 * width, 0.75 * height)
    lower_left = (0, height)
    lower_right = (width, height)
    return [upper_right, upper_left, lower_left, lower_right]


# dimensions = (height, width) tuple
def get_trapezoid_frame(trapezoid_bounds, frame_dimensions):
    trapezoid_bounds = np.array(trapezoid_bounds, dtype=np.int32)
    trapezoid_frame = np.zeros(frame_dimensions, dtype=np.uint8)
    cv2.fillConvexPoly(trapezoid_frame, trapezoid_bounds, 1)
    return trapezoid_frame


def get_birds_eye_view(road_frame, width, height):
    screen_corners = np.array(get_screen_corners_from_dimensions(width, height), dtype=np.float32)
    trapezoid_corners = np.float32(get_trapezoid_corners(width, height))
    magic_matrix = cv2.getPerspectiveTransform(trapezoid_corners, screen_corners)
    return cv2.warpPerspective(road_frame, magic_matrix, (width, height))


def combine_filtered_versions_of_frame(fr):
    fr_32 = np.float32(fr)
    sobel_vertical_frame = cv2.filter2D(fr_32, -1, VERTICAL_SOBEL)
    sobel_horizontal_frame = cv2.filter2D(fr_32, -1, HORIZONTAL_SOBEL)
    sobel_result_frame = np.sqrt(sobel_vertical_frame ** 2 + sobel_horizontal_frame ** 2)
    return cv2.convertScaleAbs(sobel_result_frame)


def get_binary_frame(filtered_frame):
    threshold = int(255 / 2) - 20
    applyThresholdOverArray = np.vectorize(lambda x: 0 if x < threshold else 255)
    for i in range(filtered_frame.shape[0]):
        filtered_frame[i] = applyThresholdOverArray(filtered_frame[i])
    return filtered_frame


def remove_redundant_columns(width, fr):
    fr_copy = fr.copy()
    columns = int(width * 0.2)
    fr_copy[0:, 0:columns] = 0
    fr_copy[0:, -columns:] = 0
    return fr_copy


def get_white_points_coordinates(fr, width, height):
    left_slice = fr[0:height, 0:int(width / 2)]
    right_slice = fr[0:height, int(width / 2):]
    left_slice_array = np.argwhere(left_slice == 255)
    right_slice_array = np.argwhere(right_slice == 255)

    left_xs = np.array([int(x) for y, x in left_slice_array])
    left_ys = np.array([int(y) for y, x in left_slice_array])
    right_xs = np.array([int(x + int(width / 2)) for y, x in right_slice_array])
    right_ys = np.array([int(y) for y, x in right_slice_array])

    return left_xs, left_ys, right_xs, right_ys


def get_line_points(fr, xs, ys, previous_top_x, previous_bottom_x):
    if xs.size == 0 or ys.size == 0:
        return tuple(map(lambda x: int(x), (previous_top_x, 0))), \
               tuple(map(lambda x: int(x), (previous_bottom_x, fr.shape[0])))

    a, b = np.polyfit(xs, ys, deg=1)

    top_x = previous_top_x
    top_y = 0

    bottom_x = previous_bottom_x
    bottom_y = fr.shape[0]

    new_top_x = -b / a
    new_bottom_x = (fr.shape[0] - b) / a
    if -10 ** 8 <= new_top_x <= 10 ** 8 and -10 ** 8 <= new_bottom_x <= 10 ** 8:
        top_x = new_top_x
        bottom_x = new_bottom_x

    return tuple(map(lambda x: int(x), (top_x, top_y))), tuple(map(lambda x: int(x), (bottom_x, bottom_y)))


def add_lines(fr):
    global previous_left_top_x, previous_left_bottom_x, previous_right_top_x, previous_right_bottom_x

    left_xs, left_ys, right_xs, right_ys = get_white_points_coordinates(fr, fr.shape[1], fr.shape[0])
    left_top_point, left_bottom_point = get_line_points(fr, left_xs, left_ys,
                                                        previous_left_top_x, previous_left_bottom_x)
    fr = cv2.line(fr, left_top_point, left_bottom_point, (100, 100, 100), 5)
    previous_left_top_x, _ = left_top_point
    previous_left_bottom_x, _ = left_bottom_point

    right_top_point, right_bottom_point = get_line_points(fr, right_xs, right_ys,
                                                          previous_right_top_x, previous_right_bottom_x)
    previous_right_top_x, _ = right_top_point
    previous_right_bottom_x, _ = right_bottom_point
    fr = cv2.line(fr, right_top_point, right_bottom_point, (100, 100, 100), 5)

    return fr, left_top_point, left_bottom_point, right_top_point, right_bottom_point


def get_final_frame(fr, left_top_point, left_bottom_point, right_top_point, right_bottom_point):
    height = fr.shape[0]
    width = fr.shape[1]

    blank_frame = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank_frame, left_top_point, left_bottom_point, (255, 0, 0), 3)

    screen_bounds = np.array(get_screen_corners_from_dimensions(width, height), dtype=np.float32)
    trapezoid_bounds = np.array(get_trapezoid_corners(width, height), dtype=np.float32)

    # left
    magic_matrix_final_left = cv2.getPerspectiveTransform(screen_bounds,
                                                          trapezoid_bounds)
    top_down_frame_final_left = cv2.warpPerspective(blank_frame, magic_matrix_final_left, (width, height))

    left_slice_final = top_down_frame_final_left[0:height, 0:int(width / 2)]
    left_slice_array_final = np.argwhere(left_slice_final == 255)

    left_xs_final = np.array([int(x) for y, x in left_slice_array_final])
    left_ys_final = np.array([int(y) for y, x in left_slice_array_final])

    # right
    blank_frame_1 = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank_frame_1, right_top_point, right_bottom_point, (255, 0, 0), 3)
    magic_matrix_final_right = cv2.getPerspectiveTransform(screen_bounds,
                                                           trapezoid_bounds)
    top_down_frame_final_right = cv2.warpPerspective(blank_frame_1, magic_matrix_final_right, (width, height))

    right_slice_final = top_down_frame_final_right[0:height, int(width / 2):]
    right_slice_array_final = np.argwhere(right_slice_final == 255)
    right_xs_final = np.array([int(x + int(width / 2)) for y, x in right_slice_array_final])
    right_ys_final = np.array([int(y) for y, x in right_slice_array_final])

    final_frame = fr.copy()
    for i in range(len(left_xs_final)):
        final_frame[left_ys_final[i]][left_xs_final[i]] = [0, 0, 250]
    for i in range(len(right_xs_final)):
        final_frame[right_ys_final[i]][right_xs_final[i]] = [0, 250, 0]
    cv2.imshow('Final', final_frame)
    return final_frame


def start_detection():
    global previous_left_top_x, previous_left_bottom_x, previous_right_top_x, previous_right_bottom_x

    while True:
        isOk, frame = cam.read()
        if not isOk:
            break

        # cv2.imshow('Original', frame)

        height, width = map(lambda x: int(x * 0.3), (frame.shape[0], frame.shape[1]))
        resized_frame = cv2.resize(frame, (width, height))
        cv2.imshow('Resized', resized_frame)

        # grayscale_frame = get_frame_grayscale(resized_frame, width, height)
        grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale', grayscale_frame)

        trapezoid_frame = get_trapezoid_frame(get_trapezoid_corners(width, height), (height, width))
        cv2.imshow('Trapezoid', trapezoid_frame * 255)

        grayscale_road_frame = grayscale_frame * trapezoid_frame
        cv2.imshow('Road', grayscale_road_frame)

        top_down_view = get_birds_eye_view(grayscale_road_frame, width, height)
        cv2.imshow('Birds eye', top_down_view)

        blurred_top_down_view = cv2.blur(top_down_view, ksize=(3, 3))
        cv2.imshow('Blurred', blurred_top_down_view)

        sobel_filtered_top_down_view = combine_filtered_versions_of_frame(blurred_top_down_view)
        cv2.imshow('Sobel-filtered', sobel_filtered_top_down_view)

        binary_filtered_top_down = get_binary_frame(sobel_filtered_top_down_view)
        cv2.imshow('Binary', binary_filtered_top_down)

        reduced_binary = remove_redundant_columns(width, binary_filtered_top_down)
        cv2.imshow('Binary and reduced', reduced_binary)

        line_frame, *points = add_lines(reduced_binary)
        cv2.imshow('lines', line_frame)

        final_frame = get_final_frame(resized_frame, *points)
        cv2.imshow('Final', final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


previous_left_top_x, previous_left_bottom_x, previous_right_top_x, previous_right_bottom_x = 0, 0, 0, 0

start_detection()
cam.release()
cv2.destroyAllWindows()
