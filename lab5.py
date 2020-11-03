import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

while True:
    ret, frame = cam.read()
    if ret is False:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    scale_percent = 0.3  # percent of original size
    width = int(frame.shape[1] * scale_percent)
    height = int(frame.shape[0] * scale_percent)
    resized = cv2.resize(frame, (width, height))
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Original', frame)
    # cv2.imshow('Resized', resized)
    # cv2.imshow('Grayscale', grayscale)

    upper_left = (0.45 * width, 0.75 * height)
    upper_right = (0.55 * width, 0.75 * height)
    lower_left = (0, height)
    lower_right = (width, height)

    trapezoid_bounds = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
    trapezoid_frame = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(trapezoid_frame, trapezoid_bounds, 1)
    # cv2.imshow('Trapezoid', trapezoid_frame * 255)

    road_frame = grayscale * trapezoid_frame
    # cv2.imshow('Road', road_frame)

    upper_left_screen = (0, 0)
    upper_right_screen = (width, 0)
    lower_left_screen = (0, height)
    lower_right_screen = (width, height)

    screen_bounds = np.array([upper_right_screen, upper_left_screen, lower_left_screen, lower_right_screen],
                             dtype=np.float32)
    trapezoid_bounds = np.float32(trapezoid_bounds)
    magic_matrix = cv2.getPerspectiveTransform(trapezoid_bounds, screen_bounds)
    top_down_frame = cv2.warpPerspective(road_frame, magic_matrix, (width, height))
    # cv2.imshow("Top-Down", top_down_frame)

    blur_frame = cv2.blur(top_down_frame, ksize=(3, 3))
    # cv2.imshow("Blur", blur_frame)

    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [+1, +2, +1]])
    sobel_horizontal = np.transpose(sobel_vertical)

    frame_as_float_32 = np.float32(blur_frame)
    sobel_vertical_frame = cv2.filter2D(frame_as_float_32, -1, sobel_vertical)
    sobel_horizontal_frame = cv2.filter2D(frame_as_float_32, -1, sobel_horizontal)

    # check - should be run all at once, independently from the part below it
    # sobel_vertical_frame = cv2.convertScaleAbs(sobel_vertical_frame)
    # cv2.imshow("Vertical", sobel_vertical_frame)
    # sobel_horizontal_frame = cv2.convertScaleAbs(sobel_horizontal_frame)
    # cv2.imshow("Horizontal", sobel_horizontal_frame)

    sobel_result_frame = np.sqrt(sobel_vertical_frame ** 2 + sobel_horizontal_frame ** 2)
    sobel_result_frame = cv2.convertScaleAbs(sobel_result_frame)
    # cv2.imshow('Sobel', sobel_result_frame)

    threshold = int(255 / 2) - 10
    for i in range(sobel_result_frame.shape[0]):
        for j in range(sobel_result_frame.shape[1]):
            if sobel_result_frame[i][j] < threshold:
                sobel_result_frame[i][j] = 0
            else:
                sobel_result_frame[i][j] = 255
    # cv2.imshow('Binarized', sobel_result_frame)

    sobel_copy = sobel_result_frame.copy()
    columns = int(width * 0.2)
    sobel_copy[0:, 0:columns] = 0
    sobel_copy[0:, -columns:] = 0
    # cv2.imshow('Binarized and reduced', sobel_copy)

    

cam.release()
cv2.destroyAllWindows()
