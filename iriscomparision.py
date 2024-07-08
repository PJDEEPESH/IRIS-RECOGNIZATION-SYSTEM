import numpy as np
import cv2
import math
import random
import copy

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    return blurred

def load_image(filepath, show=False):
    img = cv2.imread(filepath, 0)
    if show:
        cv2.imshow(filepath, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img

def get_iris_boundaries(img, show=False):
    pupil_circle = find_pupil(img)

    if not pupil_circle:
        print('ERROR: Pupil circle not found!')
        return None, None

    radius_range = int(math.ceil(pupil_circle[2] * 1.5))
    multiplier = 0.25
    center_range = int(math.ceil(pupil_circle[2] * multiplier))
    ext_iris_circle = find_ext_iris(img, pupil_circle, center_range, radius_range)

    while not ext_iris_circle and multiplier <= 0.7:
        multiplier += 0.05
        print('Searching exterior iris circle with multiplier ' + str(multiplier))
        center_range = int(math.ceil(pupil_circle[2] * multiplier))
        ext_iris_circle = find_ext_iris(img, pupil_circle, center_range, radius_range)

    if not ext_iris_circle:
        print('ERROR: Exterior iris circle not found!')
        return None, None

    if show:
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        draw_circles(cimg, pupil_circle, ext_iris_circle, center_range, radius_range)
        cv2.imshow('iris boundaries', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pupil_circle, ext_iris_circle

def find_pupil(img):
    def get_edges(image):
        edges = cv2.Canny(image, 20, 100)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        ksize = 2 * random.randrange(5, 11) + 1
        edges = cv2.GaussianBlur(edges, (ksize, ksize), 0)
        return edges

    param1 = 200
    param2 = 120
    pupil_circles = []
    while param2 > 35 and len(pupil_circles) < 100:
        for mdn, thrs in [(m, t) for m in [3, 5, 7] for t in range(20, 61, 5)]:
            median = cv2.medianBlur(img, 2 * mdn + 1)
            ret, thres = cv2.threshold(median, thrs, 255, cv2.THRESH_BINARY_INV)

            contours, hierarchy = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            edges = get_edges(thres)

            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1, np.array([]), param1, param2)

            if circles is not None and circles.size > 0:
                circles = np.round(circles[0, :]).astype("int")
                for c in circles:
                    pupil_circles.append(c)

        param2 -= 1

    if not pupil_circles:
        return None
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return get_mean_circle(pupil_circles)

def get_mean_circle(circles, draw=None):
    if not circles:
        return None
    mean_0 = int(np.mean([c[0] for c in circles]))
    mean_1 = int(np.mean([c[1] for c in circles]))
    mean_2 = int(np.mean([c[2] for c in circles]))

    if draw is not None:
        draw = draw.copy()
        cv2.circle(draw, (mean_0, mean_1), mean_2, (0, 255, 0), 1)
        cv2.circle(draw, (mean_0, mean_1), 2, (0, 255, 0), 2)
        cv2.imshow('mean circle', draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mean_0, mean_1, mean_2

def find_ext_iris(img, pupil_circle, center_range, radius_range):
    def get_edges(image, thrs2):
        thrs1 = 0
        edges = cv2.Canny(image, thrs1, thrs2, apertureSize=5)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        ksize = 2 * random.randrange(5, 11) + 1
        edges = cv2.GaussianBlur(edges, (ksize, ksize), 0)
        return edges

    param1 = 200
    param2 = 30
    iris_circles = []
    while param2 > 15 and len(iris_circles) < 100:
        for mdn in [3, 5, 7]:
            median = cv2.medianBlur(img, 2 * mdn + 1)
            edges = get_edges(median, param2)

            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1, np.array([]), param1, param2, 
                                       minRadius=int(pupil_circle[2] * 1.1), maxRadius=int(pupil_circle[2] * 2))

            if circles is not None and circles.size > 0:
                circles = np.round(circles[0, :]).astype("int")
                for c in circles:
                    iris_circles.append(c)

        param2 -= 1

    if not iris_circles:
        return None

    return get_mean_circle(iris_circles)

def draw_circles(img, pupil_circle, ext_circle, center_range, radius_range):
    cv2.circle(img, (pupil_circle[0], pupil_circle[1]), pupil_circle[2], (0, 255, 0), 2)
    cv2.circle(img, (ext_circle[0], ext_circle[1]), ext_circle[2], (255, 0, 0), 2)
    cv2.circle(img, (pupil_circle[0], pupil_circle[1]), center_range, (0, 0, 255), 2)
    cv2.circle(img, (pupil_circle[0], pupil_circle[1]), radius_range, (255, 255, 0), 2)

def get_equalized_iris(img, ext_iris_circle, pupil_circle, show=False):
    def find_roi():
        mask = img.copy()
        mask[:] = (0)

        cv2.circle(mask, (ext_iris_circle[0], ext_iris_circle[1]), ext_iris_circle[2], (255), -1)
        cv2.circle(mask, (pupil_circle[0], pupil_circle[1]), pupil_circle[2], (0), -1)

        roi = cv2.bitwise_and(img, mask)
        return roi

    roi = find_roi()

    for p_col in range(roi.shape[1]):
        for p_row in range(roi.shape[0]):
            theta = angle_v(ext_iris_circle[0], ext_iris_circle[1], p_col, p_row)
            if 50 < theta < 130:
                roi[p_row, p_col] = 0

    ret, roi = cv2.threshold(roi, 50, 255, cv2.THRESH_TOZERO)

    equ_roi = cv2.equalizeHist(roi)

    if show:
        cv2.imshow('equalized histogram iris region', equ_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return equ_roi

def get_rois(img, pupil_circle, ext_circle, show=False):
    bg = img.copy()
    bg[:] = 0

    init_dict = {'img': bg.copy(), 'pupil_circle': pupil_circle, 'ext_circle': ext_circle, 'kp': None,
                 'img_kp_init': bg.copy(), 'img_kp_filtered': bg.copy(), 'des': None}

    rois = {'right-side': copy.deepcopy(init_dict), 'left-side': copy.deepcopy(init_dict),
            'bottom': copy.deepcopy(init_dict), 'complete': copy.deepcopy(init_dict)}

    for p_col in range(img.shape[1]):
        for p_row in range(img.shape[0]):
            if not point_in_circle(pupil_circle[0], pupil_circle[1], pupil_circle[2], p_col, p_row) and \
               point_in_circle(ext_circle[0], ext_circle[1], ext_circle[2], p_col, p_row):
                theta = angle_v(ext_circle[0], ext_circle[1], p_col, p_row)
                if -50 <= theta <= 50:
                    rois['right-side']['img'][p_row, p_col] = img[p_row, p_col]
                if theta >= 130 or theta <= -130:
                    rois['left-side']['img'][p_row, p_col] = img[p_row, p_col]
                if -140 <= theta <= -40:
                    rois['bottom']['img'][p_row, p_col] = img[p_row, p_col]
                rois['complete']['img'][p_row, p_col] = img[p_row, p_col]

    rois['right-side']['ext_circle'] = (0, int(1.25 * ext_circle[2]), int(ext_circle[2]))
    rois['left-side']['ext_circle'] = (int(1.25 * ext_circle[2]), int(1.25 * ext_circle[2]), int(ext_circle[2]))
    rois['bottom']['ext_circle'] = (int(1.25 * ext_circle[2]), 0, int(ext_circle[2]))
    rois['complete']['ext_circle'] = (int(1.25 * ext_circle[2]), int(1.25 * ext_circle[2]), int(ext_circle[2]))

    for pos in ['right-side', 'left-side', 'bottom', 'complete']:
        tx = rois[pos]['ext_circle'][0] - ext_circle[0]
        ty = rois[pos]['ext_circle'][1] - ext_circle[1]
        rois[pos]['pupil_circle'] = (int(tx + pupil_circle[0]), int(ty + pupil_circle[1]), int(pupil_circle[2]))

    if show:
        for key in rois:
            cv2.imshow(key, rois[key]['img'])
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    return rois

def point_in_circle(cx, cy, radius, x, y):
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

def angle_v(cx, cy, x, y):
    dx = x - cx
    dy = y - cy
    return math.degrees(math.atan2(dy, dx))

def compare_iris(img1, img2):
    pupil_circle1, ext_iris_circle1 = get_iris_boundaries(img1)
    pupil_circle2, ext_iris_circle2 = get_iris_boundaries(img2)

    if not pupil_circle1 or not ext_iris_circle1 or not pupil_circle2 or not ext_iris_circle2:
        return False

    equ_iris1 = get_equalized_iris(img1, ext_iris_circle1, pupil_circle1)
    equ_iris2 = get_equalized_iris(img2, ext_iris_circle2, pupil_circle2)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(equ_iris1, None)
    kp2, des2 = orb.detectAndCompute(equ_iris2, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_ratio = len(matches) / max(len(kp1), len(kp2))

    return match_ratio > 0.2

def capture_and_compare():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Capturing first image. Press 'Space' to capture.")
    while True:
        ret, frame1 = cap.read()
        cv2.imshow('Capture First Image', frame1)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    print("First image captured.")

    print("Capturing second image. Press 'Space' to capture.")
    while True:
        ret, frame2 = cap.read()
        cv2.imshow('Capture Second Image', frame2)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    print("Second image captured.")

    cap.release()
    cv2.destroyAllWindows()

    processed_frame1 = preprocess_image(frame1)
    processed_frame2 = preprocess_image(frame2)

    processed_frame1 = cv2.bilateralFilter(processed_frame1, 9, 75, 75)
    processed_frame2 = cv2.bilateralFilter(processed_frame2, 9, 75, 75)

    match = compare_iris(processed_frame1, processed_frame2)

    if match:
        print("The irises match.")
    else:
        print("The irises do not match.")

capture_and_compare()
