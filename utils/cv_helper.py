import datetime

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim


def read_grayscale_images(path1, path2):
    # give second parameter as 0 will read grayscale image
    obj = cv2.imread(path1, 0)
    search = cv2.imread(path2, 0)

    return obj, search


def sift_show_keypoints_and_matches(object_path, search_image_path):
    # refer to https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html

    start = datetime.datetime.now()

    obj, search = read_grayscale_images(object_path, search_image_path)
    if search is None:
        raise Exception('search image is None')
    if obj is None:
        raise Exception('obj is None')

    print('search image size, height, width: {}, {}'.format(search.shape[0], search.shape[1]))

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(obj, None)
    kp2, des2 = sift.detectAndCompute(search, None)

    bf = cv2.BFMatcher()

    temp = []

    matches = bf.knnMatch(des1, des2, k=2)

    for m, n in matches:
        # do ratio test, range from 0.3 to 0.6 is tested to have high good match rate
        # you can refer to the original paper for more detail
        if m.distance < 0.6 * n.distance:
            temp.append([m])

    img_result = cv2.drawMatchesKnn(obj, kp1, search, kp2, temp, None, flags=2)

    elapsed = datetime.datetime.now() - start
    print('time spend by sift is : {}'.format(elapsed))

    plt.imshow(img_result)
    plt.show()


def sift_and_knn_get_object_coordinate(obj_path, search_path):
    start = datetime.datetime.now()

    search, obj = read_grayscale_images(obj_path, search_path)
    if search is None:
        raise Exception('search image is None')
    if obj is None:
        raise Exception('obj is None')

    print('search image size, height, width: {}, {}'.format(search.shape[0], search.shape[1]))

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(obj, None)
    kp2, des2 = sift.detectAndCompute(search, None)

    bf = cv2.BFMatcher()

    temp = []

    matches = bf.knnMatch(des1, des2, k=2)

    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            temp.append([m])

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in temp]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in temp]).reshape(-1, 1, 2)

    # M is 3*3 matrix to transform object location in object image to search image
    # mask is a list to mark inlier and ourlier for matches
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 5.0)

    h, w = obj.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    # transform object location in object image to search image and get coordinate on the search image
    dst = cv2.perspectiveTransform(pts, M)

    area = [np.int32(dst)]

    x = (area[0][2][0][0] - area[0][0][0][0]) / 2 + area[0][0][0][0]
    y = (area[0][1][0][1] - area[0][0][0][1]) / 2 + area[0][0][0][1]

    elapsed = datetime.datetime.now() - start
    print('time spend by sift and match is : {}'.format(elapsed))

    return x, y


def surf_and_flann_get_object_coordinate(obj_path, search_path):
    start = datetime.datetime.now()

    obj, search = read_grayscale_images(obj_path, search_path)
    if search is None:
        raise Exception('search image is None')
    if obj is None:
        raise Exception('obj is None')

    print('screenshot size, height, width: {}, {}'.format(search.shape[0], search.shape[1]))

    # descriptor for surf will have 64 dimension vectors descriptor
    # refer to https://docs.opencv.org/3.3.0/df/dd2/tutorial_py_surf_intro.html
    surf = cv2.xfeatures2d.SURF_create(400)
    kp1, des1 = surf.detectAndCompute(obj, None)
    kp2, des2 = surf.detectAndCompute(search, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    temp = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            temp.append(m)

    # refer to https://docs.opencv.org/3.3.0/d1/de0/tutorial_py_feature_homography.html
    src_pts = np.float32([kp1[m.queryIdx].pt for m in temp]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in temp]).reshape(-1, 1, 2)

    # refer to https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findhomography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = obj.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M)
    search = cv2.polylines(search, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    area = [np.int32(dst)]

    # compute object center coordinate in the search image
    x = (area[0][2][0][0] - area[0][0][0][0]) / 2 + area[0][0][0][0]
    y = (area[0][1][0][1] - area[0][0][0][1]) / 2 + area[0][0][0][1]

    # this will generate all good matched(inlier) keypoints on search image.
    # it is a demo code, will not be used in any place in this function
    good = []
    for index, k in enumerate(matchesMask):
        if k == 1:
            good.append(temp[index])

    elapsed = datetime.datetime.now() - start
    print('time spend by surf and match is : {}'.format(elapsed))

    return x, y, search.shape[0], search.shape[1]


def resize_image(a, b):
    ha, wa = a.shape
    hb, wb = b.shape
    if ha / wa != hb / wb:
        raise Exception('two images don\'t have same heigh and weight rate ')

    if ha != hb and wa != wb:
        scale = ha / hb if ha > hb else hb / ha

        # always scale smaller
        if ha > hb:
            resized = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            b = resized

        else:
            scale = scale
            resized = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            a = resized

    return a, b


def compare_diff(path1, path2):
    # refer to https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

    # load the two input images to grayscale images
    imageA, imageB = read_grayscale_images(path1, path2)

    imageA, imageB = resize_image(imageA, imageB)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(imageA, imageB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours

    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return imageA, imageB, score


if __name__ == '__main__':

    a, b, score = compare_diff('../images/loginbuttonchanged.png', '../images/loginbuttondoublesize.png')

    # check the score
    print(score)

    # show the grey scale image
    plt.imshow(a)
    plt.show()

    plt.imshow(b)
    plt.show()
