import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean


def matchAB(fileA, fileB):
    # 读取图像数据
    imgA = cv2.imread(fileA)
    imgB = cv2.imread(fileB)

    # 转换成灰色
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # akaze特征量抽出
    akaze = cv2.AKAZE_create()
    kpA, desA = akaze.detectAndCompute(grayA, None)
    kpB, desB = akaze.detectAndCompute(grayB, None)

    # BFMatcher定义和图形化
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(desB, desB)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_image = cv2.drawMatches(imgA, kpA, imgB, kpB, matches, None, flags=2, matchColor=-1, singlePointColor=1)

    cv2.imshow("matches", matched_image)
    # cv2.waitKey(10000)
    cv2.imwrite("matches.jpg", matched_image)



matchAB('/vol/grid-solar/sgeusers/sargisfinl/data/bovw_1_color_run/train/color_1/MNZ_Oyster_0191_1.jpg', '/vol/grid-solar/sgeusers/sargisfinl/data/bovw_1_color_run/train/color_1/MNZ_Oyster_0879_1.jpg')

exit(1)




# https://stackoverflow.com/questions/42253126/opencv-draw-only-objects-that-keypoints-refer-to-using-python

# img1 = cv2.imread(,0)  # queryImage
# img2 = cv2.imread('/vol/grid-solar/sgeusers/sargisfinl/data/bovw_1_color_run/train/color_1/MNZ_Oyster_0879_1.jpg',0) # trainImage

img = cv2.imread('/vol/grid-solar/sgeusers/sargisfinl/data/bovw_1_color_run/train/color_1/MNZ_Oyster_0191_1.jpg',0)
# img = cv2.imread('images/MNZ_Oyster_0034.jpg', 0)
sift = cv2.SIFT.create(nOctaveLayers=3, sigma=1.8, contrastThreshold=0.065, edgeThreshold=3)
kp, des = sift.detectAndCompute(img,None)
# kp, des = orb.compute(img, kp)

img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=0)
cv2.imshow('keypoint_on_text.jpg', img2)
cv2.imwrite('keypoint_on_text.jpg', img2)
# cv2.waitKey(1000)


mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
mask[:] = (0, 0, 0)

fmask = cv2.drawKeypoints(mask,kp,None,color=(0,255,0), flags=0)
cv2.imshow('fmask.jpg', fmask)
cv2.imwrite('fmask.jpg', fmask)
# cv2.waitKey(1000)


graymask = cv2.cvtColor(fmask,cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(graymask, 50, 255, 0)
co, contours , = cv2.findContours(th,2,1)
rep = cv2.drawContours(fmask, co, -1, (0,255,0), 5)
cv2.imshow('contours.jpg',rep)
cv2.imwrite('contours.jpg', rep)
# cv2.waitKey(1000)


repmask = cv2.cvtColor(rep,cv2.COLOR_BGR2GRAY)
ret, th1 = cv2.threshold(repmask, 50, 255, 0)
res = cv2.bitwise_and(img,img,mask = th1)
cv2.imshow('OnlyFeats.jpg',res)
cv2.imwrite('OnlyFeats.jpg',res)
# cv2.waitKey(100000)
