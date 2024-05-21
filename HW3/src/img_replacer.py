import cv2
from matplotlib import pyplot as plt
import numpy as np

class ImgReplacer:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
    
    def read(self, test_id: int):
        self.test_id = test_id
        source_img = cv2.imread(f'{self.dataset_dir}/test{test_id}/source.jpg')
        destination_img = cv2.imread(f'{self.dataset_dir}/test{test_id}/destination.jpg')
        target_img = cv2.imread(f'{self.dataset_dir}/target.jpg')
        return source_img, destination_img, target_img
    
    def sift(self, img):
        sift = cv2.SIFT_create()
        kps, des = sift.detectAndCompute(img, None)
        return kps, des
    
    def match(self, kps, kpd):
        # BFmatcher 找到兩張圖片的相同關鍵點
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(kps, kpd, k=2)
        filter_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                filter_matches.append(m)
        filter_matches = sorted(filter_matches, key=lambda x: x.distance)
        filter_matches = np.array(filter_matches)
        return filter_matches
    
    def match_img(self, img1, img2, kps1, kps2, filter_matches):
        return cv2.drawMatches(img1, kps1, img2, kps2, filter_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    def ransac(self, kps, kpd, filter_matches):
        # RANSAC 找到最佳的 Homography
        src_pts = np.float32([kps[m.queryIdx].pt for m in filter_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpd[m.trainIdx].pt for m in filter_matches]).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        return homography
    
    def warp(self, img, homography, size):
        return cv2.warpPerspective(img, homography, size)
    
    def merge(self, img1, img2):
        return img1[:, :] * (img2[:, :] == 0) + img2[:, :]
    
    def resize(self, src, target):
        return cv2.resize(src, (target.shape[1], target.shape[0]))
    
    def save(self, img, filename):
        cv2.imwrite(f'{self.dataset_dir}/test{self.test_id}/{filename}.jpg', img)