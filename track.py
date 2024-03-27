import cv2
import numpy as np


class Tracker:
    def __init__(self, first_image, first_label) -> None:
        # gpu_image = cv2.cuda_GpuMat()
        # gpu_image.upload(first_image)
        # gpu_label = cv2.cuda.GpuMat()
        # gpu_label.upload(first_label)
        self.h = first_image.shape[0]
        self.w = first_image.shape[1]
        self.images = [first_image]
        self.labels = [first_label]
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        self.delaunay = cv2.Subdiv2D((0, 0, self.w, self.h))
        mask = (first_label.sum(-1) != 0).astype(np.uint8)
        # extend the mask
        mask = cv2.dilate(mask, np.ones(5), iterations=3)
        first_kp, first_des = self.detector.detectAndCompute(first_image, mask)
        self.kp = [first_kp]
        self.des = [first_des]
        self.matchNumber = 500

    def track_label(self, next_image):
        mask = (self.labels[-1].sum(-1) != 0).astype(np.uint8)
        kp, des = self.detector.detectAndCompute(next_image, mask)
        matches = self.matcher.match(self.des[-1], des)
        matches = sorted(matches, key=lambda x: x.distance)[:self.matchNumber]
        src_pts = [self.kp[-1][m.queryIdx].pt for m in matches]
        dst_pts = [kp[m.trainIdx].pt for m in matches]
        self.delaunay.insert(src_pts)
        src_triangles = self.delaunay.getTriangleList()
        self.delaunay.initDelaunay()
        self.delaunay.insert(dst_pts)
        dst_triangles = self.delaunay.getTriangleList()
        vector = [dst_pts[i] - src_pts[i] for i in range(len(matches))]
        field = np.zeros((self.h, self.w, 2))
        return


class TrackerOF:
    def __init__(self, first_image, first_label) -> None:
        self.detector = cv2.goodFeaturesToTrack(first_image)


if __name__ == '__main__':
    import paths
    import os
    d = os.path.join(paths.DATASET_DIR, paths.ALL_CASES[0])
    fns = [fn for fn in os.listdir(os.path.join(d, 'label')) if fn.endswith('.png')]
    images = [cv2.imread(os.path.join(d, fn)) for fn in fns]
    labels = [cv2.imread(os.path.join(d, 'label', fn)) for fn in fns]
    tracker = Tracker(images[0], labels[0])
    for i in range(1, len(fns)):
        tracker.track_label(images[i])
    