import cv2
import numpy as np
from utils import LABEL_GT_CHARACTERIZER, time_it, matmul_affine_matrices, evaluate_segmentation, characterize


class TrackerSubdiv:
    def __init__(self, first_image, first_label) -> None:
        # gpu_image = cv2.cuda_GpuMat()
        # gpu_image.upload(first_image)
        # gpu_label = cv2.cuda.GpuMat()
        # gpu_label.upload(first_label)
        self.matchNumber = 500
        self.h = first_image.shape[0]
        self.w = first_image.shape[1]
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        self.delaunay = cv2.Subdiv2D((0, 0, self.w, self.h))
        self.images = [first_image]
        self.labels = [first_label]
        first_kp, first_des = self.detector.detectAndCompute(first_image, self.extend_mask(first_label))
        self.kp = [first_kp]
        self.des = [first_des]

    @staticmethod
    def extend_mask(label):
        mask = (label.sum(-1) != 0).astype(np.uint8)
        # to include the borders
        mask = cv2.dilate(mask, np.ones(5), iterations=3)
        return mask
    
    def is_match_reasonable(self, before, after):
        v = np.asarray(before) - np.asarray(after)
        l = (v[0] ** 2 + v[1] ** 2) ** 0.5
        max = (self.h ** 2 + self.w ** 2) ** 0.5
        return l < max * 0.05

    @time_it
    def track_label(self, next_image):
        kp, des = self.detector.detectAndCompute(next_image, self.extend_mask(self.labels[-1]))
        matches = self.matcher.match(self.des[-1], des)
        matches = list(filter(
            lambda m: self.is_match_reasonable(self.kp[-1][m.queryIdx].pt, kp[m.trainIdx].pt), 
            matches))
        matches = sorted(matches, key=lambda x: x.distance)[:self.matchNumber]
        src_pts = np.asarray([self.kp[-1][m.queryIdx].pt for m in matches]).astype(np.float32)
        dst_pts = np.asarray([kp[m.trainIdx].pt for m in matches]).astype(np.float32)
        dst_to_src = { tuple(dst_pts[i]) : src_pts[i] for i in range(len(matches)) }
        target_label = np.zeros_like(self.labels[-1])
        self.delaunay.insert(dst_pts)
        # warp each triangle
        for tri in self.delaunay.getTriangleList():
            dst_tri = np.asarray([tri[0:2], tri[2:4], tri[4:6]])
            src_tri = np.asarray([dst_to_src[tuple(s)] for s in dst_tri])
            mat = cv2.getAffineTransform(src_tri, dst_tri)
            # crop a rectangle
            src_rect = cv2.boundingRect(src_tri)
            dst_rect = cv2.boundingRect(dst_tri)
            src_cropped = self.labels[-1][src_rect[1] : src_rect[1]+src_rect[3], src_rect[0] : src_rect[0]+src_rect[2], :]
            dst_cropped = cv2.warpAffine(src_cropped, mat, dsize=dst_rect[2:], borderMode=cv2.BORDER_REFLECT)
            # clear the pixels inside of the rectangle but outside of the triangle
            dst_mask = np.zeros_like(dst_cropped)
            dst_mask = cv2.fillConvexPoly(dst_mask, (dst_tri - np.asarray(dst_rect[0:2])).astype(np.int32), (1, 1, 1))
            dst_cropped *= dst_mask
            # fill the target with the rectangle
            target_label[dst_rect[1] : dst_rect[1]+dst_rect[3], dst_rect[0] : dst_rect[0]+dst_rect[2], :] += dst_cropped
        self.images.append(next_image)
        self.labels.append(target_label)
        self.kp.append(kp)
        self.des.append(des)
        return target_label


class TrackerOF:
    def __init__(self, first_image, first_label) -> None:
        self.detector = cv2.goodFeaturesToTrack(first_image)


class TrackerAverage:
    def __init__(self, first_image, first_label, strategy='perspective', use_mask=False) -> None:
        self.matchNumber = 400
        self.triNumber = 100
        self.vectorThreshold = 0.1
        self.strategy = strategy
        self.useMask = use_mask
        self.h = first_image.shape[0]
        self.w = first_image.shape[1]
        self.detector = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        self.images = [first_image]
        self.labels = [first_label]
        first_kp, first_des = self.detector.detectAndCompute(first_image, 
            self.extend_mask(first_label) if self.useMask else None)
        self.kp = [first_kp]
        self.des = [first_des]
        # The matrices accroding to the first label
        first_mat = np.identity(3)
        if self.strategy == 'affine':
            first_mat = first_mat[:2, :]
        self.mats = [first_mat]
        
    @staticmethod
    def extend_mask(label):
        mask = (label.sum(-1) != 0).astype(np.uint8)
        # to include the borders
        mask = cv2.dilate(mask, np.ones(5), iterations=3)
        return mask
        
    def is_match_reasonable(self, before, after):
        v = np.asarray(before) - np.asarray(after)
        l = (v[0] ** 2 + v[1] ** 2) ** 0.5
        max = (self.h ** 2 + self.w ** 2) ** 0.5
        return l < max * self.vectorThreshold
        
    @time_it
    def track_label(self, next_image):
        kp, des = self.detector.detectAndCompute(next_image, 
                                                 self.extend_mask(self.labels[-1]) if self.useMask else None)
        matches = self.matcher.match(self.des[-1], des)
        matches = list(filter(
            lambda m: self.is_match_reasonable(self.kp[-1][m.queryIdx].pt, kp[m.trainIdx].pt), 
            matches))
        matches = sorted(matches, key=lambda x: x.distance)[:self.matchNumber]
        src_pts = np.asarray([self.kp[-1][m.queryIdx].pt for m in matches]).astype(np.float32)
        dst_pts = np.asarray([kp[m.trainIdx].pt for m in matches]).astype(np.float32)
        if self.strategy == 'perspective':
            homography, homo_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            mat = np.matmul(homography, self.mats[-1])
            new_label = cv2.warpPerspective(self.labels[0], mat, (self.w, self.h))
        else:
            mat = cv2.estimateAffine2D(src_pts, dst_pts)[0]
            mat = matmul_affine_matrices(self.mats[-1], mat)
            new_label = cv2.warpAffine(self.labels[0], mat, (self.w, self.h))
        self.images.append(next_image)
        self.labels.append(new_label)
        self.kp.append(kp)
        self.des.append(des)
        self.mats.append(mat)
        return new_label

if __name__ == '__main__':
    import paths
    import os
    d = os.path.join(paths.DATASET_DIR, paths.ALL_CASES[1])
    fns = [fn for fn in os.listdir(os.path.join(d, 'label')) if fn.endswith('.png')]
    images = [cv2.imread(os.path.join(d, fn)) for fn in fns]
    labels = [cv2.imread(os.path.join(d, 'label', fn)) for fn in fns]
    tracker = TrackerAverage(images[0], labels[0])
    for i in range(1, len(fns)):
        new_label = tracker.track_label(images[i])
        cv2.imwrite(f'tmp_label_{i}.png', new_label)
        metrics = evaluate_segmentation(
            characterize(new_label.transpose((2,0,1)), LABEL_GT_CHARACTERIZER),
            characterize(labels[i].transpose((2,0,1)), LABEL_GT_CHARACTERIZER)
        )
        print(metrics)
    