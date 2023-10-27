from torch import nn
import cv2
import numpy as np

from network.transform import Affine2dTransformer, Affine2dPredictor


class TrackerNet(nn.Module):
    def __init__(self, init_frame, init_mask):
        super().__init__()
        self.last_frame = init_frame
        self.last_mask = init_mask
        self.reg_net = Affine2dPredictor()
        self.reg_trans = Affine2dTransformer()
        return

    def forward(self, new_frame):
        reg_params = self.reg_net(self.last_frame, new_frame)
        new_mask = self.reg_trans(reg_params, new_frame)
        self.last_frame = new_frame
        return new_mask


class TrackerKP:
    """
    Use key points
    """
    def __init__(self, init_frame=None, init_mask=None):
        """

        Args:
            init_frame (np.ndarray):
            init_mask (np.ndarray):
        """
        self.orb_detector = cv2.ORB_create(500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.height, self.width = init_frame.shape[:-1]
        self.frames = [init_frame]
        self.masks = [init_mask]
        self.matrices = [np.identity(3)]
        self.kp_and_d = [self.orb_detector.detectAndCompute(init_frame, None)]
        return

    def init(self, init_frame, init_mask):
        self.height, self.width = init_frame.shape[:-1]
        self.frames = [init_frame]
        self.masks = [init_mask]
        self.matrices = [np.identity(3)]
        self.kp_and_d = [self.orb_detector.detectAndCompute(init_frame, None)]

    def __call__(self, new_frame):
        """
        Get the mask of the new frame.
        Args:
            new_frame (np.ndarray): dtype=np.uint8
        Returns:
            np.ndarray:
        """
        new_kp, new_d = self.orb_detector.detectAndCompute(new_frame, None)
        old_kp = self.kp_and_d[-1][0]
        matches = self.matcher.match(self.kp_and_d[-1][1], new_d)
        matches = matches[:int(len(matches) * 0.4)]
        p1 = np.asarray([old_kp[m.queryIdx].pt for m in matches])
        p2 = np.asarray([new_kp[m.trainIdx].pt for m in matches])
        homography, homo_mask = cv2.findHomography(p1, p2, cv2.RANSAC)
        new_matrix = np.matmul(homography, self.matrices[-1])
        new_mask = cv2.warpPerspective(self.masks[0], new_matrix, (self.width, self.height))
        self.matrices.append(new_matrix)
        self.kp_and_d.append((new_kp, new_d))
        self.frames.append(new_frame)
        self.masks.append(new_mask)
        return new_mask
