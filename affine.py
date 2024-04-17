import math

import torch
import torch.nn.functional as nnf
import numpy as np

from utils import cv2_to_tensor, tensor_to_cv2, time_it
from network.transform import Affine2dPredictor, Affine2dTransformer


class BaseAffineSolver:
    def __init__(self) -> None:
        return
    
    def solve(self, moving: np.ndarray, fixed: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """
        Compute the affine factor from src to dst, assuming they are square.
        Args:
            moving (np.ndarray): (C=2, H, W), values are 0 or 1
            fixed (np.ndarray): (C=2, H, W), values are 0 or 1
        Returns:
            mat (torch.Tensor): sample matrix
            moved (np.ndarray): (C=2, H, W)
        """
        pass
    
    def apply(self, mat, src, return_tensor=True, mode='bilinear'):
        """
        Args:
            src (np.nadarry): (from cv2)
        """
        new_in_tensor = cv2_to_tensor(src).unsqueeze(0).cuda()
        grid = nnf.affine_grid(mat.unsqueeze(0), new_in_tensor.size(), align_corners=False).cuda()
        transformed = nnf.grid_sample(new_in_tensor, grid, mode=mode, align_corners=False).squeeze()
        if not return_tensor:
            transformed = tensor_to_cv2(transformed)
        return transformed
    
    @time_it
    def solve_and_affine(self, moving: np.ndarray, fixed: np.ndarray, src: np.ndarray, return_tensor=True):
        """
        Solve the affine transform factors from moving to fixed, and apply it on src.
        Assume fixed and moving are square images, but src is might not.
        Use torch affine rather than cv2.
        Args:
            fixed (np.ndarray): (C=2, H, W), values are 0 or 1
            moving (np.ndarray): (C=2, H, W), values are 0 or 1
            src (np.ndarray): from cv2, (H', W', C'(BGR))
            is_pad (bool): whether the moving image is padded from src
        Returns:
            dst (np.ndarray): (H', W', C'(BGR))
        """
        mat, moved = self.solve(moving, fixed)
        dst = self.apply(mat, src, return_tensor)
        return dst, mat, moved


class NetworkAffineSolver(BaseAffineSolver):
    def __init__(self, weight_path) -> None:
        self.predictor = Affine2dPredictor().cuda()
        self.transformer = Affine2dTransformer().cuda()
        self.predictor.eval()
        self.transformer.eval()
        return
    
    def solve(self, src: np.ndarray, dst: np.ndarray):
        src_tensor = torch.from_numpy(src).cuda().unsqueeze(0)
        dst_tensor = torch.from_numpy(dst).cuda().unsqueeze(0)
        with torch.no_grad():
            params = self.predictor(src_tensor, dst_tensor).detach().cpu().squeeze()
        tx, ty, rot, scale = params
        tx = self.transformer.tx_lambda(tx)
        ty = self.transformer.ty_lambda(ty)
        rot = self.transformer.rot_lambda(rot)
        scale = self.transformer.scale_lambda(scale)
        return tx, ty, rot, scale


class GeometryAffineSolver(BaseAffineSolver):
    def __init__(self) -> None:
        return
    
    @staticmethod
    def eval_func(preds, target):
        inter = (preds * target).sum((-1, -2))
        psum = preds.sum((-1, -2))
        tsum = target.sum((-1, -2))
        return 2 * inter / (psum + tsum)
    
    @staticmethod
    def centroid(i, normal=True):
        '''
        Args: 
            i: shape of (H, W), (0, 1) value
            normal: if True, normalize to [-1, 1]^2 space
        '''
        if isinstance(i, np.ndarray):
            height = i.shape[0]
            width = i.shape[1]            
            centerh = (i.sum(1) * np.arange(height)).sum() / i.sum()
            centerw = (i.sum(0) * np.arange(width)).sum() / i.sum()
        elif isinstance(i, torch.Tensor):
            height = i.size(0)
            width = i.size(1)
            centerh = (i.sum(1) * torch.arange(height)).sum() / i.sum()
            centerw = (i.sum(0) * torch.arange(width)).sum() / i.sum()
        else:
            raise RuntimeError('The input of function `centroid` must be np.ndarray or torch.Tensor')
        if normal:
            centerh = centerh / height * 2 - 1
            centerw = centerw / width * 2 - 1
        return centerh, centerw
    
    def solve(self, moving: np.ndarray, fixed: np.ndarray):
        centerh_src_0, centerw_src_0 = self.centroid(moving[0])
        centerh_src_1, centerw_src_1 = self.centroid(moving[1])
        centerh_dst_0, centerw_dst_0 = self.centroid(fixed[0])
        centerh_dst_1, centerw_dst_1 = self.centroid(fixed[1])
        tx0 = -centerw_src_0
        ty0 = -centerh_src_0
        tx1 = centerw_dst_0
        ty1 = centerh_dst_0
        scale = (fixed[1].sum() / moving[1].sum()) ** 0.5
        inv_s = 1.0 / scale
        rot_batch = torch.arange(0, 360, 1, dtype=torch.float32, device='cuda') / 180 * torch.pi
        '''
        For src => dst: translate_to_origin -> scale -> rotate -> translate_to_new_position,
        Then for dst => src (sample) is reversed.
        [1, 0, -tx0]   [1/s, 0,   0]   [cos,  sin, 0]   [1, 0, -tx1]
        [0, 1, -ty0] * [0,   1/s, 0] * [-sin, cos, 0] * [0, 1, -ty1]
        [0, 0, 1   ]   [0,   0,   1]   [0,    0,   1]   [0, 0, 1   ]
        '''
        sin_a = torch.sin(rot_batch)
        cos_a = torch.cos(rot_batch)
        mat00_batch = inv_s * cos_a
        mat01_batch = inv_s * sin_a
        mat02_batch = -inv_s * (tx1 * cos_a + ty1 * sin_a) - tx0
        mat12_batch = inv_s * (tx1 * sin_a - ty1 * cos_a) - ty0
        mat0_batch = torch.stack((mat00_batch, mat01_batch, mat02_batch), dim=1)
        mat1_batch = torch.stack((-mat01_batch, mat00_batch, mat12_batch), dim=1)
        mat_batch = torch.stack((mat0_batch, mat1_batch), dim=1)
        moving_tensor = torch.from_numpy(moving).cuda().unsqueeze(0).repeat(len(rot_batch), 1, 1, 1)
        fixed_tensor = torch.from_numpy(fixed).cuda()
        grid_batch = nnf.affine_grid(mat_batch, moving_tensor.size(), align_corners=False)
        dst_batch = nnf.grid_sample(moving_tensor, grid_batch, mode='nearest', align_corners=False)
        metric = self.eval_func(dst_batch, fixed_tensor).mean(-1)
        opt_idx = metric.argmax()
        return mat_batch[opt_idx], dst_batch[opt_idx].detach().cpu().numpy()
    
    
class HybridAffineSolver(BaseAffineSolver):
    def __init__(self, weight_path) -> None:
        self.network = NetworkAffineSolver(weight_path)
        self.geometry = GeometryAffineSolver()

    def solve(self, src: np.ndarray, dst: np.ndarray):
        geo_tx, geo_ty, geo_rot, geo_scale = self.geometry.solve(src, dst)
        src_tensor = torch.from_numpy(src).unsqueeze(0).cuda()
        geo_inv_s = 1.0 / geo_scale
        geo_cos_a = math.cos(geo_rot)
        geo_sin_a = math.sin(geo_rot)
        geo_mat = torch.tensor([[
            [geo_inv_s * geo_cos_a, geo_inv_s * geo_sin_a, -geo_inv_s * (geo_tx * geo_cos_a + geo_ty * geo_sin_a)],
            [-geo_inv_s * geo_sin_a, geo_inv_s * geo_cos_a, geo_inv_s * (geo_tx * geo_sin_a - geo_ty * geo_cos_a)]
        ]], dtype=torch.float32).cuda()
        geo_grid = nnf.affine_grid(geo_mat, src_tensor.size(), align_corners=False).cuda()
        net_in_tensor = nnf.grid_sample(src_tensor, geo_grid, mode='nearest', align_corners=False)
        net_in = net_in_tensor.squeeze().detach().cpu().numpy()
        net_tx, net_ty, net_rot, net_scale = self.network.solve(net_in, dst)
        
        tx = geo_tx + net_tx
        ty = geo_ty + net_ty
        rot = geo_rot + net_rot
        scale = geo_scale * net_scale
        return tx, ty, rot, scale
