import json
import math
import os
import argparse

import numpy as np
import torch
import torch.optim
from torch.nn import CosineSimilarity
import torch.nn.functional as nnf
import cv2
import SimpleITK as sitk

import paths
from network.profen import ProFEN
from network.affine2d import Affine2dPredictor, Affine2dTransformer
from network.track import TrackerKP
from utils import crop_and_resize_square, make_channels, tensor_to_cv2, cv2_to_tensor, time_it, images_alpha_lighten
from probe import Probe, deserialize_probes, ablation_num_of_probes
from dataloaders import set_fold, TestSingleCaseDataloader
import paths
from render import PRRenderer

CASE_INFO = {
    'GongDaoming': 'type1',
    'JiangTao': 'type2',
    'JiWeifeng': 'type1',
    'LinXirong': 'type3',
    'LiuHongshan': 'type4',
    'SunYufeng': 'type1',
    'YinRentang': 'type1',
    'WuQuan': 'type5',
    'WuYong': 'type4',
}

PROBE_PRESETS = {
    # azimuth should be in (-180, 180] degrees, elevation should be in [-90, 90] degrees.
    # each lambda expression inputs a np.ndarray and outputs a bool np.ndarray
    'type1': {
        'description': 'The renal main axis is z=y, the renal hilum is face to (-1, -1, 0).',
        'azimuth': lambda x: (45 <= x) & (x <= 135),
        'elevation': lambda x: (-60 <= x) & (x <= 0)
    },
    'type2': {
        'description': 'The renal main axis is z=-x, the renal hilum is face to (-1, -1, -1).',
        'azimuth': lambda x: (90 <= x) & (x <= 180),
        'elevation': lambda x: (-60 <= x) & (x <= 0)
    },
    'type3': {
        'description': 'The renal main axis is z=y, the renal hilum is face to (1, -1, 0).',
        'azimuth': lambda x: (45 <= x) & (x <= 135),
        'elevation': lambda x: (-45 <= x) & (x <= 15)
    },
    'type4': {
        'description': 'The renal main axis is z=y, the renal hilum is face to (1, -1, 0).',
        'azimuth': lambda x: (135 <= x) & (x <= 180) | (-180 < x) & (x <= -135),
        'elevation': lambda x: (-60 <= x) & (x <= 30)
    },
    'type5': {
        'description': 'The renal main axis is z=y, the renal hilum is face to (1, -1, 0).',
        'azimuth': lambda x: (-45 <= x) & (x <= 60),
        'elevation': lambda x: (-60 <= x) & (x <= 0)
    }
}


class BaseAffineSolver:
    def __init__(self) -> None:
        return
    
    def solve(self, src: np.ndarray, dst: np.ndarray):
        """
        Compute the affine factor from src to dst
        Args:
            src (np.ndarray): (C=2, H, W), values in [0, 1]
            dst (np.ndarray): (C=2, H, W), values in [0, 1]
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): tx, ty, rot, scale
        """
        pass
    
    def solve_and_affine(self, src: np.ndarray, dst: np.ndarray, new_in: np.ndarray):
        """
        Args:
            src (np.ndarray): (C=2, H, W), values in [0, 1]
            dst (np.ndarray): (C=2, H, W), values in [0, 1]
            new_in (np.ndarray): from cv2, (H, W, C(BGR))
        Returns:
            np.ndarray: (H, W, C(BGR))
        """
        tx, ty, rot, scale = self.solve(src, dst)
        old_ratio = src.shape[1] / src.shape[2]
        new_ratio = new_in.shape[0] / new_in.shape[1]
        inv_scale = 1.0 / scale
        # because of center crop, src is always a part of new_in
        if new_ratio < old_ratio:
            tx = tx * new_ratio / old_ratio
        else:
            ty = ty / new_ratio * old_ratio
        mat = torch.stack([
            torch.stack([inv_scale * torch.cos(rot), inv_scale * new_ratio * torch.sin(rot), ty], dim=0),
            torch.stack([-inv_scale / new_ratio * torch.sin(rot), inv_scale * torch.cos(rot), tx], dim=0)
        ], dim=0).unsqueeze(0)
        new_in_tensor = cv2_to_tensor(new_in).unsqueeze(0)
        grid = nnf.affine_grid(mat, new_in_tensor.size(), align_corners=False)
        transformed = nnf.grid_sample(new_in_tensor, grid, mode='bilinear', align_corners=False).squeeze()
        transformed = tensor_to_cv2(transformed)
        return transformed, mat


class NetworkAffineSolver(BaseAffineSolver):
    def __init__(self, weight_path) -> None:
        self.predictor = Affine2dPredictor().cuda()
        self.transformer = Affine2dTransformer().cuda()
        return
    
    def solve(self, src: np.ndarray, dst: np.ndarray):
        src_tensor = torch.from_numpy(src).unsqueeze(0).cuda()
        dst_tensor = torch.from_numpy(dst).unsqueeze(0).cuda()
        params = self.predictor(src_tensor, dst_tensor).detach().cpu().squeeze()
        tx, ty, rot, scale = params
        tx = self.transformer.tx_lambda(tx)
        ty = self.transformer.ty_lambda(ty)
        rot = self.transformer.rot_lambda(rot)
        scale = self.transformer.scale_lambda(scale)
        return tx, ty, rot, scale


class GeometryAffineSolver(BaseAffineSolver):
    def __init__(self) -> None:
        self.mse_func = torch.nn.MSELoss(reduction='none')
        return
    
    @staticmethod
    def centroid(i: np.ndarray):
        '''
        Args: 
            i (np.ndarray): (H, W)
        '''
        x = (i.sum(1) * np.arange(i.shape[0])).sum() / i.sum()
        y = (i.sum(0) * np.arange(i.shape[1])).sum() / i.sum()
        return x, y
    
    def solve(self, src: np.ndarray, dst: np.ndarray):
        h = src.shape[1]
        w = src.shape[2]
        ch0, cw0 = self.centroid(src.sum(0))
        ch1, cw1 = self.centroid(dst.sum(0))
        # use tumor channel to compute scale, sometimes tumor is complete and kidney is not
        scale = torch.tensor((dst[1].sum() / src[1].sum()) ** 0.5, dtype=torch.float32)
        inv_scale = 1.0 / scale
        transl_w = torch.tensor((cw0 / w * 2 - 1) - (cw1 / w * 2 - 1) / scale, dtype=torch.float32)
        transl_h = torch.tensor((ch0 / h * 2 - 1) - (ch1 / h * 2 - 1) / scale, dtype=torch.float32)
        rot_batch = [torch.tensor(i / 180 * math.pi) for i in range(360)]
        mat_batch = torch.stack([torch.stack([
            torch.stack([inv_scale * torch.cos(rot), inv_scale * (h / w) * torch.sin(rot), transl_w], dim=0),
            torch.stack([-inv_scale / (h / w) * torch.sin(rot), inv_scale * torch.cos(rot), transl_h], dim=0)
        ], dim=0) for rot in rot_batch], dim=0).cuda()
        src_tensor = torch.from_numpy(src).unsqueeze(0).repeat(len(rot_batch), 1, 1, 1).cuda()
        dst_tensor = torch.from_numpy(dst).unsqueeze(0).repeat(len(rot_batch), 1, 1, 1).cuda()
        grid_batch = nnf.affine_grid(mat_batch, src_tensor.size(), align_corners=False)
        transformed_batch = nnf.grid_sample(src_tensor, grid_batch, mode='nearest', align_corners=False)
        errors = self.mse_func(transformed_batch.flatten(1), dst_tensor.flatten(1)).mean(1)
        rot_index = errors.argmin()
        rot = rot_batch[rot_index]
        return transl_h, transl_w, rot, scale
    
    
class HybridAffineSolver(BaseAffineSolver):
    def __init__(self, weight_path) -> None:
        self.network = NetworkAffineSolver(weight_path)
        self.geometry = GeometryAffineSolver()

    def solve(self, src: np.ndarray, dst: np.ndarray):
        geo_tx, geo_ty, geo_rot, geo_scale = self.geometry.solve(src, dst)
        src_tensor = torch.from_numpy(src).unsqueeze(0).cuda()
        inv_geo_scale = 1.0 / geo_scale
        ratio = src.shape[1] / src.shape[2]
        geo_mat = torch.stack([
            torch.stack([inv_geo_scale * torch.cos(geo_rot), inv_geo_scale * ratio * torch.sin(geo_rot), geo_ty], dim=0),
            torch.stack([-inv_geo_scale / ratio * torch.sin(geo_rot), inv_geo_scale * torch.cos(geo_rot), geo_tx], dim=0)
        ], dim=0).unsqueeze(0)
        geo_grid = nnf.affine_grid(geo_mat, src_tensor.size(), align_corners=False).cuda()
        net_in_tensor = nnf.grid_sample(src_tensor, geo_grid, mode='nearest', align_corners=False)
        net_in = net_in_tensor.squeeze().detach().cpu().numpy()
        net_tx, net_ty, net_rot, net_scale = self.network.solve(net_in, dst)
        
        # inv_net_scale = 1.0 / net_scale
        # net_mat = torch.stack([
        #     torch.stack([inv_net_scale * torch.cos(net_rot), inv_net_scale * ratio * torch.sin(net_rot), net_ty], dim=0),
        #     torch.stack([-inv_net_scale / ratio * torch.sin(net_rot), inv_net_scale * torch.cos(net_rot), net_tx], dim=0)
        # ], dim=0).unsqueeze(0)
        # net_grid = nnf.affine_grid(net_mat, net_in_tensor.size(), align_corners=False).cuda()
        # net_out_tensor = nnf.grid_sample(net_in_tensor, net_grid, mode='nearest', align_corners=False)
        # net_out = net_out_tensor.squeeze().detach().cpu().numpy()
        
        tx = geo_tx + net_tx
        ty = geo_ty + net_ty
        rot = geo_rot + net_rot
        scale = geo_scale * net_scale
        return tx, ty, rot, scale

    
class Fuser:
    def __init__(self, case_type, probes, feature_extractor=None, affine2d_solver=None,
                 image_size=512):
        """
        Do the preoperative and intraoperative image fusion.
        Args:
            probes (list of Probe):
            feature_extractor (torch.nn.Module):
            affine2d_solver (BaseAffineSolver):
            image_size (int or tuple of int):
        """
        self.image_size = image_size
        self.feature_extractor = feature_extractor
        self.affine_solver = affine2d_solver
        self.feature_sim_func = CosineSimilarity()

        self.probes = probes
        # (Np,)
        self.probe_azimuth = np.asarray([p.get_spcoord_dict()['azimuth'] for p in probes])
        # (Np,)
        self.probe_elevation = np.asarray([p.get_spcoord_dict()['elevation'] for p in probes])
        # (Np, 2, H, W)
        self.rendered_2ch_pool = np.asarray([make_channels(p.render.transpose((2, 0, 1)),
                                                            [lambda x: x[0] != 0, lambda x: (x[0] == 0) & (x.any(0))])
                                            for p in probes])
        # (Np, L) 
        # to avoid CUDA OOM, should not input all of the rendered_2ch_pool as 1 batch into feature extractor
        self.feature_pool = []
        bs = 64
        for i in range(len(self.rendered_2ch_pool) // bs + 1):
            batch = np.asarray(self.rendered_2ch_pool[i * bs : min(i * bs + bs, len(self.rendered_2ch_pool))])
            batch = torch.from_numpy(batch).cuda()
            pred = self.feature_extractor(batch).detach().cpu()
            self.feature_pool.append(pred)
        self.feature_pool = torch.cat(self.feature_pool, dim=0)

        # PPS
        self.restriction = PROBE_PRESETS[case_type]
        
        # for re-render
        self.renderer = PRRenderer(probes[0].mesh_path, out_size=image_size)

    @time_it
    def add_frame(self, frame, segment_2ch, ablation=None):
        """
        Use the closest probe in the given range.
        Args:
            frame (np.ndarray): shape of (H, W, BGR)
            segment (np.ndarray): shape of (C=2, H, W). values are 0/1
            prior_type (str): indicate which type of kidney.
        Returns:
            dict:
                'render' (np.ndarray): shape of (H, W, 3)
        """
        # find the best matching probe
        seg_feature = self.feature_extractor(torch.from_numpy(segment_2ch).unsqueeze(0).cuda()).detach().cpu()
        similarity = self.feature_sim_func(seg_feature, self.feature_pool).numpy()
        if ablation is None or ablation != 'wo_pps':
            similarity = np.where(self.restriction['azimuth'](self.probe_azimuth) & self.restriction['elevation'](self.probe_elevation),
                                  similarity, -1)
        hit_index = similarity.argmax()
        # registration src
        hit_render_2ch = self.rendered_2ch_pool[hit_index]
        # re-render with additional information, registration new_in
        re_rendered = self.probes[hit_index].re_render(renderer=self.renderer, draw_mesh=None)
        # new_out
        transformed, affine_matrix = self.affine_solver.solve_and_affine(hit_render_2ch, segment_2ch, re_rendered)
        # fuse
        fused = (images_alpha_lighten(frame / 255, transformed / 255, 0.5) * 255).astype(np.uint8)
        # information
        frame_info = {
            'original': frame,
            'fusion': fused,
            'hit index': hit_index,
            'hit render': self.probes[hit_index].render,
            're-rendered': re_rendered,
            'affine matrix': affine_matrix,
            'transformed': transformed,
        }
        return frame_info


def evaluate(predict, label):
    predict = predict[0]
    label = label[0]
    dice = 2 * (predict * label).sum() / (predict.sum() + label.sum())
    predict = predict.astype(np.float32)
    label = label.astype(np.float32)
    mask1 = sitk.GetImageFromArray(predict, isVector=False)
    mask2 = sitk.GetImageFromArray(label, isVector=False)
    contour_filter = sitk.SobelEdgeDetectionImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(contour_filter.Execute(mask1), contour_filter.Execute(mask2))
    hd = hausdorff_distance_filter.GetHausdorffDistance()
    avd = hausdorff_distance_filter.GetAverageHausdorffDistance()
    return {
        'dice': dice,
        'hd': hd,
        'avd': avd
    }


def test(fold=0, n_fold=6, ablation=None):
    base_dir = paths.DATASET_DIR
    _, test_cases = set_fold(fold, n_fold)
    for case_id in test_cases:
        # directories and dataloader
        case_dir = os.path.join(base_dir, case_id)
        result_dir = paths.RESULTS_DIR if ablation is None else paths.RESULTS_DIR + '_' + ablation
        os.makedirs(result_dir, exist_ok=True)
        fusion_dir = os.path.join(result_dir, case_id, 'fusion')
        os.makedirs(fusion_dir, exist_ok=True)
        case_dataloader = TestSingleCaseDataloader(case_dir)
        
        # probes
        probe_path = os.path.join(paths.RESULTS_DIR, case_id, paths.PROBE_FILENAME)
        probes = deserialize_probes(probe_path)
        if ablation is not None and ablation.startswith('div_'):
            probes = ablation_num_of_probes(probes, factor=int(ablation[4:]) ** 0.5)
        
        # feature extractor
        profen_weight_dir = 'profen' if ablation is None else 'profen_' + ablation
        profen_path = '{}/fold{}/{}/best.pth'.format(paths.WEIGHTS_DIR, fold, profen_weight_dir)
        profen = ProFEN().cuda()
        profen.load_state_dict(torch.load(profen_path))
        profen.eval()
        
        # affine 2d solver
        affine2d_weight_dir = 'affine2d' if ablation is None else 'affine2d_' + ablation
        affine2d_path = '{}/fold{}/{}/best.pth'.format(paths.WEIGHTS_DIR, fold, affine2d_weight_dir)
        # affine_solver = HybridAffineSolver(weight_path=affine2d_path)
        # affine_solver = NetworkAffineSolver(weight_path=affine2d_path)
        affine_solver = GeometryAffineSolver()

        # case type
        if case_id not in CASE_INFO.keys():
            case_type = 'type1'
        else:
            case_type = CASE_INFO[case_id]
        
        # fuse
        registrator = Fuser(
            probes=probes,
            case_type=case_type,
            feature_extractor=profen,
            affine2d_solver=affine_solver,
            image_size=case_dataloader.image_size(),
        )

        evaluations = {}
        for i in range(case_dataloader.length()):
            photo = case_dataloader.images[i]
            orig_segment = case_dataloader.labels[i]
            if orig_segment is None:        # some frames do not have segmented labels
                continue
            segment = crop_and_resize_square(orig_segment, out_size=512).transpose((2, 0, 1))
            segment = make_channels(segment, [
                lambda x: x[2] != 0,
                lambda x: x[1] != 0
            ])
            frame_info = registrator.add_frame(photo, segment)
            cv2.imwrite('{}/{}'.format(fusion_dir, case_dataloader.fns[i]), frame_info['fusion'])
            try:
                metrics = evaluate(
                    make_channels(frame_info['transformed'].transpose((2, 0, 1)), [
                        lambda x: x.any(0) & (x[0] < x[1] + x[2]) & (x[2] < x[0] + x[1]),
                        lambda x: (x[0] < 0.1) & (x[1] > 0.2) & (x[2] > 0.2)
                    ]),
                    make_channels(orig_segment.transpose((2, 0, 1)), [
                        lambda x: x.any(0),
                        lambda x: x[1] != 0
                    ])
                )
            except:
                # mostly because of hausdorff distance compute with no pixel
                print('Exception occurs when computing metrics of case {} frame {}.')
                continue
            evaluations[case_dataloader.fns[i]] = metrics
            print('Case: {} Frame: {} is OK.'.format(case_id, case_dataloader.fns[i]))
        average_metrics = {
            'dice':
                np.asarray([case['dice'] for case in evaluations.values()]).mean(),
            'hd':
                np.asarray([case['hd'] for case in evaluations.values()]).mean(),
            'avd':
                np.asarray([case['avd'] for case in evaluations.values()]).mean()
        }
        evaluations['average'] = average_metrics
        with open('{}/{}/metrics.json'.format(result_dir, case_id), 'w') as f:
            json.dump(evaluations, f, indent=4)
        # explicitly delete registrator, release renderer in time, avoid GL errors
        del registrator
        print('Case {} is OK.'.format(case_id))

    print('All OK')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fusion settings')
    parser.add_argument('--gpu', type=int, default=2, required=False, help='do inference on which gpu')
    parser.add_argument('--folds', type=list, default=[0, 1, 2, 3, 4, 5], required=False, 
                        help='which folds should be tested in fusion, e.g. --folds 0 2 4')
    parser.add_argument('--n_folds', type=int, default=6, required=False, 
                        help='how many folds in total')
    parser.add_argument('--ablation', type=bool, default=False, required=False, 
                        help='whether do the ablation')
    args = parser.parse_args()
    args.folds = [int(f) for f in args.folds]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    if not args.ablation:
        for fold in args.folds:
            test(fold, args.n_folds)
    else:
        for fold in args.folds:
            test(fold, args.n_folds, ablation='wo_ref_loss')
            test(fold, args.n_folds, ablation='div_4')
            test(fold, args.n_folds, ablation='div_9')
            test(fold, args.n_folds, ablation='div_16')
            test(fold, args.n_folds, ablation='wo_agent')
            test(fold, args.n_folds, ablation='wo_pps')
