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
from utils import resized_center_square, make_channels, tensor_to_cv2, cv2_to_tensor, time_it, images_alpha_lighten
from probe import Probe, deserialize_probes, ablation_num_of_probes
from dataloaders import set_fold, TestSingleCaseDataloader
import paths

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
    
    def solve(self, src: np.ndarray, dst: np.ndarray, new_in: np.ndarray):
        '''
        Compute the affine factor from src to dst, and apply it on new_in
        Args:
            src (np.ndarray): (C=2, H, W), values in [0, 1]
            dst (np.ndarray): (C=2, H, W), values in [0, 1]
            new_in (np.ndarray): (C'=3, H', W')
        Returns:
            (np.ndarray): (C', H', W')
        '''
        pass


class NetworkAffineSolver(BaseAffineSolver):
    def __init__(self, weight_path) -> None:
        self.predictor = Affine2dPredictor().cuda()
        self.transformer = Affine2dTransformer().cuda()
        return
    
    def solve(self, src: np.ndarray, dst: np.ndarray, new_in: np.ndarray):
        ref0 = torch.from_numpy(src.unsqueeze(0)).cuda()
        ref1 = torch.from_numpy(dst.unsqueeze(0)).cuda()
        params = self.predictor(ref0, ref1).detach().cpu().squeeze()
        tx, ty, rot, scale = params
        tx = self.transformer.tx_lambda(tx)
        ty = self.transformer.ty_lambda(ty)
        rot = self.transformer.rot_lambda(rot)
        scale = self.transformer.scale_lambda(scale)
        new_in_tensor = cv2_to_tensor(new_in).unsqueeze(0)
        new_ratio = new_in.size(2) / new_in.size(3)
        old_ratio = ref0.size(2) / ref0.size(3)
        if new_ratio < old_ratio:
            tx = tx * new_ratio / old_ratio
        else:
            ty = ty / new_ratio * old_ratio
        mat = torch.stack([
            torch.stack([1.0 / scale * torch.cos(rot), 1.0 / scale * new_ratio * torch.sin(rot), tx], dim=0),
            torch.stack([-1.0 / scale / new_ratio * torch.sin(rot), 1.0 / scale * torch.cos(rot), ty], dim=0)
        ], dim=0).unsqueeze(0)
        grid = nnf.affine_grid(mat, new_in_tensor.size(), align_corners=False)
        transformed = nnf.grid_sample(new_in_tensor, grid, mode='bilinear').squeeze()
        transformed = tensor_to_cv2(transformed)
        affine_factor = {
            'translation x': tx,
            'translation y': ty,
            'rotation degree': rot / np.pi * 180,
            'scale factor': scale
        }
        return transformed, affine_factor


class GeometryAffineSolver(BaseAffineSolver):
    def __init__(self) -> None:
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
    
    def solve(self, src: np.ndarray, dst: np.ndarray, new_in: np.ndarray):
        h = src.shape[1]
        w = src.shape[2]
        ch0, cw0 = self.centroid(src.sum(0))
        ch1, cw1 = self.centroid(dst.sum(0))
        s = (dst.sum() / src.sum()) ** 0.5
        new_in_tensor = cv2_to_tensor(new_in).unsqueeze(0)
        tw = torch.tensor((cw0 / w * 2 - 1) - (cw1 / w * 2 - 1) / s, dtype=torch.float32)
        th = torch.tensor((ch0 / h * 2 - 1) - (ch1 / h * 2 - 1) / s, dtype=torch.float32)
        rot_batch = [torch.tensor(i / 180 * math.pi) for i in range(-60, 60)]
        mat_batch = torch.stack([torch.stack([
            torch.stack([1.0 / s * torch.cos(rot), 1.0 / s * (h / w) * torch.sin(rot), tw], dim=0),
            torch.stack([-1.0 / s / (h / w) * torch.sin(rot), 1.0 / s * torch.cos(rot), th], dim=0)
        ], dim=0) for rot in rot_batch], dim=0)
        ref0_batch = torch.from_numpy(src).unsqueeze(0).repeat(len(rot_batch), 1, 1, 1)
        grid_batch = nnf.affine_grid(mat_batch, ref0_batch.size(), align_corners=False)
        trans0_batch = nnf.grid_sample(ref0_batch, grid_batch, mode='nearest')
        errors = np.asarray([float(nnf.mse_loss(trans0, torch.from_numpy(dst))) for trans0 in trans0_batch])
        rot_index = errors.argmin()
        rot = rot_batch[rot_index]
        new_ratio = new_in.shape[1] / new_in.shape[2]
        if new_ratio < (h / w):
            tw = tw * new_ratio / (h / w)
        else:
            th = th / new_ratio * (h / w)
        mat = torch.stack([
            torch.stack([1.0 / s * torch.cos(rot), 1.0 / s * new_ratio * torch.sin(rot), tw], dim=0),
            torch.stack([-1.0 / s / new_ratio * torch.sin(rot), 1.0 / s * torch.cos(rot), th], dim=0)
        ], dim=0).unsqueeze(0)
        grid = nnf.affine_grid(mat, new_in_tensor.size(), align_corners=False)
        transformed = nnf.grid_sample(new_in_tensor, grid, mode='bilinear').squeeze()
        transformed = tensor_to_cv2(transformed)
        affine_factor = {
            'translation x': th,
            'translation y': tw,
            'rotation degree': rot / math.pi * 180,
            'scale factor': s
        }
        return transformed, affine_factor
    
    
class MontionAffineSolver(BaseAffineSolver):
    def __init__(self) -> None:
        self.image_list = []
        self.tracker = TrackerKP()
        
    def solve(self, src: np.ndarray, dst: np.ndarray, new_in: np.ndarray):
        last_frame = self.image_list[-1]
        
        return

    
class Fuser:
    def __init__(self, case_type, probes, feature_extractor=None, affine2d_solver=None,
                 image_size=512):
        """
        Do the preoperative and intraoperative image fusion.
        Use a frame window to control the stability.
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

    @time_it
    def add_frame(self, frame, segment_2ch, prior_type='type1', ablation=None):
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
                                  similarity,
                                  -1)
        hit_index = similarity.argmax()
        # registration src
        hit_render_2ch = self.rendered_2ch_pool[hit_index]
        # re-render with additional information, registration new_in
        re_rendered = self.probes[hit_index].re_render(out_size=self.image_size, draw_mesh=None)
        # new_out
        transformed, affine_factor = self.affine_solver.solve(hit_render_2ch, segment_2ch, re_rendered)
        # fuse
        fused = (images_alpha_lighten(frame, transformed / transformed.max(), 0.5) * 255).astype(np.uint8)
        # information
        frame_info = {
            'original': frame,
            'fusion': fused,
            'hit index': hit_index,
            'hit azimuth': self.probe_azimuth[hit_index],
            'hit elevation': self.probe_elevation[hit_index],
            'hit render': self.probes[hit_index].render,
            're-rendered': re_rendered,
            'affine factor': affine_factor,
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
        affine_network = NetworkAffineSolver(weight_path=affine2d_path)
        affine_geometry = GeometryAffineSolver()
        
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
            affine2d_solver=affine_geometry,
            image_size=case_dataloader.image_size(),
        )

        evaluations = {}
        for i in range(case_dataloader.length()):
            photo = case_dataloader.images[i]
            orig_segment = case_dataloader.labels[i]
            if orig_segment is None:        # some frames do not have segmented labels
                continue
            segment = resized_center_square(orig_segment, out_size=512).transpose((2, 0, 1))
            segment = make_channels(segment, [
                lambda x: x[2] != 0,
                lambda x: x[1] != 0
            ])
            frame_info = registrator.add_frame(photo, segment, prior_type=case_type)
            cv2.imwrite('{}/{}'.format(fusion_dir, case_dataloader.fns[i]), frame_info['fusion'])
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

    print('All OK')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fusion settings')
    parser.add_argument('--gpu', type=int, default=3, required=False, help='do inference on which gpu')
    parser.add_argument('--folds', type=list, default=[0], required=False, 
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
