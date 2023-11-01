import json
import os
import argparse

import numpy as np
import torch
from torch.nn import CosineSimilarity
import cv2
import SimpleITK as sitk

import paths
from network.profen import ProFEN
from utils import crop_and_resize_square, make_channels, time_it, images_alpha_lighten, cv2_to_tensor, tensor_to_cv2
from affine import BaseAffineSolver, GeometryAffineSolver, NetworkAffineSolver, HybridAffineSolver
from probe import ProbeGroup
from dataloaders import set_fold, TestSingleCaseDataloader
import paths
from render import PRRenderer


restrictions = {
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

 
class Fuser:
    def __init__(self, 
                 case_type, 
                 probe_group, 
                 feature_extractor: torch.nn.Module = None, 
                 affine2d_solver: BaseAffineSolver = None,
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

        probes = probe_group.probes
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
        bs = 128
        with torch.no_grad():
            for i in range(len(self.rendered_2ch_pool) // bs + 1):
                batch = np.asarray(self.rendered_2ch_pool[i * bs : min(i * bs + bs, len(self.rendered_2ch_pool))])
                batch = torch.from_numpy(batch).cuda()
                pred = self.feature_extractor(batch)
                self.feature_pool.append(pred)
        self.feature_pool = torch.cat(self.feature_pool, dim=0)

        # PPS
        self.restriction = restrictions[case_type]
        self.pps_filtered = self.restriction['azimuth'](self.probe_azimuth) & self.restriction['elevation'](self.probe_elevation)
        self.pps_filtered = torch.from_numpy(self.pps_filtered).cuda()
        
        # for re-render
        self.renderer = PRRenderer(probe_group.mesh_path, out_size=image_size)
        self.extra_rendered = [self.renderer.render(mat=p.get_matrix()) for p in probes]

    @time_it
    def process_frame(self, frame, segment_2ch, ablation=None):
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
        seg_square_2ch = segment_2ch.transpose((1, 2, 0))
        seg_square_2ch = crop_and_resize_square(seg_square_2ch, out_size=512, interp='nearest')
        seg_square_2ch = seg_square_2ch.transpose((2, 0, 1))
        # find the best matching probe
        seg_feature = self.feature_extractor(torch.from_numpy(seg_square_2ch).cuda().unsqueeze(0))
        similarity = self.feature_sim_func(seg_feature, self.feature_pool)
        if ablation is None or ablation != 'wo_pps':
            # similarity - min() to keep minimum = 0, then set elements out of restriction to 0 too
            similarity = self.pps_filtered * (similarity - similarity.min())
        hit_index = similarity.argmax()
        # registration src
        hit_render_2ch = self.rendered_2ch_pool[hit_index]
        # re-render with additional information, registration new_in
        re_rendered = self.extra_rendered[hit_index]
        # new_out
        transformed, affine_matrix = self.affine_solver.solve_and_affine(hit_render_2ch, seg_square_2ch, re_rendered)
        # fuse
        fused = images_alpha_lighten(cv2_to_tensor(frame).cuda(), transformed, 0.5)
        fused = tensor_to_cv2(fused)
        # information
        frame_info = {
            'original': frame,
            'fusion': fused,
            'hit index': hit_index,
            're-rendered': re_rendered,
            'affine matrix': affine_matrix,
            'transformed': transformed,
        }
        return frame_info


def evaluate(prediction, label):
    """Compute the metrics between predict and label.
    Args:
        prediction (np.ndarray): shape of (C, H, W)
        label (np.ndarray): shape of (C, H, W)
    Returns:
        dict: includes dice, hausdorff distance and average hausdorff distance
    """
    if prediction.shape != label.shape:
        raise RuntimeError('The shape between prediction and label must be the same.')
    prediction = prediction[0]
    label = label[0]
    dice = 2 * (prediction * label).sum() / (prediction.sum() + label.sum())
    prediction = prediction.astype(np.float32)
    label = label.astype(np.float32)
    mask1 = sitk.GetImageFromArray(prediction, isVector=False)
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
        probe_group = ProbeGroup(deserialize_path=probe_path)
        if ablation is not None and ablation.startswith('div_'):
            probe_group.sparse(factor=int(ablation[4:]) ** 0.5)
        
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
        if case_dataloader.prior_info is None:
            case_type = 'type1'
        else:
            case_type = case_dataloader.prior_info['type']
        
        # fuse
        fuser = Fuser(
            probe_group=probe_group,
            case_type=case_type,
            feature_extractor=profen,
            affine2d_solver=affine_solver,
            image_size=case_dataloader.image_size(),
        )

        evaluations = {}
        for i in range(case_dataloader.length()):
            photo = case_dataloader.images[i]
            orig_segment = case_dataloader.labels[i]
            if orig_segment is None:                        # some frames do not have segmented labels
                continue
            orig_seg_2ch = make_channels(orig_segment.transpose((2, 0, 1)), [
                lambda x: x[2] != 0, lambda x: x[1] != 0
            ])
            frame_info = fuser.process_frame(photo, orig_seg_2ch)
            cv2.imwrite('{}/{}'.format(fusion_dir, case_dataloader.fns[i]), frame_info['fusion'])
            transformed_2ch = frame_info['transformed']
            if isinstance(transformed_2ch, torch.Tensor):
                transformed_2ch = transformed_2ch.detach().cpu().numpy()
            transformed_2ch = make_channels(transformed_2ch, [
                lambda x: x.any(0) & (x[0] < x[1] + x[2]) & (x[2] < x[0] + x[1]),
                lambda x: (x[0] < 0.1) & (x[1] > 0.2) & (x[2] > 0.2)
            ])                                               # transformed has some other colors
            try:
                metrics = evaluate(transformed_2ch, orig_seg_2ch)
            except:
                # mostly because of hausdorff distance compute with no pixel
                print('Exception occurs when computing metrics of case {} frame {}.'.format(case_id, case_dataloader.fns[i]))
                continue
            evaluations[case_dataloader.fns[i]] = metrics
            print('Case: {} Frame: {} is OK.'.format(case_id, case_dataloader.fns[i]))
        evaluations['average'] = {
            'dice':
                np.asarray([case['dice'] for case in evaluations.values()]).mean(),
            'hd':
                np.asarray([case['hd'] for case in evaluations.values()]).mean(),
            'avd':
                np.asarray([case['avd'] for case in evaluations.values()]).mean()
        }
        with open('{}/{}/metrics.json'.format(result_dir, case_id), 'w') as f:
            json.dump(evaluations, f, indent=4)
        # explicitly delete registrator, release renderer in time, avoid GL errors
        del fuser
        print('Case {} is OK.'.format(case_id))

    print('Fold {} all OK'.format(fold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fusion settings')
    parser.add_argument('--gpu', type=int, default=3, required=False, help='do inference on which gpu')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5], required=False, 
                        help='which folds should be trained, e.g. --folds 0 2 4')
    parser.add_argument('--n_folds', type=int, default=6, required=False, 
                        help='how many folds in total')
    parser.add_argument('--ablation', action='store_true', default=False, required=False, 
                        help='whether do the ablation')
    args = parser.parse_args()
    
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
