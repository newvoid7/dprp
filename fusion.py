import json
import os
import argparse

import numpy as np
import torch
import cv2
import SimpleITK as sitk
from tqdm import tqdm

import paths
from network.profen import ProFEN
from utils import crop_and_resize_square, make_channels, time_it, images_alpha_lighten, cv2_to_tensor, tensor_to_cv2
from affine import BaseAffineSolver, GeometryAffineSolver, NetworkAffineSolver, HybridAffineSolver
from probe import ProbeGroup
from dataloaders import set_fold, TestSingleCaseDataloader
import paths
from render import PRRenderer
from pps import PPS 


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
    },
    'type6' :{
        'destription': 'The renal main axis is z=x, the renal hilum is face to (1, -1, 1).',
        'azimuth': lambda x: (90 <= x) & (x <= 180),
        'elevation': lambda x: (-45 <= x) & (x <= 45)
    }
}

 
class Fuser:
    def __init__(self, 
                 case_type, 
                 probe_group, 
                 with_pps,
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
        if with_pps:
            restriction = restrictions[case_type]
            pps_filtered = restriction['azimuth'](self.probe_azimuth) & restriction['elevation'](self.probe_elevation)
            pps_filtered = torch.from_numpy(pps_filtered).cuda()
            self.pps = PPS(probe_group, self.feature_pool, pps_filtered)
        else:
            self.pps = PPS(probe_group, self.feature_pool)
        
        # for re-render
        self.renderer = PRRenderer(probe_group.mesh_path, out_size=image_size)
        self.extra_rendered = [self.renderer.render(mat=p.get_matrix()) for p in probes]

    @time_it
    def process_frame(self, frame, segment_2ch):
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
        hit_index = self.pps.best(seg_feature)
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
    ret_dict = {}
    for i in range(len(prediction)):
        pred_ch = prediction[i]
        gt_ch = label[i]
        dice = 2 * (pred_ch * gt_ch).sum() / (pred_ch.sum() + gt_ch.sum())
        try:
            pred_ch = pred_ch.astype(np.float32)
            gt_ch = gt_ch.astype(np.float32)
            mask1 = sitk.GetImageFromArray(pred_ch, isVector=False)
            mask2 = sitk.GetImageFromArray(gt_ch, isVector=False)
            contour_filter = sitk.SobelEdgeDetectionImageFilter()
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(contour_filter.Execute(mask1), contour_filter.Execute(mask2))
            hd = hausdorff_distance_filter.GetHausdorffDistance()
            avd = hausdorff_distance_filter.GetAverageHausdorffDistance()
        except:             # Hasudorff distance computing with no pixels
            hd = (prediction.shape[1] + prediction.shape[2]) / 2
            avd = (prediction.shape[1] + prediction.shape[2]) / 2
        ret_dict['channel ' + str(i)] = {
            'dice': dice,
            'hd': hd,
            'avd': avd
        }
    return ret_dict


def test(fold=0, n_fold=4, ablation=None, validation=False):
    base_dir = paths.DATASET_DIR
    if validation:
        test_cases, _ = set_fold(fold, n_fold)
    else:
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
            probe_group.sparse(factor=int(ablation[4:]))
        
        # feature extractor
        profen_weight_dir = 'profen' if ablation is None or ablation == 'wo_pps' else 'profen_' + ablation
        profen_path = '{}/fold{}/{}/best.pth'.format(paths.WEIGHTS_DIR, fold, profen_weight_dir)
        profen = ProFEN().cuda()
        profen.load_state_dict(torch.load(profen_path))
        profen.eval()
        
        # affine 2d solver
        # affine2d_weight_dir = 'affine2d' if ablation is None or ablation == 'wo_pps' else 'affine2d_' + ablation
        # affine2d_path = '{}/fold{}/{}/best.pth'.format(paths.WEIGHTS_DIR, fold, affine2d_weight_dir)
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
            with_pps=(ablation is None or ablation != 'wo_pps'),
            case_type=case_type,
            feature_extractor=profen,
            affine2d_solver=affine_solver,
            image_size=case_dataloader.image_size()
        )

        evaluations = {}
        for i in tqdm(range(case_dataloader.length())):
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
            metrics = evaluate(transformed_2ch, orig_seg_2ch)
            evaluations[case_dataloader.fns[i]] = metrics
            # print('Case: {} Frame: {} is OK.'.format(case_id, case_dataloader.fns[i]))
        evaluations['average'] = {
            channel: {
                metric: np.asarray([case_value[channel][metric] for case_value in evaluations.values()]).mean()
                for metric in list(evaluations.values())[0][channel].keys()
            } for channel in list(evaluations.values())[0].keys()
        }
        with open('{}/{}/metrics.json'.format(result_dir, case_id), 'w') as f:
            json.dump(evaluations, f, indent=4)
        # explicitly delete registrator, release renderer in time, avoid GL errors
        del fuser
        print('Case {} is OK.'.format(case_id))

    print('Fold {} all OK'.format(fold))


def statistic(fold=0, n_fold=4, validation=False):
    if validation:
        target_cases, _ = set_fold(fold, n_fold)
    else:
        _, target_cases = set_fold(fold, n_fold)
    json_paths = [os.path.join(paths.RESULTS_DIR, c, 'metrics.json') for c in target_cases]
    jsons = [json.load(open(p)) for p in json_paths]
    dices = np.asarray([j['average']['channel 0']['dice'] for j in jsons])
    avds = np.asarray([j['average']['channel 0']['avd'] for j in jsons])
    return {
        'dice mean': dices.mean(), 
        'dice std': dices.std(), 
        'avd mean': avds.mean(), 
        'avd std': avds.std()
    }
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fusion settings')
    parser.add_argument('--gpu', type=int, default=0, required=False, help='do inference on which gpu')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3], required=False, 
                        help='which folds should be trained, e.g. --folds 0 2 4')
    parser.add_argument('--n_folds', type=int, default=4, required=False, 
                        help='how many folds in total')
    parser.add_argument('--ablation', action='store_true', default=False, required=False, 
                        help='whether do the ablation')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    ablations = ['div_4', 'div_9', 'div_16', 'wo_ref_loss', 'wo_agent'] if args.ablation else [None]
    for abl in ablations:
        for fold in args.folds:
            test(fold=fold, n_fold=args.n_folds, ablation=abl, validation=False)
            print(statistic(fold=fold, n_fold=args.n_folds, validation=False))
