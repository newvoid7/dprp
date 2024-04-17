import json
import os
import argparse

import numpy as np
import torch
import cv2
from tqdm import tqdm

import paths
from network.profen import ProFEN
from network.tracknet import TrackNet
from utils import (LABEL_GT_CHARACTERIZER, 
                   RENDER_FLAT_CHARACTERIZER, 
                   LABEL_PRED_CHARACTERIZER,
                   resize_to_fit, 
                   characterize, 
                   time_it, 
                   images_alpha_lighten, 
                   cv2_to_tensor, 
                   tensor_to_cv2,
                   evaluate_segmentation
                )
from affine import BaseAffineSolver, GeometryAffineSolver, NetworkAffineSolver, HybridAffineSolver
from probe import ProbeGroup
from dataloaders import set_fold, TestSingleCaseDataloader
import paths
from render import PRRenderer
from pps import PPS 


restrictions = {
    # azimuth should be in [0, 360) degrees, elevation should be in [0, 180] degrees.
    # each lambda expression inputs a np.ndarray and outputs a bool np.ndarray
    'type1': {
        'description': 'The renal main axis is z=y, the renal hilum is face to (-1, -1, 0).',
        'azimuth': lambda x: (45 <= x) & (x <= 135),
        'zenith': lambda x: (90 <= x) & (x <= 150)
    },
    'type2': {
        'description': 'The renal main axis is z=-x, the renal hilum is face to (-1, -1, -1).',
        'azimuth': lambda x: (90 <= x) & (x <= 180),
        'zenith': lambda x: (90 <= x) & (x <= 150)
    },
    'type3': {
        'description': 'The renal main axis is z=y, the renal hilum is face to (1, -1, 0).',
        'azimuth': lambda x: (45 <= x) & (x <= 135),
        'zenith': lambda x: (75 <= x) & (x <= 135)
    },
    'type4': {
        'description': 'The renal main axis is z=y, the renal hilum is face to (1, -1, 0).',
        'azimuth': lambda x: (135 <= x) & (x <= 225),
        'zenith': lambda x: (60 <= x) & (x <= 150)
    },
    'type5': {
        'description': 'The renal main axis is z=y, the renal hilum is face to (1, -1, 0).',
        'azimuth': lambda x: (x <= 90) | (315 <= x),
        'zenith': lambda x: (45 <= x) & (x <= 135)
    },
    'type6' :{
        'destription': 'The renal main axis is z=x, the renal hilum is face to (1, -1, 1).',
        'azimuth': lambda x: (90 <= x) & (x <= 180),
        'zenith': lambda x: (45 <= x) & (x <= 135)
    }
}

 
class Fuser:
    def __init__(self, 
                 case_type, 
                 probe_group: ProbeGroup, 
                 with_pps: bool,
                 feature_extractor: torch.nn.Module = None, 
                 affine2d_solver: BaseAffineSolver = None,
                 tracker: torch.nn.Module = None,
                 image_size=512,
                 first_label=None):
        """
        Do the preoperative and intraoperative image fusion.
        Args:
            probes (ProbeGroup):
            feature_extractor (torch.nn.Module):
            affine2d_solver (BaseAffineSolver):
            image_size (int or tuple of int):
            first_label (np.ndarray):
        """
        self.frame_size = image_size
        self.feature_extractor = feature_extractor
        self.affine_solver = affine2d_solver
        self.tracker = tracker
        self.render_size = probe_group.render_size

        probes = probe_group.probes

        self.probe_azimuth = np.asarray([p.get_spcoord_dict()['azimuth'] for p in probes])
        self.probe_zenith = np.asarray([p.get_spcoord_dict()['zenith'] for p in probes])
        rendered_2ch_pool = np.asarray([characterize(p.render.transpose((2, 0, 1)), RENDER_FLAT_CHARACTERIZER) for p in probes])
        # to avoid CUDA OOM, should not input all of the rendered_2ch_pool as 1 batch into feature extractor
        self.feature_pool = []
        bs = 128
        for i in range(len(rendered_2ch_pool) // bs + 1):
            batch = np.asarray(rendered_2ch_pool[i * bs : min(i * bs + bs, len(rendered_2ch_pool))])
            batch = torch.from_numpy(batch).cuda()
            with torch.no_grad():
                pred = self.feature_extractor(batch)
            self.feature_pool.append(pred)
        self.feature_pool = torch.cat(self.feature_pool, dim=0)

        # PPS
        if with_pps:
            restriction = restrictions[case_type]
            pps_filtered = restriction['azimuth'](self.probe_azimuth) & restriction['zenith'](self.probe_zenith)
            pps_filtered = torch.from_numpy(pps_filtered).cuda()
            self.pps = PPS(probe_group.neighbor, self.feature_pool, pps_filtered)
        else:
            self.pps = PPS(probe_group.neighbor, self.feature_pool)
        
        # for re-render
        self.renderer = PRRenderer(probe_group.mesh_path, out_size=image_size)
        self.extra_rendered = [self.renderer.render(mat=p.get_matrix()) for p in probes]
        self.resized_rendered = [self.renderer.render(mat=p.get_matrix(), draw_mesh=probe_group.draw_mesh, mode='FLAT') for p in probes]
        self.resized_rendered = [characterize(r.transpose((2, 0, 1)), RENDER_FLAT_CHARACTERIZER) for r in self.resized_rendered]
        
        # last_label: np.ndarray (C=2, H, W)
        self.last_label = first_label
        self.last_frame = None

    def segmentation(self, new_frame, pad=True):
        if self.last_frame is None:
            return self.last_label
        else:
            last_frame_tensor = cv2_to_tensor(resize_to_fit(self.last_frame, self.render_size, pad=pad)).unsqueeze(0).cuda()
            new_frame_tensor = cv2_to_tensor(resize_to_fit(new_frame, self.render_size, pad=pad)).unsqueeze(0).cuda()
            last_label_cv2 = np.stack([resize_to_fit(c, self.render_size, pad=pad) for c in self.last_label], axis=0)
            last_label_tensor = torch.from_numpy(last_label_cv2).unsqueeze(0).cuda()
            with torch.no_grad():
                new_label_tensor, _ = self.tracker(last_frame_tensor, new_frame_tensor, last_label_tensor)
            new_label = new_label_tensor.squeeze().detach().cpu().numpy()
            new_label = characterize(new_label, LABEL_PRED_CHARACTERIZER)
            new_label = np.stack([resize_to_fit(c, self.frame_size, pad=not pad) for c in new_label], axis=0)
            return new_label

    @time_it
    def process_frame(self, frame):
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
        seg_2ch = self.segmentation(frame)
        # find the best matching probe
        seg_2ch_square = np.stack([resize_to_fit(s, self.render_size, pad=False) for s in seg_2ch], axis=0)
        seg_2ch_square[0] = seg_2ch_square.any(0)   # ProFEN receive not one hot images
        with torch.no_grad():
            seg_feature = self.feature_extractor(torch.from_numpy(seg_2ch_square).cuda().unsqueeze(0))
        hit_index = self.pps.best(seg_feature)
        # registration from moving -> fixed, apply it on source
        moving = self.resized_rendered[hit_index]
        source = self.extra_rendered[hit_index]
        dst, mat, moved = self.affine_solver.solve_and_affine(moving, seg_2ch, source)
        # fuse
        fused = images_alpha_lighten(cv2_to_tensor(frame).cuda(), dst, 0.5)
        fused = tensor_to_cv2(fused)
        # maintain last label and last frame
        self.last_label = moved
        self.last_frame = frame
        # information
        fuse_info = {
            'fused image': fused,       # np.ndarray (H, W, [BGR])
            'original': frame,
            'segmentation': seg_2ch,    # np.ndarray (C=2, H, W)
            'probe index': hit_index,   
            'matrix': mat,              # torch.Tensor (2, 3)
            'moving image': moving,     # np.ndarray (C=2, H, W)
            'fixed image': seg_2ch,     # same as segmentation
            'moved image': moved,       # np.ndarray (C=2, H, W)
            'source image': source,     # np.ndarray (H, W, [BGR])
            'destination image': dst    # np.ndarray (H, W, [BGR])
        }
        return fuse_info


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
        affine_solver = GeometryAffineSolver()

        # segmentation tracker
        tracker_path = '{}/fold{}/tracknet/best.pth'.format(paths.WEIGHTS_DIR, fold)
        tracker = TrackNet().cuda()
        tracker.load_state_dict(torch.load(tracker_path))
        tracker.eval()

        # case type
        if case_dataloader.prior_info is None:
            case_type = 'type1'
        else:
            case_type = case_dataloader.prior_info['type']
        
        # the first label
        first_idx = 0
        for first_idx in range(len(case_dataloader.labels)):
            if case_dataloader.labels[first_idx] is None:
                first_idx += 1
            else:
                break
        first_label = case_dataloader.labels[first_idx]
        first_label = characterize(first_label.transpose((2, 0, 1)), LABEL_GT_CHARACTERIZER)
        
        # fuse
        fuser = Fuser(
            probe_group=probe_group,
            with_pps=(ablation is None or ablation != 'wo_pps'),
            case_type=case_type,
            feature_extractor=profen,
            affine2d_solver=affine_solver,
            tracker=tracker,
            image_size=case_dataloader.image_size(),
            first_label=first_label
        )

        evaluations = {}
        for i in tqdm(iterable=range(first_idx, case_dataloader.length()), 
                      desc='Fusion of case {}'.format(case_id)):
            photo = case_dataloader.images[i]
            fuse_info = fuser.process_frame(photo)
            cv2.imwrite('{}/{}'.format(fusion_dir, case_dataloader.fns[i]), fuse_info['fused image'])
            segment_gt = case_dataloader.labels[i]
            if segment_gt is not None:
                seg_gt_2ch = characterize(segment_gt.transpose((2, 0, 1)), LABEL_GT_CHARACTERIZER)
                metrics = evaluate_segmentation(fuse_info['destination image'], seg_gt_2ch)
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
    parser.add_argument('-g', '--gpu', type=int, default=0, required=False, help='do inference on which gpu')
    parser.add_argument('-f', '--folds', type=int, nargs='+', default=[0, 1, 2, 3], required=False, 
                        help='which folds should be trained, e.g. --folds 0 2 4')
    parser.add_argument('-nf', '--n_folds', type=int, default=4, required=False, 
                        help='how many folds in total')
    parser.add_argument('-a', '--ablations', type=str, nargs='+', 
                        default=['none', 'div_4', 'div_9', 'div_16', 'wo_pps', 'wo_agent'], required=False, 
                        help='whether do the ablation')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    for abl in args.ablations:
        for fold in args.folds:
            test(fold=fold, n_fold=args.n_folds, ablation=abl if abl != 'none' else None, validation=False)
            print(statistic(fold=fold, n_fold=args.n_folds, validation=False))
