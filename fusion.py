import json
import math
import os

import numpy as np
import torch
import torch.optim
from torch.nn import CosineSimilarity
import torch.nn.functional as nnf
import cv2
import SimpleITK as sitk

from render import PRRenderer
from network.profen import ProFEN
from network.affine2d import Affine2dPredictor, Affine2dTransformer
from network.track import TrackerKP
from utils import resized_center_square, make_channels, tensor_to_cv2, cv2_to_tensor, time_it, images_alpha_lighten
from probe import Probe, deserialize_probes, ablation_num_of_probes
from train import set_fold
from paths import DATASET_DIR

CASE_INFO = {
    'GongDaoming': 'type1',
    'JiangTao': 'type2',
    'JiWeifeng': 'type1',
    'LinXirong': 'type3',
    'LiuHongshan': 'type4',
    'SunYufeng': 'type1',
    'WuQuan': 'type5',
    'WuYong': 'type4',
}


class Registrator:
    def __init__(self, mesh_path, probes, profen_pth=None, affine2d_pth=None,
                 image_size=512, window_size=10):
        """
        Do the preoperative and intraoperative image fusion.
        Use a frame window to control the stability.
        Args:
            mesh_path (str):
            probes (list of Probe):
            profen_pth:
            affine2d_pth:
            image_size (int or tuple of int):
            window_size: TODO use a window for calibration
        """
        self.image_size = image_size
        self.renderer = PRRenderer(mesh_path, out_size=image_size)
        self.feature_extractor = ProFEN().cuda()
        self.feature_extractor.load_state_dict(torch.load(profen_pth))
        self.feature_extractor.eval()
        self.affine_predictor = Affine2dPredictor().cuda()
        self.affine_predictor.load_state_dict(torch.load(affine2d_pth))       # TODO Affine transform
        self.affine_predictor.eval()
        self.affine_transformer = Affine2dTransformer().cuda()
        self.feature_sim_func = CosineSimilarity()

        self.probe_presets = {
            # azimuth should be in (-180, 180] degrees, elevation should be in [-90, 90] degrees.
            'type1': {
                'description': 'The renal main axis is z=y, the renal hilum is face to (-1, -1, 0).',
                'azimuth': lambda x: 45 <= x <= 135,
                'elevation': lambda x: -60 <= x <= 0
            },
            'type2': {
                'description': 'The renal main axis is z=-x, the renal hilum is face to (-1, -1, -1).',
                'azimuth': lambda x: 90 <= x <= 180,
                'elevation': lambda x: -60 <= x <= 0
            },
            'type3': {
                'description': 'The renal main axis is z=y, the renal hilum is face to (1, -1, 0).',
                'azimuth': lambda x: 45 <= x <= 135,
                'elevation': lambda x: -45 <= x <= 15
            },
            'type4': {
                'description': 'The renal main axis is z=y, the renal hilum is face to (1, -1, 0).',
                'azimuth': lambda x: 135 <= x <= 180 or -180 < x <= -135,
                'elevation': lambda x: -60 <= x <= 30
            },
            'type5': {
                'description': 'The renal main axis is z=y, the renal hilum is face to (1, -1, 0).',
                'azimuth': lambda x: -45 <= x <= 60,
                'elevation': lambda x: -60 <= x <= 0
            }
        }

        self.probes = probes
        self.probe_sphere_coord = [
            {
                # azimuth in (-180, 180] degrees, elevation in [-90, 90] degrees.
                'radius': np.linalg.norm(p.get_eye()),
                'azimuth': np.arctan2(p.get_eye()[1], p.get_eye()[0]) / np.pi * 180,
                'elevation': np.arcsin(p.get_eye()[2] / np.linalg.norm(p.get_eye())) / np.pi * 180
            }
            for p in probes
        ]
        self.probe_render_2ch = [make_channels(p.render.transpose((2, 0, 1)),
                                               [lambda x: x[0] != 0, lambda x: (x[0] == 0) & (x.any(0))])
                                 for p in probes]
        self.probe_feature = [self.feature_extractor(torch.from_numpy(r).unsqueeze(0).cuda()).detach().cpu()
                              for r in self.probe_render_2ch]

        self.tracker = None

        self.window_size = window_size
        self.frame_window = []

    def affine_transform_network(self, ref0, ref1, src):
        """
        Use a 2D affine registration network to evaluate the affine transform between ref0 and ref1,
        then apply it on src.
        Args:
            ref0 (torch.Tensor): (B, C=2, H, W), values in [0, 1], rendered from probe
            ref1 (torch.Tensor): (B, C=2, H, W), values in [0, 1], segmentation
            src (torch.Tensor): (B, C'=3, H, W), the re-rendered image
        Returns:
            (np.ndarray, dict):
                shape of (H, W, BGR), transformed image.
                dict of affine factors (translation, rotation and scale).
        """
        params = self.affine_predictor(ref0.cuda(), ref1.cuda()).detach().cpu().squeeze()
        params = params.detach().cpu().squeeze()
        tx, ty, rot, scale = params
        tx = self.affine_transformer.tx_lambda(tx)
        ty = self.affine_transformer.ty_lambda(ty)
        rot = self.affine_transformer.rot_lambda(rot)
        scale = self.affine_transformer.scale_lambda(scale)
        new_ratio = src.size(2) / src.size(3)
        old_ratio = ref0.size(2) / ref0.size(3)
        if new_ratio < old_ratio:
            tx = tx * new_ratio / old_ratio
        else:
            ty = ty / new_ratio * old_ratio
        mat = torch.stack([
            torch.stack([1.0 / scale * torch.cos(rot), 1.0 / scale * new_ratio * torch.sin(rot), tx], dim=0),
            torch.stack([-1.0 / scale / new_ratio * torch.sin(rot), 1.0 / scale * torch.cos(rot), ty], dim=0)
        ], dim=0).unsqueeze(0)
        grid = nnf.affine_grid(mat, src.size(), align_corners=False)
        transformed = nnf.grid_sample(src, grid, mode='bilinear').numpy().squeeze().transpose((1, 2, 0))
        affine_factor = {
            'translation x': tx,
            'translation y': ty,
            'rotation degree': rot / np.pi * 180,
            'scale factor': scale
        }
        return transformed, affine_factor

    @staticmethod
    def affine_transform_geometry(ref0, ref1, src):
        """
        Use a geometric based method to evaluate the affine transform between ref0 and ref1, then apply it on src.
        Args:
            ref0 (torch.Tensor): (B=1, C=2, H, W), values are 0/1, rendered from probe
            ref1 (torch.Tensor): (B=1, C=2, H, W), values are 0/1, segmentation
            src (torch.Tensor): (B=1, C'=3, H, W), the re-rendered image
        Returns:
            (np.ndarray, dict):
                shape of (H, W, BGR), transformed image.
                dict of affine factors (translation, rotation and scale).
        """
        def centroid(i: torch.Tensor):
            x = (i.sum(1) * torch.arange(i.size(0))).sum() / i.sum()
            y = (i.sum(0) * torch.arange(i.size(1))).sum() / i.sum()
            return x, y
        h = ref0.size(2)
        w = ref0.size(3)
        ch0, cw0 = centroid(ref0[0, ...].sum(0))
        ch1, cw1 = centroid(ref1[0, ...].sum(0))
        s = (ref1.sum() / ref0.sum()) ** 0.5
        tw = (cw0 / w * 2 - 1) - (cw1 / w * 2 - 1) / s
        th = (ch0 / h * 2 - 1) - (ch1 / h * 2 - 1) / s
        rot_batch = [torch.tensor(i / 180 * math.pi) for i in range(-60, 60)]
        mat_batch = torch.stack([torch.stack([
            torch.stack([1.0 / s * torch.cos(rot), 1.0 / s * (h / w) * torch.sin(rot), tw], dim=0),
            torch.stack([-1.0 / s / (h / w) * torch.sin(rot), 1.0 / s * torch.cos(rot), th], dim=0)
        ], dim=0) for rot in rot_batch], dim=0)
        ref0_batch = ref0.repeat(len(rot_batch), 1, 1, 1)
        grid_batch = nnf.affine_grid(mat_batch, ref0_batch.size(), align_corners=False)
        trans0_batch = nnf.grid_sample(ref0_batch, grid_batch, mode='nearest')
        errors = np.asarray([float(nnf.mse_loss(trans0, ref1.squeeze())) for trans0 in trans0_batch])
        rot_index = errors.argmin()
        rot = rot_batch[rot_index]
        new_ratio = src.size(2) / src.size(3)
        if new_ratio < (h / w):
            tw = tw * new_ratio / (h / w)
        else:
            th = th / new_ratio * (h / w)
        mat = torch.stack([
            torch.stack([1.0 / s * torch.cos(rot), 1.0 / s * new_ratio * torch.sin(rot), tw], dim=0),
            torch.stack([-1.0 / s / new_ratio * torch.sin(rot), 1.0 / s * torch.cos(rot), th], dim=0)
        ], dim=0).unsqueeze(0)
        grid = nnf.affine_grid(mat, src.size(), align_corners=False)
        transformed = nnf.grid_sample(src, grid, mode='bilinear').squeeze().numpy().transpose((1, 2, 0))
        affine_factor = {
            'translation x': th,
            'translation y': tw,
            'rotation degree': rot / math.pi * 180,
            'scale factor': s
        }
        return transformed, affine_factor

    def overlap(self, frame, segmentation, index):
        """
        Overlap a frame with a rgb rendered from position of anchor_index
        Args:
            frame (np.ndarray): original picture, 0-255
            segmentation (torch.Tensor): Tensor (C=2, H, W), values are 0/1,
                first channel is kidney, second channel is tumor
            index (int):
        Returns:
            (np.ndarray, np.ndarray, np.ndarray, dict):
                aligned frame and re-rendered, shape of (H, W, BGR), 0-255, the overlapped image.
                shape of (H, W, BGR), the re-rendered image using parameters from probes[index].
                shape of (H, W, BGR), the re-rendered image after transformation, values in 0-1
        """
        frame = (frame / 255.0).astype(np.float32)
        probe_render = torch.from_numpy(self.probe_render_2ch[index])
        re_rendered = self.renderer.render(self.probes[index].get_matrix(), mode='RGB')[..., ::-1]
        transformed, affine_factor = self.affine_transform_geometry(
            probe_render.unsqueeze(0), segmentation.unsqueeze(0), cv2_to_tensor(re_rendered).unsqueeze(0))
        aligned = (images_alpha_lighten(frame, transformed / transformed.max(), 0.5) * 255).astype(np.uint8)
        return aligned, re_rendered, transformed, affine_factor

    def overlap_tracker(self, frame, index):
        segmentation = self.tracker(frame)
        probe_render = torch.from_numpy(self.probe_render_2ch[index])
        re_rendered = self.renderer.render(self.probes[index].get_matrix(), mode='RGB')[..., ::-1]
        transformed, affine_factor = self.affine_transform_geometry(
            probe_render.unsqueeze(0), segmentation.unsqueeze(0), cv2_to_tensor(re_rendered).unsqueeze(0))
        aligned = (images_alpha_lighten(frame, transformed / transformed.max(), 0.5) * 255).astype(np.uint8)
        return aligned, re_rendered, transformed, affine_factor

    def init_tracker(self, image, segmentation):
        self.tracker = TrackerKP(image, segmentation)
        return

    @time_it
    def add_frame(self, frame, segment, prior_type='type1'):
        """
        Use the closest probe in the given range.
        Args:
            frame (np.ndarray):
            segment (torch.Tensor): shape of (C=2, H, W). values are 0/1
            prior_type (str): indicate which type of kidney.
        Returns:
            dict:
                'render' (np.ndarray): shape of (H, W, 3)
        """
        seg_feature = self.feature_extractor(segment.unsqueeze(0).cuda()).detach().cpu()
        feature_sim = np.asarray([self.feature_sim_func(seg_feature, f).numpy() for f in self.probe_feature])
        restriction = self.probe_presets[prior_type]
        out_restricted_area = [
            not restriction['azimuth'](c['azimuth'])
            or not restriction['elevation'](c['elevation'])
            for c in self.probe_sphere_coord
        ]
        feature_sim[out_restricted_area] = -1               # TODO ablation PPS
        hit_index = feature_sim.argmax()
        hit_render = self.probes[hit_index].render
        fused, re_rendered, transformed, affine_factor = self.overlap(frame, segment, hit_index)

        frame_info = {
            'raw': frame,
            'fusion': fused,
            'hit index': hit_index,
            'hit parameter': self.probe_sphere_coord[hit_index],
            'hit render': hit_render,
            're-rendered': re_rendered,
            'affine factor': affine_factor,
            'transformed': transformed,
        }
        return frame_info

    def add_frame_use_neighbor(self, frame, prior_type='type1'):
        segment = self.tracker()
        seg_feature = self.feature_extractor(segment.unsqueeze(0).cuda()).detach().cpu()
        feature_sim = np.asarray([self.feature_sim_func(seg_feature, f).numpy() for f in self.probe_feature])
        restriction = self.probe_presets[prior_type]
        out_restricted_area = [
            not restriction['azimuth'](c['azimuth'])
            or not restriction['elevation'](c['elevation'])
            for c in self.probe_sphere_coord
        ]
        feature_sim[out_restricted_area] = -1               # TODO ablation PPS
        hit_index = feature_sim.argmax()
        hit_render = self.probes[hit_index].render
        fused, re_rendered, transformed, affine_factor = self.overlap(frame, segment, hit_index)

        frame_info = {
            'fusion': fused,
            'hit index': hit_index,
            'hit parameter': self.probe_sphere_coord[hit_index],
            'hit render': hit_render,
            're-rendered': re_rendered,
            'affine factor': affine_factor,
            'transformed': transformed,
        }
        return frame_info


# def evaluate_registration(render, gt):
#     """
#     Compute: dice, hausdorff distance
#     Args:
#         render (np.ndarray): (C=2, H, W), the transformed render result, 2 channels for 2 targets.
#         gt (np.ndarray): (C=2, H, W), the label image.
#     """
#     # first extract different parts of the render image and ground truth
#     render_kidney = render[0]
#     render_tumor = render[1]
#     gt_kidney = gt[0]
#     gt_tumor = gt[1]
#
#     dice_kidney = 2 * ((render_kidney * gt_kidney).sum()) / (render_kidney.sum() + gt_kidney.sum())
#     dice_tumor = 2 * ((render_tumor * gt_tumor).sum()) / (render_tumor.sum() + gt_tumor.sum())
#
#     render_kidney_border, _ = cv2.findContours(
#         (render_kidney * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     render_tumor_border, _ = cv2.findContours(
#         (render_tumor * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     gt_kidney_border, _ = cv2.findContours(
#         (gt_kidney * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     gt_tumor_border, _ = cv2.findContours(
#         (gt_tumor * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#     hausdorff_de = cv2.createHausdorffDistanceExtractor()
#     hd_kidney = hausdorff_de.computeDistance(render_kidney_border[0], gt_kidney_border[0])
#     # hd_tumor = hausdorff_de.computeDistance(render_tumor_border[0], gt_tumor_border[0])
#
#     distance_matrix = np.asarray([
#         ((gt_kidney_border[0][0].squeeze() - render_pt) ** 2).sum(-1) ** 0.5
#         for render_pt in render_kidney_border[0][0].squeeze()
#     ])
#     avd = distance_matrix.min(-1).mean()
#
#     return {
#         'dice':
#             {'kidney': dice_kidney, 'tumor': dice_tumor},
#         'hd':
#             {'kidney': hd_kidney},
#         'avd':
#             {'kidney': avd}
#     }

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


def alpha_test(fold=0, **kwargs):
    base_dir = DATASET_DIR
    _, test_cases = set_fold(fold)
    for case_id in test_cases:
        case_dir = os.path.join(base_dir, case_id)
        case_type = CASE_INFO[case_id]
        filenames = [fn for fn in os.listdir(case_dir) if fn.endswith('.jpg') or fn.endswith('.png')]
        filenames.sort(key=lambda x: int(x[:-4]))
        probes = deserialize_probes('results/{}/probes.pk'.format(case_id))
        # give a new mesh to re-render in the fusion result
        mesh_path = os.path.join(case_dir, 'kidney_tumor_artery_vein.obj')
        height, width = cv2.imread(os.path.join(case_dir, filenames[0])).shape[:-1]
        profen_path = 'weights/fold{}/profen_best.pth'.format(fold)
        result_dir = 'results'
        if 'ablation_number_of_probes' in kwargs.keys():
            factor = kwargs['ablation_number_of_probes']
            probes = ablation_num_of_probes(probes, factor=factor)
            profen_path = 'weights/fold{}/profen_abl{}_best.pth'.format(fold, factor)
            result_dir = 'results_abl{}'.format(factor)
        if 'ablation_loss_function' in kwargs.keys():
            name = kwargs['ablation_loss_function']
            profen_path = 'weights/fold{}/profen_{}_best.pth'.format(fold, name)
            result_dir = 'result_{}'.format(name)
        registrator = Registrator(
            mesh_path=mesh_path, probes=probes,
            profen_pth=profen_path,
            affine2d_pth='weights/fold{}/affine2d_best.pth'.format(fold),
            image_size=(height, width), window_size=3
        )
        os.makedirs(os.path.join('{}/{}/fusion'.format(result_dir, case_id)), exist_ok=True)
        evaluations = {}
        for i, fn in enumerate(filenames):
            photo_path = os.path.join(case_dir, fn)
            label_path = os.path.join(case_dir, 'label', fn)
            if not os.path.exists(photo_path) or not os.path.exists(label_path):
                continue
            photo = cv2.imread(photo_path)
            orig_segment = cv2.imread(label_path)
            if i == 0:
                registrator.init_tracker(photo, orig_segment)
            segment = resized_center_square(orig_segment, out_size=512).transpose((2, 0, 1))
            segment = make_channels(segment, [
                lambda x: x[2] != 0,
                lambda x: x[1] != 0
            ])
            segment = torch.from_numpy(segment).float()
            frame_info = registrator.add_frame(photo, segment, prior_type=case_type)
            cv2.imwrite('{}/{}/fusion/{}'.format(result_dir, case_id, fn), frame_info['fusion'])
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
            evaluations[fn] = metrics
            print('Case: {} Frame: {} is OK.'.format(case_id, fn))
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
        registrator.renderer.destroy()

    print('All OK')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    for n_fold in range(4):
        alpha_test(n_fold)
        alpha_test(n_fold, ablation_loss_function='infonce')
        # alpha_test(n_fold, ablation_number_of_probes=2)
