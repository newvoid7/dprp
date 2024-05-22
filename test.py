import os
import time
import json

import torch
import numpy as np
import cv2

from network.profen import ProFEN
from network.tracknet import TrackNet
from dataloaders import set_fold, SimulateDataloader, RESTRICTIONS
import paths
from probe import ProbeGroup
from fusion import Fuser
from nr_test import NRRenderer
from utils import (RENDER_FLAT_CHARACTERIZER, 
                   LABEL_GT_CHARACTERIZER, WHITE, YELLOW, 
                   characterize, 
                   colorize,
                   cosine_similarity, 
                   timer, 
                   cv2_to_tensor, 
                   evaluate_segmentation, 
                   normalize_vec,
                   resize_to_fit)
from pps import PPS


def generate_test_data(test_times=10):
    for case in paths.ALL_CASES:
        os.makedirs(os.path.join(paths.TEST_DATA_DIR, case), exist_ok=True)
        case_dataloader = SimulateDataloader(os.path.join(paths.DATASET_DIR, case))
        viewpoint_gt = {}
        for i in range(test_times):
            image, viewpoint, azimuth, zenith = case_dataloader.get_image()
            cv2.imwrite(f'{paths.TEST_DATA_DIR}/{case}/{i}.png', image)
            viewpoint_gt[i] = viewpoint.tolist()
        with open(f'{paths.TEST_DATA_DIR}/{case}/viewpoint_ground_truth.json', 'w') as f:
            json.dump(viewpoint_gt, f)
        del case_dataloader
            

def test_profen(fold, n_fold, device=0, ablation=None):
    """ Simulate the intraoperative images with Probe generated images.
    Args:
        fold (int, optional): _description_. Defaults to 0.
        n_fold (int, optional): _description_. Defaults to 6.
        ablation (_type_, optional): _description_. Defaults to None.
    Returns:
        []
    """
    # feature extractor
    profen_weight_dir = 'profen' if ablation is None or ablation == 'wo_pps' else 'profen_' + ablation
    profen_path = f'{paths.WEIGHTS_DIR}/fold{fold}/{profen_weight_dir}/best.pth'
    profen = ProFEN().cuda(device)
    profen.load_state_dict(torch.load(profen_path, map_location=f'cuda:{device}'))
    profen.eval()
    
    _, test_cases = set_fold(fold, n_fold)
    for case in test_cases:
        # like in fusion.py
        # restriction
        with open(os.path.join(paths.DATASET_DIR, case, paths.PRIOR_INFO_FILENAME)) as f:
            case_type = json.load(f)['type']
        restriction = RESTRICTIONS[case_type]
        # probes
        probe_path = os.path.join(paths.RESULTS_DIR, case, paths.PROBE_FILENAME)
        probe_group = ProbeGroup(deserialize_path=probe_path)
        if ablation is not None and ablation.startswith('div_'):
            probe_group.sparse(factor=int(ablation[4:]))
        fuser = Fuser(
            restriction=restriction,
            with_pps=ablation is None or ablation != 'wo_pps',
            probe_group=probe_group,
            feature_extractor=profen,
            device=device
            # following affine2d_solver, tracker, etc are not needed
        )
        # test_data
        fns = [fn for fn in os.listdir(f'{paths.TEST_DATA_DIR}/{case}') if fn.endswith('.png')]
        with open(f'{paths.TEST_DATA_DIR}/{case}/viewpoint_ground_truth.json') as f:
            gt = json.load(f)
        for i in range(len(fns)):
            image = cv2.imread(f'{paths.TEST_DATA_DIR}/{case}/{i}.png')
            with torch.no_grad():
                feature = fuser.feature_extractor(torch.from_numpy(image).cuda(fuser.device).unsqueeze(0))
            hit_index = fuser.pps.best(feature)
            pred = probe_group.probes[hit_index]
            viewpoint = normalize_vec(-pred.get_eye())
            image = pred.render
            cv2.imwrite(f'{paths.TEST_DATA_DIR}/{case}/profen_{ablation}_{i}.png', image)


def test_nr(device=0):
    test_cases = [c for c in os.listdir(paths.TEST_DATA_DIR) 
                  if os.path.isdir(os.path.join(paths.TEST_DATA_DIR, c))]
    for case in test_cases:
        fns = [fn for fn in os.listdir(f'{paths.TEST_DATA_DIR}/{case}') if fn.endswith('.png')]
        with open(f'{paths.TEST_DATA_DIR}/{case}/viewpoint_ground_truth.json') as f:
            gt = json.load(f)
        for i in range(len(fns)):
            nr = NRRenderer(
                filename_obj=os.path.join(paths.DATASET_DIR, case, paths.MESH_FILENAME), 
                meshes=[0, 1], 
                init_eye=0.02-np.asarray(gt[str(i)]),
                init_up=[0, 0, 1],
                path_ref_image=f'{paths.TEST_DATA_DIR}/{case}/{fns[i]}',
            )
            nr.to(device)
            optimizer = torch.optim.Adam(nr.parameters(), lr=0.1)
            for iter in range(20):
                optimizer.zero_grad()
                loss = nr()
                loss.backward()
                optimizer.step()
            viewpoint = normalize_vec(-nr.camera_position.data.detach().cpu().numpy())
            image = nr.renderer(nr.vertices, nr.faces, nr.textures, mode='label').flip(dims=[1])
            image = image.detach().cpu().numpy().squeeze().transpose((1, 2, 0)) * 255
            image = image.astype(np.uint8)
            cv2.imwrite(f'{paths.TEST_DATA_DIR}/{case}/nr_{i}.png', image)


def test_tracknet(fold=0, n_fold=4, device=0):
    tracknet = TrackNet().to(device)
    tracknet.load_state_dict(torch.load(os.path.join(paths.WEIGHTS_DIR, f'fold{fold}', 'tracknet', 'best.pth'),
                                        map_location=f'cuda:{device}'))
    tracknet.eval()
    test_cases = set_fold(fold, n_fold)[1]
    ret_dict = {}
    for c in test_cases:
        ret_dict[c] = {}
        os.makedirs(os.path.join('tmp', c), exist_ok=True)
        case_dir = os.path.join(paths.DATASET_DIR, c)
        fns = [fn for fn in os.listdir(case_dir) if fn.endswith('.png') or fn.endswith('.jpg')]
        fns.sort(key=lambda x: int(x[:-4]))
        images = [cv2.imread(os.path.join(case_dir, fn)) for fn in fns]
        size = images[0].shape[:2]
        # images = [crop_patches(cv2_to_tensor(i), (320, 320)) for i in images]
        images = [cv2_to_tensor(resize_to_fit(i, (400, 400))).unsqueeze(0) for i in images]
        labels = [cv2.imread(os.path.join(case_dir, 'label', fn)) for fn in fns]
        labels = [resize_to_fit(l, (400, 400), interp='nearest') if l is not None else None for l in labels]
        labels = [characterize(l.transpose((2, 0, 1)), LABEL_GT_CHARACTERIZER) if l is not None else None for l in labels]
        # last_label = crop_patches(torch.from_numpy(labels[0]), (320, 320))
        first_label_idx = 0
        for i in range(len(labels)):
            if labels[i] is None:
                first_label_idx += 1
            else:
                break
        last_label = torch.from_numpy(labels[first_label_idx]).unsqueeze(0)
        for i, fn in enumerate(fns[first_label_idx + 1:]):
            with torch.no_grad():
                pred_label, _ = tracknet(images[i + first_label_idx].cuda(), images[i + 1 + first_label_idx].cuda(), last_label.cuda())
            # pred_label_cv2 =  merge_patches((320, 320), size, pred_label).detach().cpu().numpy().squeeze()
            pred_label_cv2 = pred_label.detach().cpu().numpy().squeeze()
            cv2.imwrite(os.path.join('tmp', c, fn), colorize(pred_label_cv2, (WHITE, YELLOW)))
            if labels[i + 1] is not None:
                metrics = evaluate_segmentation(pred_label_cv2, labels[i + 1])
                ret_dict[c][fn] = metrics
            last_label = pred_label
    return ret_dict


if __name__ == '__main__':
    # generate_test_data()
    # for f in range(4):
        # test_profen(f, 4, 0)
    test_nr()
