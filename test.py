import os

import torch
import numpy as np
import cv2
import time

from network.profen import ProFEN
from network.tracknet import TrackNet
from dataloaders import set_fold, SimulateDataloader
import paths
from probe import ProbeGroup
from agent import AgentTask
from utils import (RENDER_CHARACTERIZER, LABEL_GT_CHARACTERIZER, WHITE, YELLOW, 
                   make_channels, 
                   make_colorful,
                   cosine_similarity, 
                   time_it, 
                   cv2_to_tensor, 
                   evaluate_segmentation, 
                   crop_patches, 
                   merge_patches, 
                   resize_to_fit)
from fusion import restrictions
from pps import PPS


@time_it
def test_profen(fold=0, n_fold=4, ablation=None):
    """ Simulate the intraoperative images with Probe generated images.
    Args:
        fold (int, optional): _description_. Defaults to 0.
        n_fold (int, optional): _description_. Defaults to 6.
        ablation (_type_, optional): _description_. Defaults to None.
    Returns:
        []
    """
    _, test_cases = set_fold(fold, n_fold)
    agent_task = AgentTask(occlusion_dir=os.path.join(paths.DATASET_DIR, '.mask'))
    test_loss = []
    time_cost = []
    for case_id in test_cases:
        case_loss = []
        # probes
        probe_path = os.path.join(paths.RESULTS_DIR, case_id, paths.PROBE_FILENAME)
        probe_group = ProbeGroup(deserialize_path=probe_path)
        if ablation is not None and ablation.startswith('div_'):
            factor = int(ablation[4:])
            probe_group.sparse(factor)
        probes = probe_group.probes
        probe_azimuth = np.asarray([p.get_spcoord_dict()['azimuth'] for p in probes])
        probe_elevation = np.asarray([p.get_spcoord_dict()['zenith'] for p in probes])
        
        # feature extractor
        profen_weight_dir = 'profen' if ablation is None or ablation == 'wo_pps' else 'profen_' + ablation
        profen_path = '{}/fold{}/{}/best.pth'.format(paths.WEIGHTS_DIR, fold, profen_weight_dir)
        profen = ProFEN().cuda()
        profen.load_state_dict(torch.load(profen_path))
        profen.eval()
        
        # PPS
        case_dataloader = SimulateDataloader(os.path.join(paths.DATASET_DIR, case_id))
        if case_dataloader.prior_info is None:
            case_type = 'type1'
        else:
            case_type = case_dataloader.prior_info['type']
        restriction = restrictions[case_type]
        pps_filtered = restriction['azimuth'](probe_azimuth) & restriction['zenith'](probe_elevation)
        pps_filtered = torch.from_numpy(pps_filtered).cuda()
        
        # feature pool
        image_pool = np.asarray([make_channels(p.render.transpose((2, 0, 1)), RENDER_CHARACTERIZER)
                                        for p in probes])
        image_pool = torch.from_numpy(image_pool)
        feature_pool = []
        bs = 128
        with torch.no_grad():
            for i in range(len(probes) // bs + 1):
                batch = image_pool[i * bs : min(i * bs + bs, len(probes))].cuda()
                pred = profen(batch)
                feature_pool.append(pred)
        feature_pool = torch.cat(feature_pool, dim=0)
        
        if ablation != 'wo_pps':
            pps = PPS(probe_group.neighbor, feature_pool, pps_filtered)
        else:
            pps = PPS(probe_group.neighbor, feature_pool)
        
        # test
        ts = 100                # test how many times
        bs = 16                 # batch size
        start_time = time.time()
        for t in range(ts):
            input = []
            target = []
            for _ in range(bs):
                while True:
                    i, t, a, z = case_dataloader.get_image()
                    if restriction['azimuth'](a / np.pi * 180) and restriction['zenith'](z / np.pi * 180):
                        break
                i = make_channels(i.transpose((2, 0, 1)), RENDER_CHARACTERIZER)
                input.append(torch.from_numpy(i))
                target.append(torch.from_numpy(t.copy()))
            input = torch.stack(input, dim=0).cuda()
            target = np.stack(target, axis=0)
            input = agent_task.apply(input)
            feature = profen(input)
            pred = [probes[pps.best(f)].get_orientation() for f in feature]
            pred = np.stack(pred, axis=0)
            loss = cosine_similarity(target, pred, dim=1).mean()
            case_loss.append(loss)
        test_loss.append(case_loss)
        end_time = time.time()
        time_cost.append(end_time - start_time)
        del case_dataloader
    return test_loss, time_cost


@time_it
def test_tracknet(fold=0, n_fold=4):
    tracknet = TrackNet().cuda()
    tracknet.load_state_dict(torch.load(os.path.join(paths.WEIGHTS_DIR, 'fold{}'.format(fold), 'tracknet', 'best.pth')))
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
        labels = [resize_to_fit(l, (400, 400), interp=cv2.INTER_NEAREST) if l is not None else None for l in labels]
        labels = [make_channels(l.transpose((2, 0, 1)), LABEL_GT_CHARACTERIZER) if l is not None else None for l in labels]
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
            cv2.imwrite(os.path.join('tmp', c, fn), make_colorful(pred_label_cv2, (WHITE, YELLOW)))
            if labels[i + 1] is not None:
                metrics = evaluate_segmentation(pred_label_cv2, labels[i + 1])
                ret_dict[c][fn] = metrics
            last_label = pred_label
    return ret_dict


def test_ablations():
    ablations = ['div_4', 'div_9', 'div_16', 'wo_pps', 'wo_agent', 'wo_ref_loss']
    loss = []
    time_cost = []
    loss_abl = {k: [] for k in ablations}
    time_cost_abl = {k: [] for k in ablations}
    for fold in range(4):
        l, t = test_profen(fold)
        loss += l
        time_cost += t
        for k in ablations:
            l, t = test_profen(fold, ablation=k)
            loss_abl[k] += l
            time_cost_abl[k] += t
    loss = np.asarray([np.asarray(l).mean() for l in loss])
    time_cost = np.asarray([np.asarray(t).mean() for t in time_cost])
    for k in ablations:
        loss_abl[k] = np.asarray([np.asarray(l).mean() for l in loss_abl[k]])
        time_cost_abl[k] = np.asarray([np.asarray(t).mean() for t in time_cost_abl[k]])
    print('OK')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    for f in range(4):
        m = test_tracknet(f)
        dices = np.asarray([
            [m[c][fn]['channel 0']['dice'] 
            for fn in m[c].keys()] 
            for c in m.keys()
        ])
        print(dices.mean())
