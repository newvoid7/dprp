import os
import random
import time

import torch
from torch.nn import CosineSimilarity
import numpy as np

from network.profen import ProFEN
from dataloaders import set_fold, TestSingleCaseDataloader
import paths
from probe import ProbeGroup
from agent import AgentTask
from utils import make_channels, cosine_similarity, time_it, make_colorful
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
        probes = probe_group.probes
        probe_azimuth = np.asarray([p.get_spcoord_dict()['azimuth'] for p in probes])
        probe_elevation = np.asarray([p.get_spcoord_dict()['elevation'] for p in probes])
        
        # feature extractor
        profen_weight_dir = 'profen' if ablation is None or ablation == 'wo_pps' else 'profen_' + ablation
        profen_path = '{}/fold{}/{}/best.pth'.format(paths.WEIGHTS_DIR, fold, profen_weight_dir)
        profen = ProFEN().cuda()
        profen.load_state_dict(torch.load(profen_path))
        profen.eval()
        
        # PPS
        case_dataloader = TestSingleCaseDataloader(os.path.join(paths.DATASET_DIR, case_id))
        if case_dataloader.prior_info is None:
            case_type = 'type1'
        else:
            case_type = case_dataloader.prior_info['type']
        restriction = restrictions[case_type]
        input_filtered = restriction['azimuth'](probe_azimuth) & restriction['elevation'](probe_elevation)
        input_filtered = torch.from_numpy(input_filtered).cuda()
        pps_filtered = input_filtered
        
        # feature pool
        input_pool = np.asarray([make_channels(p.render.transpose((2, 0, 1)),
                                            [lambda x: x[0] != 0, lambda x: (x[0] == 0) & (x.any(0))])
                                        for p in probes])
        input_pool = torch.from_numpy(input_pool)
        feature_pool = []
        bs = 128
        if ablation.startswith('div_'):
            factor = int(ablation[4:])
            probe_group.sparse(factor)
            candidates_azimuth = np.asarray([p.get_spcoord_dict()['azimuth'] for p in probe_group.probes])
            candidates_elevation = np.asarray([p.get_spcoord_dict()['elevation'] for p in probe_group.probes])
            candidates = np.asarray([make_channels(p.render.transpose((2, 0, 1)),
                                            [lambda x: x[0] != 0, lambda x: (x[0] == 0) & (x.any(0))])
                                        for p in probe_group.probes])
            candidates = torch.from_numpy(candidates)
            pps_filtered = restriction['azimuth'](candidates_azimuth) & restriction['elevation'](candidates_elevation)
            pps_filtered = torch.from_numpy(pps_filtered).cuda()
        else:
            candidates = input_pool
        with torch.no_grad():
            for i in range(len(candidates) // bs + 1):
                batch = np.asarray(candidates[i * bs : min(i * bs + bs, len(candidates))])
                batch = torch.from_numpy(batch).cuda()
                pred = profen(batch)
                feature_pool.append(pred)
        feature_pool = torch.cat(feature_pool, dim=0)
        
        if ablations != 'wo_pps':
            pps = PPS(probe_group, feature_pool, pps_filtered)
        else:
            pps = PPS(probe_group, feature_pool)
        
        # test
        ts = 100                # test how many times
        bs = 16                 # batch size
        start_time = time.time()
        for i in range(ts):
            input = []
            target = []
            pred = []
            for _ in range(bs):
                index = random.randint(0, len(input_pool) - 1)
                while not input_filtered[index]:          # only select probes in resitriction
                    index = random.randint(0, len(input_pool) - 1)
                input.append(input_pool[index])
                target.append(probes[index].get_orientation())
            input = torch.stack(input, dim=0).cuda()
            target = np.stack(target, axis=0)
            input = agent_task.apply(input)
            feature = profen(input)
            for f in feature:
                hit_index = pps.best(f)
                pred.append(probes[hit_index].get_orientation())
            pred = np.stack(pred, axis=0)
            loss = cosine_similarity(target, pred, dim=1).mean()
            case_loss.append(loss)
        test_loss.append(case_loss)
        end_time = time.time()
        time_cost.append(end_time - start_time)
    return test_loss, time_cost


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # ablations = ['div_4', 'div_9', 'div_16', 'wo_pps', 'wo_agent']
    ablations = []
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
