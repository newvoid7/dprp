import os
import random

import torch
from torch.nn import CosineSimilarity
import numpy as np

from network.profen import ProFEN
from dataloaders import set_fold, TestSingleCaseDataloader
import paths
from probe import ProbeGroup
from agent import AgentTask
from utils import make_channels, cosine_similarity
from fusion import restrictions


def test_profen(fold=0, n_fold=6, ablation=None):
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
        
        # sim func
        sim_func = CosineSimilarity()
        
        # PPS
        case_dataloader = TestSingleCaseDataloader(os.path.join(paths.DATASET_DIR, case_id))
        if case_dataloader.prior_info is None:
            case_type = 'type1'
        else:
            case_type = case_dataloader.prior_info['type']
        restriction = restrictions[case_type]
        pps_filtered = restriction['azimuth'](probe_azimuth) & restriction['elevation'](probe_elevation)
        pps_filtered = torch.from_numpy(pps_filtered).cuda()
        
        # feature pool
        rendered_2ch_pool = np.asarray([make_channels(p.render.transpose((2, 0, 1)),
                                            [lambda x: x[0] != 0, lambda x: (x[0] == 0) & (x.any(0))])
                                        for p in probes])
        rendered_2ch_pool = torch.from_numpy(rendered_2ch_pool)
        feature_pool = []
        bs = 128
        with torch.no_grad():
            for i in range(len(rendered_2ch_pool) // bs + 1):
                batch = np.asarray(rendered_2ch_pool[i * bs : min(i * bs + bs, len(rendered_2ch_pool))])
                batch = torch.from_numpy(batch).cuda()
                pred = profen(batch)
                feature_pool.append(pred)
        feature_pool = torch.cat(feature_pool, dim=0)
        
        # test
        ts = 100                # test how many times
        bs = 16                 # batch size
        for i in range(ts):
            input = []
            target = []
            pred = []
            for _ in range(bs):
                index = random.randint(0, len(rendered_2ch_pool) - 1)
                while not pps_filtered[index]:          # only select probes in resitriction
                    index = random.randint(0, len(rendered_2ch_pool) - 1)
                input.append(rendered_2ch_pool[index])
                target.append(probes[index].get_orientation())
            input = torch.stack(input, dim=0).cuda()
            target = np.stack(target, axis=0)
            input = agent_task.apply(input)
            feature = profen(input)
            for f in feature:
                similarity = sim_func(f, feature_pool)
                if ablation is None or ablation != 'wo_pps':
                    similarity = pps_filtered * (similarity - similarity.min())
                hit_index = similarity.argmax()
                pred.append(probes[hit_index].get_orientation())
            pred = np.stack(pred, axis=0)
            loss = cosine_similarity(target, pred, dim=1).mean()
            loss = 0.5 - loss / 2
            case_loss.append(loss)
        test_loss.append(case_loss)
    return test_loss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    ablations = ['div_4', 'div_9', 'div_16', 'wo_ref_loss', 'wo_pps', 'wo_agent']
    loss = []
    loss_abl = {
        'div_4': [],
        'div_9': [],
        'div_16': [],
        'wo_pps': [],
        'wo_agent': []
    }
    for fold in range(6):
        loss += test_profen(fold)
        for k, v in loss_abl.items():
            v += test_profen(fold, ablation=k)
    loss = np.asarray([np.asarray(l).mean() for l in loss])
    for k, v in loss_abl.items():
        loss_abl[k] = np.asarray([np.asarray(l).mean() for l in v])
    print('OK')
