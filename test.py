import os

import torch
from torch.nn import CosineSimilarity
import numpy as np

from network.profen import ProFEN
from dataloaders import set_fold, TestSingleCaseDataloader
import paths
from probe import ProbeGroup
from agent import AgentTask
from utils import make_channels
from fusion import restrictions


def test_profen(fold=0, n_fold=6, ablation=None):
    """ Simulate the intraoperative images with Probe generated images.
    Args:
        fold (int, optional): _description_. Defaults to 0.
        n_fold (int, optional): _description_. Defaults to 6.
        ablation (_type_, optional): _description_. Defaults to None.
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
        for i, p in enumerate(probes):
            if not pps_filtered[i]:
                continue
            input = torch.from_numpy(rendered_2ch_pool[i]).cuda().unsqueeze(0)
            input = agent_task.apply(input)
            target = p.get_orientation()
            feature = profen(input)
            similarity = sim_func(feature, feature_pool)
            if ablation is None or ablation != 'wo_pps':
                similarity = pps_filtered * (similarity - similarity.min())
            hit_index = similarity.argmax()
            pred = probes[hit_index].get_orientation()
            loss = (pred * target).sum() 
            loss /= (pred ** 2).sum() ** 0.5 
            loss /= (target ** 2).sum() ** 0.5
            case_loss.append(1.0 - loss)
        test_loss.append(case_loss)
    return test_loss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    loss = []
    loss_wo_ref_loss = []
    loss_div_4 = []
    loss_div_9 = []
    loss_div_16 = []
    loss_wo_pps = []
    for fold in range(6):
        loss.append(test_profen(fold))
        loss_wo_ref_loss.append(test_profen(fold, ablation='wo_ref_loss'))
        loss_div_4.append(test_profen(fold, ablation='div_4'))
        loss_div_9.append(test_profen(fold, ablation='div_9'))
        loss_div_16.append(test_profen(fold, ablation='div_16'))
        loss_wo_pps.append(test_profen(fold, ablation='wo_pps'))
    loss = [np.asarray(l) for l in loss]
    loss_wo_ref_loss = [np.asarray(l) for l in loss_wo_ref_loss]
    loss_div_4 = [np.asarray(l) for l in loss_div_4]
    loss_div_9 = [np.asarray(l) for l in loss_div_9]
    loss_div_16 = [np.asarray(l) for l in loss_div_16]
    loss_wo_pps = [np.asarray(l) for l in loss_wo_pps]
    print('OK')
