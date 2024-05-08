import os
import json
from typing import List

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
import numpy as np
import cv2

from utils import LABEL_GT_CHARACTERIZER, RENDER_FLAT_CHARACTERIZER, cosine_similarity, characterize, normalize_vec
from probe import Probe, DEFAULT_UP, ProbeGroup
import paths
from render import PRRenderer

                
def set_fold(fold, num_all_folds):
    if fold == -1:      # train with all cases
        return paths.ALL_CASES, []
    num_all_cases = len(paths.ALL_CASES)
    if num_all_cases % num_all_folds != 0:
        raise Warning(f'The number of cases ({num_all_cases}) could not be divided into {num_all_folds} folds.')
    fold_size = num_all_cases // num_all_folds
    test_indices = [fold * fold_size + i for i in range(fold_size)]
    test_cases = [paths.ALL_CASES[i] for i in test_indices]
    train_indices = [i for i in range(len(paths.ALL_CASES)) if i not in test_indices]
    train_cases = [paths.ALL_CASES[i] for i in train_indices]
    return train_cases, test_cases

RESTRICTIONS = None

def read_restrictions():
    restrictions_info_path = os.path.join(paths.DATASET_DIR, paths.RESTRICTIONS_INFO_FILENAME)
    print(f'Reading restrictions in {restrictions_info_path}.')
    with open(restrictions_info_path) as f:
        RESTRICTIONS = json.load(restrictions_info_path)
        for k, v in RESTRICTIONS.items():
            # azimuth should be in [0, 360) degrees, elevation should be in [0, 180] degrees.
            # each lambda expression inputs a np.ndarray and outputs a bool np.ndarray
            if 0 <= v['azimuth'][0] < v['azimuth'][1] <= 360: 
                RESTRICTIONS[k]['azimuth'] = lambda x: (v['azimuth'][0] <= x) & (x < v['azimuth'][1])
            else:
                RESTRICTIONS[k]['azimuth'] = lambda x: (360 + v['azimuth'][0] <= x) | (x < v['azimuth'][1])
            RESTRICTIONS[k]['zenith'] = lambda x: (v['zenith'][0] <= x) & (x < v['zenith'][1])
    return


class ProbeDataloader(SlimDataLoaderBase):
    """
    Generate a batch of images, they are in the same case, and on the same side to avoid false negative pair.
    False negative pair: a pair of probes on 2 distant positions,
        but they render the similar images due to the symmetry of the object.
        It is very hard to tell that they are rendered from a negative pair, if we have only the 2 images,
        so we need to avoid generating these pairs.
    """
    def __init__(self, probe_groups: List[ProbeGroup], batch_size=8, batch_same_side=True, num_threads=None):
        """
        Args:
            probes (list of (list of Probe)): list of case, each case is a number of probes.
            batch_size (int):
            num_threads:
        """
        self.num_cases = len(probe_groups)
        self.num_total = sum([pg.amount for pg in probe_groups])
        self.batch_same_side = batch_same_side
        renders = [
            [characterize(p.render.transpose((2, 0, 1)), RENDER_FLAT_CHARACTERIZER) for p in pg.probes]
            for pg in probe_groups
        ]
        positions = [
            [p.get_eye() for p in pg.probes]
            for pg in probe_groups
        ]
        data = {
            'renders': renders,
            'positions': positions
        }
        super(ProbeDataloader, self).__init__(data, batch_size, num_threads)

    def generate_train_batch(self):
        """
        Returns:
            dict of (str, np.ndarray):
                'data' (np.ndarray): shape of (B, 2, H, W), value of 0/1
                'position' (np.ndarray): shape of (B, 3)
        """
        case_id = np.random.randint(self.num_cases)
        if self.batch_same_side:
            first_id = np.random.randint(len(self._data['positions'][case_id]))
            first_pos = self._data['positions'][case_id][first_id]
            images = [self._data['renders'][case_id][first_id]]
            positions = [first_pos]
            for _ in range(1, self.batch_size):
                same_side = False
                while not same_side:
                    probe_id = np.random.randint(len(self._data['positions'][case_id]))
                    this_pos = self._data['positions'][case_id][probe_id]
                    if cosine_similarity(first_pos, this_pos) >= 0:
                        same_side = True
                images.append(self._data['renders'][case_id][probe_id])
        else:
            indices = [np.random.randint(len(self._data['renders'][case_id])) for _ in range(self.batch_size)]
            images = [self._data['renders'][case_id][i] for i in indices]
            positions = [self._data['positions'][case_id][i] for i in indices]
        data = {
            'data': np.stack(images, axis=0),
            'position': np.stack(positions, axis=0)
        }
        return data
    
    
class PracticalDataloader:
    """ Single Case """
    def __init__(self, case_dir):
        image_fns = [fn for fn in os.listdir(case_dir) if fn.endswith('.png') or fn.endswith('.jpg')]
        image_fns.sort(key=lambda x: int(x[:-4]))
        label_dir = os.path.join(case_dir, 'label')
        label_fns = [fn for fn in os.listdir(label_dir) if fn.endswith('.png') or fn.endswith('.jpg')]
        self.images = [cv2.imread(os.path.join(case_dir, fn)) for fn in image_fns]
        # in some cases, not all images have corresponding labels
        # but still keep length of 2 lists the same
        self.labels = [cv2.imread(os.path.join(label_dir, fn)) if fn in label_fns else None for fn in image_fns]
        prior_info_path = os.path.join(case_dir, paths.PRIOR_INFO_FILENAME)
        if os.path.exists(prior_info_path):
            with open(prior_info_path) as f:
                prior_info = json.load(f)
            if RESTRICTIONS is None:
                read_restrictions()
            self.restriction = RESTRICTIONS[self.prior_info['type']]
        else:
            self.restriction = None
        self.fns = image_fns
        
    def image_size(self):
        return self.images[0].shape[:-1]
    
    def length(self):
        return len(self.images)
    

class SimulateDataloader:
    """ Single case """
    def __init__(self, case_dir) -> None:
        self.renderer = PRRenderer(os.path.join(case_dir, paths.MESH_FILENAME))
        self.radius_range = (1.2, 3.6)
        self.focus_deviation = 0.0
        prior_info_path = os.path.join(case_dir, paths.PRIOR_INFO_FILENAME)
        if os.path.exists(prior_info_path):
            with open(prior_info_path) as f:
                prior_info = json.load(f)
            if RESTRICTIONS is None:
                read_restrictions()
            self.restriction = RESTRICTIONS[self.prior_info['type']]
        else:
            self.restriction = None
        
    def get_image(self):
        radius = np.random.uniform(*self.radius_range)
        azimuth = np.random.rand() * 2.0 * np.pi
        zenith = np.random.rand() * np.pi
        inside_pps = False
        while not inside_pps:
            if self.restriction['azimuth'](azimuth / np.pi * 180) and self.restriction['zenith'](zenith / np.pi * 180):
                inside_pps = True
            else:
                azimuth = np.random.rand() * 2.0 * np.pi
                zenith = np.random.rand() * np.pi
        eye = np.asarray([
            radius * np.sin(zenith) * np.cos(azimuth),
            radius * np.sin(zenith) * np.sin(azimuth),
            radius * np.cos(zenith)
        ])
        focus = (np.random.rand(3) - 0.5) * 2.0 * self.focus_deviation
        direction = normalize_vec(focus - eye)
        roll = np.random.rand() * 2.0 * np.pi
        right = normalize_vec(np.cross(direction, DEFAULT_UP))
        up = normalize_vec(np.cross(right, direction))
        up = np.cos(roll) * up + np.sin(roll) * right
        probe = Probe(None, eye=eye, focus=focus, up=up)
        label = self.renderer.render(probe.get_matrix(), mode='FLAT', draw_mesh=[0, 1])[..., ::-1]
        return label, normalize_vec(-eye), azimuth, zenith
    
    def __del__(self):
        del self.renderer


class TrackLabelDataloader(SlimDataLoaderBase):
    def __init__(self, cases, batch_size, number_of_threads_in_multithreaded=None):
        data = []
        for c in cases:
            fns = [fn for fn in os.listdir(os.path.join(paths.DATASET_DIR, c, 'label')) 
                   if fn.endswith('.png') or fn.endswith('.jpg')]
            fns.sort(key=lambda x: int(x[:-4]))
            images = [cv2.imread(os.path.join(paths.DATASET_DIR, c, fn)) for fn in fns]
            labels = [cv2.imread(os.path.join(paths.DATASET_DIR, c, 'label', fn)) for fn in fns]
            factor = 400.0 / images[0].shape[0]
            images = [cv2.resize(i, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC) for i in images]
            labels = [cv2.resize(l, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST) for l in labels]
            images = [i.transpose((2, 0, 1)) / 255 for i in images]
            labels = [characterize(l.transpose((2, 0, 1)), LABEL_GT_CHARACTERIZER) for l in labels]
            data.append({
                'image': images,
                'label': labels,
                'count': len(fns)
            })
        self.num_cases = len(cases)
        self.num_total = sum([d['count'] for d in data])
        super().__init__(data, batch_size, number_of_threads_in_multithreaded)
        
    def generate_train_batch(self):
        c = np.random.randint(self.num_cases)
        idx = np.random.randint(low=0, high=self._data[c]['count'] - 1, size=self.batch_size)
        ret_dict = {
            # (B, C=6, H, W)
            'data': np.stack([
                np.concatenate(self._data[c]['image'][i: i+2], axis=0)
                for i in idx
            ], axis=0),
            # (B, C=4, H, W)
            'seg': np.stack([
                np.concatenate(self._data[c]['label'][i: i+2], axis=0)
                for i in idx
            ], axis=0)
        }
        return ret_dict


if __name__ == '__main__':
    from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
    from batchgenerators.transforms.spatial_transforms import SpatialTransform
    train, test = set_fold(0, 4)
    tl = TrackLabelDataloader(train, batch_size=8, number_of_threads_in_multithreaded=8)
    trans = SpatialTransform(
        patch_size=(512, 512),
        do_elastic_deform=False,
        angle_x=(-np.pi / 3, np.pi / 3)
    )
    ag = MultiThreadedAugmenter(tl, trans, num_processes=4)
    d = next(tl)
    dd = next(ag)
    print('ok')
