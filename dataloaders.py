from calendar import setfirstweekday
import os
import json

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
        raise Warning('The number of cases ({}) could not be divided into {} folds.'
                      .format(num_all_cases, num_all_folds))
    fold_size = num_all_cases // num_all_folds
    test_indices = [fold * fold_size + i for i in range(fold_size)]
    test_cases = [paths.ALL_CASES[i] for i in test_indices]
    train_indices = [i for i in range(len(paths.ALL_CASES)) if i not in test_indices]
    train_cases = [paths.ALL_CASES[i] for i in train_indices]
    return train_cases, test_cases


class ProbeSingleCaseDataloader(SlimDataLoaderBase):
    """
    Generate a batch of images, they are in the same case, and on the same side to avoid false negative pair.
    False negative pair: a pair of probes on 2 distant positions,
        but they render the similar images due to the symmetry of the object.
        It is very hard to tell that they are rendered from a negative pair, if we have only the 2 images,
        so we need to avoid generating these pairs.
    """
    def __init__(self, probes, batch_size=8, batch_same_side=True, num_threads=None):
        """
        Args:
            probes (list of (list of Probe)): list of case, each case is a number of probes.
            batch_size (int):
            num_threads:
        """
        self.num_cases = len(probes)
        self.num_total = sum([len(p) for p in probes])
        self.batch_same_side = batch_same_side
        super(ProbeSingleCaseDataloader, self).__init__(probes, batch_size, num_threads)

    def generate_train_batch(self):
        """
        Returns:
            dict of (str, np.ndarray):
                'data' (np.ndarray): shape of (B, 2, H, W), value of 0/1
                'position' (np.ndarray): shape of (B, 3)
        """
        case_id = np.random.randint(self.num_cases)
        if self.batch_same_side:
            first = self._data[case_id][np.random.randint(len(self._data[case_id]))]
            probes = [first]
            for _ in range(1, self.batch_size):
                same_side = False
                while not same_side:
                    image_id = np.random.randint(len(self._data[case_id]))
                    this = self._data[case_id][image_id]
                    if cosine_similarity(first.get_eye(), this.get_eye()) >= 0:
                        same_side = True
                probes.append(this)
        else:
            probes = [self._data[case_id][np.random.randint(len(self._data[case_id]))] for _ in range(self.batch_size)]
        data = {
            'data': np.asarray([characterize(p.render.transpose((2, 0, 1)), RENDER_FLAT_CHARACTERIZER) for p in probes]),
            'position': np.asarray([p.get_eye() for p in probes])
        }
        return data
    
    
class TestSingleCaseDataloader:
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
                self.prior_info = json.load(f)
        else:
            self.prior_info = None
        self.fns = image_fns
        
    def image_size(self):
        return self.images[0].shape[:-1]
    
    def length(self):
        return len(self.images)
    

class SimulateDataloader:
    """ Single case
    """
    def __init__(self, case_dir) -> None:
        prior_info_path = os.path.join(case_dir, paths.PRIOR_INFO_FILENAME)
        if os.path.exists(prior_info_path):
            with open(prior_info_path) as f:
                self.prior_info = json.load(f)
        else:
            self.prior_info = None
        self.renderer = PRRenderer(os.path.join(case_dir, paths.MESH_FILENAME))
        self.radius_range = (1.2, 3.6)
        self.focus_deviation = 0.0
        
    def get_image(self):
        radius = np.random.uniform(*self.radius_range)
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
