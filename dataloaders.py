import os
import json

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
import numpy as np
import cv2

from utils import cosine_similarity, make_channels, normalize_vec, stitch_images, resize_to_fit
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
            'data': np.asarray([make_channels(p.render.transpose((2, 0, 1)), [
                    lambda x: x[0] != 0,
                    lambda x: (x[0] == 0) & (x.any(0))]) for p in probes]),
            'position': np.asarray([p.get_eye() for p in probes])
        }
        return data
    
    
class RealTimeProbeDataloader(SlimDataLoaderBase):
    """
    Render a batch of images real-time.
    Args:
        SlimDataLoaderBase (_type_): _description_
    """
    def __init__(self, batch_size=8, batch_same_side=True, num_threads=None):
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


def test_dataloader():
    from paths import DATASET_DIR
    for c in os.listdir(DATASET_DIR):
        cd = os.path.join(DATASET_DIR, c)
        sd = SimulateDataloader(case_dir=cd)
        results = [sd.get_image()[0] for _ in range(16)]
        results = [resize_to_fit(r, out_size=200) for r in results]


if __name__ == '__main__':
    from utils import cosine_similarity
    def consine_sim_pairs(l):
        ret = []
        for i in range(len(l) - 1):
            for j in range(i + 1, len(l)):
                ret.append(cosine_similarity(l[i], l[j]))
        return ret
    
    probes = [ProbeGroup(deserialize_path=os.path.join(paths.RESULTS_DIR, c, paths.PROBE_FILENAME)).probes for c in paths.ALL_CASES]
    dl = ProbeSingleCaseDataloader(probes, batch_same_side=True)
    observation_positions = [consine_sim_pairs(next(dl)['position']) for _ in range(100000)]
    observation_positions = np.asarray(observation_positions)
    print(observation_positions.mean())
