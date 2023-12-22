import os
import json

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
import numpy as np
import cv2

from utils import cosine_similarity, make_channels
from probe import Probe
import paths

                
def set_fold(fold, num_all_folds):
    if fold == -1:
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
    def __init__(self, probes, batch_size=8, num_threads=None):
        """
        Args:
            probes (list of (list of Probe)): list of case, each case is a number of probes.
            batch_size (int):
            num_threads:
        """
        self.num_cases = len(probes)
        self.num_total = sum([len(p) for p in probes])
        super(ProbeSingleCaseDataloader, self).__init__(probes, batch_size, num_threads)

    def generate_train_batch(self):
        """
        Returns:
            dict of (str, np.ndarray):
                'data' (np.ndarray): shape of (B, 2, H, W), value of 0/1
                'position' (np.ndarray): shape of (B, 3)
        """
        case_id = np.random.randint(self.num_cases)
        probes = []
        for i in range(self.batch_size):
            same_side = False
            this = None
            while not same_side:
                image_id = np.random.randint(len(self._data[case_id]))
                # case_id = 0             # debug
                # image_id = 0            # debug
                this = self._data[case_id][image_id]
                if len(probes) == 0:
                    same_side = True
                else:
                    same_side = all([cosine_similarity(this.get_eye(), e.get_eye()) >= 0
                                     for e in probes])
            probes.append(this)
        data = {
            'data': np.asarray([make_channels(p.render.transpose((2, 0, 1)), [
                    lambda x: x[0] != 0,
                    lambda x: (x[0] == 0) & (x.any(0))]) for p in probes]),
            'position': np.asarray([p.camera_position for p in probes])
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


if __name__ == '__main__':
    print('ok')
