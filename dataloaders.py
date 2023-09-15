from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
import numpy as np

from utils import cosine_similarity, make_channels
from probe import Probe


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
                    same_side = all([cosine_similarity(this.camera_position, e.camera_position) > -0.2
                                     for e in probes])
            probes.append(this)
        data = {
            'data': np.asarray([make_channels(p.render.transpose((2, 0, 1)), [
                    lambda x: x[0] != 0,
                    lambda x: (x[0] == 0) & (x.any(0))]) for p in probes]),
            'position': np.asarray([p.camera_position for p in probes])
        }
        return data


if __name__ == '__main__':
    print('ok')
