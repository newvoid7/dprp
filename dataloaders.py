import os

import torch
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
import numpy as np
import cv2

from utils import resized_center_square, cosine_similarity, make_channels, cv2_to_tensor
from probe import Probe


class VOSDataloader(SlimDataLoaderBase):
    """
    Video Object Segment dataloader.
    """
    def __init__(self, image_dirs, label_dirs, seq_len=6, batch_size=4, num_threads=None):
        images = []
        labels = []
        self.num_case = len(image_dirs)
        self.seq_len = seq_len
        self.num_total = 0
        for case_id in range(self.num_case):
            case_image_dir = image_dirs[case_id]
            case_label_dir = label_dirs[case_id]
            fns = [fn for fn in os.listdir(case_label_dir) if fn.endswith('.jpg') or fn.endswith('.png')]
            fns.sort(key=lambda x: int(x[:-4]))
            self.num_total += len(fns)
            if len(fns) < self.seq_len:
                raise RuntimeError('VOSDataloader: sequence not long enough')
            case_images = [cv2.imread(os.path.join(case_image_dir, fn)) for fn in fns]
            case_images = [resized_center_square(i, out_size=512) for i in case_images]
            case_images = [cv2_to_tensor(i) for i in case_images]
            case_labels = [cv2.imread(os.path.join(case_label_dir, fn)) for fn in fns]
            case_labels = [resized_center_square(l, out_size=512) for l in case_labels]
            case_labels = [torch.from_numpy(make_channels(l.transpose((2, 0, 1)),
                                            [lambda x: ~x.any(0), lambda x: x[2] != 0, lambda x: x[1] != 0])).float()
                           for l in case_labels]
            images.append(case_images)
            labels.append(case_labels)
        data = {
            'images': images,
            'labels': labels
        }
        super(VOSDataloader, self).__init__(data, batch_size, num_threads)
        return

    def generate_train_batch(self):
        """
        Each batch should include [self.seq_len] rgb images and labels.
        Returns:
            dict:
                'data' (torch.Tensor): B * [self.seq_len] * 3(BGR) * H * W.
                'seg' (torch.Tensor): B * [self.seq_len] * 3(background, kidney, tumor) * H * W
        """
        batch_images = []
        batch_labels = []
        for i in range(self.batch_size):
            case_id = np.random.randint(self.num_case)
            start = np.random.randint(len(self._data['images'][case_id]) - self.seq_len)
            # case_id = 0         # TODO Debug
            # start = 0           # TODO Debug
            case_images = torch.stack(self._data['images'][case_id][start: start + self.seq_len], dim=0)
            case_labels = torch.stack(self._data['labels'][case_id][start: start + self.seq_len], dim=0)
            batch_images.append(case_images)
            batch_labels.append(case_labels)
        batch_images = torch.stack(batch_images, dim=0)
        batch_labels = torch.stack(batch_labels, dim=0)
        return {'data': batch_images, 'seg': batch_labels}

    def get_test_data(self):
        """
        Returns:
            list of (list of torch.Tensor), list of (list of torch.Tensor)
        """
        return self._data['images'], self._data['labels']


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
                # case_id = 0             # TODO debug
                # image_id = 0            # TODO debug
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
