from torch.nn import CosineSimilarity
import numpy as np


class PPS:
    def __init__(self, probe_group, features, mask=None) -> None:
        """ Prior Probe Selecting
        Args:
            probe_group (ProbeGroup): _description_
            features (torch.Tensor): _description_
            mask (torch.Tensor, optional): prior knowledge masked. Defaults to None.
        """
        self.probe_group = probe_group
        self.features = features
        self.sim_func = CosineSimilarity().cuda()
        self.mask = mask
        
    def best(self, target, with_neighbor=True):
        similarity = self.sim_func(target, self.features)
        if with_neighbor:
            neighbors = similarity[np.asarray(self.probe_group.neighbor)]
            neighbors = neighbors.mean(-1)
            similarity += neighbors
        if self.mask is not None:
            # similarity - min() to keep minimum = 0, then set elements out of mask to 0 too
            similarity = self.mask * (similarity - similarity.min())
            hit_index = similarity.argmax()
        return hit_index
