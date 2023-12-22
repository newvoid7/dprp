from torch.nn import CosineSimilarity
import numpy as np

from utils import time_it


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
        
    @time_it
    def best(self, target, neighbor_factor=0.5):
        """Find the best matching feature. Considering neighbors.
        Args:
            target (torch.Tensor): features of target
            neighbor_factor (float, optional): How important the neighbor features are. 
                Set to zero to omit neighbors. Defaults to 0.5.
        Returns:
            int: the index of the best matching feature in the probe group.
        """
        similarity = self.sim_func(target, self.features)
        if neighbor_factor != 0:
            neighbors = similarity[np.asarray(self.probe_group.neighbor)]
            neighbors = neighbors.mean(-1)
            similarity = (1 - neighbor_factor) * similarity + neighbor_factor * neighbors
        if self.mask is not None:
            # similarity - min() to keep minimum = 0, then set elements out of mask to 0 too
            similarity = self.mask * (similarity - similarity.min())
        hit_index = similarity.argmax()
        return hit_index
