import os
import random

import cv2
from torchvision import transforms
import torch.nn.functional as nnf
import torch

from utils import crop_and_resize_square, cv2_to_tensor
from network.transform import Affine2dTransformer

class AgentTask:
    """ Simulate the noisy context in intraoperative images. Should only apply on segmented images.
    """
    def __init__(self, occlusion_dir):
        occlusion_paths = [os.path.join(occlusion_dir, fn) for fn in os.listdir(occlusion_dir)
                           if fn.endswith('.png') or fn.endswith('.jpg')]
        self.occlusions = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in occlusion_paths]
        self.occlusions = [crop_and_resize_square(o, out_size=512) for o in self.occlusions]
        self.occlusions = [cv2_to_tensor(o) for o in self.occlusions]
        # self.transformer = Affine2dTransformer()
        self.real_params = None

    def apply(self, i):
        """
        Actually the agent task should be rendering from a near position, but we use 2d data augmentation instead.
        Because profen reduces 2 DoF, the agent task should reduce the remaining 4 DoF (3 translation and 1 rotation).
        Also, in endoscopic images, some mask (due to the viewport or something else) exist, so we use a mask to help.
        Args:
            i (torch.Tensor): shape of (B, C, H, W)
        Returns:
            torch.Tensor: shape of (B, C, H, W)
        """
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=40, interpolation=transforms.InterpolationMode.NEAREST),
            # for RandomResizedCrop, the scale is based on area
            transforms.RandomResizedCrop(size=512, scale=(0.2, 1.1), ratio=(1.0, 1.0),
                                         interpolation=transforms.InterpolationMode.NEAREST)
        ])
        i = transform(i)
        # deprecated
        # params = torch.rand((i.size(0), 4)).to(i.device)
        # # tx, ty: -0.5 ~ 0.5 -> 0.25 ~ 0.75
        # params[:, 0: 2] = params[:, 0: 2] * 0.5 + 0.25
        # # rot: -pi/3 ~ pi/3 -> 1/3 ~ 2/3
        # params[:, 2] = params[:, 2] * (1 / 3.0) + (1 / 3.0)
        # # scale: 8/9 ~ 5 -> 0.46 ~ 1
        # params[:, 3] = params[:, 3] * 0.54 + 0.46
        # self.real_params = params
        # i = self.transformer(i, params)
        hw = i.size()[-2:]
        mask = torch.ones(hw)
        for occ in self.occlusions:
            if random.random() > 0.5:           # each mask has a random chance to be applied
                mask *= occ if occ.size() == hw else nnf.interpolate(occ, size=hw)
        i *= mask.to(i.device)
        return i
    
    
if __name__ == '__main__':
    # for test
    import paths
    from utils import cv2_to_tensor, tensor_to_cv2
    from probe import ProbeGroup
    agent_task = AgentTask(paths.MASK_DIR)
    pg = ProbeGroup(deserialize_path=os.path.join(paths.RESULTS_DIR, paths.ALL_CASES[0], paths.PROBE_FILENAME))
    in_ = pg.probes[10].render
    cv2.imwrite('before.png', in_)
    in_ = cv2_to_tensor(in_).unsqueeze(0)
    out_ = agent_task.apply(in_)
    params = agent_task.real_params
    out_ = tensor_to_cv2(out_.squeeze())
    cv2.imwrite('after.png', out_)
    print('params: {}'.format(params))
    