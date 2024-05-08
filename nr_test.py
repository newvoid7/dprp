import os
import random
import time

import torch
from torch import nn
import neural_renderer as nr
import trimesh
import numpy as np
import cv2

from dataloaders import set_fold
from paths import DATASET_DIR, ALL_CASES, MESH_FILENAME, PROBE_FILENAME, RESULTS_DIR
from probe import ProbeGroup
from utils import cosine_similarity


class NRRenderer(nn.Module):
    def __init__(self, filename_obj, init_eye, init_up, meshes=None, path_ref_image=None) -> None:
        super(NRRenderer, self).__init__()
        orig_mesh = trimesh.load(filename_obj)
        vertices = []
        faces = []
        textures = []
        index = 0
        v_base = 0
        texture_size = 2
        for m in orig_mesh.geometry.values():
            if (meshes is None) or (index in meshes):
                index += 1
                vertices.append(np.asarray(m.vertices, dtype=np.float32))
                faces.append(np.asarray(m.faces) + v_base)
                t = np.asarray(m.visual.material.baseColorFactor[:-1]).astype(np.float32) / 255
                t = np.tile(t, [m.faces.shape[0], texture_size, texture_size, texture_size, 1])
                # t = t[np.newaxis, ...].repeat(m.faces.shape[0] * texture_size ** 3, axis=0)
                # t = t.reshape(m.faces.shape[0], texture_size, texture_size, texture_size, 3)
                textures.append(t)
                v_base += len(m.vertices)
            else:
                index += 1
                continue
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)
        textures = np.concatenate(textures, axis=0)
        vertices = torch.from_numpy(vertices)[None, ...]
        faces = torch.from_numpy(faces)[None, ...]
        textures = torch.from_numpy(textures)[None, ...]
        
        # normalize
        v_max = vertices.max(dim=1)[0]
        v_min = vertices.min(dim=1)[0]
        center = (v_max + v_min) / 2
        vertices -= center
        max_range = (v_max - v_min).max()
        vertices /= max_range / 2
        
        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)
        self.register_buffer('textures', textures)

        # load reference image
        image_ref = torch.from_numpy(cv2.imread(path_ref_image).transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)
        image_ref / 255
        self.register_buffer('image_ref', image_ref)

        # camera parameters
        self.camera_position = nn.Parameter(torch.tensor(init_eye, dtype=torch.float32))
        # self.camera_direction = torch.tensor(init_dir, dtype=torch.float32)
        self.camera_up = torch.tensor(init_up, dtype=torch.float32).cuda()

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', image_size=512)
        renderer.eye = self.camera_position
        # renderer.camera_direction = self.camera_direction
        renderer.camera_up = self.camera_up
        self.renderer = renderer

        
    def forward(self):
        image = self.renderer(self.vertices, self.faces, self.textures, mode='label').flip(dims=[1])
        # cv2.imwrite('tmp_nr.png', (image.detach().cpu().numpy().squeeze().transpose((1, 2, 0)) * 255).astype(np.uint8))
        loss = torch.sum((image - self.image_ref[None, :, :]) ** 2)
        return loss
    