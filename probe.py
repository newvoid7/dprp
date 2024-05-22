from ctypes import ArgumentError
import json
import math
import os

import cv2
import numpy as np
from render import PRRenderer
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (timer, 
                   quaternion_from_view_up, 
                   quaternion_rotate_vec, 
                   quaternion_to_matrix, 
                   cartesian_product, 
                   cosine_similarity, 
                   resize_to_fit,
                   stitch_images)
import paths


os.environ['PYOPENGL_PLATFORM'] = 'egl'

class Probe:
    """
    Attributes:
        mesh_path (str): the mesh file that was rendered.
        camera_position (np.ndarray): shape of (3,)
        camera_quaternion (np.ndarray): shape of (4,), use a quaternion to show how the camera is rotated.
            Note that the default view direction is [0, 0, -1] and view up is [0, 1, 0].
        render (np.ndarray): out from PRRenderer, ready for cv2, (H, W, [BGR]), dtype=np.uint8, values in [0, 255]
    """
    DEFAULT_ORIENTATION = [0, 0, -1]
    DEFAULT_UP = [0, 1, 0]
    
    def __init__(self, mesh_path, eye=None, focus=None, up=None, 
                 camera_position=None, camera_quaternion=None, render=None):
        self.mesh_path = mesh_path
        self.render = render
        if camera_position is not None and camera_quaternion is not None:
            self.camera_position = camera_position
            self.camera_quaternion = camera_quaternion
            return
        if isinstance(eye, list):
            self.camera_position = np.asarray(eye).astype(float)
        elif isinstance(eye, np.ndarray):
            self.camera_position = eye.astype(float)
        else:
            self.camera_position = None
        if eye is not None and focus is not None and up is not None:
            self.camera_quaternion = quaternion_from_view_up(
                new_view=[focus[i] - eye[i] for i in range(3)], new_up=up,
                orig_view=Probe.DEFAULT_ORIENTATION, orig_up=Probe.DEFAULT_UP
            )
        else:
            self.camera_quaternion = None
        return

    def get_eye(self):
        return self.camera_position
    
    def get_radius(self):
        return np.linalg.norm(self.camera_position)

    def get_orientation(self):
        return quaternion_rotate_vec(np.asarray(Probe.DEFAULT_ORIENTATION, dtype=float), self.camera_quaternion)

    def get_up(self):
        return quaternion_rotate_vec(np.asarray(Probe.DEFAULT_UP, dtype=float), self.camera_quaternion)
    
    def get_spcoord_dict(self):
        """
        Sphere coordinate of camera position
        Returns:
            dict: including 3 items specifying radius, azimuth and elevation
        """
        radius = np.linalg.norm(self.camera_position)
        azimuth = np.arctan2(self.camera_position[1], self.camera_position[0]) / np.pi * 180
        azimuth = (azimuth + 360) % 360
        zenith = np.arccos(self.camera_position[2] / radius) / np.pi * 180
        return {
            # azimuth in [0, 360) degrees, elevation in [0, 180] degrees.
            'radius': radius,
            'azimuth': azimuth,
            'zenith': zenith
        }

    def get_matrix(self):
        rot_mat = quaternion_to_matrix(self.camera_quaternion)
        final_mat = np.concatenate([rot_mat, self.camera_position[..., np.newaxis]], axis=1, dtype=float)
        final_mat = np.concatenate([final_mat, np.asarray([[0, 0, 0, 1]])], axis=0, dtype=float)
        return final_mat


class ProbeGroup:
    def __init__(self, mesh_path=None, deserialize_path=None) -> None:
        """ Default focus is [0, 0, 0], default up is z axis. Right hand coordinate system.
        The azimuth is from x-axis to y-axis.
        Args:
            mesh_path:
            deserialize_path: 
        """
        self.mesh_path = mesh_path
        self.probes = []
        self.amount = 0
        self.render_size = None
        self.grid_type = None
        self.neighbor = []
        self.draw_mesh = [0, 1]
        if mesh_path is not None and deserialize_path is None:
            self.generate()
        elif deserialize_path is not None:
            self.deserialize(deserialize_path)

    @timer
    def generate(self, amount=600, radius=2.5, grid_type='fib'):
        """
        Generate flat-textured render from the azimuth and elevation samples
        """
        self.probes = []
        renderer = PRRenderer(self.mesh_path, out_size=self.render_size)
        if grid_type == 'sph':
            # use sphere coordinate system
            z_total = int((amount / 2) ** 0.5)
            a_total = amount / z_total
            azimuth_sample = np.linspace(start=0, stop=2*np.pi, num=a_total, endpoint=False)
            zenith_sample = np.linspace(start=np.pi/(z_total + 1), stop=np.pi, num=z_total, endpoint=False)
            zenith, azimuth = np.meshgrid(zenith_sample, azimuth_sample)
            azimuth = azimuth.flatten()
            zenith = zenith.flatten()
            positions = np.stack([
                radius * np.sin(zenith) * np.cos(azimuth),
                radius * np.sin(zenith) * np.sin(azimuth),
                radius * np.cos(zenith)
            ], axis=1)
            product = cosine_similarity(*cartesian_product(positions, positions))
            product = product.reshape(len(positions), len(positions))
            self.neighbor = product.argsort(axis=1)[..., -5:-1]
        elif grid_type == 'fib':
            # Spherical Fibonacci grid
            n = amount
            golden_ratio = (1 + np.sqrt(5)) / 2
            i = np.arange(n)
            phi = 2 * np.pi * (i / golden_ratio % 1)
            theta = np.arccos(1 - (2 * i + 1) / n)
            positions = np.stack([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta)
            ], axis=1)
            product = cosine_similarity(*cartesian_product(positions, positions), dim=1)
            product = product.reshape(len(positions), len(positions))
            self.neighbor = product.argsort(axis=1)[..., -5:-1]
        else:
            raise ArgumentError(f'Expect gird type in [sph|fib], get {grid_type} instead.')
        for p in tqdm(iterable=positions, desc='Generating probes'):
            probe = Probe(self.mesh_path, eye=p, focus=[0, 0, 0], up=[0, 0, 1], render=None)
            label = renderer.render(probe.get_matrix(), mode='FLAT', draw_mesh=self.draw_mesh)[..., ::-1]
            probe.render = label
            self.probes.append(probe)
        self.amount = len(self.probes)
        self.render_size = self.probes[0].render.shape[:-1]
        self.grid_type = grid_type

    @timer
    def visualize(self, result_dir, stitch=True, cell_width=200, gap=5):
        """
        Visualize the render of probes to a single image (if stitch) or separate images.
        Meanwhile, write the render parameters to a json file.
        Args:
            result_dir (str): the output image is written to result_dir/vis.png
            stitch (bool): put all the render together, try to make output as square as possible,
                also make the number of columns times of 10 to help count.
            cell_width (int): indicate how many pixels a cell (if stitch) 
                or a probe sampled image (if not stitch)'s width is.
            gap (int): only effective if stitch is True
        """
        # draw images rendered from probes
        if stitch:
            resized = [resize_to_fit(p.render, out_size=cell_width-gap) for p in self.probes]
            img, coords = stitch_images(resized, gap=gap)
            for i in range(self.amount):
                cv2.putText(img, text=str(i), org=(coords[i][1] + cell_width - 55, coords[i][0] + cell_width),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.7, color=(0, 0, 255))
            cv2.imwrite(os.path.join(result_dir, 'vis.png'), img)
        else:
            for i, p in enumerate(self.probes):
                image = cv2.resize(p.render, dsize=(cell_width, cell_width))
                cv2.imwrite(os.path.join(result_dir, f'probe_{i}.jpg'), image)
        # save position info
        params = {
            'mesh_path': self.mesh_path,
            'draw_mesh': self.draw_mesh,
            'total': self.amount
        }
        for i, p in enumerate(self.probes):
            params[i] = {
                'position': p.camera_position.tolist(),
                'quaternion': p.camera_quaternion.tolist()
            }
            
        with open(os.path.join(result_dir, 'info.json'), 'w') as f:
            json.dump(params, f, indent=4)
        # draw sample figure
        radius = np.linalg.norm(self.probes[0].camera_position) - 0.01
        fig = plt.figure(figsize=(10, 10), dpi=120)
        ax = fig.add_subplot(projection='3d')
        ax.xaxis.set_pane_color((0, 0, 0, 0))
        ax.yaxis.set_pane_color((0, 0, 0, 0))
        ax.zaxis.set_pane_color((0, 0, 0, 0))
        u = np.linspace(start=0, stop=2*np.pi, num=100)
        v = np.linspace(start=-np.pi/2, stop=np.pi/2, num=100)
        x = radius * np.outer(np.cos(v), np.cos(u))
        y = radius * np.outer(np.cos(v), np.sin(u))
        z = radius * np.outer(np.sin(v), np.ones(u.shape))
        ax.plot_surface(x, y, z, cmap='gray')
        probe_coords = np.asarray([p.camera_position for p in self.probes])
        ax.scatter(probe_coords[..., 0], probe_coords[..., 1], probe_coords[..., 2], c='red', s=2)
        ax.set_aspect('equal')
        fig.savefig(os.path.join(result_dir, 'sample.png'), bbox_inches='tight', pad_inches=0.0)
        plt.close()
        return

    @timer
    def serialize(self, write_path):
        """
        Save to file, convert to a dict of np.ndarray to compress efficiently
        """
        prepared_dict = {
            'mesh_path': self.mesh_path,
            'grid_type': self.grid_type,
            'camera_position': np.stack([p.camera_position for p in self.probes], axis=0),
            'camera_quaternion': np.stack([p.camera_quaternion for p in self.probes], axis=0),
            'render': np.stack([p.render for p in self.probes], axis=0),
            'neighbor': self.neighbor
        }
        np.savez_compressed(write_path, **prepared_dict)

    @timer
    def deserialize(self, read_path):
        """
        Read from file
        """
        read_dict = np.load(read_path)
        self.mesh_path = str(read_dict['mesh_path'])
        camera_positions = read_dict['camera_position']
        camera_quaternions = read_dict['camera_quaternion']
        renders = read_dict['render']
        self.amount = len(renders)
        self.probes = [Probe(mesh_path=self.mesh_path,
                            camera_position=camera_positions[i],
                            camera_quaternion=camera_quaternions[i],
                            render=renders[i])
                        for i in range(self.amount)]
        self.grid_type = str(read_dict['grid_type'])
        self.neighbor = read_dict['neighbor']
        self.render_size = self.probes[0].render.shape[:-1]

    def sparse(self, factor=4):
        """
        Decrease probes by a factor (azimuth and zenith together),
        note that the factor > 1, and the final number of probes is about original / factor
        Args:
            factor: 
        Returns:
            list of Probe:
        """
        if factor is None:
            return
        new_amount = int(self.amount / factor)
        radius = self.probes[0].get_radius()
        self.generate(amount=new_amount, radius=radius, grid_type=self.grid_type)


if __name__ == '__main__':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    base_dir = paths.DATASET_DIR
    for case in paths.ALL_CASES:
        case_result_dir = os.path.join(paths.RESULTS_DIR, case)
        os.makedirs(case_result_dir, exist_ok=True)
        pg = ProbeGroup(
            mesh_path=os.path.join(base_dir, case, paths.MESH_FILENAME)
        )
        pg.serialize(os.path.join(case_result_dir, paths.PROBE_FILENAME))
        pg.visualize(case_result_dir)
        # probes = deserialize_probes(os.path.join(case_result_dir, paths.PROBE_FILENAME))
        print(f'Case {case} is OK.')
