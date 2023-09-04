import json
import math
import os
import pickle

import cv2
import numpy as np
from render import PRRenderer

from utils import time_it, quaternion_from_view_up, quaternion_rotate_vec, quaternion_to_matrix
import paths


DEFAULT_ORIENTATION = [0, 0, -1]
DEFAULT_UP = [0, 1, 0]


class Probe:
    """
    Attributes:
        mesh_path (str): the mesh file that was rendered.
        camera_position (np.ndarray): shape of (3,)
        camera_quaternion (np.ndarray): shape of (4,), use a quaternion to show how the camera is rotated.
            Note that the default view direction is [0, 0, -1] and view up is [0, 1, 0].
        render (np.ndarray): out from PRRenderer, ready for cv2, (H, W, [BGR]), dtype=np.uint8, values in [0, 255]
    """
    def __init__(self, mesh_path, eye, focus, up, render):
        self.mesh_path = mesh_path
        if isinstance(eye, list):
            self.camera_position = np.asarray(eye).astype(float)
        elif isinstance(eye, np.ndarray):
            self.camera_position = eye.astype(float)
        self.camera_quaternion = quaternion_from_view_up(
            new_view=[focus[i] - eye[i] for i in range(3)], new_up=up,
            orig_view=DEFAULT_ORIENTATION, orig_up=DEFAULT_UP
        )
        self.render = render
        return

    def img_write(self, path):
        cv2.imwrite(path, self.render)
        return

    def get_eye(self):
        return self.camera_position

    def get_orientation(self):
        return quaternion_rotate_vec(np.asarray(DEFAULT_ORIENTATION, dtype=float), self.camera_quaternion)

    def get_up(self):
        return quaternion_rotate_vec(np.asarray(DEFAULT_UP, dtype=float), self.camera_quaternion)

    def get_matrix(self):
        rot_mat = quaternion_to_matrix(self.camera_quaternion)
        final_mat = np.concatenate([rot_mat, self.camera_position[..., np.newaxis]], axis=1, dtype=float)
        final_mat = np.concatenate([final_mat, np.asarray([[0, 0, 0, 1]])], axis=0, dtype=float)
        return final_mat

    def get_render_height(self):
        return self.render.shape[0]

    def get_render_width(self):
        return self.render.shape[1]


def visualize_probes(probe_list, result_dir, stitch=True, cell_width=100, gap=5):
    """
    Visualize the render of probes to a single image (if stitch) or separate images.
    Meanwhile, write the render parameters to a json file.
    Args:
        probe_list (list of Probe):
        result_dir (str): the output image is written to result_dir/vis.png
        stitch (bool): put all the render together, try to make output as square as possible,
            also make the number of columns times of 10 to help count.
        cell_width (int): only effective if stitch is True, indicate how big a render is by pixels.
        gap (int): only effective if stitch is True
    """
    params = {}
    if stitch:
        w_num = round((len(probe_list) ** 0.5) / 10) * 10
        h_num = math.ceil(len(probe_list) / w_num)
        pic_w = cell_width - gap
        pic_h = round(pic_w / probe_list[0].get_render_width() * probe_list[0].get_render_height())
        cell_height = pic_h + gap
        out_width = cell_width * w_num
        out_height = cell_height * h_num
        img = np.zeros((out_height, out_width, 3)).astype(np.uint8)
        for i, p in enumerate(probe_list):
            params[i] = {
                'mesh_path': p.mesh_path,
                'eye': p.get_eye().tolist(),
                'orientation': p.get_orientation().tolist(),
                'up': p.get_up().tolist(),
            }
            h_count = i // w_num
            w_count = i - w_num * h_count
            h_coord = h_count * cell_height
            w_coord = w_count * cell_width
            img[h_coord: h_coord + pic_h, w_coord: w_coord + pic_w, ...] = cv2.resize(
                p.render,
                dsize=(pic_w, pic_h)
            )
            cv2.putText(img, text=str(i), org=(w_coord + pic_w + gap - 55, h_coord + pic_h + gap),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.7, color=(0, 0, 255))
        cv2.imwrite(os.path.join(result_dir, 'vis.png'), img)
    else:
        for i, p in enumerate(probe_list):
            params[i] = {
                'mesh_path': p.mesh_path,
                'eye': p.get_eye().tolist(),
                'orientation': p.get_orientation().tolist(),
                'up': p.get_up().tolist(),
            }
            p.img_write(os.path.join(result_dir, 'sample_{}.jpg'.format(i)))
    with open(os.path.join(result_dir, 'info.json'), 'w') as f:
        json.dump(params, f, indent=4)
    return


def serialize_probes(write_path, probe_list: list):
    """
    Save to file
    """
    with open(write_path, 'wb') as f:
        pickle.dump(probe_list, f)


def deserialize_probes(read_path):
    """
    Read from file
    """
    with open(read_path, 'rb') as f:
        probe_list = pickle.load(f)
    return probe_list


@time_it
def generate_probes(mesh_path=None, radius=2.7, azimuth_sample=None, elevation_sample=None, dump_path=None):
    """
    Generate flat-textured render from the azimuth and elevation samples,
    default focus is [0, 0, 0], default up is z axis. Right hand coordinate system.
    The azimuth is from x-axis to y-axis.
    Args:
        mesh_path:
        radius:
        azimuth_sample:
        elevation_sample:
        dump_path:
    Returns:
        list of Probe:
    """
    k = 0
    probes = []
    renderer = PRRenderer(mesh_path)
    if azimuth_sample is None:
        azimuth_sample = [a / 180 * math.pi for a in range(0, 360, 10)]
    if elevation_sample is None:
        elevation_sample = [a / 180 * math.pi for a in range(-80, 90, 10)]
    for azimuth in azimuth_sample:
        for elevation in elevation_sample:
            position = [radius * math.cos(elevation) * math.cos(azimuth),
                        radius * math.cos(elevation) * math.sin(azimuth),
                        radius * math.sin(elevation)]
            probe = Probe(mesh_path, eye=position, focus=[0, 0, 0], up=[0, 0, 1], render=None)
            label = renderer.render(probe.get_matrix(), mode='FLAT')[..., ::-1]
            probe.render = label
            probes.append(probe)
            k += 1
            print('Generated probe [{}/{}]'.format(k, len(azimuth_sample) * len(elevation_sample)))
    if dump_path is not None:
        serialize_probes(dump_path, probes)
    return probes


def ablation_num_of_probes(probe_list, factor=2):
    """
    Decrease probes by a factor (azimuth and elevation together),
    note that the factor > 1, and the final number of probes is about original / factor**2
    Args:
        probe_list (list of Probe): azimuth in range(0, 360, 10), elevation in (-80, 90, 10)
        factor:
    Returns:
        list of Probe:
    """
    if factor is None:
        return probe_list
    ret_list = []
    for i, p in enumerate(probe_list):
        if (i // 17) % factor == 0 and (i % 17 % factor) == 0:
            ret_list.append(p)
    return ret_list


if __name__ == '__main__':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    base_dir = paths.DATASET_DIR
    for case in paths.ALL_CASES:
        case_result_dir = os.path.join(paths.RESULTS_DIR, case)
        os.makedirs(case_result_dir, exist_ok=True)
        case_probes = generate_probes(
            mesh_path=os.path.join(base_dir, case, paths.MESH_FILENAME),
            dump_path=os.path.join(case_result_dir, paths.PROBE_FILENAME)
        )
        visualize_probes(case_probes, case_result_dir)
        print('Case {} is OK.'.format(case))
